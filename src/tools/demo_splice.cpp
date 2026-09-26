#include <base/dbg.h>
#include <base/logger.h>
#include <base/mem.h>
#include <base/os.h>
#include <base/str.h>

#include <engine/shared/demo.h>
#include <engine/shared/network.h>
#include <engine/shared/snapshot.h>
#include <engine/storage.h>

#include <generated/protocol.h>

#include <algorithm>
#include <memory>
#include <vector>

static const char *TOOL_NAME = "demo_splice";

// Moves the absolute ticks inside a snapshot along with the snapshot itself.
// A part is written under new tick numbers, and an item that kept the numbers
// of its own recording is read as something that is happening now: a
// character drawn firing for the rest of the demo, a projectile that starts
// over every tick, a freeze that never ends.
static void ShiftTicks(CSnapshot *pSnapshot, int Delta)
{
	if(Delta == 0)
		return;
	for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
	{
		// The listener shifts its own copy of the snapshot
		int *pItemData = const_cast<int *>(pSnapshot->GetItem(Index)->Data());
		const size_t Size = pSnapshot->GetItemSize(Index);
		switch(pSnapshot->GetItemType(Index))
		{
		case NETOBJTYPE_CHARACTER:
		{
			if(Size < sizeof(CNetObj_Character))
				break;
			CNetObj_Character *pCharacter = (CNetObj_Character *)pItemData;
			pCharacter->m_Tick += Delta;
			pCharacter->m_AttackTick += Delta;
			break;
		}
		case NETOBJTYPE_PROJECTILE:
		{
			if(Size < sizeof(CNetObj_Projectile))
				break;
			((CNetObj_Projectile *)pItemData)->m_StartTick += Delta;
			break;
		}
		case NETOBJTYPE_LASER:
		{
			if(Size < sizeof(CNetObj_Laser))
				break;
			((CNetObj_Laser *)pItemData)->m_StartTick += Delta;
			break;
		}
		case NETOBJTYPE_DDNETPROJECTILE:
		{
			if(Size < sizeof(CNetObj_DDNetProjectile))
				break;
			((CNetObj_DDNetProjectile *)pItemData)->m_StartTick += Delta;
			break;
		}
		case NETOBJTYPE_DDNETLASER:
		{
			if(Size < sizeof(CNetObj_DDNetLaser))
				break;
			((CNetObj_DDNetLaser *)pItemData)->m_StartTick += Delta;
			break;
		}
		case NETOBJTYPE_DDNETCHARACTER:
		{
			if(Size < sizeof(CNetObj_DDNetCharacter))
				break;
			CNetObj_DDNetCharacter *pCharacter = (CNetObj_DDNetCharacter *)pItemData;
			// Not every value in these is a tick: no freeze is 0, a deep
			// freeze is -1 and no ninja is -1, and a tick that would land on
			// one of those after the shift is kept off it
			if(pCharacter->m_FreezeEnd > 0)
				pCharacter->m_FreezeEnd = std::max(pCharacter->m_FreezeEnd + Delta, 1);
			if(pCharacter->m_FreezeStart > 0)
				pCharacter->m_FreezeStart = std::max(pCharacter->m_FreezeStart + Delta, 1);
			if(pCharacter->m_NinjaActivationTick >= 0)
				pCharacter->m_NinjaActivationTick = std::max(pCharacter->m_NinjaActivationTick + Delta, 0);
			break;
		}
		case NETOBJTYPE_GAMEINFO:
		{
			if(Size < sizeof(CNetObj_GameInfo))
				break;
			((CNetObj_GameInfo *)pItemData)->m_RoundStartTick += Delta;
			break;
		}
		default:
			break;
		}
	}
}

// Writes what it is given into the output demo, shifting the ticks so this
// part continues where the last one ended.
class CSpliceListener : public CDemoPlayer::IListener
{
	CDemoPlayer *m_pDemoPlayer;
	CDemoRecorder *m_pDemoRecorder;
	int m_FirstTick = -1;
	int m_Offset;
	int m_LastTick = -1;
	int m_NumSnapshots = 0;
	CSnapshotBuffer m_Snapshot;

public:
	CSpliceListener(CDemoPlayer *pDemoPlayer, CDemoRecorder *pDemoRecorder, int Offset) :
		m_pDemoPlayer(pDemoPlayer), m_pDemoRecorder(pDemoRecorder), m_Offset(Offset)
	{
	}

	void OnDemoPlayerSnapshot(void *pData, int Size) override
	{
		const int Tick = m_pDemoPlayer->Info()->m_Info.m_CurrentTick;
		if(m_FirstTick < 0)
			m_FirstTick = Tick;
		// A demo the recorder accepts has strictly rising ticks, a part that
		// starts where the last one ended keeps that
		m_LastTick = Tick - m_FirstTick + m_Offset;
		// The player reads the next snapshot as a delta against this one, so
		// the shifted ticks go into a copy of it
		dbg_assert(Size >= 0 && (size_t)Size <= sizeof(m_Snapshot.m_aData), "snapshot does not fit");
		mem_copy(m_Snapshot.m_aData, pData, Size);
		ShiftTicks(m_Snapshot.AsSnapshot(), m_LastTick - Tick);
		m_pDemoRecorder->RecordSnapshot(m_LastTick, m_Snapshot.m_aData, Size);
		m_NumSnapshots++;
	}

	// Messages belong to the tick that was written last, which is where the
	// recorder puts them, so they need no shifting of their own
	void OnDemoPlayerMessage(void *pData, int Size) override
	{
		if(m_LastTick >= 0)
			m_pDemoRecorder->RecordMessage(pData, Size);
	}

	int FirstTick() const { return m_FirstTick; }
	int LastTick() const { return m_LastTick; }
	int NumSnapshots() const { return m_NumSnapshots; }
};

static std::unique_ptr<CSnapshotDelta> CreateSnapshotDelta()
{
	std::unique_ptr<CSnapshotDelta> pResult = std::make_unique<CSnapshotDelta>();
	CNetObjHandler NetObjHandler;
	for(int i = 0; i < NUM_NETOBJTYPES; i++)
	{
		pResult->SetStaticsize(i, NetObjHandler.GetObjSize(i));
	}
	return pResult;
}

// Plays one input into the recorder. The first one also decides the map the
// output demo is of, the ones after it have to be of the same map.
static bool Splice(IStorage *pStorage, CSnapshotDelta *pSnapshotDelta, const char *pPath,
	CDemoRecorder *pRecorder, const SHA256_DIGEST *pWantedSha256, int Offset, int *pLastTick)
{
	CDemoPlayer DemoPlayer(pSnapshotDelta, nullptr, false);
	if(DemoPlayer.Load(pStorage, nullptr, pPath, IStorage::TYPE_ALL_OR_ABSOLUTE) != 0)
	{
		log_error(TOOL_NAME, "Failed to load demo '%s'", pPath);
		return false;
	}
	if(DemoPlayer.IsSixup())
	{
		log_error(TOOL_NAME, "0.7 demos are not supported");
		DemoPlayer.Stop();
		return false;
	}
	const CMapInfo *pMapInfo = DemoPlayer.GetMapInfo();
	if(!pMapInfo->m_Sha256.has_value())
	{
		log_error(TOOL_NAME, "Demo '%s' has no map SHA256", pPath);
		DemoPlayer.Stop();
		return false;
	}
	if(pWantedSha256 != nullptr && pMapInfo->m_Sha256.value() != *pWantedSha256)
	{
		log_error(TOOL_NAME, "Demo '%s' is of another map than the first one", pPath);
		DemoPlayer.Stop();
		return false;
	}
	CSpliceListener Listener(&DemoPlayer, pRecorder, Offset);
	DemoPlayer.SetListener(&Listener);
	DemoPlayer.Play();
	while(DemoPlayer.IsPlaying())
	{
		DemoPlayer.Update(false);
		if(DemoPlayer.Info()->m_Info.m_Paused)
			break;
	}
	DemoPlayer.Stop();
	if(Listener.NumSnapshots() == 0)
	{
		log_error(TOOL_NAME, "Demo '%s' holds no snapshot", pPath);
		return false;
	}
	log_info(TOOL_NAME, "'%s': %d snapshots, ticks %d to %d of the output",
		pPath, Listener.NumSnapshots(), Offset, Listener.LastTick());
	*pLastTick = Listener.LastTick();
	return true;
}

int main(int argc, const char *argv[])
{
	std::unique_ptr<IStorage> pStorage = CreateLocalStorage();
	CCmdlineFix CmdlineFix(&argc, &argv);
	log_set_global_logger_default();
	CNetBase::Init();

	if(!pStorage)
	{
		log_error(TOOL_NAME, "Error creating local storage");
		return -1;
	}
	if(argc < 4)
	{
		log_error(TOOL_NAME, "Usage: %s <first.demo> <second.demo>... <out.demo>", TOOL_NAME);
		log_error(TOOL_NAME, "Joins demos of one map into a single demo, one part after the other.");
		log_error(TOOL_NAME, "A run that was saved and loaded again is recorded in two files, the");
		log_error(TOOL_NAME, "replay of its rank is both halves in a row. The parts keep their client");
		log_error(TOOL_NAME, "ids, the converter writes both halves with the same ones.");
		return -1;
	}
	const int NumInputs = argc - 2;
	const char *pOutputPath = argv[argc - 1];

	std::unique_ptr<CSnapshotDelta> pSnapshotDelta = CreateSnapshotDelta();

	// The output demo is of the map of the first part, taken from its header
	CDemoPlayer HeaderPlayer(pSnapshotDelta.get(), nullptr, false);
	if(HeaderPlayer.Load(pStorage.get(), nullptr, argv[1], IStorage::TYPE_ALL_OR_ABSOLUTE) != 0)
	{
		log_error(TOOL_NAME, "Failed to load demo '%s'", argv[1]);
		return -1;
	}
	const CMapInfo *pMapInfo = HeaderPlayer.GetMapInfo();
	const CDemoPlayer::CPlaybackInfo *pInfo = HeaderPlayer.Info();
	if(!pMapInfo->m_Sha256.has_value())
	{
		log_error(TOOL_NAME, "Demo '%s' has no map SHA256", argv[1]);
		HeaderPlayer.Stop();
		return -1;
	}
	const SHA256_DIGEST MapSha256 = pMapInfo->m_Sha256.value();
	CDemoRecorder DemoRecorder(pSnapshotDelta.get());
	unsigned char *pMapData = HeaderPlayer.GetMapData(pStorage.get());
	const int Error = DemoRecorder.Start(pStorage.get(), nullptr, pOutputPath, pInfo->m_Header.m_aNetversion,
		pMapInfo->m_aName, MapSha256, pMapInfo->m_Crc, pInfo->m_Header.m_aType, pMapInfo->m_Size, pMapData,
		nullptr, nullptr, nullptr);
	free(pMapData);
	HeaderPlayer.Stop();
	if(Error != 0)
	{
		log_error(TOOL_NAME, "Failed to start demo recorder for '%s'", pOutputPath);
		return -1;
	}

	int Offset = 0;
	int LastTick = -1;
	std::vector<int> vSeams;
	for(int i = 0; i < NumInputs; i++)
	{
		if(i > 0)
			vSeams.push_back(Offset);
		if(!Splice(pStorage.get(), pSnapshotDelta.get(), argv[1 + i], &DemoRecorder,
			   i == 0 ? nullptr : &MapSha256, Offset, &LastTick))
		{
			DemoRecorder.Stop(IDemoRecorder::EStopMode::REMOVE_FILE);
			return -1;
		}
		Offset = LastTick + 1;
	}
	// Where one part ends and the next begins, which is the /save and /load
	// of the run. The recorder only takes a marker of a tick it has written,
	// so they are added once everything is in.
	for(const int Seam : vSeams)
	{
		DemoRecorder.AddDemoMarker(Seam);
	}
	DemoRecorder.Stop(IDemoRecorder::EStopMode::KEEP_FILE);
	log_info(TOOL_NAME, "wrote '%s', %d parts covering %d ticks", pOutputPath, NumInputs, LastTick + 1);
	return 0;
}
