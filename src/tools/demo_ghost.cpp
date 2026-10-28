#include <base/logger.h>
#include <base/mem.h>
#include <base/os.h>
#include <base/str.h>

#include <engine/client/ghost.h>
#include <engine/kernel.h>
#include <engine/shared/demo.h>
#include <engine/shared/network.h>
#include <engine/shared/snapshot.h>
#include <engine/storage.h>

#include <generated/protocol.h>

#include <game/gamecore.h>

#include <memory>
#include <optional>
#include <vector>

static const char *TOOL_NAME = "demo_ghost";

// The items of a ghost file, as the client reads them in
// src/game/client/components/ghost.h
enum
{
	GHOSTDATA_TYPE_SKIN = 0,
	GHOSTDATA_TYPE_CHARACTER_NO_TICK,
	GHOSTDATA_TYPE_CHARACTER,
	GHOSTDATA_TYPE_START_TICK
};

struct CGhostSkin
{
	int m_aSkin[6];
	int m_UseCustomColor;
	int m_ColorBody;
	int m_ColorFeet;
};

struct CGhostCharacter
{
	int m_X;
	int m_Y;
	int m_VelX;
	int m_VelY;
	int m_Angle;
	int m_Direction;
	int m_Weapon;
	int m_HookState;
	int m_HookX;
	int m_HookY;
	int m_AttackTick;
	int m_Tick;
};

// Collects one tee's run the way the client records its own, see
// CGhost::GetGhostCharacter
class CGhostListener : public CDemoPlayer::IListener
{
	CDemoPlayer *m_pDemoPlayer;
	int m_ClientId;
	int m_StartTick;
	int m_FinishTick;

public:
	char m_aName[MAX_NAME_LENGTH] = "";
	std::optional<CGhostSkin> m_Skin;
	std::vector<CGhostCharacter> m_vPath;
	bool m_Failed = false;

	CGhostListener(CDemoPlayer *pDemoPlayer, int ClientId, int StartTick, int FinishTick) :
		m_pDemoPlayer(pDemoPlayer), m_ClientId(ClientId), m_StartTick(StartTick), m_FinishTick(FinishTick)
	{
	}

	void OnDemoPlayerSnapshot(void *pData, int Size) override
	{
		const int Tick = m_pDemoPlayer->Info()->m_Info.m_CurrentTick;
		if(Tick < m_StartTick || Tick > m_FinishTick || m_Failed)
			return;
		const CSnapshot *pSnapshot = (const CSnapshot *)pData;
		const CNetObj_Character *pCharacter = nullptr;
		const CNetObj_DDNetCharacter *pDDNetCharacter = nullptr;
		const CNetObj_ClientInfo *pClientInfo = nullptr;
		for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
		{
			if(pSnapshot->GetItem(Index)->Id() != m_ClientId)
				continue;
			const void *pItemData = pSnapshot->GetItem(Index)->Data();
			const size_t ItemSize = pSnapshot->GetItemSize(Index);
			switch(pSnapshot->GetItemType(Index))
			{
			case NETOBJTYPE_CHARACTER:
				if(ItemSize >= sizeof(CNetObj_Character))
					pCharacter = (const CNetObj_Character *)pItemData;
				break;
			case NETOBJTYPE_DDNETCHARACTER:
				if(ItemSize >= sizeof(CNetObj_DDNetCharacter))
					pDDNetCharacter = (const CNetObj_DDNetCharacter *)pItemData;
				break;
			case NETOBJTYPE_CLIENTINFO:
				if(ItemSize >= sizeof(CNetObj_ClientInfo))
					pClientInfo = (const CNetObj_ClientInfo *)pItemData;
				break;
			}
		}
		if(!m_Skin.has_value() && pClientInfo != nullptr)
		{
			IntsToStr(pClientInfo->m_aName, std::size(pClientInfo->m_aName), m_aName, sizeof(m_aName));
			CGhostSkin Skin;
			static_assert(sizeof(Skin.m_aSkin) == sizeof(pClientInfo->m_aSkin));
			mem_copy(Skin.m_aSkin, pClientInfo->m_aSkin, sizeof(Skin.m_aSkin));
			Skin.m_UseCustomColor = pClientInfo->m_UseCustomColor;
			Skin.m_ColorBody = pClientInfo->m_ColorBody;
			Skin.m_ColorFeet = pClientInfo->m_ColorFeet;
			m_Skin = Skin;
		}
		if(pCharacter == nullptr)
			return;
		// The client moves a character sent with an older tick on to the
		// snapshot's, which needs the map. The converter never sends one.
		if(pCharacter->m_Tick != Tick)
		{
			log_error(TOOL_NAME, "The character of %d at tick %d is of tick %d", m_ClientId, Tick, pCharacter->m_Tick);
			m_Failed = true;
			return;
		}
		CGhostCharacter Ghost;
		Ghost.m_X = pCharacter->m_X;
		Ghost.m_Y = pCharacter->m_Y;
		Ghost.m_VelX = pCharacter->m_VelX;
		Ghost.m_VelY = 0;
		Ghost.m_Angle = pCharacter->m_Angle;
		Ghost.m_Direction = pCharacter->m_Direction;
		Ghost.m_Weapon = pDDNetCharacter != nullptr && pDDNetCharacter->m_FreezeEnd != 0 ? WEAPON_NINJA : pCharacter->m_Weapon;
		Ghost.m_HookState = pCharacter->m_HookState;
		Ghost.m_HookX = pCharacter->m_HookX;
		Ghost.m_HookY = pCharacter->m_HookY;
		Ghost.m_AttackTick = pCharacter->m_AttackTick;
		Ghost.m_Tick = Tick;
		m_vPath.push_back(Ghost);
	}

	void OnDemoPlayerMessage(void *pData, int Size) override
	{
	}
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
	if(argc != 6)
	{
		log_error(TOOL_NAME, "Usage: %s <demo> <client id> <start tick> <finish tick> <out.gho>", TOOL_NAME);
		log_error(TOOL_NAME, "Writes the run of one tee in a demo as a ghost the client can race against.");
		log_error(TOOL_NAME, "The run is the ticks from the start line to the finish, its time is theirs.");
		return -1;
	}
	const int ClientId = str_toint(argv[2]);
	const int StartTick = str_toint(argv[3]);
	const int FinishTick = str_toint(argv[4]);
	if(ClientId < 0 || ClientId >= MAX_CLIENTS || FinishTick <= StartTick)
	{
		log_error(TOOL_NAME, "Invalid client id or ticks");
		return -1;
	}

	std::unique_ptr<CSnapshotDelta> pSnapshotDelta = CreateSnapshotDelta();
	CDemoPlayer DemoPlayer(pSnapshotDelta.get(), nullptr, false);
	if(DemoPlayer.Load(pStorage.get(), nullptr, argv[1], IStorage::TYPE_ALL_OR_ABSOLUTE) != 0)
	{
		log_error(TOOL_NAME, "Failed to load demo '%s'", argv[1]);
		return -1;
	}
	if(DemoPlayer.IsSixup())
	{
		log_error(TOOL_NAME, "0.7 demos are not supported");
		DemoPlayer.Stop();
		return -1;
	}
	const CMapInfo *pMapInfo = DemoPlayer.GetMapInfo();
	if(!pMapInfo->m_Sha256.has_value())
	{
		log_error(TOOL_NAME, "Demo '%s' has no map SHA256", argv[1]);
		DemoPlayer.Stop();
		return -1;
	}
	char aMapName[sizeof(pMapInfo->m_aName)];
	str_copy(aMapName, pMapInfo->m_aName);
	const SHA256_DIGEST MapSha256 = pMapInfo->m_Sha256.value();

	CGhostListener Listener(&DemoPlayer, ClientId, StartTick, FinishTick);
	DemoPlayer.SetListener(&Listener);
	DemoPlayer.Play();
	while(DemoPlayer.IsPlaying())
	{
		DemoPlayer.Update(false);
		if(DemoPlayer.Info()->m_Info.m_Paused)
			break;
	}
	DemoPlayer.Stop();
	if(Listener.m_Failed)
		return -1;
	if(Listener.m_vPath.empty() || !Listener.m_Skin.has_value())
	{
		log_error(TOOL_NAME, "Demo '%s' does not show %d between ticks %d and %d", argv[1], ClientId, StartTick, FinishTick);
		return -1;
	}
	// A recording that begins in the middle of a run has only its end
	if(Listener.m_vPath.front().m_Tick != StartTick)
	{
		log_error(TOOL_NAME, "Demo '%s' shows %d from tick %d, not from the start tick %d", argv[1], ClientId, Listener.m_vPath.front().m_Tick, StartTick);
		return -1;
	}

	std::unique_ptr<IKernel> pKernel(IKernel::Create());
	pKernel->RegisterInterface(pStorage.get(), false);
	CGhostRecorder Recorder;
	pKernel->RegisterInterface(static_cast<IGhostRecorder *>(&Recorder), false);
	Recorder.Init();
	if(Recorder.Start(argv[5], aMapName, MapSha256, Listener.m_aName) != 0)
		return -1;
	// A client on a DDNet server starts the ghosts one tick before the start
	// line (m_BugDDRaceGhost), its own as well as the ones it plays
	const int GhostStartTick = StartTick - 1;
	Recorder.WriteData(GHOSTDATA_TYPE_START_TICK, &GhostStartTick, sizeof(GhostStartTick));
	Recorder.WriteData(GHOSTDATA_TYPE_SKIN, &Listener.m_Skin.value(), sizeof(CGhostSkin));
	for(const CGhostCharacter &Character : Listener.m_vPath)
		Recorder.WriteData(GHOSTDATA_TYPE_CHARACTER, &Character, sizeof(Character));
	Recorder.Stop((int)Listener.m_vPath.size(), (FinishTick - StartTick) * 1000 / SERVER_TICK_SPEED);
	log_info(TOOL_NAME, "wrote '%s', %d ticks of '%s' on %s", argv[5], (int)Listener.m_vPath.size(), Listener.m_aName, aMapName);
	return 0;
}
