// Throwaway: prints what a demo says about its players at a tick.
#include <base/dbg.h>
#include <base/logger.h>
#include <base/os.h>
#include <base/str.h>

#include <engine/shared/demo.h>
#include <engine/shared/network.h>
#include <engine/shared/packer.h>
#include <engine/shared/snapshot.h>
#include <engine/shared/uuid_manager.h>
#include <engine/storage.h>

#include <generated/protocol.h>

#include <game/gamecore.h>

#include <cmath>
#include <memory>

static int g_WantTick = 0;
static int g_WantEnd = 0;

class CInspectListener : public CDemoPlayer::IListener
{
	CDemoPlayer *m_pDemoPlayer;
	int m_aTeams[MAX_CLIENTS] = {0};
	char m_aaNames[MAX_CLIENTS][MAX_NAME_LENGTH] = {{0}};
	int m_PreviousTick = -1;
	int m_NumSnapshots = 0;
	int m_NumMessages = 0;
	int m_NumExtended = 0;
	// Scan mode (-2): the previous two positions and the hook state of every
	// character, to find one-tick moves of 192 px or more that continue the
	// tick before (a speedup, not a placement)
	int m_aPosTick[MAX_CLIENTS] = {0};
	int m_aX[MAX_CLIENTS] = {0};
	int m_aY[MAX_CLIENTS] = {0};
	int m_aPrevX[MAX_CLIENTS] = {0};
	int m_aPrevY[MAX_CLIENTS] = {0};
	int m_aHook[MAX_CLIENTS] = {0};
	int m_Jumps = 0;
	int m_FalseJumps = 0;

public:
	int Jumps() const { return m_Jumps; }
	int FalseJumps() const { return m_FalseJumps; }
	CInspectListener(CDemoPlayer *pDemoPlayer) :
		m_pDemoPlayer(pDemoPlayer) {}

	int LastTick() const { return m_PreviousTick; }
	int NumSnapshots() const { return m_NumSnapshots; }
	int NumMessages() const { return m_NumMessages; }
	int NumExtended() const { return m_NumExtended; }

	void OnDemoPlayerSnapshot(void *pData, int Size) override
	{
		const int Tick = m_pDemoPlayer->Info()->m_Info.m_CurrentTick;
		m_NumSnapshots++;
		CSnapshot *pSnapshot = (CSnapshot *)pData;
		for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
		{
			const CSnapshotItem *pItem = pSnapshot->GetItem(Index);
			if(pSnapshot->GetItemType(Index) != NETOBJTYPE_CLIENTINFO)
				continue;
			const int Cid = pItem->Id() % MAX_CLIENTS;
			IntsToStr(((const CNetObj_ClientInfo *)pItem->Data())->m_aName, 4, m_aaNames[Cid], sizeof(m_aaNames[Cid]));
		}
		if(g_WantTick == -2)
		{
			for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
			{
				if(pSnapshot->GetItemType(Index) != NETOBJTYPE_CHARACTER)
					continue;
				const CSnapshotItem *pItem = pSnapshot->GetItem(Index);
				const int Cid = pItem->Id() % MAX_CLIENTS;
				const CNetObj_Character *pCharacter = (const CNetObj_Character *)pItem->Data();
				if(m_aPosTick[Cid] == Tick - 1)
				{
					const float Dx = pCharacter->m_X - m_aX[Cid];
					const float Dy = pCharacter->m_Y - m_aY[Cid];
					const float Dist = std::sqrt(Dx * Dx + Dy * Dy);
					if(Dist >= 6 * 32)
					{
						const float Ex = m_aX[Cid] + (m_aX[Cid] - m_aPrevX[Cid]) - pCharacter->m_X;
						const float Ey = m_aY[Cid] + (m_aY[Cid] - m_aPrevY[Cid]) - pCharacter->m_Y;
						const float Dev = std::sqrt(Ex * Ex + Ey * Ey);
						m_Jumps++;
						if(Dev < 6 * 32)
							m_FalseJumps++;
						printf("jump tick %d id %d '%s' dist %.0f dev %.0f hook %d -> %d\n", Tick, Cid, m_aaNames[Cid], Dist, Dev, m_aHook[Cid], pCharacter->m_HookState);
					}
				}
				m_aPrevX[Cid] = m_aPosTick[Cid] == Tick - 1 ? m_aX[Cid] : pCharacter->m_X;
				m_aPrevY[Cid] = m_aPosTick[Cid] == Tick - 1 ? m_aY[Cid] : pCharacter->m_Y;
				m_aX[Cid] = pCharacter->m_X;
				m_aY[Cid] = pCharacter->m_Y;
				m_aPosTick[Cid] = Tick;
				m_aHook[Cid] = pCharacter->m_HookState;
			}
			m_PreviousTick = Tick;
			return;
		}
		if(g_WantTick < 0)
		{
			if(m_PreviousTick >= 0 && Tick != m_PreviousTick + 1)
				printf("snapshot tick jump %d -> %d\n", m_PreviousTick, Tick);
			m_PreviousTick = Tick;
			return;
		}
		m_PreviousTick = Tick;
		if(Tick < g_WantTick || Tick > g_WantEnd)
			return;
		int NumCharacters = 0;
		for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
			if(pSnapshot->GetItemType(Index) == NETOBJTYPE_CHARACTER)
				NumCharacters++;
		printf("tick %d, %d items, %d characters\n", Tick, pSnapshot->NumItems(), NumCharacters);
		for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
		{
			const CSnapshotItem *pItem = pSnapshot->GetItem(Index);
			const int Type = pSnapshot->GetItemType(Index);
			const int Cid = pItem->Id() % MAX_CLIENTS;
			if(Type == NETOBJTYPE_CHARACTER)
			{
				const CNetObj_Character *pCharacter = (const CNetObj_Character *)pItem->Data();
				printf("  id %2d '%s' team %d char tick %d (%+d) pos %d,%d vel %d,%d hook %d at %d,%d tick %d player %d attack %d (%d ago) weapon %d\n",
					pItem->Id(), m_aaNames[Cid], m_aTeams[Cid], pCharacter->m_Tick, pCharacter->m_Tick - Tick,
					pCharacter->m_X, pCharacter->m_Y, pCharacter->m_VelX, pCharacter->m_VelY,
					pCharacter->m_HookState, pCharacter->m_HookX, pCharacter->m_HookY, pCharacter->m_HookTick, pCharacter->m_HookedPlayer,
					pCharacter->m_AttackTick, Tick - pCharacter->m_AttackTick, pCharacter->m_Weapon);
			}
			else if(Type == NETOBJTYPE_DDNETCHARACTER)
			{
				const CNetObj_DDNetCharacter *pDDNet = (const CNetObj_DDNetCharacter *)pItem->Data();
				printf("  id %2d ddnet flags 0x%x freeze %d..%d jumps %d tune %d\n",
					pItem->Id(), pDDNet->m_Flags, pDDNet->m_FreezeStart, pDDNet->m_FreezeEnd,
					pDDNet->m_Jumps, pDDNet->m_TuneZoneOverride);
			}
			else if(Type == NETOBJTYPE_PLAYERINFO)
			{
				const CNetObj_PlayerInfo *pInfo = (const CNetObj_PlayerInfo *)pItem->Data();
				printf("  id %2d playerinfo local %d team %d score %d\n",
					pItem->Id(), pInfo->m_Local, pInfo->m_Team, pInfo->m_Score);
			}
		}
	}

	void OnDemoPlayerMessage(void *pData, int Size) override
	{
		m_NumMessages++;
		CUnpacker Unpacker;
		Unpacker.Reset(pData, Size);
		if(Unpacker.GetInt() != 0)
			return;
		m_NumExtended++;
		const CUuid *pUuid = (const CUuid *)Unpacker.GetRaw(sizeof(CUuid));
		if(pUuid == nullptr || g_UuidManager.LookupUuid(*pUuid) != NETMSGTYPE_SV_TEAMSSTATE)
			return;
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
			m_aTeams[Cid] = Unpacker.GetInt();
		const int Tick = m_pDemoPlayer->Info()->m_Info.m_CurrentTick;
		if(g_WantTick < 0 || (Tick >= g_WantTick - 500 && Tick <= g_WantEnd))
		{
			printf("tick %d teams:", Tick);
			for(int Cid = 0; Cid < 8; Cid++)
				printf(" %d:%d", Cid, m_aTeams[Cid]);
			printf("\n");
		}
	}
};

int main(int argc, const char *argv[])
{
	CCmdlineFix CmdlineFix(&argc, &argv);
	log_set_global_logger_default();
	std::unique_ptr<IStorage> pStorage = std::unique_ptr<IStorage>(CreateLocalStorage());
	if(argc < 3)
	{
		printf("Usage: %s <demo> <tick> [end tick]\n", argv[0]);
		return -1;
	}
	g_WantTick = str_toint(argv[2]);
	g_WantEnd = argc > 3 ? str_toint(argv[3]) : g_WantTick + 1;

	CSnapshotDelta SnapshotDelta;
	CNetObjHandler NetObjHandler;
	for(int Type = 0; Type < NUM_NETOBJTYPES; Type++)
		SnapshotDelta.SetStaticsize(Type, NetObjHandler.GetObjSize(Type));

	CDemoPlayer DemoPlayer(&SnapshotDelta, nullptr, false);
	if(DemoPlayer.Load(pStorage.get(), nullptr, argv[1], IStorage::TYPE_ALL_OR_ABSOLUTE) != 0)
	{
		printf("failed to load demo: %s\n", DemoPlayer.ErrorMessage());
		return -1;
	}
	CInspectListener Listener(&DemoPlayer);
	DemoPlayer.SetListener(&Listener);
	const CDemoPlayer::CPlaybackInfo *pInfo = DemoPlayer.Info();
	CNetBase::Init();
	DemoPlayer.Play();
	while(DemoPlayer.IsPlaying())
	{
		DemoPlayer.Update(false);
		if(pInfo->m_Info.m_Paused)
			break;
	}
	DemoPlayer.Stop();
	if(g_WantTick == -2)
		printf("jumps %d false %d\n", Listener.Jumps(), Listener.FalseJumps());
	printf("last snapshot tick %d, %d snapshots, %d messages (%d extended)\n",
		Listener.LastTick(), Listener.NumSnapshots(), Listener.NumMessages(), Listener.NumExtended());
	return 0;
}
