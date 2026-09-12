/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#include "multi_server.h"

#include <base/log.h>

#include <engine/graphics.h>
#include <engine/shared/config.h>
#include <engine/textrender.h>

#include <game/client/gameclient.h>
#include <game/client/render.h>
#include <game/gamecore.h>
#include <game/localization.h>

#include <algorithm>
#include <limits>

void CMultiServer::OnConsoleInit()
{
	Console()->Register("multi_server_connect", "s[host|ip:port]", CFGFLAG_CLIENT, ConMultiServerConnect, this, "Watch another server that runs the same map");
	Console()->Register("multi_server_spectate", "i[server] i[id]", CFGFLAG_CLIENT, ConMultiServerSpectate, this, "Watch a player on one of the observed servers, server -1 is the one you are connected to");
	Console()->Register("multi_server_disconnect", "", CFGFLAG_CLIENT, ConMultiServerDisconnect, this, "Stop observing the other servers");
}

void CMultiServer::OnNewSnapshot()
{
	m_ScoreboardDirty = true;
	if(IsActive())
	{
		KeepSpectating();
	}
}

void CMultiServer::OnStateChange(int NewState, int OldState)
{
	// The map is loaded once the state is online, which is what the observed servers
	// are checked against.
	if(NewState == IClient::STATE_ONLINE)
	{
		AttachPending();
	}
	else if(NewState == IClient::STATE_OFFLINE)
	{
		m_vPending.clear();
	}
}

void CMultiServer::OnReset()
{
	for(CRemoteServer &Server : m_aServers)
	{
		Server = CRemoteServer();
	}
	m_NumServers = 0;
	m_ScoreboardScroll = 0;
	m_ScoreboardMaxScroll = 0;
	m_SpectateServer = -1;
	m_SpectateClientId = -1;
	m_ScoreboardDirty = true;
	m_NextLocalSpectateTime = 0;
	m_NextShowDistanceTime = 0;
	m_LastShowDistanceX = 0.0f;
	m_LastShowDistanceY = 0.0f;
}

void CMultiServer::ConnectAll(const char *pConnectAddress, const std::vector<CServerEntry> &vObserve)
{
	DetachAll();
	m_vPending = vObserve;
	Client()->Connect(pConnectAddress);
}

void CMultiServer::Observe(const char *pAddress)
{
	// Takes the port apart and resolves host names as well as numeric addresses.
	NETADDR Addr;
	if(net_host_lookup(pAddress, &Addr, NETTYPE_ALL) != 0)
	{
		log_error("multi_server", "could not find the address of '%s'", pAddress);
		return;
	}
	if(Addr.port == 0)
	{
		Addr.port = 8303;
	}

	const int Conn = Client()->ObserverConnect(Addr, pAddress);
	if(Conn >= 0)
	{
		Attach(Conn, pAddress);
	}
}

void CMultiServer::Attach(int Conn, const char *pName)
{
	CRemoteServer &Server = this->Server(Conn);
	Server = CRemoteServer();
	Server.m_Conn = Conn;
	str_copy(Server.m_aName, pName);
	m_NumServers++;
}

void CMultiServer::AttachPending()
{
	for(const CServerEntry &Entry : m_vPending)
	{
		const int Conn = Client()->ObserverConnect(Entry.m_Addr, Entry.m_aName);
		if(Conn < 0)
		{
			continue;
		}

		Attach(Conn, Entry.m_aName);
	}
	m_vPending.clear();
}

void CMultiServer::DetachAll()
{
	// Resets every observed server through OnObserverDisconnect.
	Client()->ObserverDisconnectAll();
	// Not part of OnReset, the pending servers have to survive the state changes of the
	// main connection coming up.
	m_vPending.clear();
	m_SpectateServer = -1;
	m_SpectateClientId = -1;
}

void CMultiServer::OnObserverDisconnect(int Conn)
{
	if(Server(Conn).IsActive())
	{
		m_NumServers--;
	}
	Server(Conn) = CRemoteServer();
	m_ScoreboardDirty = true;
	if(m_SpectateServer == Conn - IClient::CONN_OBSERVER_FIRST)
	{
		m_SpectateServer = -1;
		m_SpectateClientId = -1;
	}
}

void CMultiServer::OnObserverEnterGame(int Conn)
{
	CRemoteServer &Server = this->Server(Conn);
	if(!Server.IsActive())
	{
		return;
	}
	// Reached again when that server changes its map, which recreates our player there,
	// so everything it told us before is gone, including the view box it keeps for us.
	Server.m_ShowDistanceSent = false;
	Server.m_NextSpectateTime = 0;
	Server.m_SpectateAttempts = 0;
	Server.m_HasSpectatorInfo = false;
	Server.m_LocalInGame = false;
	Server.m_NumVisible = 0;
	m_ScoreboardDirty = true;
	for(CRemotePlayer &Player : Server.m_aPlayers)
	{
		Player = CRemotePlayer();
	}
}

int CMultiServer::SnapInput(int Conn, int *pData)
{
	// The observed server clips the snapshot around the view position of its spectator,
	// so point it at our camera to receive the players we are actually looking at.
	CNetObj_PlayerInput Input = {};
	Input.m_PlayerFlags = PLAYERFLAG_SPEC_CAM;
	if(GameClient()->m_Scoreboard.IsActive())
	{
		// Servers only refresh the latency they report about their other players while
		// this flag is set, see CPlayer::PostTick. Without it every ping of an observed
		// server stays at zero.
		Input.m_PlayerFlags |= PLAYERFLAG_SCOREBOARD;
	}
	Input.m_TargetX = round_to_int(GameClient()->m_Camera.m_Center.x);
	Input.m_TargetY = round_to_int(GameClient()->m_Camera.m_Center.y);
	mem_copy(pData, &Input, sizeof(Input));
	return sizeof(Input);
}

// Ticks of an observed server are rebased into our own tick base. They can hold any
// value the server sends, so the sum is clamped instead of overflowing.
static int RebaseTick(int Tick, int Offset)
{
	return (int)std::clamp<int64_t>((int64_t)Tick + Offset, std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
}

void CMultiServer::OnObserverSnapshot(int Conn)
{
	CRemoteServer &Server = this->Server(Conn);
	if(!Server.IsActive())
	{
		return;
	}

	Server.m_HasSpectatorInfo = false;
	Server.m_LocalInGame = false;
	m_ScoreboardDirty = true;
	Server.m_NumVisible = 0;
	for(CRemotePlayer &Player : Server.m_aPlayers)
	{
		Player.m_Active = false;
		Player.m_HasCharacter = false;
		Player.m_Local = false;
		Player.m_Frozen = false;
		Player.m_LiveFrozen = false;
		Player.m_Invincible = false;
	}

	// The observed server runs its own tick counter. Rebase the ticks that the renderer
	// compares against our own game tick, everything else is interpolated per connection.
	const int CurTick = Client()->GameTick(Conn);
	const int PrevTick = Client()->PrevGameTick(Conn);
	const int TickOffset = Client()->GameTick(g_Config.m_ClDummy) - CurTick;

	// Looking an item up is a linear scan of the whole snapshot, so collect the previous
	// characters in one pass instead of searching for each of them.
	const CNetObj_Character *apPrevCharacters[MAX_CLIENTS] = {nullptr};
	const int NumPrevItems = Client()->ObserverSnapNumItems(Conn, IClient::SNAP_PREV);
	for(int i = 0; i < NumPrevItems; i++)
	{
		const IClient::CSnapItem PrevItem = Client()->ObserverSnapGetItem(Conn, IClient::SNAP_PREV, i);
		if(PrevItem.m_Type == NETOBJTYPE_CHARACTER && in_range(PrevItem.m_Id, 0, MAX_CLIENTS - 1))
		{
			apPrevCharacters[PrevItem.m_Id] = (const CNetObj_Character *)PrevItem.m_pData;
		}
	}

	const int NumItems = Client()->ObserverSnapNumItems(Conn, IClient::SNAP_CURRENT);
	for(int i = 0; i < NumItems; i++)
	{
		const IClient::CSnapItem Item = Client()->ObserverSnapGetItem(Conn, IClient::SNAP_CURRENT, i);
		if(Item.m_Id < 0 || Item.m_Id >= MAX_CLIENTS)
		{
			continue;
		}
		CRemotePlayer &Player = Server.m_aPlayers[Item.m_Id];

		if(Item.m_Type == NETOBJTYPE_CLIENTINFO)
		{
			const CNetObj_ClientInfo *pInfo = (const CNetObj_ClientInfo *)Item.m_pData;
			// Client info is snapped every tick but hardly ever changes.
			if(mem_comp(&Player.m_ClientInfo, pInfo, sizeof(Player.m_ClientInfo)) == 0)
			{
				continue;
			}
			Player.m_ClientInfo = *pInfo;
			if(!IntsToStr(pInfo->m_aName, std::size(pInfo->m_aName), Player.m_aName, std::size(Player.m_aName)))
			{
				str_copy(Player.m_aName, "nameless tee");
			}
			IntsToStr(pInfo->m_aClan, std::size(pInfo->m_aClan), Player.m_aClan, std::size(Player.m_aClan));
			char aSkinName[MAX_SKIN_LENGTH];
			IntsToStr(pInfo->m_aSkin, std::size(pInfo->m_aSkin), aSkinName, std::size(aSkinName));
			if(!CSkin::IsValidName(aSkinName) ||
				(!GameClient()->m_GameInfo.m_AllowXSkins && CSkins::IsSpecialSkin(aSkinName)))
			{
				str_copy(aSkinName, "default");
			}
			UpdateSkinInfo(Player, aSkinName);
		}
		else if(Item.m_Type == NETOBJTYPE_PLAYERINFO)
		{
			const CNetObj_PlayerInfo *pInfo = (const CNetObj_PlayerInfo *)Item.m_pData;
			if(pInfo->m_ClientId != Item.m_Id)
			{
				continue;
			}
			Player.m_Active = true;
			Player.m_Local = pInfo->m_Local != 0;
			Player.m_Team = pInfo->m_Team;
			Player.m_Score = pInfo->m_Score;
			Player.m_Latency = pInfo->m_Latency;
			if(Player.m_Local && Player.m_Team != TEAM_SPECTATORS)
			{
				Server.m_LocalInGame = true;
			}
		}
		else if(Item.m_Type == NETOBJTYPE_DDNETPLAYER)
		{
			const CNetObj_DDNetPlayer *pInfo = (const CNetObj_DDNetPlayer *)Item.m_pData;
			Player.m_FinishTimeSeconds = pInfo->m_FinishTimeSeconds;
			Player.m_FinishTimeMillis = pInfo->m_FinishTimeMillis;
		}
		else if(Item.m_Type == NETOBJTYPE_CHARACTER)
		{
			const CNetObj_Character *pPrev = apPrevCharacters[Item.m_Id];
			if(pPrev == nullptr)
			{
				Player.m_Evolved.m_Tick = -1;
				continue;
			}
			Player.m_HasCharacter = true;
			Player.m_Cur = *((const CNetObj_Character *)Item.m_pData);
			Player.m_Prev = *pPrev;

			// Servers keep sending the same dead reckoned character with an old tick
			// while its movement stays predictable, so evolve it to the tick we render,
			// in the tick base of the server it came from.
			// The observed server picks these ticks, so the distance is measured in 64 bit.
			// A tick far in the past would pass the check and evolve for hours.
			const bool EvolvePrev = (int64_t)PrevTick - Player.m_Prev.m_Tick <= 3 * Client()->GameTickSpeed();
			const bool EvolveCur = (int64_t)CurTick - Player.m_Cur.m_Tick <= 3 * Client()->GameTickSpeed();
			if(EvolveCur && Player.m_Evolved.m_Tick == PrevTick)
			{
				if(mem_comp(&Player.m_Prev, &Player.m_Snapped, sizeof(CNetObj_Character)) == 0)
					Player.m_Prev = Player.m_Evolved;
				if(mem_comp(&Player.m_Cur, &Player.m_Snapped, sizeof(CNetObj_Character)) == 0)
					Player.m_Cur = Player.m_Evolved;
			}
			if(EvolvePrev && Player.m_Prev.m_Tick)
				GameClient()->Evolve(&Player.m_Prev, PrevTick);
			if(EvolveCur && Player.m_Cur.m_Tick)
				GameClient()->Evolve(&Player.m_Cur, CurTick);
			Player.m_Snapped = *((const CNetObj_Character *)Item.m_pData);
			Player.m_Evolved = Player.m_Cur;

			for(CNetObj_Character *pCharacter : {&Player.m_Cur, &Player.m_Prev})
			{
				pCharacter->m_Tick = RebaseTick(pCharacter->m_Tick, TickOffset);
				pCharacter->m_AttackTick = RebaseTick(pCharacter->m_AttackTick, TickOffset);
				// The hooked player id belongs to the other server, the renderer would
				// look it up in our own snapshot. The hook position is snapped anyway.
				pCharacter->m_HookedPlayer = -1;
			}
		}
		else if(Item.m_Type == NETOBJTYPE_SPECTATORINFO)
		{
			const CNetObj_SpectatorInfo *pInfo = (const CNetObj_SpectatorInfo *)Item.m_pData;
			Server.m_HasSpectatorInfo = true;
			Server.m_SpectatorPos = vec2(pInfo->m_X, pInfo->m_Y);
		}
		else if(Item.m_Type == NETOBJTYPE_DDNETCHARACTER)
		{
			const CNetObj_DDNetCharacter *pCharacter = (const CNetObj_DDNetCharacter *)Item.m_pData;
			Player.m_Frozen = pCharacter->m_FreezeEnd != 0;
			Player.m_LiveFrozen = (pCharacter->m_Flags & CHARACTERFLAG_MOVEMENTS_DISABLED) != 0;
			Player.m_Invincible = (pCharacter->m_Flags & CHARACTERFLAG_INVINCIBLE) != 0;
		}
	}

	// Collect who is in the world once instead of walking all the slots per frame, and
	// look the friends up per snapshot like CGameClient::OnNewSnapshot does.
	for(int ClientId = 0; ClientId < MAX_CLIENTS; ClientId++)
	{
		CRemotePlayer &Player = Server.m_aPlayers[ClientId];
		if(!Player.m_Active)
		{
			continue;
		}
		Player.m_Friend = GameClient()->Friends()->IsFriend(Player.m_aName, Player.m_aClan, true);
		if(Player.m_HasCharacter && !Player.m_Local)
		{
			Server.m_aVisibleIds[Server.m_NumVisible] = ClientId;
			Server.m_NumVisible++;
		}
	}
}

void CMultiServer::OnObserverMessage(int MsgId, CUnpacker *pUnpacker, int Conn)
{
	// The DDRace team of a player is not in the snapshot, it only arrives in this
	// message. Everything else an observed server says is none of our business.
	if(MsgId != NETMSGTYPE_SV_TEAMSSTATE && MsgId != NETMSGTYPE_SV_TEAMSSTATELEGACY)
	{
		return;
	}

	CRemoteServer &Server = this->Server(Conn);
	if(!Server.IsActive())
	{
		return;
	}
	for(CRemotePlayer &Player : Server.m_aPlayers)
	{
		const int Team = pUnpacker->GetInt();
		if(pUnpacker->Error() || Team < TEAM_FLOCK || Team >= NUM_DDRACE_TEAMS)
		{
			Player.m_DdTeam = TEAM_FLOCK;
			break;
		}
		Player.m_DdTeam = Team;
	}
}

void CMultiServer::UpdateSkinInfo(CRemotePlayer &Player, const char *pSkinName)
{
	CSkinDescriptor SkinDescriptor;
	SkinDescriptor.m_Flags = CSkinDescriptor::FLAG_SIX;
	str_copy(SkinDescriptor.m_aSkinName, pSkinName);

	if(Player.m_pSkinInfo == nullptr || Player.m_pSkinInfo->SkinDescriptor() != SkinDescriptor)
	{
		Player.m_pSkinInfo = GameClient()->CreateManagedTeeRenderInfo(CTeeRenderInfo(), SkinDescriptor);
	}
	Player.m_pSkinInfo->TeeRenderInfo().ApplyColors(Player.m_ClientInfo.m_UseCustomColor != 0,
		Player.m_ClientInfo.m_ColorBody, Player.m_ClientInfo.m_ColorFeet);
	Player.m_pSkinInfo->TeeRenderInfo().m_Size = 64.0f;
}

vec2 CMultiServer::RenderPos(int Server, const CRemotePlayer &Player) const
{
	const float Intra = Client()->IntraGameTick(m_aServers[Server].m_Conn);
	return mix(vec2(Player.m_Prev.m_X, Player.m_Prev.m_Y), vec2(Player.m_Cur.m_X, Player.m_Cur.m_Y), Intra);
}

void CMultiServer::OnRender()
{
	if(Client()->State() != IClient::STATE_ONLINE)
	{
		return;
	}
	if(!IsActive())
	{
		return;
	}

	// The watched player left or joined the spectators.
	if(m_SpectateServer >= 0 && !IsSpectatingRemote())
	{
		ClearRemoteWatch();
	}

	CScreenRect ScreenRect = Graphics()->GetScreen();
	ScreenRect.Expand(100.0f);

	ForEachRemotePlayer([&](int ServerIndex, int ClientId, const CRemotePlayer &Player) {
		if(Player.m_pSkinInfo == nullptr)
		{
			return;
		}

		// Each remote player owns its render info, so set the flags in place instead of
		// copying the whole thing per player and frame. The same rules as in
		// CPlayers::OnRender, without the prediction that only local players get.
		CTeeRenderInfo &RenderInfo = Player.m_pSkinInfo->TeeRenderInfo();
		RenderInfo.m_TeeRenderFlags = 0;
		if(Player.m_Frozen)
		{
			RenderInfo.m_TeeRenderFlags |= TEE_EFFECT_FROZEN | TEE_NO_WEAPON;
		}
		if(Player.m_LiveFrozen)
		{
			RenderInfo.m_TeeRenderFlags |= TEE_EFFECT_FROZEN;
		}
		if(Player.m_Invincible)
		{
			RenderInfo.m_TeeRenderFlags |= TEE_EFFECT_SPARKLE;
		}

		// Wearing the ninja skin replaces the player's own, so that one needs the copy.
		const CTeeRenderInfo *pRenderInfo = &RenderInfo;
		CTeeRenderInfo NinjaRenderInfo;
		const bool Ninja = Player.m_Cur.m_Weapon == WEAPON_NINJA ||
				   (Player.m_Frozen && !GameClient()->m_GameInfo.m_NoSkinChangeForFrozen);
		if(Ninja && g_Config.m_ClShowNinja)
		{
			NinjaRenderInfo = RenderInfo;
			NinjaRenderInfo.m_aSixup[g_Config.m_ClDummy].Reset();
			NinjaRenderInfo.ApplySkin(GameClient()->m_Players.NinjaTeeRenderInfo()->TeeRenderInfo());
			NinjaRenderInfo.m_CustomColoredSkin = GameClient()->IsTeamPlay();
			if(!NinjaRenderInfo.m_CustomColoredSkin)
			{
				NinjaRenderInfo.m_ColorBody = ColorRGBA(1, 1, 1);
				NinjaRenderInfo.m_ColorFeet = ColorRGBA(1, 1, 1);
			}
			pRenderInfo = &NinjaRenderInfo;
		}

		// The id tells the renderer which server this player is on, so that it can pick
		// the alpha, see CPlayers::RenderPlayer.
		const int RenderId = RenderClientId(ServerIndex);
		const float Intra = Client()->IntraGameTick(m_aServers[ServerIndex].m_Conn);
		GameClient()->m_Players.RenderHook(ScreenRect, &Player.m_Prev, &Player.m_Cur, pRenderInfo, RenderId, Intra);
		GameClient()->m_Players.RenderPlayer(ScreenRect, &Player.m_Prev, &Player.m_Cur, pRenderInfo, RenderId, Intra);
	});
}

void CMultiServer::SendShowDistance(float Zoom)
{
	if(!IsActive())
	{
		return;
	}

	// The observed servers clip their snapshots to this box around our camera. The zoom
	// is the one CGameClient::OnNewSnapshot tells our own server about, it keeps the
	// larger distance while zooming in.
	float x, y;
	Graphics()->CalcScreenParams(Graphics()->ScreenAspect(), Zoom, &x, &y);

	// Cl_ShowDistance is vital, so it must not be sent per frame. A stream of vital
	// messages that outruns the acks fills the resend buffer of the connection, and
	// chunks dropped from a full buffer can never be recovered.
	const int64_t Now = time_get();
	bool Changed = false;
	if((x != m_LastShowDistanceX || y != m_LastShowDistanceY) && Now >= m_NextShowDistanceTime)
	{
		m_LastShowDistanceX = x;
		m_LastShowDistanceY = y;
		m_NextShowDistanceTime = Now + time_freq() / 2;
		Changed = true;
	}

	CNetMsg_Cl_ShowDistance Msg;
	Msg.m_X = m_LastShowDistanceX;
	Msg.m_Y = m_LastShowDistanceY;
	CMsgPacker Packer(&Msg);
	Msg.Pack(&Packer);
	for(CRemoteServer &Server : m_aServers)
	{
		if(!Server.IsActive() || !Client()->ObserverOnline(Server.m_Conn) || (!Changed && Server.m_ShowDistanceSent))
		{
			continue;
		}
		Client()->SendMsg(Server.m_Conn, &Packer, MSGFLAG_VITAL);
		Server.m_ShowDistanceSent = true;
	}
}

void CMultiServer::KeepSpectating()
{
	// Watching several servers at once means playing on none of them, so every
	// connection joins the spectators. Servers put a joining client into the game and
	// rate limit team changes (sv_team_change_delay, 3 seconds by default), so retry
	// slowly. Cl_SetTeam is vital, retrying it per second would flood the connection.
	bool AnyOnline = false;
	for(const CRemoteServer &Server : m_aServers)
	{
		AnyOnline = AnyOnline || (Server.IsActive() && Client()->ObserverOnline(Server.m_Conn));
	}
	if(!AnyOnline)
	{
		// Every observed server may still turn out to run another map and be dropped
		// again, leaving the player in the spectators of their own server for nothing.
		return;
	}

	const int64_t Now = time_get();
	const int64_t RetryInterval = 5 * time_freq();

	CNetMsg_Cl_SetTeam Msg;
	Msg.m_Team = TEAM_SPECTATORS;
	CMsgPacker Packer(&Msg);
	Msg.Pack(&Packer);

	// m_pLocalInfo describes whichever connection is active, so ask the snapshot about
	// each local tee instead. Both of them have to leave the game, not just the active
	// one, or the other keeps playing and this retries forever.
	if(Now >= m_NextLocalSpectateTime)
	{
		for(int Local = 0; Local < NUM_DUMMIES; Local++)
		{
			const int LocalId = GameClient()->m_aLocalIds[Local];
			if(Local == 1 && !Client()->DummyConnected())
			{
				break;
			}
			if(!in_range(LocalId, 0, MAX_CLIENTS - 1))
			{
				continue;
			}
			const CNetObj_PlayerInfo *pInfo = GameClient()->m_Snap.m_apPlayerInfos[LocalId];
			if(pInfo == nullptr || pInfo->m_Team == TEAM_SPECTATORS)
			{
				continue;
			}
			Client()->SendMsg(Local == 0 ? IClient::CONN_MAIN : IClient::CONN_DUMMY, &Packer, MSGFLAG_VITAL);
			m_NextLocalSpectateTime = Now + RetryInterval;
		}
	}

	for(CRemoteServer &Server : m_aServers)
	{
		if(!Server.IsActive() || !Client()->ObserverOnline(Server.m_Conn) || Now < Server.m_NextSpectateTime)
		{
			continue;
		}

		if(!Server.m_LocalInGame)
		{
			continue;
		}

		Server.m_SpectateAttempts++;
		if(Server.m_SpectateAttempts == 3)
		{
			// As a player the server clips its snapshots around our own tee instead of
			// our camera, so the other players barely update. Worth saying out loud.
			log_warn("multi_server", "'%s' keeps us in the game, its players will update poorly", Server.m_aName);
		}
		Client()->SendMsg(Server.m_Conn, &Packer, MSGFLAG_VITAL);
		Server.m_NextSpectateTime = Now + RetryInterval;
	}
}

void CMultiServer::SendSpectatorMode(int Server, int SpectatorId)
{
	const CRemoteServer &RemoteServer = m_aServers[Server];
	if(!RemoteServer.IsActive() || !Client()->ObserverOnline(RemoteServer.m_Conn))
	{
		return;
	}
	CNetMsg_Cl_SetSpectatorMode Msg;
	Msg.m_SpectatorId = SpectatorId;
	Client()->SendPackMsg(RemoteServer.m_Conn, &Msg, MSGFLAG_VITAL);
}

void CMultiServer::Spectate(int Server, int ClientId)
{
	if(Server < 0)
	{
		// Clears the remote watch and guards against watching who we already watch.
		GameClient()->m_Spectator.Spectate(ClientId);
		return;
	}
	if(Server >= MAX_OBSERVERS || !m_aServers[Server].IsActive() || !in_range(ClientId, 0, MAX_CLIENTS - 1))
	{
		return;
	}
	if(Server == m_SpectateServer && ClientId == m_SpectateClientId)
	{
		return;
	}

	// Watching another server means free view here, the server we play on has no player
	// to follow.
	GameClient()->m_Spectator.Spectate(SPEC_FREEVIEW);
	m_SpectateServer = Server;
	m_SpectateClientId = ClientId;
	// An observed server only snaps the players around the view position it keeps for
	// us. Watching a player it does not send us can never work, the camera would have
	// to be there already, so let the server follow that player instead.
	SendSpectatorMode(Server, ClientId);
}

void CMultiServer::ClearRemoteWatch()
{
	if(m_SpectateServer < 0)
	{
		return;
	}
	// Let that server go back to following our camera.
	SendSpectatorMode(m_SpectateServer, SPEC_FREEVIEW);
	m_SpectateServer = -1;
	m_SpectateClientId = -1;
}

void CMultiServer::SpectateNext(bool Reverse)
{
	std::vector<const CScoreboardRow *> vpRows;
	for(const CScoreboardRow &Row : BuildScoreboard())
	{
		if(!Row.IsHeader() && !Row.m_Spectator)
		{
			vpRows.push_back(&Row);
		}
	}
	if(vpRows.empty())
	{
		return;
	}

	const int CurrentServer = IsSpectatingRemote() ? m_SpectateServer : -1;
	const int CurrentClientId = IsSpectatingRemote() ? m_SpectateClientId : GameClient()->m_Snap.m_SpecInfo.m_SpectatorId;
	int Current = -1;
	for(size_t i = 0; i < vpRows.size(); i++)
	{
		if(vpRows[i]->m_Server == CurrentServer && vpRows[i]->m_ClientId == CurrentClientId)
		{
			Current = i;
			break;
		}
	}

	const int Num = vpRows.size();
	const int Next = Current < 0 ? (Reverse ? Num - 1 : 0) : (Current + (Reverse ? -1 : 1) + Num) % Num;
	Spectate(vpRows[Next]->m_Server, vpRows[Next]->m_ClientId);
}

bool CMultiServer::IsSpectatingRemote() const
{
	if(m_SpectateServer < 0)
	{
		return false;
	}
	const CRemotePlayer &Player = m_aServers[m_SpectateServer].m_aPlayers[m_SpectateClientId];
	return m_aServers[m_SpectateServer].IsActive() && Player.m_Active && Player.m_Team != TEAM_SPECTATORS;
}

int CMultiServer::WatchedServer() const
{
	if(IsSpectatingRemote())
	{
		return m_SpectateServer;
	}
	if(GameClient()->m_Snap.m_SpecInfo.m_SpectatorId != SPEC_FREEVIEW)
	{
		return -1;
	}
	return -2;
}

float CMultiServer::PlayerAlpha(int Server) const
{
	const int Watched = WatchedServer();
	if(!IsActive() || Watched == -2 || Watched == Server)
	{
		return 1.0f;
	}
	return g_Config.m_ClMultiServerAlpha / 100.0f;
}

bool CMultiServer::IsWatching(int Server, int ClientId) const
{
	if(IsSpectatingRemote())
	{
		return Server == m_SpectateServer && ClientId == m_SpectateClientId;
	}
	return Server < 0 && ClientId == GameClient()->m_Snap.m_SpecInfo.m_SpectatorId;
}

const CTeeRenderInfo *CMultiServer::PlayerRenderInfo(int Server, int ClientId) const
{
	if(!in_range(ClientId, 0, MAX_CLIENTS - 1))
	{
		return nullptr;
	}
	if(Server < 0)
	{
		return &GameClient()->m_aClients[ClientId].m_RenderInfo;
	}
	if(Server >= MAX_OBSERVERS || !m_aServers[Server].IsActive())
	{
		return nullptr;
	}
	const CRemotePlayer &Player = m_aServers[Server].m_aPlayers[ClientId];
	return Player.m_pSkinInfo == nullptr ? nullptr : &Player.m_pSkinInfo->TeeRenderInfo();
}

vec2 CMultiServer::SpectatePosition()
{
	const CRemoteServer &Server = m_aServers[m_SpectateServer];
	const CRemotePlayer &Player = Server.m_aPlayers[m_SpectateClientId];
	if(Player.m_HasCharacter)
	{
		m_SpectatePos = RenderPos(m_SpectateServer, Player);
	}
	else if(Server.m_HasSpectatorInfo)
	{
		// The character has not arrived yet, but the server already follows the player.
		m_SpectatePos = Server.m_SpectatorPos;
	}
	return m_SpectatePos;
}

const std::vector<CMultiServer::CScoreboardRow> &CMultiServer::BuildScoreboard()
{
	// Sorting more than a thousand rows is not something to do several times per frame.
	if(!m_ScoreboardDirty)
	{
		return m_vScoreboardRows;
	}
	m_ScoreboardDirty = false;

	std::vector<CScoreboardRow> &vRows = m_vScoreboardRows;
	vRows.clear();

	// Spectators last, teams stay together, and inside a team the same order the normal
	// scoreboard uses.
	const auto &&Compare = CGameClient::GetScoreComparator(GameClient()->m_GameInfo.m_TimeScore, GameClient()->m_ReceivedDDNetPlayerFinishTimes, false);
	const auto &&Less = [&](const CScoreboardRow &Left, const CScoreboardRow &Right) {
		if(Left.m_Spectator != Right.m_Spectator)
			return Right.m_Spectator;
		if(Left.m_DdTeam != Right.m_DdTeam)
			return Left.m_DdTeam < Right.m_DdTeam;
		if(GameClient()->m_ReceivedDDNetPlayerFinishTimes)
			return Compare(Left.m_FinishTimeSeconds, Right.m_FinishTimeSeconds, Left.m_FinishTimeMillis, Right.m_FinishTimeMillis);
		return Compare(Left.m_Score, Right.m_Score, 0, 0);
	};

	const auto &&AppendServer = [&](int ServerIndex, const char *pName) {
		// The header row goes in first, inserting it afterwards would move every row of
		// the group again.
		vRows.push_back({ServerIndex, -1, TEAM_FLOCK, pName, 0,
			FinishTime::UNSET, FinishTime::NOT_FINISHED_MILLIS, 0, false, false});
		const size_t First = vRows.size();
		if(ServerIndex < 0)
		{
			for(int ClientId = 0; ClientId < MAX_CLIENTS; ClientId++)
			{
				const CNetObj_PlayerInfo *pInfo = GameClient()->m_Snap.m_apPlayerInfos[ClientId];
				if(pInfo == nullptr)
				{
					continue;
				}
				const CGameClient::CClientData &Client = GameClient()->m_aClients[ClientId];
				vRows.push_back({-1, ClientId, GameClient()->m_Teams.Team(ClientId), Client.m_aName,
					pInfo->m_Latency, Client.m_FinishTimeSeconds, Client.m_FinishTimeMillis, pInfo->m_Score,
					GameClient()->m_Snap.m_LocalClientId == ClientId, pInfo->m_Team == TEAM_SPECTATORS});
			}
		}
		else
		{
			const CRemoteServer &Server = m_aServers[ServerIndex];
			for(int ClientId = 0; ClientId < MAX_CLIENTS; ClientId++)
			{
				const CRemotePlayer &Player = Server.m_aPlayers[ClientId];
				if(!Player.m_Active || Player.m_Local)
				{
					continue;
				}
				vRows.push_back({ServerIndex, ClientId, Player.m_DdTeam, Player.m_aName,
					Player.m_Latency, Player.m_FinishTimeSeconds, Player.m_FinishTimeMillis,
					Player.m_Score, false, Player.m_Team == TEAM_SPECTATORS});
			}
		}
		if(vRows.size() == First)
		{
			// A server without players does not get a group.
			vRows.pop_back();
			return;
		}
		std::stable_sort(vRows.begin() + First, vRows.end(), Less);
	};

	str_copy(m_aOwnServerName, Client()->ServerInfo().m_aName[0] != '\0' ? Client()->ServerInfo().m_aName : Localize("Your server"));
	AppendServer(-1, m_aOwnServerName);
	for(int ServerIndex = 0; ServerIndex < MAX_OBSERVERS; ServerIndex++)
	{
		if(m_aServers[ServerIndex].IsActive())
		{
			AppendServer(ServerIndex, m_aServers[ServerIndex].m_aName);
		}
	}
	return vRows;
}

void CMultiServer::RenderScoreboard(const CUIRect &Screen)
{
	const std::vector<CScoreboardRow> &vRows = BuildScoreboard();

	const float TitleHeight = 24.0f;
	const float ColumnWidth = 210.0f;
	const float Padding = 5.0f;

	CUIRect Board;
	Screen.Margin(20.0f, &Board);
	Board.Draw(ColorRGBA(0.0f, 0.0f, 0.0f, 0.5f), IGraphics::CORNER_ALL, 7.5f);

	CUIRect Title, Body;
	Board.Margin(Padding, &Body);
	Body.HSplitTop(TitleHeight, &Title, &Body);

	int NumPlayers = 0;
	for(const CScoreboardRow &Row : vRows)
	{
		if(!Row.IsHeader())
		{
			NumPlayers++;
		}
	}

	// A header repeated at the top of a column costs a row, so the layout is computed
	// for the worst case of one extra row per column.
	const int MaxColumns = std::max(1, (int)(Body.w / ColumnWidth));
	const float FullRowHeight = 13.0f;
	float RowHeight = FullRowHeight;
	int RowsPerColumn = std::max(2, (int)(Body.h / RowHeight));
	const auto &&FitsOnePage = [&]() { return (RowsPerColumn - 1) * MaxColumns >= (int)vRows.size(); };
	while(!FitsOnePage() && RowHeight > 7.0f)
	{
		RowHeight -= 1.0f;
		RowsPerColumn = (int)(Body.h / RowHeight);
	}
	if(!FitsOnePage())
	{
		// Squeezing does not save the second page here, so keep the rows readable and
		// let them be scrolled through instead.
		RowHeight = FullRowHeight;
		RowsPerColumn = std::max(2, (int)(Body.h / RowHeight));
	}
	const float FontSize = RowHeight - 3.0f;

	// Lay the rows out once to know how many columns they need, the same way the
	// spectator selector does, see CSpectator::RenderMultiServerSelector.
	const int NeededColumns = FlowRows(vRows, RowsPerColumn, false, [](const CFlowSlot &Slot) {});
	m_ScoreboardMaxScroll = std::max(0, NeededColumns - MaxColumns);
	// Only clamp for drawing, writing it back would throw the page away whenever the row
	// count dips for a frame.
	const int Scroll = std::clamp(m_ScoreboardScroll, 0, m_ScoreboardMaxScroll);

	char aBuf[128];
	str_format(aBuf, sizeof(aBuf), Localize("%d players on %d servers"), NumPlayers, NumServers() + 1);
	Ui()->DoLabel(&Title, aBuf, 18.0f, TEXTALIGN_MC);
	if(m_ScoreboardMaxScroll > 0)
	{
		str_format(aBuf, sizeof(aBuf), Localize("Scroll for more (%d/%d)"), Scroll + 1, m_ScoreboardMaxScroll + 1);
		CUIRect Page;
		Title.VSplitRight(5.0f, &Page, nullptr);
		Ui()->DoLabel(&Page, aBuf, 10.0f, TEXTALIGN_MR);
	}

	// Same rules as the normal scoreboard, see CScoreboard::RenderScoreboard.
	const bool MillisecondScore = GameClient()->m_ReceivedDDNetPlayerFinishTimes;
	const bool TimeScore = GameClient()->m_GameInfo.m_TimeScore;
	const char *pSpectatorLabel = Localize("spec");

	const auto &&RectAt = [&](int Col, int InColumn) {
		return CUIRect{Body.x + (Col - Scroll) * ColumnWidth, Body.y + InColumn * RowHeight, ColumnWidth - Padding, RowHeight};
	};

	FlowRows(vRows, RowsPerColumn, false, [&](const CFlowSlot &Slot) {
		if(Slot.m_Column < Scroll || Slot.m_Column >= Scroll + MaxColumns)
		{
			return;
		}
		if(Slot.m_pHeader != nullptr)
		{
			CUIRect Header = RectAt(Slot.m_Column, Slot.m_RowInColumn);
			Header.Draw(ColorRGBA(1.0f, 1.0f, 1.0f, 0.15f), IGraphics::CORNER_ALL, 2.0f);
			Header.VSplitLeft(3.0f, nullptr, &Header);
			Ui()->DoLabel(&Header, Slot.m_pHeader, FontSize, TEXTALIGN_ML);
			return;
		}

		const CScoreboardRow &ScoreboardRow = *Slot.m_pRow;
		CUIRect Row = RectAt(Slot.m_Column, Slot.m_RowInColumn);
		if(ScoreboardRow.m_Local)
		{
			Row.Draw(ColorRGBA(1.0f, 1.0f, 1.0f, 0.25f), IGraphics::CORNER_ALL, 2.0f);
		}
		else if(IsWatching(ScoreboardRow.m_Server, ScoreboardRow.m_ClientId))
		{
			Row.Draw(ColorRGBA(1.0f, 1.0f, 0.5f, 0.25f), IGraphics::CORNER_ALL, 2.0f);
		}

		CUIRect Team, Name, Time, Ping;
		Row.VSplitLeft(20.0f, &Team, &Row);
		Row.VSplitRight(24.0f, &Row, &Ping);
		Row.VSplitRight(50.0f, &Name, &Time);

		if(ScoreboardRow.m_DdTeam != TEAM_FLOCK)
		{
			Team.Draw(GameClient()->GetDDTeamColor(ScoreboardRow.m_DdTeam).WithAlpha(0.5f), IGraphics::CORNER_ALL, 2.0f);
			if(ScoreboardRow.m_DdTeam == GameClient()->m_Teams.TeamSuper())
			{
				str_copy(aBuf, "S");
			}
			else
			{
				str_format(aBuf, sizeof(aBuf), "%d", ScoreboardRow.m_DdTeam);
			}
			Ui()->DoLabel(&Team, aBuf, FontSize, TEXTALIGN_MC);
		}

		Name.VSplitLeft(3.0f, nullptr, &Name);
		Ui()->DoLabel(&Name, ScoreboardRow.m_pName, FontSize, TEXTALIGN_ML);

		aBuf[0] = '\0';
		if(ScoreboardRow.m_Spectator)
		{
			str_copy(aBuf, pSpectatorLabel);
		}
		else if(MillisecondScore)
		{
			if(ScoreboardRow.m_FinishTimeSeconds != FinishTime::NOT_FINISHED_MILLIS)
			{
				str_time((int64_t)ScoreboardRow.m_FinishTimeSeconds * 100 + ScoreboardRow.m_FinishTimeMillis / 10, ETimeFormat::MINS_CENTISECS, aBuf, sizeof(aBuf));
			}
		}
		else if(TimeScore)
		{
			if(ScoreboardRow.m_Score != FinishTime::NOT_FINISHED_TIMESCORE)
			{
				str_time((int64_t)ScoreboardRow.m_Score * 100, ETimeFormat::MINS_CENTISECS, aBuf, sizeof(aBuf));
			}
		}
		else
		{
			str_format(aBuf, sizeof(aBuf), "%d", std::clamp(ScoreboardRow.m_Score, -999, 99999));
		}
		Ui()->DoLabel(&Time, aBuf, FontSize, TEXTALIGN_MR);

		str_format(aBuf, sizeof(aBuf), "%d", std::clamp(ScoreboardRow.m_Latency, 0, 999));
		Ui()->DoLabel(&Ping, aBuf, FontSize, TEXTALIGN_MR);
	});
}

void CMultiServer::ConMultiServerConnect(IConsole::IResult *pResult, void *pUserData)
{
	((CMultiServer *)pUserData)->Observe(pResult->GetString(0));
}

void CMultiServer::ConMultiServerSpectate(IConsole::IResult *pResult, void *pUserData)
{
	CMultiServer *pSelf = (CMultiServer *)pUserData;
	pSelf->Spectate(pResult->GetInteger(0), pResult->GetInteger(1));
}

void CMultiServer::ConMultiServerDisconnect(IConsole::IResult *pResult, void *pUserData)
{
	((CMultiServer *)pUserData)->DetachAll();
}
