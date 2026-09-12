/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef GAME_CLIENT_COMPONENTS_MULTI_SERVER_H
#define GAME_CLIENT_COMPONENTS_MULTI_SERVER_H

#include <base/dbg.h>
#include <base/mem.h>
#include <base/time.h>
#include <base/vmath.h>

#include <engine/client.h>
#include <engine/client/enums.h>
#include <engine/console.h>
#include <engine/shared/packer.h>
#include <engine/shared/protocol.h>

#include <generated/protocol.h>

#include <game/client/component.h>
#include <game/client/render.h>
#include <game/client/ui_rect.h>
#include <game/teamscore.h>

#include <algorithm>
#include <memory>
#include <vector>

/**
 * Observes other servers that run the same map as the server the client is connected to.
 *
 * The observed servers are joined as spectators over their own connection, see
 * IClient::ObserverConnect. Their players are shown in the world and in a merged
 * scoreboard, so more than `MAX_CLIENTS` players can be watched at the same time.
 * There is no shared physics, players of different servers cannot interact.
 */
class CMultiServer : public CComponent
{
public:
	class CRemotePlayer
	{
	public:
		bool m_Active = false;
		bool m_Local = false;

		char m_aName[MAX_NAME_LENGTH] = "";
		char m_aClan[MAX_CLAN_LENGTH] = "";
		int m_Team = TEAM_SPECTATORS;
		int m_DdTeam = TEAM_FLOCK;
		int m_Score = 0;
		int m_Latency = 0;
		int m_FinishTimeSeconds = FinishTime::UNSET;
		int m_FinishTimeMillis = FinishTime::NOT_FINISHED_MILLIS;

		// Kept to notice when the skin of a player changes, it is snapped every tick.
		CNetObj_ClientInfo m_ClientInfo = {};
		std::shared_ptr<CManagedTeeRenderInfo> m_pSkinInfo = nullptr;
		bool m_Friend = false;

		bool m_HasCharacter = false;
		bool m_Frozen = false;
		bool m_LiveFrozen = false;
		bool m_Invincible = false;
		CNetObj_Character m_Prev = {};
		CNetObj_Character m_Cur = {};
		// Last snapped and evolved character, to reuse the evolve result like
		// CGameClient::OnNewSnapshot does for the server we are connected to.
		CNetObj_Character m_Snapped = {};
		CNetObj_Character m_Evolved = {};
	};

	class CRemoteServer
	{
	public:
		int m_Conn = -1;
		bool IsActive() const { return m_Conn >= 0; }

		char m_aName[64] = "";
		int64_t m_NextSpectateTime = 0;
		int m_SpectateAttempts = 0;
		bool m_ShowDistanceSent = false;
		// Where the observed server thinks our camera is, it snaps the players around it.
		bool m_HasSpectatorInfo = false;
		// Whether one of our own tees is still playing there, updated per snapshot.
		bool m_LocalInGame = false;
		vec2 m_SpectatorPos = vec2(0.0f, 0.0f);
		CRemotePlayer m_aPlayers[MAX_CLIENTS];
		// The ids of m_aPlayers that are in the world, so that the render paths do not
		// have to walk 128 slots per server and frame.
		int m_aVisibleIds[MAX_CLIENTS] = {0};
		int m_NumVisible = 0;
	};

	/**
	 * One line of the merged scoreboard. Rows are grouped per server, each group
	 * starting with a header row and ordered by DDRace team like the normal scoreboard.
	 */
	class CScoreboardRow
	{
	public:
		int m_Server; // index into m_aServers, or -1 for the server we are connected to
		int m_ClientId; // -1 marks the header row of a server
		int m_DdTeam;
		const char *m_pName; // the server name on a header row
		int m_Latency;
		int m_FinishTimeSeconds;
		int m_FinishTimeMillis;
		int m_Score;
		bool m_Local;
		bool m_Spectator;

		bool IsHeader() const { return m_ClientId < 0; }
	};

	int Sizeof() const override { return sizeof(*this); }
	void OnConsoleInit() override;
	void OnReset() override;
	void OnRender() override;
	void OnNewSnapshot() override;
	void OnStateChange(int NewState, int OldState) override;

	void OnObserverSnapshot(int Conn);
	void OnObserverMessage(int MsgId, CUnpacker *pUnpacker, int Conn);
	void OnObserverDisconnect(int Conn);
	void OnObserverEnterGame(int Conn);
	/**
	 * Tells the observed servers how far our camera sees, they clip their snapshots to
	 * it. `Zoom` is the same value our own server is told about.
	 */
	void SendShowDistance(float Zoom);
	int SnapInput(int Conn, int *pData);

	class CServerEntry
	{
	public:
		NETADDR m_Addr;
		char m_aName[64];
	};

	/**
	 * Connects to `pConnectAddress` and observes every server in `vObserve`.
	 *
	 * The caller has to make sure that all of them run the same map, mismatches are
	 * dropped once the observed server announces its map.
	 *
	 * @param pConnectAddress All addresses of the server to join, as the browser lists
	 * them, so that the 0.7 prefix and the fallback to the other addresses survive.
	 */
	void ConnectAll(const char *pConnectAddress, const std::vector<CServerEntry> &vObserve);
	void ClearRemoteWatch();
	void Observe(const char *pAddress);
	void DetachAll();
	bool IsActive() const { return m_NumServers > 0; }
	int NumServers() const { return m_NumServers; }

	/**
	 * Moves the merged scoreboard by `Direction` columns. The page is kept while the
	 * scoreboard is closed.
	 */
	void ScrollScoreboard(int Direction) { m_ScoreboardScroll = std::clamp(m_ScoreboardScroll + Direction, 0, m_ScoreboardMaxScroll); }

	void Spectate(int Server, int ClientId);
	void SpectateNext(bool Reverse);
	bool IsSpectatingRemote() const;
	bool IsWatching(int Server, int ClientId) const;
	/**
	 * Tee of a scoreboard row, `nullptr` while its skin is not loaded yet.
	 */
	const CTeeRenderInfo *PlayerRenderInfo(int Server, int ClientId) const;

	/**
	 * Client id the player renderer is called with for a player of an observed server.
	 *
	 * The renderer already treats negative ids as tees that are not in `m_aClients`,
	 * `-1` and `-2` are taken by the spectator char and the race ghost.
	 */
	static constexpr int RenderClientId(int Server) { return -3 - Server; }
	static constexpr int RenderServer(int ClientId) { return -3 - ClientId; }
	static constexpr bool IsRenderClientId(int ClientId) { return ClientId <= -3; }

	/**
	 * How opaque the players of a server are rendered, `Server` is `-1` for the server
	 * we are connected to.
	 *
	 * Players of a server other than the one being watched cannot be interacted with,
	 * so they are dimmed like players of another team. Free view follows nobody, so
	 * there nothing is dimmed.
	 */
	float PlayerAlpha(int Server) const;

	/**
	 * Calls `Callback(Server, ClientId, Player)` for every player of an observed server
	 * that is currently part of the world.
	 */
	template<typename F>
	void ForEachRemotePlayer(F &&Callback) const
	{
		for(int Server = 0; Server < MAX_OBSERVERS; Server++)
		{
			const CRemoteServer &RemoteServer = m_aServers[Server];
			if(!RemoteServer.IsActive() || !Client()->ObserverOnline(RemoteServer.m_Conn))
			{
				continue;
			}
			for(int i = 0; i < RemoteServer.m_NumVisible; i++)
			{
				const int ClientId = RemoteServer.m_aVisibleIds[i];
				Callback(Server, ClientId, RemoteServer.m_aPlayers[ClientId]);
			}
		}
	}

	vec2 RenderPos(int Server, const CRemotePlayer &Player) const;
	vec2 SpectatePosition();

	const std::vector<CScoreboardRow> &BuildScoreboard();

	/**
	 * One cell of a column layout of the scoreboard rows, either a player or the name of
	 * the server the players below it are on.
	 */
	class CFlowSlot
	{
	public:
		int m_Column;
		int m_RowInColumn;
		const char *m_pHeader; // nullptr unless this cell holds a server name
		const CScoreboardRow *m_pRow; // nullptr unless this cell holds a player
	};
	/**
	 * Walks the rows in column order, repeating the server name at the top of every
	 * column its players spill into.
	 *
	 * @return The number of columns the rows need.
	 */
	template<typename F>
	static int FlowRows(const std::vector<CScoreboardRow> &vRows, int PerColumn, bool SkipSpectators, F &&Callback)
	{
		int Column = 0;
		int RowInColumn = 0;
		const char *pHeaderName = "";
		bool HeaderPending = false;

		for(const CScoreboardRow &Row : vRows)
		{
			if(Row.IsHeader())
			{
				pHeaderName = Row.m_pName;
				HeaderPending = true;
				continue;
			}
			if(SkipSpectators && Row.m_Spectator)
			{
				continue;
			}

			// The repeated name and the first of its rows have to fit together.
			if(RowInColumn + (HeaderPending ? 2 : 1) > PerColumn)
			{
				Column++;
				RowInColumn = 0;
				HeaderPending = true;
			}
			if(HeaderPending)
			{
				Callback({Column, RowInColumn, pHeaderName, nullptr});
				RowInColumn++;
				HeaderPending = false;
			}
			Callback({Column, RowInColumn, nullptr, &Row});
			RowInColumn++;
		}
		return Column + 1;
	}
	void RenderScoreboard(const CUIRect &Screen);

private:
	CRemoteServer m_aServers[MAX_OBSERVERS];

	int m_SpectateServer = -1;
	int m_SpectateClientId = -1;
	vec2 m_SpectatePos = vec2(0.0f, 0.0f);

	int64_t m_NextLocalSpectateTime = 0;
	int64_t m_NextShowDistanceTime = 0;
	float m_LastShowDistanceX = 0.0f;
	float m_LastShowDistanceY = 0.0f;

	// Observers can only attach once the map of the main connection is loaded, so the
	// servers picked in the browser wait here while the main connection comes up.
	int m_NumServers = 0;
	// Scoreboard rows keep the name, so it has to outlive the frame that built them.
	char m_aOwnServerName[64] = "";
	// First shown column of the merged scoreboard and the last one that can be scrolled
	// to, both in columns like the spectator selector.
	int m_ScoreboardScroll = 0;
	int m_ScoreboardMaxScroll = 0;
	std::vector<CServerEntry> m_vPending;

	// Rebuilt once per snapshot, kept to reuse its capacity.
	std::vector<CScoreboardRow> m_vScoreboardRows;
	bool m_ScoreboardDirty = true;

	CRemoteServer &Server(int Conn) { return m_aServers[Conn - IClient::CONN_OBSERVER_FIRST]; }
	void Attach(int Conn, const char *pName);
	void UpdateSkinInfo(CRemotePlayer &Player, const char *pSkinName);
	void AttachPending();
	// Server being watched, -1 is the one we are connected to, -2 is free view.
	int WatchedServer() const;
	void KeepSpectating();
	void SendSpectatorMode(int Server, int SpectatorId);

	static void ConMultiServerConnect(IConsole::IResult *pResult, void *pUserData);
	static void ConMultiServerSpectate(IConsole::IResult *pResult, void *pUserData);
	static void ConMultiServerDisconnect(IConsole::IResult *pResult, void *pUserData);
};

#endif
