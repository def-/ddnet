#include <base/dbg.h>
#include <base/io.h>
#include <base/logger.h>
#include <base/math.h>
#include <base/mem.h>
#include <base/os.h>
#include <base/str.h>

#include <engine/console.h>
#include <engine/shared/config.h>
#include <engine/shared/demo.h>
#include <engine/shared/json.h>
#include <engine/shared/map.h>
#include <engine/shared/network.h>
#include <engine/shared/packer.h>
#include <engine/shared/protocol.h>
#include <engine/shared/protocol_ex.h>
#include <engine/shared/snapshot.h>
#include <engine/shared/teehistorian_ex.h>
#include <engine/shared/uuid_manager.h>
#include <engine/storage.h>

#include <generated/protocol.h>

#include <game/collision.h>
#include <game/gamecore.h>
#include <game/layers.h>
#include <game/mapitems.h>
#include <game/teamscore.h>
#include <game/version.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <vector>

// Converts a teehistorian recording into a demo. Positions come straight from
// the recording, everything the recording lacks (hooks, velocities, weapons,
// freeze, projectiles) is reconstructed by replaying the recorded inputs
// through the shared physics, snapped back to the recorded positions every
// tick so the simulation can never drift away from the truth.

static const char *TOOL_NAME = "teehistorian2demo";

// Extra demo time around the run in --rank mode
static constexpr int RUN_PRE_SECONDS = 5;
static constexpr int RUN_POST_SECONDS = 3;
// Widened margins when the finish position is only known from the timestamp
static constexpr int APPROX_EXTRA_SECONDS = 15;
// Freeze duration of freeze tiles (sv_freeze_delay, virtually never changed)
static constexpr int FREEZE_SECONDS = 3;

// Chunk types, see CTeeHistorian in src/game/server/teehistorian.cpp
enum
{
	TEEHISTORIAN_NONE,
	TEEHISTORIAN_FINISH,
	TEEHISTORIAN_TICK_SKIP,
	TEEHISTORIAN_PLAYER_NEW,
	TEEHISTORIAN_PLAYER_OLD,
	TEEHISTORIAN_INPUT_DIFF,
	TEEHISTORIAN_INPUT_NEW,
	TEEHISTORIAN_MESSAGE,
	TEEHISTORIAN_JOIN,
	TEEHISTORIAN_DROP,
	TEEHISTORIAN_CONSOLE_COMMAND,
	TEEHISTORIAN_EX,
};

// A finish event matching the searched rank, see CConverter::ScanForRank.
struct CRankCandidate
{
	int m_FinishTick;
	int m_Cid;
	int m_Team;
};

class CConverter
{
	struct CPlayer
	{
		bool m_Connected = false;
		bool m_Alive = false;
		int m_X = 0;
		int m_Y = 0;
		int m_PrevX = 0;
		int m_PrevY = 0;
		int m_PrevTick = -1;
		CNetObj_PlayerInput m_Input = {};
		CNetObj_PlayerInput m_SimInput = {};
		int m_Weapon = WEAPON_GUN;
		bool m_aGotWeapons[NUM_WEAPONS] = {true, true}; // everyone has hammer and gun
		int m_LastNextWeapon = 0;
		int m_LastPrevWeapon = 0;
		int m_AttackTick = 0;
		int m_LastFire = 0;
		int m_Score = -9999;

		char m_aName[MAX_NAME_LENGTH] = "";
		char m_aClan[MAX_CLAN_LENGTH] = "";
		int m_Country = -1;
		char m_aSkin[24] = "default";
		int m_UseCustomColor = 0;
		int m_ColorBody = 0;
		int m_ColorFeet = 0;

		// Freeze inferred from the map tiles at the recorded positions
		int m_FreezeEndTick = 0;
		int m_FreezeStartTick = 0;
		bool m_DeepFrozen = false;
		bool m_InFreezeTile = false;

		CCharacterCore m_Core;

		// What a character starts with on every spawn
		void ResetSpawnState()
		{
			m_FreezeEndTick = 0;
			m_FreezeStartTick = 0;
			m_DeepFrozen = false;
			m_InFreezeTile = false;
			m_Weapon = WEAPON_GUN;
			std::fill(std::begin(m_aGotWeapons), std::end(m_aGotWeapons), false);
			m_aGotWeapons[WEAPON_HAMMER] = true;
			m_aGotWeapons[WEAPON_GUN] = true;
		}

		// Name, clan, country and skin: kept across joins on the same client
		// id, since players carried over a map change do not resend them
		void CopyIdentityFrom(const CPlayer &Other)
		{
			str_copy(m_aName, Other.m_aName);
			str_copy(m_aClan, Other.m_aClan);
			m_Country = Other.m_Country;
			str_copy(m_aSkin, Other.m_aSkin);
			m_UseCustomColor = Other.m_UseCustomColor;
			m_ColorBody = Other.m_ColorBody;
			m_ColorFeet = Other.m_ColorFeet;
		}
	};

	IStorage *m_pStorage;
	CDemoRecorder m_Recorder;
	CNetObjHandler m_NetObjHandler;

	CMap m_Map;
	CLayers m_Layers;
	CCollision m_Collision;
	CWorldCore m_WorldCore;
	CTeamsCore m_TeamsCore;
	CTuningParams m_Tuning;

	CPlayer m_aPlayers[MAX_CLIENTS];
	std::vector<std::vector<unsigned char>> m_vTickMessages;

	int m_Tick = 0;
	int m_LastSimTick = 0;
	int m_FirstTick = -1;
	bool m_TickDirty = false;
	int m_CurMaxCid = -1;
	bool m_ExpectPlayers = false;
	int m_StartTick = 0;
	int m_EndTick = std::numeric_limits<int>::max();
	bool m_Done = false;

	// Rank targeting (--rank)
	const std::vector<const char *> *m_pvRankNames = nullptr;
	int m_RankTimeTicks = 0;
	int m_RankExpectedTick = -1;
	std::vector<CRankCandidate> m_vRankCandidates;
	CRankCandidate m_ApproxCandidate = {-1, -1, -1};
	bool m_ApproxChecked = false;
	int m_FilterTeam = -1;
	// Server-side effects are not recorded, reconstructed ones queue here
	struct CPendingEvent
	{
		int m_Type;
		int m_X;
		int m_Y;
		int m_SoundId;
	};
	std::vector<CPendingEvent> m_vPendingEvents;
	struct CGrenade
	{
		int m_Id;
		int m_Owner;
		int m_Weapon;
		vec2 m_StartPos;
		vec2 m_Direction;
		int m_StartTick;
		int m_LastTick;
	};
	std::vector<CGrenade> m_vGrenades;
	struct CLaserBeam
	{
		int m_Id;
		int m_Owner;
		int m_Weapon;
		vec2 m_From;
		vec2 m_Pos;
		vec2 m_Dir;
		float m_Energy;
		int m_EvalTick;
		int m_Bounces;
		bool m_Done;
	};
	std::vector<CLaserBeam> m_vLasers;
	std::vector<vec2> m_vSpawnPoints;
	// Weapon pickups in the map: tile index -> weapon, and which weapons the
	// map can grant at all
	std::map<int, int> m_PickupTiles;
	bool m_aMapWeapons[NUM_WEAPONS] = {};
	int m_NextEventId = 0;
	int m_NextGrenadeId = 0;
	int m_NextLaserId = 0;
	int m_MarkerStartTick = -1;
	int m_MarkerFinishTick = -1;

	// stats
	int m_NumChunks = 0;
	int m_NumTicks = 0;
	int m_NumSnapshots = 0;
	int m_NumChatMessages = 0;
	int m_MaxPlayersSeen = 0;
	int m_NumPlayerHookTicks = 0;
	int m_NumFrozenTicks = 0;

public:
	CConverter(IStorage *pStorage, CSnapshotDelta *pSnapshotDelta) :
		m_pStorage(pStorage),
		m_Recorder(pSnapshotDelta)
	{
	}

	bool LoadMap(const char *pMapPath, const char *pExpectedSha256)
	{
		if(!m_Map.Load(m_pStorage, pMapPath, IStorage::TYPE_ALL_OR_ABSOLUTE))
		{
			log_error(TOOL_NAME, "Failed to load map '%s'", pMapPath);
			return false;
		}
		if(pExpectedSha256 != nullptr)
		{
			char aMapSha256[SHA256_MAXSTRSIZE];
			sha256_str(m_Map.Sha256(), aMapSha256, sizeof(aMapSha256));
			if(str_comp(aMapSha256, pExpectedSha256) != 0)
			{
				log_warn(TOOL_NAME, "Map sha256 mismatch: teehistorian expects %s, map file has %s", pExpectedSha256, aMapSha256);
			}
		}
		m_Layers.Init(&m_Map, false, false);
		m_Collision.Init(&m_Layers);
		// Spawn points: an instant respawn (kill bind, /r, death tiles) has no
		// chunk of its own in the recording, it is detected as a position
		// jump onto a spawn point
		for(int y = 0; y < m_Collision.GetHeight(); y++)
		{
			for(int x = 0; x < m_Collision.GetWidth(); x++)
			{
				const int Index = y * m_Collision.GetWidth() + x;
				for(const CTile *pTile : {m_Collision.GameLayer(), m_Collision.FrontLayer()})
				{
					if(pTile == nullptr)
						continue;
					const int TileIndex = pTile[Index].m_Index;
					if(TileIndex >= ENTITY_OFFSET + ENTITY_SPAWN && TileIndex <= ENTITY_OFFSET + ENTITY_SPAWN_BLUE)
						m_vSpawnPoints.emplace_back(x * 32 + 16, y * 32 + 16);
					int PickupWeapon = -1;
					if(TileIndex == ENTITY_OFFSET + ENTITY_WEAPON_SHOTGUN)
						PickupWeapon = WEAPON_SHOTGUN;
					else if(TileIndex == ENTITY_OFFSET + ENTITY_WEAPON_GRENADE)
						PickupWeapon = WEAPON_GRENADE;
					else if(TileIndex == ENTITY_OFFSET + ENTITY_WEAPON_LASER)
						PickupWeapon = WEAPON_LASER;
					if(PickupWeapon >= 0)
					{
						m_PickupTiles[Index] = PickupWeapon;
						m_aMapWeapons[PickupWeapon] = true;
					}
				}
			}
		}
		return true;
	}

	void ApplyTuning(const json_value *pTuning)
	{
		if(pTuning == nullptr || pTuning->type != json_object)
			return;
		for(unsigned i = 0; i < pTuning->u.object.length; i++)
		{
			const json_value *pValue = pTuning->u.object.values[i].value;
			if(pValue->type != json_string)
				continue;
			const int Value = str_toint(pValue->u.string.ptr);
			if(!m_Tuning.Set(pTuning->u.object.values[i].name, Value / 100.0f))
			{
				log_warn(TOOL_NAME, "Unknown tuning parameter '%s'", pTuning->u.object.values[i].name);
			}
		}
	}

	bool StartDemo(const char *pOutputPath, const char *pMapName)
	{
		void *pMapData;
		unsigned MapSize;
		IOHANDLE MapFile = io_open(m_Map.Path(), IOFLAG_READ);
		if(!MapFile || !io_read_all(MapFile, &pMapData, &MapSize))
		{
			log_error(TOOL_NAME, "Failed to read map file for embedding");
			if(MapFile)
				io_close(MapFile);
			return false;
		}
		io_close(MapFile);

		const int Error = m_Recorder.Start(m_pStorage, nullptr, pOutputPath, GAME_NETVERSION, pMapName,
			m_Map.Sha256(), m_Map.Crc(), "server", MapSize, (unsigned char *)pMapData, nullptr, nullptr, nullptr);
		free(pMapData);
		if(Error != 0)
		{
			log_error(TOOL_NAME, "Failed to start demo recorder for '%s'", pOutputPath);
			return false;
		}
		return true;
	}

	void SetTickRange(int StartTick, int EndTick)
	{
		m_StartTick = StartTick;
		m_EndTick = EndTick;
	}

	// Only scan for finishes of the given rank instead of recording: one name
	// matches PLAYER_FINISH events, multiple names TEAM_FINISH events. The
	// matches are collected in RankCandidates(), parsing stops after LastTick.
	void ScanForRank(const std::vector<const char *> *pvNames, int TimeTicks, int ExpectedTick, int LastTick)
	{
		m_pvRankNames = pvNames;
		m_RankTimeTicks = TimeTicks;
		m_RankExpectedTick = ExpectedTick;
		m_StartTick = std::numeric_limits<int>::max();
		m_EndTick = LastTick;
	}

	const std::vector<CRankCandidate> &RankCandidates() const { return m_vRankCandidates; }

	// Fallback for recordings older than April 2024 without finish events: the
	// player found by name when the scan passes the rank's wall-clock offset.
	const CRankCandidate &ApproxRankCandidate() const { return m_ApproxCandidate; }

	// Hide all players outside the given team, including their messages.
	void SetTeamFilter(int Team) { m_FilterTeam = Team; }

	// Add demo timeline markers at the run's start and finish.
	void SetRankMarkers(int StartTick, int FinishTick)
	{
		m_MarkerStartTick = StartTick;
		m_MarkerFinishTick = FinishTick;
	}

	// Seed player identities into another converter: they are not resent on
	// map changes, so they can only be learned from previous recordings of
	// the same server.
	void CopyPlayerIdentitiesTo(CConverter *pTarget) const
	{
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			const CPlayer &Player = m_aPlayers[Cid];
			if(Player.m_aName[0] == '\0')
				continue;
			pTarget->m_aPlayers[Cid].CopyIdentityFrom(Player);
		}
	}

	// Returns false when the stream ends (finish chunk, end of data, end of
	// the requested tick range or parse error).
	bool ParseChunk(CUnpacker *pUnpacker)
	{
		if(m_Done)
			return false;
		m_NumChunks++;
		if((m_NumChunks & 0xFFFFFF) == 0)
		{
			const int Seconds = m_Tick / SERVER_TICK_SPEED;
			log_info(TOOL_NAME, "progress: parsed up to %d:%02d:%02d", Seconds / 3600, Seconds / 60 % 60, Seconds % 60);
		}
		const int TypeOrCid = pUnpacker->GetInt();
		if(pUnpacker->Error())
		{
			log_info(TOOL_NAME, "stream end after %d chunks (no finish chunk)", m_NumChunks);
			return false;
		}

		if(TypeOrCid >= 0)
		{
			const int Dx = pUnpacker->GetInt();
			const int Dy = pUnpacker->GetInt();
			if(pUnpacker->Error() || TypeOrCid >= MAX_CLIENTS)
				return false;
			OnPlayerChunk(TypeOrCid);
			CPlayer *pPlayer = &m_aPlayers[TypeOrCid];
			pPlayer->m_X += Dx;
			pPlayer->m_Y += Dy;
			return true;
		}

		switch(-TypeOrCid)
		{
		case TEEHISTORIAN_FINISH:
			log_info(TOOL_NAME, "finish chunk after %d chunks", m_NumChunks);
			return false;
		case TEEHISTORIAN_TICK_SKIP:
		{
			const int Dt = pUnpacker->GetInt();
			if(pUnpacker->Error() || Dt < 0)
				return false;
			FlushTick();
			m_Tick += Dt + 1;
			m_CurMaxCid = -1;
			m_ExpectPlayers = true;
			m_TickDirty = true;
			break;
		}
		case TEEHISTORIAN_PLAYER_NEW:
		{
			const int Cid = pUnpacker->GetInt();
			const int X = pUnpacker->GetInt();
			const int Y = pUnpacker->GetInt();
			if(pUnpacker->Error() || Cid < 0 || Cid >= MAX_CLIENTS)
				return false;
			OnPlayerChunk(Cid);
			CPlayer *pPlayer = &m_aPlayers[Cid];
			pPlayer->m_Connected = true; // JOIN might be missing at the start of the file
			pPlayer->m_Alive = true;
			pPlayer->m_X = X;
			pPlayer->m_Y = Y;
			pPlayer->m_PrevX = X;
			pPlayer->m_PrevY = Y;
			pPlayer->m_PrevTick = -1;
			pPlayer->ResetSpawnState();
			pPlayer->m_Core.Init(&m_WorldCore, &m_Collision, &m_TeamsCore);
			pPlayer->m_Core.Reset();
			// The id gates the team checks: without it hooks on players are
			// released again on the next tick (CanKeepHook against id -1).
			// Set only after Reset(), which must run like a fresh server core
			// (with the id set it would unhook via the uninitialized
			// m_HookedPlayer of a newly constructed core).
			pPlayer->m_Core.m_Id = Cid;
			pPlayer->m_Core.m_Tuning = m_Tuning;
			pPlayer->m_Core.m_Pos = vec2(X, Y);
			m_WorldCore.m_apCharacters[Cid] = &pPlayer->m_Core;
			break;
		}
		case TEEHISTORIAN_PLAYER_OLD:
		{
			const int Cid = pUnpacker->GetInt();
			if(pUnpacker->Error() || Cid < 0 || Cid >= MAX_CLIENTS)
				return false;
			OnPlayerChunk(Cid);
			m_aPlayers[Cid].m_Alive = false;
			m_WorldCore.m_apCharacters[Cid] = nullptr;
			break;
		}
		case TEEHISTORIAN_INPUT_DIFF:
		case TEEHISTORIAN_INPUT_NEW:
		{
			const int Cid = pUnpacker->GetInt();
			int aInput[10];
			for(int &Value : aInput)
			{
				Value = pUnpacker->GetInt();
			}
			if(pUnpacker->Error() || Cid < 0 || Cid >= MAX_CLIENTS)
				return false;
			m_TickDirty = true;
			CPlayer *pPlayer = &m_aPlayers[Cid];
			int *pInput = (int *)&pPlayer->m_Input;
			for(size_t i = 0; i < std::size(aInput); i++)
			{
				if(-TypeOrCid == TEEHISTORIAN_INPUT_DIFF)
					pInput[i] += aInput[i];
				else
					pInput[i] = aInput[i];
			}
			break;
		}
		case TEEHISTORIAN_MESSAGE:
		{
			const int Cid = pUnpacker->GetInt();
			const int MsgSize = pUnpacker->GetInt();
			if(pUnpacker->Error() || MsgSize < 0)
				return false;
			const unsigned char *pMsg = pUnpacker->GetRaw(MsgSize);
			if(pUnpacker->Error() || Cid < 0 || Cid >= MAX_CLIENTS)
				return false;
			m_TickDirty = true;
			OnMessage(Cid, pMsg, MsgSize);
			break;
		}
		case TEEHISTORIAN_JOIN:
		{
			const int Cid = pUnpacker->GetInt();
			if(pUnpacker->Error() || Cid < 0 || Cid >= MAX_CLIENTS)
				return false;
			m_TickDirty = true;
			// Keep the identity: players carried over a map change do not
			// resend Cl_StartInfo and old recordings have no player-name
			// chunks, so it may have been seeded from a previous recording.
			// Fresh players overwrite it with their Cl_StartInfo right away.
			CPlayer Fresh;
			Fresh.CopyIdentityFrom(m_aPlayers[Cid]);
			Fresh.m_Connected = true;
			m_aPlayers[Cid] = Fresh;
			m_TeamsCore.Team(Cid, 0);
			m_WorldCore.m_apCharacters[Cid] = nullptr;
			break;
		}
		case TEEHISTORIAN_DROP:
		{
			const int Cid = pUnpacker->GetInt();
			pUnpacker->GetString();
			if(pUnpacker->Error() || Cid < 0 || Cid >= MAX_CLIENTS)
				return false;
			m_TickDirty = true;
			m_aPlayers[Cid].m_Connected = false;
			m_aPlayers[Cid].m_Alive = false;
			m_WorldCore.m_apCharacters[Cid] = nullptr;
			break;
		}
		case TEEHISTORIAN_CONSOLE_COMMAND:
		{
			const int Cid = pUnpacker->GetInt();
			const int FlagMask = pUnpacker->GetInt();
			const char *pCommand = pUnpacker->GetString();
			char aChat[512];
			str_format(aChat, sizeof(aChat), "/%s", pCommand);
			// Save, load and timeout codes are secrets, the demo must not
			// hand them to whoever watches it
			const bool Secret = str_comp_nocase(pCommand, "save") == 0 || str_comp_nocase(pCommand, "load") == 0 || str_comp_nocase(pCommand, "timeout") == 0;
			const int NumArgs = pUnpacker->GetInt();
			if(pUnpacker->Error() || NumArgs < 0 || NumArgs > 128)
				return false;
			for(int i = 0; i < NumArgs; i++)
			{
				const char *pArg = pUnpacker->GetString();
				if(!Secret)
				{
					str_append(aChat, " ");
					str_append(aChat, pArg);
				}
			}
			if(pUnpacker->Error())
				return false;
			m_TickDirty = true;
			// Chat is not recorded in teehistorian, but chat commands are
			// executed through the console, show them as chat messages.
			if(Cid >= 0 && Cid < MAX_CLIENTS && (FlagMask & CFGFLAG_CHAT))
			{
				QueueChat(Cid, 0, aChat);
			}
			break;
		}
		case TEEHISTORIAN_EX:
		{
			const unsigned char *pUuidData = pUnpacker->GetRaw(sizeof(CUuid));
			const int Size = pUnpacker->GetInt();
			if(pUnpacker->Error() || Size < 0)
				return false;
			CUuid Uuid;
			mem_copy(&Uuid, pUuidData, sizeof(Uuid));
			const unsigned char *pData = pUnpacker->GetRaw(Size);
			if(pUnpacker->Error())
				return false;
			OnExChunk(Uuid, pData, Size);
			break;
		}
		default:
			log_error(TOOL_NAME, "Unknown chunk type %d at tick %d after %d chunks, stopping", -TypeOrCid, m_Tick, m_NumChunks);
			return false;
		}
		return true;
	}

	bool Finish()
	{
		FlushTick();
		if(m_NumSnapshots == 0)
		{
			m_Recorder.Stop(IDemoRecorder::EStopMode::REMOVE_FILE);
			log_error(TOOL_NAME, "No ticks in the selected time range, recording covers %d:%02d:%02d hours",
				m_Tick / SERVER_TICK_SPEED / 3600, m_Tick / SERVER_TICK_SPEED / 60 % 60, m_Tick / SERVER_TICK_SPEED % 60);
			return false;
		}
		m_Recorder.Stop(IDemoRecorder::EStopMode::KEEP_FILE);
		log_info(TOOL_NAME, "Wrote %d snapshots covering %d ticks (%d:%02d min), %d chat messages, %d players seen, %d ticks with player hooks, %d ticks frozen",
			m_NumSnapshots, m_NumTicks,
			m_NumTicks / SERVER_TICK_SPEED / 60, m_NumTicks / SERVER_TICK_SPEED % 60,
			m_NumChatMessages, m_MaxPlayersSeen, m_NumPlayerHookTicks, m_NumFrozenTicks);
		return true;
	}

private:
	// Player position chunks (PLAYER_NEW, PLAYER_OLD and position diffs) advance
	// the tick implicitly when their client id is not increasing within a tick.
	void OnPlayerChunk(int Cid)
	{
		if(m_ExpectPlayers)
		{
			m_ExpectPlayers = false;
		}
		else if(m_CurMaxCid < 0 || Cid <= m_CurMaxCid)
		{
			FlushTick();
			m_Tick += 1;
		}
		m_CurMaxCid = Cid;
		m_TickDirty = true;
	}

	void OnMessage(int Cid, const unsigned char *pMsgData, int MsgSize)
	{
		CUnpacker Unpacker;
		Unpacker.Reset(pMsgData, MsgSize);
		CMsgPacker UuidPacker(NETMSG_EX, true);
		int Msg;
		bool Sys;
		CUuid Uuid;
		if(UnpackMessageId(&Msg, &Sys, &Uuid, &Unpacker, &UuidPacker) != UNPACKMESSAGE_OK || Sys)
			return;

		void *pRawMsg = m_NetObjHandler.SecureUnpackMsg(Msg, &Unpacker);
		if(!pRawMsg)
			return;

		CPlayer *pPlayer = &m_aPlayers[Cid];
		switch(Msg)
		{
		case NETMSGTYPE_CL_STARTINFO:
		case NETMSGTYPE_CL_CHANGEINFO:
		{
			// Cl_StartInfo and Cl_ChangeInfo have identical layouts
			const CNetMsg_Cl_StartInfo *pInfo = (const CNetMsg_Cl_StartInfo *)pRawMsg;
			str_copy(pPlayer->m_aName, pInfo->m_pName);
			str_copy(pPlayer->m_aClan, pInfo->m_pClan);
			pPlayer->m_Country = pInfo->m_Country;
			str_copy(pPlayer->m_aSkin, pInfo->m_pSkin);
			pPlayer->m_UseCustomColor = pInfo->m_UseCustomColor;
			pPlayer->m_ColorBody = pInfo->m_ColorBody;
			pPlayer->m_ColorFeet = pInfo->m_ColorFeet;
			break;
		}
		case NETMSGTYPE_CL_SAY:
		{
			// Not recorded by current servers (TeeHistorianRecordMsg filters
			// it), but present in old recordings.
			const CNetMsg_Cl_Say *pSay = (const CNetMsg_Cl_Say *)pRawMsg;
			QueueChat(Cid, pSay->m_Team ? 1 : 0, pSay->m_pMessage);
			break;
		}
		case NETMSGTYPE_CL_KILL:
		{
			if(!IncludePlayer(Cid))
				break;
			CPacker Packer;
			Packer.Reset();
			Packer.AddInt((NETMSGTYPE_SV_KILLMSG << 1) | 0);
			Packer.AddInt(Cid); // killer
			Packer.AddInt(Cid); // victim
			Packer.AddInt(WEAPON_SELF);
			Packer.AddInt(0); // mode special
			QueueMessage(&Packer);
			break;
		}
		case NETMSGTYPE_CL_EMOTICON:
		{
			if(!IncludePlayer(Cid))
				break;
			const CNetMsg_Cl_Emoticon *pEmoticon = (const CNetMsg_Cl_Emoticon *)pRawMsg;
			CPacker Packer;
			Packer.Reset();
			Packer.AddInt((NETMSGTYPE_SV_EMOTICON << 1) | 0);
			Packer.AddInt(Cid);
			Packer.AddInt(pEmoticon->m_Emoticon);
			QueueMessage(&Packer);
			break;
		}
		default:
			break;
		}
	}

	bool IncludePlayer(int Cid) const
	{
		return m_FilterTeam < 0 || m_TeamsCore.Team(Cid) == m_FilterTeam;
	}

	void QueueMessage(const CPacker *pPacker)
	{
		if(pPacker->Error())
			return;
		std::vector<unsigned char> vData(pPacker->Size());
		mem_copy(vData.data(), pPacker->Data(), pPacker->Size());
		m_vTickMessages.push_back(std::move(vData));
	}

	void QueueChat(int Cid, int Team, const char *pMessage)
	{
		if(!IncludePlayer(Cid))
			return;
		CPacker Packer;
		Packer.Reset();
		Packer.AddInt((NETMSGTYPE_SV_CHAT << 1) | 0);
		Packer.AddInt(Team);
		Packer.AddInt(Cid);
		Packer.AddString(pMessage, -1);
		QueueMessage(&Packer);
		if(m_Tick >= m_StartTick && m_Tick <= m_EndTick)
			m_NumChatMessages++;
	}

	void OnExChunk(CUuid Uuid, const unsigned char *pData, int Size)
	{
		const int Type = g_UuidManager.LookupUuid(Uuid);
		CUnpacker Unpacker;
		Unpacker.Reset(pData, Size);
		switch(Type)
		{
		case TEEHISTORIAN_PLAYER_TEAM:
		{
			const int Cid = Unpacker.GetInt();
			const int Team = Unpacker.GetInt();
			if(!Unpacker.Error() && Cid >= 0 && Cid < MAX_CLIENTS)
				m_TeamsCore.Team(Cid, Team);
			break;
		}
		case TEEHISTORIAN_PLAYER_NAME:
		{
			const int Cid = Unpacker.GetInt();
			const char *pName = Unpacker.GetString();
			if(!Unpacker.Error() && Cid >= 0 && Cid < MAX_CLIENTS)
				str_copy(m_aPlayers[Cid].m_aName, pName);
			break;
		}
		case TEEHISTORIAN_PLAYER_FINISH:
		{
			const int Cid = Unpacker.GetInt();
			const int TimeTicks = Unpacker.GetInt();
			if(!Unpacker.Error() && Cid >= 0 && Cid < MAX_CLIENTS)
			{
				m_aPlayers[Cid].m_Score = -TimeTicks / SERVER_TICK_SPEED;
				if(m_pvRankNames != nullptr && m_pvRankNames->size() == 1 && absolute(TimeTicks - m_RankTimeTicks) <= 1 &&
					str_comp(m_aPlayers[Cid].m_aName, (*m_pvRankNames)[0]) == 0)
				{
					m_vRankCandidates.push_back({m_Tick, Cid, m_TeamsCore.Team(Cid)});
				}
			}
			break;
		}
		case TEEHISTORIAN_TEAM_FINISH:
		{
			const int Team = Unpacker.GetInt();
			const int TimeTicks = Unpacker.GetInt();
			if(Unpacker.Error())
				break;
			for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
			{
				if(m_aPlayers[Cid].m_Connected && m_TeamsCore.Team(Cid) == Team)
					m_aPlayers[Cid].m_Score = -TimeTicks / SERVER_TICK_SPEED;
			}
			if(m_pvRankNames != nullptr && m_pvRankNames->size() >= 2 && absolute(TimeTicks - m_RankTimeTicks) <= 1)
			{
				// Camera target: the first roster name found in the team
				int CandidateCid = -1;
				for(const char *pName : *m_pvRankNames)
				{
					for(int Cid = 0; Cid < MAX_CLIENTS && CandidateCid < 0; Cid++)
					{
						if(m_aPlayers[Cid].m_Connected && m_TeamsCore.Team(Cid) == Team && str_comp(m_aPlayers[Cid].m_aName, pName) == 0)
							CandidateCid = Cid;
					}
					if(CandidateCid >= 0)
						break;
				}
				if(CandidateCid >= 0)
					m_vRankCandidates.push_back({m_Tick, CandidateCid, Team});
			}
			break;
		}
		default:
			break;
		}
	}

	// An instant respawn produces no join/leave chunks, only a position jump
	// onto a spawn point: reset the state a fresh character starts with
	void DetectRespawns()
	{
		if(m_vSpawnPoints.empty())
			return;
		for(auto &Player : m_aPlayers)
		{
			if(!Player.m_Alive || Player.m_PrevTick != m_Tick - 1)
				continue;
			const vec2 Pos(Player.m_X, Player.m_Y);
			if(distance(vec2(Player.m_PrevX, Player.m_PrevY), Pos) < 6 * 32)
				continue;
			bool AtSpawn = false;
			for(const vec2 &Spawn : m_vSpawnPoints)
			{
				if(distance(Spawn, Pos) < 64.0f)
				{
					AtSpawn = true;
					break;
				}
			}
			if(!AtSpawn)
				continue;
			Player.ResetSpawnState();
			Player.m_Core.SetHookedPlayer(-1);
			Player.m_Core.m_HookState = HOOK_IDLE;
		}
	}

	// Freeze is not part of the recording, but it is almost always caused by
	// the map: infer it from the tiles at the recorded positions. Freeze from
	// the switch layer or from entities (freeze lasers) is not covered.
	void UpdateFreeze()
	{
		// Scan passes run without a map
		if(m_Layers.GameLayer() == nullptr)
			return;
		for(auto &Player : m_aPlayers)
		{
			if(!Player.m_Alive)
				continue;
			const int Index = m_Collision.GetPureMapIndex(vec2(Player.m_X, Player.m_Y));
			const int Tile = m_Collision.GetTileIndex(Index);
			const int FrontTile = m_Collision.GetFrontTileIndex(Index);
			// The switch layer also holds freeze tiles: freezing ones are only
			// honored when unconditional (switch number 0, always active),
			// unfreezing ones always (a possibly-off switch must not leave a
			// player visibly frozen forever)
			const auto PickupTile = m_PickupTiles.find(Index);
			if(PickupTile != m_PickupTiles.end())
				Player.m_aGotWeapons[PickupTile->second] = true;
			const int SwitchType = m_Collision.GetSwitchType(Index);
			const bool SwitchAlways = m_Collision.GetSwitchNumber(Index) == 0;
			Player.m_InFreezeTile = Tile == TILE_FREEZE || FrontTile == TILE_FREEZE || (SwitchType == TILE_FREEZE && SwitchAlways);
			if(Tile == TILE_DFREEZE || FrontTile == TILE_DFREEZE || (SwitchType == TILE_DFREEZE && SwitchAlways))
				Player.m_DeepFrozen = true;
			else if(Tile == TILE_DUNFREEZE || FrontTile == TILE_DUNFREEZE || SwitchType == TILE_DUNFREEZE)
				Player.m_DeepFrozen = false;
			if(Player.m_InFreezeTile)
			{
				// The server only refreshes the freeze timer once per second
				// while standing in freeze (CCharacter::Freeze), players
				// unfreeze up to a second after their last refresh, not after
				// their last tile contact
				if(!IsFrozen(Player) || Player.m_FreezeStartTick < m_Tick - SERVER_TICK_SPEED)
				{
					Player.m_FreezeStartTick = m_Tick;
					Player.m_FreezeEndTick = m_Tick + FREEZE_SECONDS * SERVER_TICK_SPEED;
				}
			}
			else if(Tile == TILE_UNFREEZE || FrontTile == TILE_UNFREEZE || SwitchType == TILE_UNFREEZE)
			{
				Player.m_FreezeEndTick = 0;
			}
		}
	}

	bool IsFrozen(const CPlayer &Player) const
	{
		return Player.m_DeepFrozen || Player.m_FreezeEndTick > m_Tick;
	}

	// Advance the guided simulation to the current tick: physics gives us hook
	// state, velocity and angle, the recorded positions are the ground truth
	// that the simulation is snapped back to.
	void Simulate()
	{
		const int SimTicks = std::clamp(m_Tick - m_LastSimTick, 0, 50);
		for(int t = 0; t < SimTicks; t++)
		{
			for(auto &Player : m_aPlayers)
			{
				if(!Player.m_Alive)
					continue;
				// Snap to the recorded positions before ticking: hooks flying
				// at other players test against their position this tick, one
				// tick of drift at high speeds is enough to miss the attach
				Player.m_Core.m_Pos = vec2(Player.m_X, Player.m_Y);
			}
			for(auto &Player : m_aPlayers)
			{
				if(!Player.m_Alive)
					continue;
				Player.m_Core.m_Input = Player.m_SimInput;
				if(IsFrozen(Player))
				{
					// Frozen tees cannot move, jump or hook
					Player.m_Core.m_Input.m_Direction = 0;
					Player.m_Core.m_Input.m_Jump = 0;
					Player.m_Core.m_Input.m_Hook = 0;
				}
				Player.m_Core.Tick(true);
			}
			for(auto &Player : m_aPlayers)
			{
				if(!Player.m_Alive)
					continue;
				Player.m_Core.Move();
				Player.m_Core.Quantize();
			}
		}
		m_LastSimTick = m_Tick;

		for(auto &Player : m_aPlayers)
		{
			if(!Player.m_Alive)
				continue;
			const vec2 RecordedPos(Player.m_X, Player.m_Y);
			if(Player.m_PrevTick >= 0 && m_Tick > Player.m_PrevTick)
			{
				const float Gap = m_Tick - Player.m_PrevTick;
				Player.m_Core.m_Vel = vec2(Player.m_X - Player.m_PrevX, Player.m_Y - Player.m_PrevY) / Gap;
			}
			Player.m_Core.m_Pos = RecordedPos;
		}

		// Assist pass: re-test hooks against the snapped ground-truth
		// positions with a slightly generous radius. Small timing offsets
		// between the replayed inputs and the recorded positions otherwise
		// make hooks on fast-moving players miss: a flying hook is checked at
		// its tip, and a hook that already missed and retracted while the
		// button is still held is re-tested along the aim ray (the recorded
		// motion of such pairs shows the real hook connected).
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			CPlayer &Player = m_aPlayers[Cid];
			if(!Player.m_Alive || IsFrozen(Player))
				continue;
			CCharacterCore &Core = Player.m_Core;
			if(Core.m_Input.m_Hook == 0 || Core.m_HookState == HOOK_IDLE || Core.m_HookState == HOOK_GRABBED)
				continue;
			vec2 From;
			vec2 To;
			if(Core.m_HookState == HOOK_FLYING)
			{
				From = Core.m_HookPos - Core.m_HookDir * m_Tuning.m_HookFireSpeed;
				To = Core.m_HookPos;
			}
			else
			{
				const vec2 TargetDirection = normalize(vec2(Core.m_Input.m_TargetX, Core.m_Input.m_TargetY));
				From = Core.m_Pos + TargetDirection * CCharacterCore::PhysicalSize() * 1.5f;
				To = Core.m_Pos + TargetDirection * m_Tuning.m_HookLength;
			}
			int ClosestCid = -1;
			float ClosestDistance = 0.0f;
			for(int OtherCid = 0; OtherCid < MAX_CLIENTS; OtherCid++)
			{
				const CPlayer &Other = m_aPlayers[OtherCid];
				if(OtherCid == Cid || !Other.m_Alive || !m_TeamsCore.CanCollide(Cid, OtherCid))
					continue;
				const float TargetDistance = distance(Core.m_Pos, Other.m_Core.m_Pos);
				// A real hook cannot reach beyond its length or through walls
				if(TargetDistance > m_Tuning.m_HookLength - 20.0f)
					continue;
				if(m_Collision.IntersectLine(Core.m_Pos, Other.m_Core.m_Pos, nullptr, nullptr) != 0)
					continue;
				vec2 ClosestPoint;
				if(!closest_point_on_line(From, To, Other.m_Core.m_Pos, ClosestPoint))
					continue;
				if(distance(Other.m_Core.m_Pos, ClosestPoint) < CCharacterCore::PhysicalSize() * 1.5f && (ClosestCid == -1 || TargetDistance < ClosestDistance))
				{
					ClosestCid = OtherCid;
					ClosestDistance = TargetDistance;
				}
			}
			if(ClosestCid >= 0)
			{
				Core.m_HookState = HOOK_GRABBED;
				Core.SetHookedPlayer(ClosestCid);
				Core.m_HookTick = 0;
			}
		}
	}

	void CycleWeapon(CPlayer &Player, int Step)
	{
		for(int Offset = 1; Offset < NUM_WEAPONS; Offset++)
		{
			const int Weapon = (Player.m_Weapon + Step * Offset + Offset * NUM_WEAPONS) % NUM_WEAPONS;
			if(Player.m_aGotWeapons[Weapon] && Weapon != WEAPON_NINJA)
			{
				Player.m_Weapon = Weapon;
				return;
			}
		}
	}

	bool InRecordWindow() const { return m_Tick >= m_StartTick && m_Tick <= m_EndTick; }

	// The recording has no server-side events, reconstruct the visible
	// effects of firing: hammer hits (which also unfreeze the target) and
	// grenades (rendered as projectiles, exploding in UpdateGrenades)
	void OnFire(int Cid)
	{
		CPlayer &Player = m_aPlayers[Cid];
		const vec2 Pos(Player.m_X, Player.m_Y);
		vec2 Direction = normalize(vec2(Player.m_Input.m_TargetX, Player.m_Input.m_TargetY));
		if(Direction.x == 0.0f && Direction.y == 0.0f)
			Direction = vec2(1.0f, 0.0f);
		const bool Emit = InRecordWindow() && IncludePlayer(Cid);
		if(Player.m_Weapon == WEAPON_HAMMER)
		{
			const vec2 HitPos = Pos + Direction * CCharacterCore::PhysicalSize() * 0.75f;
			for(int OtherCid = 0; OtherCid < MAX_CLIENTS; OtherCid++)
			{
				CPlayer &Other = m_aPlayers[OtherCid];
				if(OtherCid == Cid || !Other.m_Alive || !m_TeamsCore.CanCollide(Cid, OtherCid))
					continue;
				// Slightly wider than the server (radius + proximity): the
				// replayed inputs are up to a tick offset from the recorded
				// positions, borderline hits must not lose their unfreeze
				if(distance(vec2(Other.m_X, Other.m_Y), HitPos) > CCharacterCore::PhysicalSize() * 2.0f)
					continue;
				// A hammer hit unfreezes the target
				Other.m_FreezeEndTick = 0;
				if(Emit)
					m_vPendingEvents.push_back({NETEVENTTYPE_HAMMERHIT, Other.m_X, Other.m_Y, 0});
			}
			if(Emit)
				m_vPendingEvents.push_back({NETEVENTTYPE_SOUNDWORLD, Player.m_X, Player.m_Y, SOUND_HAMMER_FIRE});
		}
		else if(Player.m_Weapon == WEAPON_GRENADE || Player.m_Weapon == WEAPON_GUN)
		{
			m_vGrenades.push_back({m_NextGrenadeId, Cid, Player.m_Weapon, Pos, Direction, m_Tick, m_Tick});
			m_NextGrenadeId = (m_NextGrenadeId + 1) % 512;
			if(Emit)
				m_vPendingEvents.push_back({NETEVENTTYPE_SOUNDWORLD, Player.m_X, Player.m_Y, Player.m_Weapon == WEAPON_GRENADE ? SOUND_GRENADE_FIRE : SOUND_GUN_FIRE});
		}
		else if(Player.m_Weapon == WEAPON_LASER || Player.m_Weapon == WEAPON_SHOTGUN)
		{
			CLaserBeam Beam = {m_NextLaserId, Cid, Player.m_Weapon, Pos, Pos, Direction, m_Tuning.m_LaserReach, m_Tick, 0, false};
			m_NextLaserId = (m_NextLaserId + 1) % 512;
			BeamBounce(Beam);
			m_vLasers.push_back(Beam);
			if(Emit)
				m_vPendingEvents.push_back({NETEVENTTYPE_SOUNDWORLD, Player.m_X, Player.m_Y, Player.m_Weapon == WEAPON_LASER ? SOUND_LASER_FIRE : SOUND_SHOTGUN_FIRE});
		}
	}

	// One bounce step of a laser/shotgun beam, mirroring CLaser::DoBounce:
	// stop on a player of the same team (a laser hit unfreezes them) or
	// reflect off the wall until the energy or bounce budget runs out
	void BeamBounce(CLaserBeam &Beam)
	{
		Beam.m_EvalTick = m_Tick;
		if(Beam.m_Energy < 0.0f)
		{
			Beam.m_Done = true;
			return;
		}
		vec2 To = Beam.m_Pos + Beam.m_Dir * Beam.m_Energy;
		const bool HitWall = m_Collision.IntersectLine(Beam.m_Pos, To, nullptr, &To) != 0;
		int HitCid = -1;
		float HitDistance = 0.0f;
		for(int OtherCid = 0; OtherCid < MAX_CLIENTS; OtherCid++)
		{
			const CPlayer &Other = m_aPlayers[OtherCid];
			if(!Other.m_Alive || (OtherCid == Beam.m_Owner && Beam.m_Bounces == 0) || !m_TeamsCore.CanCollide(Beam.m_Owner, OtherCid))
				continue;
			vec2 ClosestPoint;
			if(!closest_point_on_line(Beam.m_Pos, To, Other.m_Core.m_Pos, ClosestPoint))
				continue;
			const float Distance = distance(Beam.m_Pos, Other.m_Core.m_Pos);
			if(distance(Other.m_Core.m_Pos, ClosestPoint) < CCharacterCore::PhysicalSize() && (HitCid == -1 || Distance < HitDistance))
			{
				HitCid = OtherCid;
				HitDistance = Distance;
			}
		}
		if(HitCid >= 0)
		{
			Beam.m_From = Beam.m_Pos;
			Beam.m_Pos = vec2(m_aPlayers[HitCid].m_X, m_aPlayers[HitCid].m_Y);
			Beam.m_Energy = -1.0f;
			if(Beam.m_Weapon == WEAPON_LASER)
				m_aPlayers[HitCid].m_FreezeEndTick = 0; // a laser hit unfreezes
		}
		else if(HitWall)
		{
			Beam.m_From = Beam.m_Pos;
			Beam.m_Pos = To;
			vec2 TempPos = Beam.m_Pos;
			vec2 TempDir = Beam.m_Dir * 4.0f;
			m_Collision.MovePoint(&TempPos, &TempDir, 1.0f, nullptr);
			Beam.m_Pos = TempPos;
			Beam.m_Dir = normalize(TempDir);
			Beam.m_Energy -= distance(Beam.m_From, Beam.m_Pos) + m_Tuning.m_LaserBounceCost;
			Beam.m_Bounces++;
			if(Beam.m_Bounces > m_Tuning.m_LaserBounceNum)
				Beam.m_Energy = -1.0f;
			if(InRecordWindow() && IncludePlayer(Beam.m_Owner))
				m_vPendingEvents.push_back({NETEVENTTYPE_SOUNDWORLD, round_to_int(Beam.m_Pos.x), round_to_int(Beam.m_Pos.y), SOUND_LASER_BOUNCE});
		}
		else
		{
			Beam.m_From = Beam.m_Pos;
			Beam.m_Pos = To;
			Beam.m_Energy = -1.0f;
		}
	}

	void UpdateLasers()
	{
		if(m_Layers.GameLayer() == nullptr)
			return;
		const int DelayTicks = std::max(1, round_to_int(m_Tuning.m_LaserBounceDelay * (float)SERVER_TICK_SPEED / 1000.0f));
		for(size_t i = 0; i < m_vLasers.size();)
		{
			CLaserBeam &Beam = m_vLasers[i];
			if(m_Tick > Beam.m_EvalTick + DelayTicks)
			{
				if(Beam.m_Done)
				{
					m_vLasers.erase(m_vLasers.begin() + i);
					continue;
				}
				BeamBounce(Beam);
			}
			i++;
		}
	}

	// Advance the reconstructed projectiles along their ballistic curve until
	// they hit the map, a player of the same team (grenades) or their
	// lifetime ends
	void UpdateGrenades()
	{
		if(m_Layers.GameLayer() == nullptr)
			return;
		for(size_t i = 0; i < m_vGrenades.size();)
		{
			CGrenade &Grenade = m_vGrenades[i];
			const bool IsGrenade = Grenade.m_Weapon == WEAPON_GRENADE;
			const float Curvature = IsGrenade ? m_Tuning.m_GrenadeCurvature : m_Tuning.m_GunCurvature;
			const float Speed = IsGrenade ? m_Tuning.m_GrenadeSpeed : m_Tuning.m_GunSpeed;
			const float Lifetime = IsGrenade ? m_Tuning.m_GrenadeLifetime : m_Tuning.m_GunLifetime;
			bool Gone = false;
			vec2 HitPos = Grenade.m_StartPos;
			for(int Tick = Grenade.m_LastTick + 1; Tick <= m_Tick && !Gone; Tick++)
			{
				const vec2 PrevPos = CalcPos(Grenade.m_StartPos, Grenade.m_Direction, Curvature, Speed, (Tick - 1 - Grenade.m_StartTick) / (float)SERVER_TICK_SPEED);
				vec2 CurPos = CalcPos(Grenade.m_StartPos, Grenade.m_Direction, Curvature, Speed, (Tick - Grenade.m_StartTick) / (float)SERVER_TICK_SPEED);
				if(m_Collision.IntersectLine(PrevPos, CurPos, &CurPos, nullptr))
				{
					Gone = true;
				}
				for(int OtherCid = 0; OtherCid < MAX_CLIENTS && IsGrenade && !Gone; OtherCid++)
				{
					const CPlayer &Other = m_aPlayers[OtherCid];
					if(OtherCid == Grenade.m_Owner || !Other.m_Alive || !m_TeamsCore.CanCollide(Grenade.m_Owner, OtherCid))
						continue;
					if(distance(vec2(Other.m_X, Other.m_Y), CurPos) < CCharacterCore::PhysicalSize() + 6.0f)
						Gone = true;
				}
				if(Tick - Grenade.m_StartTick > Lifetime * (float)SERVER_TICK_SPEED)
					Gone = true;
				HitPos = CurPos;
			}
			Grenade.m_LastTick = m_Tick;
			if(Gone)
			{
				if(IsGrenade && InRecordWindow() && IncludePlayer(Grenade.m_Owner))
					m_vPendingEvents.push_back({NETEVENTTYPE_EXPLOSION, round_to_int(HitPos.x), round_to_int(HitPos.y), 0});
				m_vGrenades.erase(m_vGrenades.begin() + i);
			}
			else
			{
				i++;
			}
		}
	}

	void FlushTick()
	{
		if(!m_TickDirty)
			return;

		if(m_pvRankNames != nullptr && !m_ApproxChecked && m_RankExpectedTick >= 0 && m_Tick >= m_RankExpectedTick)
		{
			m_ApproxChecked = true;
			for(const char *pName : *m_pvRankNames)
			{
				for(int Cid = 0; Cid < MAX_CLIENTS && m_ApproxCandidate.m_Cid < 0; Cid++)
				{
					if(m_aPlayers[Cid].m_Connected && str_comp(m_aPlayers[Cid].m_aName, pName) == 0)
						m_ApproxCandidate = {m_RankExpectedTick, Cid, m_TeamsCore.Team(Cid)};
				}
				if(m_ApproxCandidate.m_Cid >= 0)
					break;
			}
		}

		DetectRespawns();
		UpdateFreeze();
		UpdateGrenades();
		UpdateLasers();

		if(m_Tick > m_EndTick)
		{
			m_Done = true;
			m_TickDirty = false;
			m_vTickMessages.clear();
			return;
		}
		const bool Record = m_Tick >= m_StartTick;
		if(Record)
		{
			Simulate();
			RecordSnapshot();
			if(m_MarkerStartTick >= 0 && m_Tick >= m_MarkerStartTick)
			{
				m_Recorder.AddDemoMarker();
				m_MarkerStartTick = -1;
			}
			if(m_MarkerFinishTick >= 0 && m_Tick >= m_MarkerFinishTick)
			{
				m_Recorder.AddDemoMarker();
				m_MarkerFinishTick = -1;
			}
			for(const auto &vMessage : m_vTickMessages)
			{
				m_Recorder.RecordMessage(vMessage.data(), vMessage.size());
			}
		}
		m_vTickMessages.clear();
		if(!Record)
			m_vPendingEvents.clear();

		int NumPlayers = 0;
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			CPlayer &Player = m_aPlayers[Cid];
			Player.m_SimInput = Player.m_Input;
			if(Player.m_Alive)
			{
				NumPlayers++;
				if(Player.m_Input.m_WantedWeapon > 0)
				{
					const int Wanted = std::clamp(Player.m_Input.m_WantedWeapon - 1, (int)WEAPON_HAMMER, (int)WEAPON_NINJA);
					// Selecting only works for owned weapons; treat a selection
					// as proof of ownership only if the map can grant it
					if(m_aMapWeapons[Wanted])
						Player.m_aGotWeapons[Wanted] = true;
					if(Player.m_aGotWeapons[Wanted])
						Player.m_Weapon = Wanted;
				}
				// Mouse-wheel switching cycles through the owned weapons.
				// The input fields are press+release counters wrapping at 64,
				// count actual presses like the server does.
				for(int Press = CountInput(Player.m_LastNextWeapon, Player.m_Input.m_NextWeapon).m_Presses; Press > 0; Press--)
					CycleWeapon(Player, 1);
				Player.m_LastNextWeapon = Player.m_Input.m_NextWeapon;
				for(int Press = CountInput(Player.m_LastPrevWeapon, Player.m_Input.m_PrevWeapon).m_Presses; Press > 0; Press--)
					CycleWeapon(Player, -1);
				Player.m_LastPrevWeapon = Player.m_Input.m_PrevWeapon;
				if(CountInput(Player.m_LastFire, Player.m_Input.m_Fire).m_Presses > 0 && !IsFrozen(Player))
				{
					Player.m_AttackTick = m_Tick;
					OnFire(Cid);
				}
				Player.m_LastFire = Player.m_Input.m_Fire;
				Player.m_PrevX = Player.m_X;
				Player.m_PrevY = Player.m_Y;
				Player.m_PrevTick = m_Tick;
			}
		}
		m_MaxPlayersSeen = std::max(m_MaxPlayersSeen, NumPlayers);

		if(Record)
		{
			if(m_FirstTick < 0)
				m_FirstTick = m_Tick;
			m_NumTicks = m_Tick - m_FirstTick + 1;
		}
		m_TickDirty = false;
	}

	void RecordSnapshot()
	{
		CSnapshotBuilder Builder;
		Builder.Init();

		CNetObj_GameInfo *pGameInfo = (CNetObj_GameInfo *)Builder.NewItemRaw(NETOBJTYPE_GAMEINFO, 0, sizeof(CNetObj_GameInfo));
		if(pGameInfo)
		{
			mem_zero(pGameInfo, sizeof(*pGameInfo));
			pGameInfo->m_RoundStartTick = m_FirstTick < 0 ? m_Tick : m_FirstTick;
			pGameInfo->m_RoundNum = 1;
			pGameInfo->m_RoundCurrent = 1;
		}

		CNetObj_GameInfoEx *pGameInfoEx = (CNetObj_GameInfoEx *)Builder.NewItemRaw(NETOBJTYPE_GAMEINFOEX, 0, sizeof(CNetObj_GameInfoEx));
		if(pGameInfoEx)
		{
			pGameInfoEx->m_Flags = GAMEINFOFLAG_TIMESCORE |
					       GAMEINFOFLAG_GAMETYPE_RACE |
					       GAMEINFOFLAG_GAMETYPE_DDRACE |
					       GAMEINFOFLAG_GAMETYPE_DDNET |
					       GAMEINFOFLAG_RACE |
					       GAMEINFOFLAG_UNLIMITED_AMMO |
					       GAMEINFOFLAG_RACE_RECORD_MESSAGE |
					       GAMEINFOFLAG_ALLOW_EYE_WHEEL |
					       GAMEINFOFLAG_ALLOW_HOOK_COLL |
					       GAMEINFOFLAG_ALLOW_ZOOM |
					       GAMEINFOFLAG_ENTITIES_DDNET |
					       GAMEINFOFLAG_ENTITIES_DDRACE |
					       GAMEINFOFLAG_ENTITIES_RACE;
			pGameInfoEx->m_Version = GAMEINFO_CURVERSION;
			pGameInfoEx->m_Flags2 = GAMEINFOFLAG2_HUD_DDRACE | GAMEINFOFLAG2_DDRACE_TEAM;
		}

		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			CPlayer *pPlayer = &m_aPlayers[Cid];
			if(!pPlayer->m_Connected || !IncludePlayer(Cid))
				continue;

			CNetObj_ClientInfo *pClientInfo = (CNetObj_ClientInfo *)Builder.NewItemRaw(NETOBJTYPE_CLIENTINFO, Cid, sizeof(CNetObj_ClientInfo));
			if(pClientInfo)
			{
				StrToInts(pClientInfo->m_aName, std::size(pClientInfo->m_aName), pPlayer->m_aName);
				StrToInts(pClientInfo->m_aClan, std::size(pClientInfo->m_aClan), pPlayer->m_aClan);
				pClientInfo->m_Country = pPlayer->m_Country;
				StrToInts(pClientInfo->m_aSkin, std::size(pClientInfo->m_aSkin), pPlayer->m_aSkin);
				pClientInfo->m_UseCustomColor = pPlayer->m_UseCustomColor;
				pClientInfo->m_ColorBody = pPlayer->m_ColorBody;
				pClientInfo->m_ColorFeet = pPlayer->m_ColorFeet;
			}

			CNetObj_PlayerInfo *pPlayerInfo = (CNetObj_PlayerInfo *)Builder.NewItemRaw(NETOBJTYPE_PLAYERINFO, Cid, sizeof(CNetObj_PlayerInfo));
			if(pPlayerInfo)
			{
				pPlayerInfo->m_Local = 0;
				pPlayerInfo->m_ClientId = Cid;
				pPlayerInfo->m_Team = pPlayer->m_Alive ? TEAM_RED : TEAM_SPECTATORS;
				pPlayerInfo->m_Score = pPlayer->m_Score;
				pPlayerInfo->m_Latency = 0;
			}

			if(pPlayer->m_Alive)
			{
				const bool Frozen = IsFrozen(*pPlayer);
				CNetObj_Character *pCharacter = (CNetObj_Character *)Builder.NewItemRaw(NETOBJTYPE_CHARACTER, Cid, sizeof(CNetObj_Character));
				if(pCharacter)
				{
					mem_zero(pCharacter, sizeof(*pCharacter));
					pPlayer->m_Core.Write(pCharacter);
					pCharacter->m_Tick = m_Tick;
					pCharacter->m_X = pPlayer->m_X;
					pCharacter->m_Y = pPlayer->m_Y;
					pCharacter->m_PlayerFlags = pPlayer->m_Input.m_PlayerFlags;
					pCharacter->m_Health = 10;
					pCharacter->m_Armor = 0;
					pCharacter->m_AmmoCount = 0;
					pCharacter->m_Weapon = pPlayer->m_Weapon;
					pCharacter->m_Emote = Frozen ? EMOTE_PAIN : EMOTE_NORMAL;
					pCharacter->m_AttackTick = pPlayer->m_AttackTick;
				}

				CNetObj_DDNetCharacter *pDDNetCharacter = (CNetObj_DDNetCharacter *)Builder.NewItemRaw(NETOBJTYPE_DDNETCHARACTER, Cid, sizeof(CNetObj_DDNetCharacter));
				if(pDDNetCharacter)
				{
					mem_zero(pDDNetCharacter, sizeof(*pDDNetCharacter));
					pDDNetCharacter->m_Flags = CHARACTERFLAG_WEAPON_HAMMER | CHARACTERFLAG_WEAPON_GUN;
					if(pPlayer->m_Weapon == WEAPON_SHOTGUN)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_SHOTGUN;
					else if(pPlayer->m_Weapon == WEAPON_GRENADE)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_GRENADE;
					else if(pPlayer->m_Weapon == WEAPON_LASER)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_LASER;
					if(pPlayer->m_InFreezeTile)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_IN_FREEZE;
					pDDNetCharacter->m_FreezeEnd = pPlayer->m_DeepFrozen ? -1 : (Frozen ? pPlayer->m_FreezeEndTick : 0);
					pDDNetCharacter->m_FreezeStart = Frozen ? pPlayer->m_FreezeStartTick : 0;
					pDDNetCharacter->m_Jumps = 2;
					pDDNetCharacter->m_JumpedTotal = pPlayer->m_Core.m_JumpedTotal;
					pDDNetCharacter->m_NinjaActivationTick = -1;
					pDDNetCharacter->m_TargetX = pPlayer->m_Input.m_TargetX;
					pDDNetCharacter->m_TargetY = pPlayer->m_Input.m_TargetY;
					pDDNetCharacter->m_TuneZoneOverride = TuneZone::OVERRIDE_NONE;
				}
			}
		}

		for(const CGrenade &Grenade : m_vGrenades)
		{
			if(!IncludePlayer(Grenade.m_Owner))
				continue;
			CNetObj_Projectile *pProjectile = (CNetObj_Projectile *)Builder.NewItemRaw(NETOBJTYPE_PROJECTILE, Grenade.m_Id, sizeof(CNetObj_Projectile));
			if(pProjectile)
			{
				pProjectile->m_X = round_to_int(Grenade.m_StartPos.x);
				pProjectile->m_Y = round_to_int(Grenade.m_StartPos.y);
				pProjectile->m_VelX = round_to_int(Grenade.m_Direction.x * 100.0f);
				pProjectile->m_VelY = round_to_int(Grenade.m_Direction.y * 100.0f);
				pProjectile->m_Type = Grenade.m_Weapon;
				pProjectile->m_StartTick = Grenade.m_StartTick;
			}
		}
		for(const CLaserBeam &Beam : m_vLasers)
		{
			if(!IncludePlayer(Beam.m_Owner))
				continue;
			CNetObj_Laser *pLaser = (CNetObj_Laser *)Builder.NewItemRaw(NETOBJTYPE_LASER, Beam.m_Id, sizeof(CNetObj_Laser));
			if(pLaser)
			{
				pLaser->m_X = round_to_int(Beam.m_Pos.x);
				pLaser->m_Y = round_to_int(Beam.m_Pos.y);
				pLaser->m_FromX = round_to_int(Beam.m_From.x);
				pLaser->m_FromY = round_to_int(Beam.m_From.y);
				pLaser->m_StartTick = Beam.m_EvalTick;
			}
		}
		for(const CPendingEvent &Event : m_vPendingEvents)
		{
			if(Event.m_Type == NETEVENTTYPE_SOUNDWORLD)
			{
				CNetEvent_SoundWorld *pEvent = (CNetEvent_SoundWorld *)Builder.NewItemRaw(NETEVENTTYPE_SOUNDWORLD, m_NextEventId, sizeof(CNetEvent_SoundWorld));
				if(pEvent)
				{
					pEvent->m_X = Event.m_X;
					pEvent->m_Y = Event.m_Y;
					pEvent->m_SoundId = Event.m_SoundId;
				}
			}
			else
			{
				CNetEvent_Common *pEvent = (CNetEvent_Common *)Builder.NewItemRaw(Event.m_Type, m_NextEventId, Event.m_Type == NETEVENTTYPE_HAMMERHIT ? sizeof(CNetEvent_HammerHit) : sizeof(CNetEvent_Explosion));
				if(pEvent)
				{
					pEvent->m_X = Event.m_X;
					pEvent->m_Y = Event.m_Y;
				}
			}
			m_NextEventId = (m_NextEventId + 1) % 1024;
		}
		m_vPendingEvents.clear();

		CSnapshotBuffer Buffer;
		const int SnapshotSize = Builder.Finish(&Buffer);
		m_Recorder.RecordSnapshot(m_Tick, Buffer.AsSnapshot(), SnapshotSize);
		m_NumSnapshots++;
		for(const auto &Player : m_aPlayers)
		{
			if(Player.m_Alive && Player.m_Core.m_HookState == HOOK_GRABBED && Player.m_Core.HookedPlayer() >= 0)
			{
				m_NumPlayerHookTicks++;
				break;
			}
		}
		for(const auto &Player : m_aPlayers)
		{
			if(Player.m_Alive && IsFrozen(Player))
			{
				m_NumFrozenTicks++;
				break;
			}
		}
	}
};

// Accepts plain seconds, M:SS or H:MM:SS. Returns -1 on invalid input.
static int ParseTimeSeconds(const char *pStr)
{
	int Total = 0;
	int Current = 0;
	int NumDigits = 0;
	int NumParts = 0;
	for(const char *p = pStr;; p++)
	{
		if(*p >= '0' && *p <= '9')
		{
			Current = Current * 10 + (*p - '0');
			NumDigits++;
		}
		else if(*p == ':' || *p == '\0')
		{
			if(NumDigits == 0 || ++NumParts > 3)
				return -1;
			Total = Total * 60 + Current;
			Current = 0;
			NumDigits = 0;
			if(*p == '\0')
				return Total;
		}
		else
		{
			return -1;
		}
	}
}

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

// Opens a teehistorian input and feeds its chunks through a converter. A
// sliding window is used so the input is only read sequentially, teehistorian
// files can be larger than memory. Single chunks are limited by the 64 KiB
// packer buffer on the writing side, so as long as the window is refilled
// before it runs lower than that, chunks never appear truncated before the
// actual end of the file.
class CTeehistorianReader
{
	static constexpr size_t WINDOW_SIZE = 16 * 1024 * 1024;
	static constexpr size_t REFILL_THRESHOLD = 256 * 1024;

	IOHANDLE m_File = nullptr;
	std::vector<unsigned char> m_vWindow;
	size_t m_Fill = 0;
	size_t m_Pos = 0;
	bool m_EndOfFile = false;

	void Refill()
	{
		memmove(m_vWindow.data(), m_vWindow.data() + m_Pos, m_Fill - m_Pos);
		m_Fill -= m_Pos;
		m_Pos = 0;
		while(m_Fill < m_vWindow.size())
		{
			const unsigned Read = io_read(m_File, m_vWindow.data() + m_Fill, m_vWindow.size() - m_Fill);
			if(Read == 0)
			{
				m_EndOfFile = true;
				break;
			}
			m_Fill += Read;
		}
	}

public:
	~CTeehistorianReader()
	{
		if(m_File != nullptr)
			io_close(m_File);
	}

	// Opens the file and validates the magic bytes. Returns the parsed json
	// header on success, which the caller frees.
	json_value *Open(const char *pPath)
	{
		m_File = io_open(pPath, IOFLAG_READ);
		if(!m_File)
		{
			log_error(TOOL_NAME, "Failed to open '%s'", pPath);
			return nullptr;
		}

		m_vWindow.resize(WINDOW_SIZE);
		Refill();

		// Magic bytes, then the json header terminated by a null byte, both
		// must fit into the initial window.
		static const CUuid TEEHISTORIAN_UUID = CalculateUuid("teehistorian@ddnet.tw");
		if(m_Fill < sizeof(CUuid) + 1 || mem_comp(m_vWindow.data(), &TEEHISTORIAN_UUID, sizeof(CUuid)) != 0)
		{
			log_error(TOOL_NAME, "'%s' is not a teehistorian file", pPath);
			return nullptr;
		}
		const unsigned char *pHeaderEnd = (const unsigned char *)memchr(m_vWindow.data() + sizeof(CUuid), 0, m_Fill - sizeof(CUuid));
		if(pHeaderEnd == nullptr)
		{
			log_error(TOOL_NAME, "Missing or unterminated teehistorian header");
			return nullptr;
		}
		json_value *pHeader = JsonParse((const char *)m_vWindow.data() + sizeof(CUuid), pHeaderEnd - (m_vWindow.data() + sizeof(CUuid)));
		if(pHeader == nullptr)
		{
			log_error(TOOL_NAME, "Failed to parse teehistorian header");
			return nullptr;
		}
		m_Pos = pHeaderEnd - m_vWindow.data() + 1;
		return pHeader;
	}

	// Feeds all chunks to the converter until it stops or the input ends.
	void ParseChunks(CConverter *pConverter)
	{
		CUnpacker Unpacker;
		Unpacker.Reset(m_vWindow.data() + m_Pos, m_Fill - m_Pos);
		while(true)
		{
			if(!m_EndOfFile && m_Fill - m_Pos < REFILL_THRESHOLD)
			{
				Refill();
				Unpacker.Reset(m_vWindow.data(), m_Fill);
			}
			if(!pConverter->ParseChunk(&Unpacker))
				break;
			m_Pos = m_Fill - Unpacker.RemainingSize();
		}
		io_close(m_File);
		m_File = nullptr;
	}
};

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

	// Optional previous recordings of the same server (oldest first), only
	// used to learn the names of players carried over map changes
	int ArgIndex = 4;
	std::vector<const char *> vPrevPaths;
	while(ArgIndex + 1 < argc && str_comp(argv[ArgIndex], "--prev") == 0)
	{
		vPrevPaths.push_back(argv[ArgIndex + 1]);
		ArgIndex += 2;
	}
	const bool RankMode = ArgIndex < argc && str_comp(argv[ArgIndex], "--rank") == 0;
	if(argc < 4 || (!RankMode && argc - ArgIndex > 2) || (RankMode && argc - ArgIndex < 4))
	{
		log_error(TOOL_NAME, "Usage: %s <input.teehistorian> <map.map> <output.demo> [--prev <old.teehistorian>]... [start] [end]", TOOL_NAME);
		log_error(TOOL_NAME, "       %s <input.teehistorian> <map.map> <output.demo> [--prev <old.teehistorian>]... --rank <time> <offset|-> <name> [name] ...", TOOL_NAME);
		log_error(TOOL_NAME, "start/end limit the converted time range, given as seconds, M:SS or H:MM:SS");
		log_error(TOOL_NAME, "--rank converts only the run of the player (or team of players) finishing in");
		log_error(TOOL_NAME, "<time> seconds around <offset> into the recording, hiding all other teams");
		log_error(TOOL_NAME, "--prev recordings (oldest first) provide the names of players that joined");
		log_error(TOOL_NAME, "before the recording started");
		return -1;
	}
	int StartSeconds = 0;
	int EndSeconds = -1;
	int RankTimeTicks = 0;
	int RankExpectedTick = -1;
	std::vector<const char *> vRankNames;
	if(RankMode)
	{
		float RankTime;
		if(!str_tofloat(argv[ArgIndex + 1], &RankTime) || RankTime <= 0.0f)
		{
			log_error(TOOL_NAME, "Invalid rank time '%s'", argv[ArgIndex + 1]);
			return -1;
		}
		RankTimeTicks = round_to_int(RankTime * (float)SERVER_TICK_SPEED);
		if(str_comp(argv[ArgIndex + 2], "-") != 0)
		{
			const int Offset = ParseTimeSeconds(argv[ArgIndex + 2]);
			if(Offset < 0)
			{
				log_error(TOOL_NAME, "Invalid rank offset '%s'", argv[ArgIndex + 2]);
				return -1;
			}
			RankExpectedTick = Offset * SERVER_TICK_SPEED;
		}
		for(int i = ArgIndex + 3; i < argc; i++)
		{
			vRankNames.push_back(argv[i]);
		}
	}
	else
	{
		if(argc - ArgIndex >= 1)
		{
			StartSeconds = ParseTimeSeconds(argv[ArgIndex]);
			if(StartSeconds < 0)
			{
				log_error(TOOL_NAME, "Invalid start time '%s'", argv[ArgIndex]);
				return -1;
			}
		}
		if(argc - ArgIndex >= 2)
		{
			EndSeconds = ParseTimeSeconds(argv[ArgIndex + 1]);
			if(EndSeconds < 0 || EndSeconds <= StartSeconds)
			{
				log_error(TOOL_NAME, "Invalid end time '%s'", argv[ArgIndex + 1]);
				return -1;
			}
		}
	}

	std::unique_ptr<CSnapshotDelta> pSnapshotDelta = CreateSnapshotDelta();

	// Learn the identities of players carried over from before the recording
	// started by parsing the previous recordings, oldest first
	CConverter NameScanner(pStorage.get(), pSnapshotDelta.get());
	NameScanner.SetTickRange(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
	for(const char *pPrevPath : vPrevPaths)
	{
		CTeehistorianReader PrevReader;
		json_value *pPrevHeader = PrevReader.Open(pPrevPath);
		if(pPrevHeader == nullptr)
		{
			return -1;
		}
		json_value_free(pPrevHeader);
		PrevReader.ParseChunks(&NameScanner);
	}

	int DemoStartTick = StartSeconds * SERVER_TICK_SPEED;
	int DemoEndTick = EndSeconds < 0 ? std::numeric_limits<int>::max() : EndSeconds * SERVER_TICK_SPEED;
	CRankCandidate RankTarget = {-1, -1, -1};
	if(RankMode)
	{
		// First pass: find the rank's finish event. Without simulation and
		// demo output this only parses the stream. Allow for the server tick
		// falling behind wall-clock time during long sessions.
		constexpr int ScanSlackTicks = 30 * 60 * SERVER_TICK_SPEED;
		CConverter Scanner(pStorage.get(), pSnapshotDelta.get());
		NameScanner.CopyPlayerIdentitiesTo(&Scanner);
		Scanner.ScanForRank(&vRankNames, RankTimeTicks, RankExpectedTick,
			RankExpectedTick < 0 ? std::numeric_limits<int>::max() : RankExpectedTick + ScanSlackTicks);
		CTeehistorianReader ScanReader;
		json_value *pScanHeader = ScanReader.Open(argv[1]);
		if(pScanHeader == nullptr)
		{
			return -1;
		}
		json_value_free(pScanHeader);
		ScanReader.ParseChunks(&Scanner);

		const CRankCandidate *pBest = nullptr;
		for(const CRankCandidate &Candidate : Scanner.RankCandidates())
		{
			if(pBest == nullptr || (RankExpectedTick >= 0 && absolute(Candidate.m_FinishTick - RankExpectedTick) < absolute(pBest->m_FinishTick - RankExpectedTick)))
				pBest = &Candidate;
		}
		int PreSeconds = RUN_PRE_SECONDS;
		int PostSeconds = RUN_POST_SECONDS;
		if(pBest == nullptr && Scanner.ApproxRankCandidate().m_Cid >= 0)
		{
			// Recordings older than April 2024 have no finish events, position
			// the window on the rank's wall-clock offset with wider margins.
			pBest = &Scanner.ApproxRankCandidate();
			PreSeconds += APPROX_EXTRA_SECONDS;
			PostSeconds += APPROX_EXTRA_SECONDS;
			log_warn(TOOL_NAME, "No finish event found (old recording), using the rank timestamp instead");
		}
		if(pBest == nullptr)
		{
			log_error(TOOL_NAME, "No finish in %.2f seconds by '%s' found", RankTimeTicks / (float)SERVER_TICK_SPEED, vRankNames[0]);
			return -1;
		}
		RankTarget = *pBest;
		const int FinishSeconds = RankTarget.m_FinishTick / SERVER_TICK_SPEED;
		log_info(TOOL_NAME, "found finish at %d:%02d:%02d cid=%d team=%d", FinishSeconds / 3600, FinishSeconds / 60 % 60, FinishSeconds % 60, RankTarget.m_Cid, RankTarget.m_Team);
		DemoStartTick = std::max(0, RankTarget.m_FinishTick - RankTimeTicks - PreSeconds * SERVER_TICK_SPEED);
		DemoEndTick = RankTarget.m_FinishTick + PostSeconds * SERVER_TICK_SPEED;
	}

	CTeehistorianReader Reader;
	json_value *pHeader = Reader.Open(argv[1]);
	if(pHeader == nullptr)
	{
		return -1;
	}

	char aMapName[128] = "unknown";
	const json_value &MapName = (*pHeader)["map_name"];
	if(MapName.type == json_string)
		str_copy(aMapName, MapName);
	const json_value &MapSha256 = (*pHeader)["map_sha256"];
	const json_value &GameType = (*pHeader)["game_type"];
	const json_value &StartTime = (*pHeader)["start_time"];
	const json_value &ServerName = (*pHeader)["server_name"];
	log_info(TOOL_NAME, "map='%s' game_type='%s' start_time='%s' server='%s'",
		aMapName,
		GameType.type == json_string ? (const char *)GameType : "?",
		StartTime.type == json_string ? (const char *)StartTime : "?",
		ServerName.type == json_string ? (const char *)ServerName : "?");

	CConverter Converter(pStorage.get(), pSnapshotDelta.get());
	NameScanner.CopyPlayerIdentitiesTo(&Converter);
	if(!Converter.LoadMap(argv[2], MapSha256.type == json_string ? (const char *)MapSha256 : nullptr))
	{
		json_value_free(pHeader);
		return -1;
	}

	Converter.ApplyTuning(json_object_get(pHeader, "tuning"));

	Converter.SetTickRange(DemoStartTick, DemoEndTick);
	if(RankMode)
	{
		Converter.SetTeamFilter(RankTarget.m_Team);
		Converter.SetRankMarkers(RankTarget.m_FinishTick - RankTimeTicks, RankTarget.m_FinishTick);
	}

	if(!Converter.StartDemo(argv[3], aMapName))
	{
		json_value_free(pHeader);
		return -1;
	}

	Reader.ParseChunks(&Converter);

	const bool Success = Converter.Finish();

	json_value_free(pHeader);
	if(Success && RankMode)
	{
		// Machine-readable result for scripts driving the tool
		printf("{\"cid\":%d,\"team\":%d,\"demo_start_tick\":%d,\"run_start_tick\":%d,\"finish_tick\":%d}\n",
			RankTarget.m_Cid, RankTarget.m_Team, DemoStartTick, RankTarget.m_FinishTick - RankTimeTicks, RankTarget.m_FinishTick);
	}
	return Success ? 0 : -1;
}
