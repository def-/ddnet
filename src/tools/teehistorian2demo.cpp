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
#include <memory>
#include <optional>
#include <vector>

#if defined(CONF_PLATFORM_EMSCRIPTEN)
#include <emscripten/emscripten.h>
#endif

static const char *TOOL_NAME = "teehistorian2demo";

// Extra demo time around the run in --rank mode
// How long a snapshot id stays out of use after the entity that had it went
// away, the same as the server's id pool
static constexpr int SNAP_ID_HOLD_SECONDS = 5;
static constexpr int RUN_PRE_SECONDS = 5;
static constexpr int RUN_POST_SECONDS = 3;
// Widened margins when the finish position is only known from the timestamp
static constexpr int APPROX_EXTRA_SECONDS = 15;
// How far from a player entities are still sent, CPlayer::m_ShowDistance
static constexpr vec2 SHOW_DISTANCE = vec2(1200.0f, 800.0f);

// SERVER_TICK_SPEED is an enum, which cannot be used in float arithmetic
static constexpr int TICK_SPEED = SERVER_TICK_SPEED;

// The ninja weapon, from datasrc/content.py
static constexpr int NINJA_DURATION_MS = 15000;
static constexpr int NINJA_MOVETIME_MS = 200;
static constexpr int NINJA_VELOCITY = 50;

class CInputSource
{
public:
	virtual ~CInputSource() = default;
	// Returns the number of bytes read, 0 on end of file or error.
	virtual unsigned Read(void *pBuffer, unsigned Size) = 0;
};

class CFileInputSource : public CInputSource
{
	IOHANDLE m_File;

public:
	CFileInputSource(IOHANDLE File) :
		m_File(File)
	{
	}
	~CFileInputSource() override
	{
		io_close(m_File);
	}
	unsigned Read(void *pBuffer, unsigned Size) override
	{
		return io_read(m_File, pBuffer, Size);
	}
};

#if defined(CONF_PLATFORM_EMSCRIPTEN)
// Stream the recording from a URL. The converter only reads sequentially and
// stops early when the requested time range has been converted, so only the
// needed prefix is downloaded and never held in memory as a whole. Uses plain
// fetch streaming without range requests, so on-the-fly decompressing servers
// work as well. The functions suspend the wasm runtime via ASYNCIFY.
// clang-format off
EM_ASYNC_JS(int, teehistorian_stream_open, (const char *pUrl), {
	try
	{
		const response = await fetch(UTF8ToString(pUrl));
		if(!response.ok || !response.body)
		{
			return response.status === 0 ? -1 : -response.status;
		}
		Module.teehistorianStream = {reader: response.body.getReader(), buffer: null, offset: 0, done: false};
		return 0;
	}
	catch(error)
	{
		console.error(error);
		return -1;
	}
});

EM_ASYNC_JS(int, teehistorian_stream_read, (void *pBuffer, int Size), {
	const stream = Module.teehistorianStream;
	if(!stream)
	{
		return 0;
	}
	try
	{
		while(stream.buffer === null || stream.offset >= stream.buffer.length)
		{
			if(stream.done)
			{
				return 0;
			}
			const {value, done} = await stream.reader.read();
			if(done)
			{
				stream.done = true;
				return 0;
			}
			stream.buffer = value;
			stream.offset = 0;
		}
		const n = Math.min(Size, stream.buffer.length - stream.offset);
		HEAPU8.set(stream.buffer.subarray(stream.offset, stream.offset + n), pBuffer);
		stream.offset += n;
		return n;
	}
	catch(error)
	{
		console.error(error);
		return 0;
	}
});

EM_JS(void, teehistorian_stream_close, (), {
	if(Module.teehistorianStream)
	{
		try
		{
			Module.teehistorianStream.reader.cancel();
		}
		catch(error)
		{
		}
		Module.teehistorianStream = null;
	}
});

// clang-format on

class CHttpInputSource : public CInputSource
{
public:
	~CHttpInputSource() override
	{
		teehistorian_stream_close();
	}
	bool Open(const char *pUrl)
	{
		const int Result = teehistorian_stream_open(pUrl);
		if(Result != 0)
		{
			log_error(TOOL_NAME, "Failed to open '%s' (%d)", pUrl, -Result);
			return false;
		}
		return true;
	}
	unsigned Read(void *pBuffer, unsigned Size) override
	{
		const int Result = teehistorian_stream_read(pBuffer, Size);
		return Result <= 0 ? 0 : Result;
	}
};
#endif

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

// The server settings the replayed entities depend on. Teehistorian writes
// every setting that differs from the built-in default into its header, so
// these defaults are the ones from config_variables.h.
struct CServerConfig
{
	int m_SvHit = 1;
	int m_SvFreezeDelay = 3;
	int m_SvDeepfly = 1;
	int m_SvDraggerRange = 700;
	int m_SvPlasmaRange = 700;
	int m_SvPlasmaPerSec = 3;
	int m_SvShotgunBulletSound = 0;
	int m_SvDestroyBulletsOnDeath = 1;
	int m_SvDestroyLasersOnDeath = 0;
	int m_SvOldTeleportWeapons = 0;
	// Set by a tile of the map as well, see CConverter::LoadMap
	int m_SvOldLaser = 0;
	int m_SvEndlessDrag = 0;
};

class CConverter;

// The entities the server builds from the map's tile layers (pickups, doors,
// freeze lasers, draggers, plasma and shotgun turrets) and the ones a shot
// creates. None of this is in the recording, so the replay runs the same code
// the server runs: these entities freeze and unfreeze players, take their
// weapons away and are half of what a run looks like. Ported from
// src/game/server/entities, kept close to the original so the two can be
// compared. Everything a server entity does to a character goes through the
// converter, which owns the players.
class CReplayEntity
{
public:
	// The order CGameWorld ticks and snaps its entity types in
	enum EType
	{
		PROJECTILE = 0,
		LASER,
		PICKUP,
		NUM_TYPES
	};

	CReplayEntity(CConverter *pConverter, EType EntityType, vec2 Pos, int Layer, int Number);
	virtual ~CReplayEntity();
	CReplayEntity(const CReplayEntity &Other) = delete;
	CReplayEntity &operator=(const CReplayEntity &Other) = delete;

	virtual void Tick() {}
	virtual void Snap() {}

	EType EntityType() const { return m_EntityType; }
	bool MarkedForDestroy() const { return m_MarkedForDestroy; }

protected:
	CConverter *m_pConverter;
	EType m_EntityType;
	int m_Id;
	vec2 m_Pos;
	int m_Layer;
	int m_Number;
	bool m_MarkedForDestroy = false;

	void Reset() { m_MarkedForDestroy = true; }
	CCollision *Collision() const;
	std::vector<SSwitchers> &Switchers() const;
	int ServerTick() const;
	const CTuningParams &Tuning(int Zone) const;
};

// CPickup: weapons, the armor that takes them away again, freeze and ninja
class CReplayPickup : public CReplayEntity
{
	static constexpr int ms_PhysicsRadius = 14;
	static constexpr int ms_CollisionExtraSize = 6;

	vec2 m_Core = vec2(0.0f, 0.0f);
	int m_Type;
	int m_Subtype;
	int m_Flags;

	void Move();

public:
	CReplayPickup(CConverter *pConverter, int Type, int SubType, vec2 Pos, int Layer, int Number, int Flags);

	void Tick() override;
	void Snap() override;
};

// CDoor: blocks movement while its switcher is off, drawn as the laser
// between its tile and the wall it ends at
class CReplayDoor : public CReplayEntity
{
	vec2 m_To;
	vec2 m_Direction;
	int m_Length;

	void ResetCollision();

public:
	CReplayDoor(CConverter *pConverter, vec2 Pos, float Rotation, int Length, int Number);

	void Snap() override;
};

// CLight: the rotating freeze laser
class CReplayLight : public CReplayEntity
{
	float m_Rotation;
	vec2 m_To = vec2(0.0f, 0.0f);
	vec2 m_Core = vec2(0.0f, 0.0f);
	int m_EvalTick;
	int m_Tick;

	bool m_ToDirty = true;

	bool HitCharacter();
	void Move();
	void Step();
	vec2 To();

public:
	// Set from the map after construction, like the server does, which is
	// why the constructor's Step() runs with all of them still zero
	int m_CurveLength = 0;
	int m_LengthL = 0;
	float m_AngularSpeed = 0.0f;
	int m_Speed = 0;
	int m_Length;

	CReplayLight(CConverter *pConverter, vec2 Pos, float Rotation, int Length, int Layer, int Number);

	void Tick() override;
	void Snap() override;
};

class CReplayDragger;

// CDraggerBeam: the beam that actually pulls one player
class CReplayDraggerBeam : public CReplayEntity
{
	CReplayDragger *m_pDragger;
	float m_Strength;
	bool m_IgnoreWalls;
	int m_ForClientId;
	int m_EvalTick;
	bool m_Active;

public:
	CReplayDraggerBeam(CConverter *pConverter, CReplayDragger *pDragger, vec2 Pos, float Strength, bool IgnoreWalls,
		int ForClientId, int Layer, int Number);

	void SetPos(vec2 Pos) { m_Pos = Pos; }
	void Deactivate();

	void Tick() override;
	void Snap() override;
};

// CDragger: picks the player each team's beam pulls
class CReplayDragger : public CReplayEntity
{
	vec2 m_Core = vec2(0.0f, 0.0f);
	float m_Strength;
	bool m_IgnoreWalls;
	int m_EvalTick;

	int m_aTargetIdInTeam[MAX_CLIENTS];
	CReplayDraggerBeam *m_apDraggerBeam[MAX_CLIENTS] = {};

	void LookForPlayersToDrag();
	std::optional<int> DraggerBeamUsingDraggerId(int SnappingClientId);

public:
	CReplayDragger(CConverter *pConverter, vec2 Pos, float Strength, bool IgnoreWalls, int Layer, int Number);

	int SnapId() const { return m_Id; }
	void RemoveDraggerBeam(int ClientId) { m_apDraggerBeam[ClientId] = nullptr; }
	bool WillDraggerBeamUseDraggerId(int TargetClientId, int SnappingClientId);

	void Tick() override;
	void Snap() override;
};

// CPlasma: one plasma bullet of a turret, freezes or unfreezes on contact
class CReplayPlasma : public CReplayEntity
{
	vec2 m_Core;
	bool m_Freeze;
	bool m_Explosive;
	int m_ForClientId;
	int m_EvalTick;
	int m_LifeTime;

	void Move();
	bool HitCharacter();
	bool HitObstacle();

public:
	CReplayPlasma(CConverter *pConverter, vec2 Pos, vec2 Dir, bool Freeze, bool Explosive, int ForClientId);

	void Tick() override;
	void Snap() override;
};

// CGun: the plasma turret
class CReplayGun : public CReplayEntity
{
	vec2 m_Core = vec2(0.0f, 0.0f);
	bool m_Freeze;
	bool m_Explosive;
	int m_EvalTick;
	int m_aLastFireTeam[MAX_CLIENTS] = {};
	int m_aLastFireSolo[MAX_CLIENTS] = {};

	void Fire();

public:
	CReplayGun(CConverter *pConverter, vec2 Pos, bool Freeze, bool Explosive, int Layer, int Number);

	void Tick() override;
	void Snap() override;
};

// CProjectile: gun and grenade shots, and the bouncing bullets of a shotgun
// turret (which have no owner and freeze what they hit)
class CReplayProjectile : public CReplayEntity
{
	vec2 m_Direction;
	vec2 m_InitDir;
	int m_LifeSpan;
	int m_Owner;
	int m_Type;
	int m_SoundImpact;
	int m_StartTick;
	bool m_Explosive;
	bool m_Freeze;
	int m_Bouncing = 0;
	int m_TuneZone;
	int m_DdraceTeam;
	bool m_IsSolo;

	vec2 GetPos(float Time) const;

public:
	CReplayProjectile(CConverter *pConverter, int Type, int Owner, vec2 Pos, vec2 Dir, int Span, bool Freeze,
		bool Explosive, int SoundImpact, vec2 InitDir, int Layer, int Number);

	void SetBouncing(int Value) { m_Bouncing = Value; }

	void Tick() override;
	void Snap() override;
};

// CLaser: a laser or shotgun shot, bouncing off the walls
class CReplayLaser : public CReplayEntity
{
	vec2 m_From;
	vec2 m_Dir;
	vec2 m_PrevPos;
	vec2 m_TelePos = vec2(0.0f, 0.0f);
	bool m_WasTele = false;
	float m_Energy;
	int m_Bounces = 0;
	int m_EvalTick = 0;
	int m_Owner;
	int m_Type;
	int m_TuneZone;
	bool m_ZeroEnergyBounceInLastTick = false;

	bool HitCharacter(vec2 From, vec2 To);
	void DoBounce();

public:
	CReplayLaser(CConverter *pConverter, vec2 Pos, vec2 Direction, float StartEnergy, int Owner, int Type);

	void Tick() override;
	void Snap() override;
};

class CConverter
{
public:
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
		CNetObj_PlayerInput m_PrevSimInput = {};
		CNetObj_PlayerInput m_DelayedInput = {};
		// The first input recorded in the current tick, see FlushTick
		CNetObj_PlayerInput m_TickInput = {};
		bool m_HasTickInput = false;
		// The weapons themselves are in m_Core, where the server keeps them
		int m_LastWeapon = WEAPON_HAMMER;
		int m_LastNextWeapon = 0;
		int m_LastPrevWeapon = 0;
		int m_AttackTick = 0;
		int m_LastFire = 0;
		int m_QueuedWeapon = -1;
		int m_ReloadTimer = 0;
		int m_PainSoundTimer = 0;
		bool m_FrozenLastTick = false;
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
		bool m_LastRefillJumps = false;
		// The teleporter the tee last stepped on and when, see
		// DetectPositionJumps. A checkpoint teleporter sends it to the outs of
		// the last checkpoint it passed instead of to a numbered one.
		int m_TeleNumber = 0;
		bool m_TeleCheck = false;
		int m_TeleTick = -1;
		int m_TeleCheckpoint = 0;
		int m_TileScanX = 0;
		int m_TileScanY = 0;
		int m_MoveRestrictions = 0;
		int m_TuneZone = 0;
		// When this character spawned, the server's entity list is ordered by
		// it and that decides who wins a hook or a push
		int m_SpawnOrder = 0;
		int m_aHitObjects[MAX_CLIENTS] = {};
		int m_NumObjectsHit = 0;

		CCharacterCore m_Core;
	};

private:
	IStorage *m_pStorage;
	CDemoRecorder m_Recorder;
	CNetObjHandler m_NetObjHandler;

	CMap m_Map;
	CLayers m_Layers;
	CCollision m_Collision;
	CWorldCore m_WorldCore;
	CTeamsCore m_TeamsCore;
	CTuningParams m_Tuning;
	// Per tune zone, filled from the map's own tune_zone settings the way the
	// server runs them at startup. Zone 0 is what the recording's header says.
	CTuningParams m_aTuneZones[TuneZone::NUM];

	CPlayer m_aPlayers[MAX_CLIENTS];
	std::vector<std::vector<unsigned char>> m_vTickMessages;

	int m_Tick = 0;
	int m_LastSimTick = 0;
	int m_FirstTick = -1;
	char m_aOutputPath[IO_MAX_PATH_LENGTH] = "";
	char m_aMapName[128] = "";
	bool m_TickDirty = false;
	int m_CurMaxCid = -1;
	bool m_ExpectPlayers = false;
	int m_StartTick = 0;
	int m_EndTick = std::numeric_limits<int>::max();
	bool m_Done = false;

	// The players of the team at the finish, for the pages that show which
	// ones the rank belongs to
	std::vector<int> m_vFinishCids;
	bool m_aFinisher[MAX_CLIENTS] = {false};
	bool m_FinishLatched = false;
	bool m_TeamsDirty = true;
	std::vector<int> m_vTeamCids;
	// The teams as they were before this tick's chunks, see TEAM_FINISH
	int m_aTeamBeforeTick[MAX_CLIENTS] = {0};
	int m_TeamBeforeTickTick = -1;

	// Rank targeting (--rank)
	const std::vector<const char *> *m_pvRankNames = nullptr;
	int m_RankTimeTicks = 0;
	int m_RankExpectedTick = -1;
	std::vector<CRankCandidate> m_vRankCandidates;
	CRankCandidate m_ApproxCandidate = {-1, -1, -1};
	std::vector<int> m_vApproxCids;
	int m_FilterTeam = -1;
	// Server-side effects are not recorded, reconstructed ones queue here.
	// m_Data is the sound for a sound event and the angle for a damage
	// indicator.
	struct CPendingEvent
	{
		int m_Type;
		int m_X;
		int m_Y;
		int m_Data;
	};
	std::vector<CPendingEvent> m_vPendingEvents;
	std::vector<vec2> m_vSpawnPoints;
	// The entities of the world, per type in the order CGameWorld ticks and
	// snaps them. New entities go to the back, and the tick walks the list
	// backwards, which is the order the server's per-type lists have: it
	// prepends, and an entity created during a tick is not ticked in it.
	std::vector<std::unique_ptr<CReplayEntity>> m_avpEntities[CReplayEntity::NUM_TYPES];
	// Snapshot ids for the entities, handed out like the server's id pool
	// Freed ids and the tick they may be handed out again on. The server
	// parks them for five seconds (CSnapIdPool), and an id that goes straight
	// from a laser that died to a shot fired in the same tick is read by the
	// client as one entity that jumped.
	std::vector<std::pair<int, int>> m_vFreeSnapIds;
	size_t m_FirstFreeSnapId = 0;
	int m_NextSnapId = 0;
	bool m_SnapIdsExhausted = false;
	bool m_SnapItemsDropped = false;
	// The player the replay follows, for the entities that look different per
	// team (doors, lights, draggers), and the builder they snap into
	int m_SnapCid = -1;
	CSnapshotBuilder *m_pSnapBuilder = nullptr;
	CSnapshotBuilder m_SnapshotBuilder;
	bool m_TickEndPositions = false;
	struct CCharacterRef
	{
		vec2 m_Pos;
		int m_Cid;
	};
	std::vector<CCharacterRef> m_vAliveChars;
	int m_NextSpawnOrder = 0;
	vec2 m_CharacterBoxMin = vec2(0.0f, 0.0f);
	vec2 m_CharacterBoxMax = vec2(0.0f, 0.0f);
	std::vector<vec2> m_vViewerPositions;
	vec2 m_ViewerBoxMin = vec2(0.0f, 0.0f);
	vec2 m_ViewerBoxMax = vec2(0.0f, 0.0f);
	CServerConfig m_Config;
	// What the map is made of, for the log line after loading it
	int m_NumPickups = 0;
	int m_NumNinjaPickups = 0;
	int m_NumDoors = 0;
	int m_NumLights = 0;
	int m_NumDraggers = 0;
	int m_NumGuns = 0;
	int m_NumShotgunTurrets = 0;
	int m_NextEventId = 0;
	int m_MarkerStartTick = -1;
	int m_MarkerFinishTick = -1;
	bool m_InputAppliedNextTick = true;
	double m_ErrSum = 0.0;
	double m_ErrMax = 0.0;
	int m_ErrCount = 0;
	int m_ErrOver4 = 0;
	int m_DebugStartTick = -1;
	int m_DebugEndTick = -1;

	// Dataset output (--dataset)
	IOHANDLE m_DatasetFile = nullptr;
	int m_NumDatasetRows = 0;

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

	~CConverter()
	{
		// Entities hand their snapshot id back when they die, so they have to
		// go before the pool they hand it to
		for(auto &vpEntities : m_avpEntities)
			vpEntities.clear();
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
		m_WorldCore.InitSwitchers(m_Collision.m_HighestSwitchNumber);
		LoadMapSettings();
		return true;
	}

	// CGameContext::CreateAllEntities, run once the recording's config and
	// tuning are known: the server also builds its entities after the header
	// was written, and the tiles below override both.
	void CreateAllEntities()
	{
		const CTile *pGame = m_Collision.GameLayer();
		const CTile *pFront = m_Collision.FrontLayer();
		const CSwitchTile *pSwitch = m_Collision.SwitchLayer();
		if(pGame == nullptr)
			return;
		for(int y = 0; y < m_Collision.GetHeight(); y++)
		{
			for(int x = 0; x < m_Collision.GetWidth(); x++)
			{
				const int Index = y * m_Collision.GetWidth() + x;
				OnLayerTile(pGame[Index].m_Index, x, y, LAYER_GAME, pGame[Index].m_Flags, 0);
				if(pFront != nullptr)
					OnLayerTile(pFront[Index].m_Index, x, y, LAYER_FRONT, pFront[Index].m_Flags, 0);
				if(pSwitch != nullptr)
					OnLayerTile(pSwitch[Index].m_Type, x, y, LAYER_SWITCH, pSwitch[Index].m_Flags, pSwitch[Index].m_Number);
			}
		}
		int NumTuneTiles = 0;
		if(m_Collision.TuneLayer() != nullptr)
		{
			for(int i = 0; i < m_Collision.GetWidth() * m_Collision.GetHeight(); i++)
			{
				if(m_Collision.IsTune(i) != 0)
					NumTuneTiles++;
			}
		}
		char aEntities[256] = "";
		const struct
		{
			const char *m_pName;
			int m_Count;
		} aCounts[] = {
			{"spawns", (int)m_vSpawnPoints.size()},
			{"pickups", m_NumPickups},
			{"ninja pickups", m_NumNinjaPickups},
			{"doors", m_NumDoors},
			{"freeze lasers", m_NumLights},
			{"draggers", m_NumDraggers},
			{"plasma turrets", m_NumGuns},
			{"shotgun turrets", m_NumShotgunTurrets},
			{"tune zone tiles", NumTuneTiles}};
		for(const auto &Count : aCounts)
		{
			if(Count.m_Count == 0)
				continue;
			char aOne[64];
			str_format(aOne, sizeof(aOne), "%s%d %s", aEntities[0] == '\0' ? "" : ", ", Count.m_Count, Count.m_pName);
			str_append(aEntities, aOne);
		}
		log_info(TOOL_NAME, "map has %s", aEntities[0] == '\0' ? "no entities" : aEntities);
	}

	// The tiles that change the rules of the whole map, and everything else
	// goes on to become an entity
	void OnLayerTile(int TileIndex, int x, int y, int Layer, int Flags, int Number)
	{
		if(Layer != LAYER_SWITCH)
		{
			switch(TileIndex)
			{
			case TILE_OLDLASER: m_Config.m_SvOldLaser = 1; return;
			case TILE_NPC: m_Tuning.Set("player_collision", 0.0f); return;
			case TILE_EHOOK: m_Config.m_SvEndlessDrag = 1; return;
			case TILE_NOHIT: m_Config.m_SvHit = 0; return;
			case TILE_NPH: m_Tuning.Set("player_hooking", 0.0f); return;
			}
		}
		if(TileIndex >= ENTITY_OFFSET)
			OnEntity(TileIndex - ENTITY_OFFSET, x, y, Layer, Flags, Number);
	}

	// IGameController::OnEntity
	void OnEntity(int Index, int x, int y, int Layer, int Flags, int Number)
	{
		const vec2 Pos(x * 32.0f + 16.0f, y * 32.0f + 16.0f);

		int aSides[8];
		aSides[0] = m_Collision.Entity(x, y + 1, Layer);
		aSides[1] = m_Collision.Entity(x + 1, y + 1, Layer);
		aSides[2] = m_Collision.Entity(x + 1, y, Layer);
		aSides[3] = m_Collision.Entity(x + 1, y - 1, Layer);
		aSides[4] = m_Collision.Entity(x, y - 1, Layer);
		aSides[5] = m_Collision.Entity(x - 1, y - 1, Layer);
		aSides[6] = m_Collision.Entity(x - 1, y, Layer);
		aSides[7] = m_Collision.Entity(x - 1, y + 1, Layer);

		if(Index >= ENTITY_SPAWN && Index <= ENTITY_SPAWN_BLUE)
		{
			// An instant respawn (kill bind, /r, death tiles) has no chunk of
			// its own in the recording, it is detected as a position jump
			// onto a spawn point
			m_vSpawnPoints.push_back(Pos);
		}
		else if(Index == ENTITY_DOOR)
		{
			for(int i = 0; i < 8; i++)
			{
				if(aSides[i] >= ENTITY_LASER_SHORT && aSides[i] <= ENTITY_LASER_LONG)
				{
					new CReplayDoor(this, Pos, pi / 4 * i, 32 * 3 + 32 * (aSides[i] - ENTITY_LASER_SHORT) * 3, Number);
					m_NumDoors++;
				}
			}
		}
		else if(Index == ENTITY_CRAZY_SHOTGUN_EX)
		{
			int Dir;
			if(!Flags)
				Dir = 0;
			else if(Flags == ROTATION_90)
				Dir = 1;
			else if(Flags == ROTATION_180)
				Dir = 2;
			else
				Dir = 3;
			const float Deg = Dir * (pi / 2);
			const vec2 Direction(std::sin(Deg), std::cos(Deg));
			CReplayProjectile *pBullet = new CReplayProjectile(this, WEAPON_SHOTGUN, -1, Pos, Direction, -2, true, true,
				m_Config.m_SvShotgunBulletSound ? SOUND_GRENADE_EXPLODE : -1, Direction, Layer, Number);
			pBullet->SetBouncing(2 - (Dir % 2));
			m_NumShotgunTurrets++;
		}
		else if(Index == ENTITY_CRAZY_SHOTGUN)
		{
			int Dir;
			if(!Flags)
				Dir = 0;
			else if(Flags == TILEFLAG_ROTATE)
				Dir = 1;
			else if(Flags == (TILEFLAG_XFLIP | TILEFLAG_YFLIP))
				Dir = 2;
			else
				Dir = 3;
			const float Deg = Dir * (pi / 2);
			const vec2 Direction(std::sin(Deg), std::cos(Deg));
			CReplayProjectile *pBullet = new CReplayProjectile(this, WEAPON_SHOTGUN, -1, Pos, Direction, -2, true, false,
				SOUND_GRENADE_EXPLODE, Direction, Layer, Number);
			pBullet->SetBouncing(2 - (Dir % 2));
			m_NumShotgunTurrets++;
		}

		int Type = -1;
		int SubType = 0;

		if(Index == ENTITY_ARMOR_1)
			Type = POWERUP_ARMOR;
		else if(Index == ENTITY_ARMOR_SHOTGUN)
			Type = POWERUP_ARMOR_SHOTGUN;
		else if(Index == ENTITY_ARMOR_GRENADE)
			Type = POWERUP_ARMOR_GRENADE;
		else if(Index == ENTITY_ARMOR_NINJA)
			Type = POWERUP_ARMOR_NINJA;
		else if(Index == ENTITY_ARMOR_LASER)
			Type = POWERUP_ARMOR_LASER;
		else if(Index == ENTITY_HEALTH_1)
			Type = POWERUP_FREEZE;
		else if(Index == ENTITY_WEAPON_SHOTGUN)
		{
			Type = POWERUP_WEAPON;
			SubType = WEAPON_SHOTGUN;
		}
		else if(Index == ENTITY_WEAPON_GRENADE)
		{
			Type = POWERUP_WEAPON;
			SubType = WEAPON_GRENADE;
		}
		else if(Index == ENTITY_WEAPON_LASER)
		{
			Type = POWERUP_WEAPON;
			SubType = WEAPON_LASER;
		}
		else if(Index == ENTITY_POWERUP_NINJA)
		{
			Type = POWERUP_NINJA;
			SubType = WEAPON_NINJA;
		}
		else if(Index >= ENTITY_LASER_FAST_CCW && Index <= ENTITY_LASER_FAST_CW)
		{
			int aSides2[8];
			aSides2[0] = m_Collision.Entity(x, y + 2, Layer);
			aSides2[1] = m_Collision.Entity(x + 2, y + 2, Layer);
			aSides2[2] = m_Collision.Entity(x + 2, y, Layer);
			aSides2[3] = m_Collision.Entity(x + 2, y - 2, Layer);
			aSides2[4] = m_Collision.Entity(x, y - 2, Layer);
			aSides2[5] = m_Collision.Entity(x - 2, y - 2, Layer);
			aSides2[6] = m_Collision.Entity(x - 2, y, Layer);
			aSides2[7] = m_Collision.Entity(x - 2, y + 2, Layer);

			int Ind = Index - ENTITY_LASER_STOP;
			int M;
			if(Ind < 0)
			{
				Ind = -Ind;
				M = 1;
			}
			else if(Ind == 0)
				M = 0;
			else
				M = -1;

			float AngularSpeed = 0.0f;
			if(Ind == 1)
				AngularSpeed = pi / 360;
			else if(Ind == 2)
				AngularSpeed = pi / 180;
			else if(Ind == 3)
				AngularSpeed = pi / 90;
			AngularSpeed *= M;

			for(int i = 0; i < 8; i++)
			{
				if(aSides[i] >= ENTITY_LASER_SHORT && aSides[i] <= ENTITY_LASER_LONG)
				{
					CReplayLight *pLight = new CReplayLight(this, Pos, pi / 4 * i, 32 * 3 + 32 * (aSides[i] - ENTITY_LASER_SHORT) * 3, Layer, Number);
					m_NumLights++;
					pLight->m_AngularSpeed = AngularSpeed;
					if(aSides2[i] >= ENTITY_LASER_C_SLOW && aSides2[i] <= ENTITY_LASER_C_FAST)
					{
						pLight->m_Speed = 1 + (aSides2[i] - ENTITY_LASER_C_SLOW) * 2;
						pLight->m_CurveLength = pLight->m_Length;
					}
					else if(aSides2[i] >= ENTITY_LASER_O_SLOW && aSides2[i] <= ENTITY_LASER_O_FAST)
					{
						pLight->m_Speed = 1 + (aSides2[i] - ENTITY_LASER_O_SLOW) * 2;
						pLight->m_CurveLength = 0;
					}
					else
						pLight->m_CurveLength = pLight->m_Length;
				}
			}
		}
		else if(Index >= ENTITY_DRAGGER_WEAK && Index <= ENTITY_DRAGGER_STRONG)
		{
			new CReplayDragger(this, Pos, Index - ENTITY_DRAGGER_WEAK + 1, false, Layer, Number);
			m_NumDraggers++;
		}
		else if(Index >= ENTITY_DRAGGER_WEAK_NW && Index <= ENTITY_DRAGGER_STRONG_NW)
		{
			new CReplayDragger(this, Pos, Index - ENTITY_DRAGGER_WEAK_NW + 1, true, Layer, Number);
			m_NumDraggers++;
		}
		else if(Index == ENTITY_PLASMAE)
		{
			new CReplayGun(this, Pos, false, true, Layer, Number);
			m_NumGuns++;
		}
		else if(Index == ENTITY_PLASMAF)
		{
			new CReplayGun(this, Pos, true, false, Layer, Number);
			m_NumGuns++;
		}
		else if(Index == ENTITY_PLASMA)
		{
			new CReplayGun(this, Pos, true, true, Layer, Number);
			m_NumGuns++;
		}
		else if(Index == ENTITY_PLASMAU)
		{
			new CReplayGun(this, Pos, false, false, Layer, Number);
			m_NumGuns++;
		}

		if(Type != -1)
		{
			int PickupFlags = 0;
			if(Flags & TILEFLAG_XFLIP)
				PickupFlags |= PICKUPFLAG_XFLIP;
			if(Flags & TILEFLAG_YFLIP)
				PickupFlags |= PICKUPFLAG_YFLIP;
			if(Flags & TILEFLAG_ROTATE)
				PickupFlags |= PICKUPFLAG_ROTATE;
			new CReplayPickup(this, Type, SubType, Pos, Layer, Number, PickupFlags);
			if(Type == POWERUP_NINJA)
				m_NumNinjaPickups++;
			else
				m_NumPickups++;
		}
	}

	// The map carries its own server settings, and the ones that change what
	// the replay has to reproduce are the tune zones and the switch defaults
	// (CGameContext::LoadMapSettings runs all of them through the console)
	void LoadMapSettings()
	{
		int Start;
		int Num;
		m_Map.GetType(MAPITEMTYPE_INFO, &Start, &Num);
		for(int i = Start; i < Start + Num; i++)
		{
			int ItemId;
			const CMapItemInfoSettings *pItem = (CMapItemInfoSettings *)m_Map.GetItem(i, nullptr, &ItemId);
			if(pItem == nullptr || ItemId != 0 || m_Map.GetItemSize(i) < (int)sizeof(CMapItemInfoSettings) || pItem->m_Settings < 0)
				continue;
			const int Size = m_Map.GetDataSize(pItem->m_Settings);
			const char *pSettings = (char *)m_Map.GetData(pItem->m_Settings);
			if(pSettings == nullptr)
				continue;
			for(const char *pLine = pSettings; pLine < pSettings + Size; pLine += str_length(pLine) + 1)
			{
				ApplyMapSetting(pLine);
			}
			m_Map.UnloadData(pItem->m_Settings);
			break;
		}
	}

	void ApplyMapSetting(const char *pLine)
	{
		char aCommand[32];
		const char *pRest = str_next_token(pLine, " ", aCommand, sizeof(aCommand));
		if(pRest == nullptr)
			return;
		char aArgs[3][64];
		int NumArgs = 0;
		while(NumArgs < 3 && pRest != nullptr)
		{
			pRest = str_next_token(pRest, " ", aArgs[NumArgs], sizeof(aArgs[NumArgs]));
			if(aArgs[NumArgs][0] == '\0')
				break;
			NumArgs++;
		}
		if(str_comp(aCommand, "tune_zone") == 0 && NumArgs == 3)
		{
			const int Zone = str_toint(aArgs[0]);
			float Value;
			if(Zone <= 0 || Zone >= (int)TuneZone::NUM || !str_tofloat(aArgs[2], &Value))
				return;
			if(!m_aTuneZones[Zone].Set(aArgs[1], Value))
			{
				log_warn(TOOL_NAME, "Unknown tuning parameter '%s' in the map's tune_zone setting", aArgs[1]);
			}
		}
		else if(str_comp(aCommand, "switch_open") == 0 && NumArgs == 1)
		{
			const int Number = str_toint(aArgs[0]);
			if(Number >= 0 && Number < (int)m_WorldCore.m_vSwitchers.size())
			{
				m_WorldCore.m_vSwitchers[Number].m_Initial = false;
				for(bool &Status : m_WorldCore.m_vSwitchers[Number].m_aStatus)
					Status = false;
			}
		}
	}

	// The settings the recording ran with, which teehistorian writes into its
	// header whenever they differ from the built-in default
	void ApplyConfig(const json_value *pConfig)
	{
		if(pConfig == nullptr || pConfig->type != json_object)
			return;
		const struct
		{
			const char *m_pName;
			int *m_pValue;
		} aSettings[] = {
			{"sv_hit", &m_Config.m_SvHit},
			{"sv_freeze_delay", &m_Config.m_SvFreezeDelay},
			{"sv_deepfly", &m_Config.m_SvDeepfly},
			{"sv_dragger_range", &m_Config.m_SvDraggerRange},
			{"sv_plasma_range", &m_Config.m_SvPlasmaRange},
			{"sv_plasma_per_sec", &m_Config.m_SvPlasmaPerSec},
			{"sv_shotgun_bullet_sound", &m_Config.m_SvShotgunBulletSound},
			{"sv_destroy_bullets_on_death", &m_Config.m_SvDestroyBulletsOnDeath},
			{"sv_destroy_lasers_on_death", &m_Config.m_SvDestroyLasersOnDeath},
			{"sv_old_teleport_weapons", &m_Config.m_SvOldTeleportWeapons},
			{"sv_old_laser", &m_Config.m_SvOldLaser},
			{"sv_endless_drag", &m_Config.m_SvEndlessDrag}};
		for(unsigned i = 0; i < pConfig->u.object.length; i++)
		{
			const json_value *pValue = pConfig->u.object.values[i].value;
			if(pValue->type != json_string)
				continue;
			for(const auto &Setting : aSettings)
			{
				if(str_comp(pConfig->u.object.values[i].name, Setting.m_pName) == 0)
				{
					*Setting.m_pValue = str_toint(pValue->u.string.ptr);
					log_info(TOOL_NAME, "recording ran with %s %d", Setting.m_pName, *Setting.m_pValue);
				}
			}
		}
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
		str_copy(m_aOutputPath, pOutputPath);
		str_copy(m_aMapName, pMapName);
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
	const std::vector<int> &ApproxRunCids() const { return m_vApproxCids; }

	// Hide all players outside the given team, including their messages.
	void SetTeamFilter(int Team) { m_FilterTeam = Team; }

	// Recordings without team chunks: the run is the players holding the
	// rank's names, published as a team of their own so the client frames them
	void SetRunCids(const std::vector<int> &vCids)
	{
		m_FilterTeam = 1;
		m_vTeamCids = vCids;
		LatchRun();
	}

	// Add demo timeline markers at the run's start and finish.
	void SetRankMarkers(int StartTick, int FinishTick)
	{
		m_MarkerStartTick = StartTick;
		m_MarkerFinishTick = FinishTick;
	}

	// Log the per-player state of every tick in the range to stderr.
	void SetDebugRange(int StartTick, int EndTick)
	{
		m_DebugStartTick = StartTick;
		m_DebugEndTick = EndTick;
	}

	// Additionally write per-tick state and input rows to the given file as
	// JSON lines, for training on the recorded play.
	void SetDatasetOutput(IOHANDLE File) { m_DatasetFile = File; }
	// Recordings before teehistorian version_minor 24 store an input in the
	// section of the tick it ARRIVED in, while the server applies it in the
	// following tick. Newer ones store it with the tick that applied it.
	void SetInputAppliedNextTick(bool Value) { m_InputAppliedNextTick = Value; }
	int ErrCount() const { return m_ErrCount; }
	double ErrMean() const { return m_ErrCount ? m_ErrSum / m_ErrCount : 0.0; }
	double ErrMax() const { return m_ErrMax; }
	int ErrOver4() const { return m_ErrOver4; }
	int NumDatasetRows() const { return m_NumDatasetRows; }
	const std::vector<int> &FinishCids() const { return m_vFinishCids; }
	// The tick the demo really starts at, which is later than the requested
	// one when the run was preceded by a kill
	int FirstTick() const { return m_FirstTick; }

	// --- The world the entities run in, standing in for CGameWorld and
	// CGameContext. Their characters are the recorded players, at the position
	// they had at the end of the previous tick: that is what the server's
	// entities see, they tick before the characters do.
	CCollision *Collision() { return &m_Collision; }
	int ServerTick() const { return m_Tick; }
	std::vector<SSwitchers> &Switchers() { return m_WorldCore.m_vSwitchers; }
	const CTuningParams &Tuning(int Zone) const { return Zone <= 0 || Zone >= (int)TuneZone::NUM ? m_Tuning : m_aTuneZones[Zone]; }
	const CServerConfig &Config() const { return m_Config; }
	int ClientId(const CPlayer *pChar) const { return pChar - m_aPlayers; }
	int Team(int Cid) const { return m_TeamsCore.Team(Cid); }
	bool GetSolo(int Cid) const { return m_TeamsCore.GetSolo(Cid); }
	bool CanCollide(int Cid, int OtherCid) const { return m_TeamsCore.CanCollide(Cid, OtherCid); }
	CPlayer *GetPlayerChar(int Cid) { return Cid >= 0 && Cid < MAX_CLIENTS && m_aPlayers[Cid].m_Alive ? &m_aPlayers[Cid] : nullptr; }
	// Entities run at the start of the server's tick and see the positions of
	// the end of the previous one. Weapons are fired from the input handler
	// at the end of the tick instead, and the snapshot is built there too,
	// when the tee already stands where this tick recorded it.
	vec2 CharPos(const CPlayer *pChar) const
	{
		return m_TickEndPositions ? vec2(pChar->m_X, pChar->m_Y) : vec2(pChar->m_PrevX, pChar->m_PrevY);
	}
	// The player the replay follows and its team, used by the entities that
	// are drawn differently per team. Without a rank the whole recording is
	// converted and there is no such player, then team 0 is shown.
	int SnapCid() const { return m_SnapCid; }
	int SnapTeam() const { return m_SnapCid >= 0 ? m_TeamsCore.Team(m_SnapCid) : TEAM_FLOCK; }
	void SetSnapCid(int Cid) { m_SnapCid = Cid; }

	void InsertEntity(CReplayEntity *pEntity)
	{
		m_avpEntities[pEntity->EntityType()].emplace_back(pEntity);
	}

	int NewSnapId()
	{
		if(m_FirstFreeSnapId < m_vFreeSnapIds.size() && m_vFreeSnapIds[m_FirstFreeSnapId].second <= m_Tick)
		{
			const int Id = m_vFreeSnapIds[m_FirstFreeSnapId].first;
			m_FirstFreeSnapId++;
			if(m_FirstFreeSnapId == m_vFreeSnapIds.size())
			{
				m_vFreeSnapIds.clear();
				m_FirstFreeSnapId = 0;
			}
			return Id;
		}
		if(m_NextSnapId > CSnapshot::MAX_ID)
		{
			if(!m_SnapIdsExhausted)
			{
				log_warn(TOOL_NAME, "more than %d entities, the ones beyond that are not drawn", CSnapshot::MAX_ID);
				m_SnapIdsExhausted = true;
			}
			return -1;
		}
		return m_NextSnapId++;
	}

	void FreeSnapId(int Id)
	{
		if(Id >= 0)
			m_vFreeSnapIds.emplace_back(Id, m_Tick + SNAP_ID_HOLD_SECONDS * SERVER_TICK_SPEED);
	}

	// Where a character stands in the world's tick order, which is what the
	// client shows as its strong/weak id
	int StrongWeakId(int Cid) const
	{
		for(size_t i = 0; i < m_vAliveChars.size(); i++)
		{
			if(m_vAliveChars[i].m_Cid == Cid)
				return (int)i;
		}
		return 0;
	}

	// Whether a character can be within Radius of the segment at all
	bool CharactersNear(vec2 Pos0, vec2 Pos1, float Radius) const
	{
		const float Reach = Radius + CCharacterCore::PhysicalSize();
		return std::min(Pos0.x, Pos1.x) - Reach <= m_CharacterBoxMax.x &&
		       std::max(Pos0.x, Pos1.x) + Reach >= m_CharacterBoxMin.x &&
		       std::min(Pos0.y, Pos1.y) - Reach <= m_CharacterBoxMax.y &&
		       std::max(Pos0.y, Pos1.y) + Reach >= m_CharacterBoxMin.y;
	}

	// The characters an entity can reach, mirroring CGameWorld::FindEntities
	// for ENTTYPE_CHARACTER. The box around all characters is the same for
	// every entity of a tick, so it cheaply skips the far ones.
	int FindCharacters(vec2 Pos, float Radius, CPlayer **ppChars, int Max)
	{
		if(!CharactersNear(Pos, Pos, Radius))
			return 0;
		int Num = 0;
		for(const CCharacterRef &Char : m_vAliveChars)
		{
			if(Num == Max)
				break;
			if(distance(Char.m_Pos, Pos) < Radius + CCharacterCore::PhysicalSize())
			{
				ppChars[Num] = &m_aPlayers[Char.m_Cid];
				Num++;
			}
		}
		return Num;
	}

	// CGameWorld::IntersectCharacter
	CPlayer *IntersectCharacter(vec2 Pos0, vec2 Pos1, float Radius, vec2 &NewPos, const CPlayer *pNotThis, int CollideWith, const CPlayer *pThisOnly = nullptr)
	{
		if(!CharactersNear(Pos0, Pos1, Radius))
			return nullptr;
		float ClosestLen = distance(Pos0, Pos1) * 100.0f;
		CPlayer *pClosest = nullptr;
		for(const CCharacterRef &Char : m_vAliveChars)
		{
			CPlayer *pChar = &m_aPlayers[Char.m_Cid];
			if(pChar == pNotThis)
				continue;
			if(pThisOnly != nullptr && pChar != pThisOnly)
				continue;
			if(CollideWith != -1 && !CanCollide(Char.m_Cid, CollideWith))
				continue;
			vec2 IntersectPos;
			if(!closest_point_on_line(Pos0, Pos1, Char.m_Pos, IntersectPos))
				continue;
			if(distance(Char.m_Pos, IntersectPos) >= CCharacterCore::PhysicalSize() + Radius)
				continue;
			const float Len = distance(Pos0, IntersectPos);
			if(Len < ClosestLen)
			{
				NewPos = IntersectPos;
				ClosestLen = Len;
				pClosest = pChar;
			}
		}
		return pClosest;
	}

	// CGameWorld::IntersectedCharacters
	std::vector<CPlayer *> IntersectedCharacters(vec2 Pos0, vec2 Pos1, float Radius, const CPlayer *pNotThis)
	{
		std::vector<CPlayer *> vpCharacters;
		if(!CharactersNear(Pos0, Pos1, Radius))
			return vpCharacters;
		for(const CCharacterRef &Char : m_vAliveChars)
		{
			CPlayer *pChar = &m_aPlayers[Char.m_Cid];
			if(pChar == pNotThis)
				continue;
			vec2 IntersectPos;
			if(!closest_point_on_line(Pos0, Pos1, Char.m_Pos, IntersectPos))
				continue;
			if(distance(Char.m_Pos, IntersectPos) < CCharacterCore::PhysicalSize() + Radius)
				vpCharacters.push_back(pChar);
		}
		return vpCharacters;
	}

	// --- What the entities do to a character, mirroring CCharacter

	// CCharacter::Freeze, the freeze timer is only refreshed once a second
	bool Freeze(CPlayer *pChar) { return Freeze(pChar, m_Config.m_SvFreezeDelay); }

	bool Freeze(CPlayer *pChar, int Seconds)
	{
		if(Seconds <= 0 || m_TeamsCore.Team(pChar - m_aPlayers) == TEAM_SUPER || pChar->m_FreezeEndTick - m_Tick > Seconds * SERVER_TICK_SPEED)
			return false;
		if(pChar->m_FreezeEndTick <= m_Tick || pChar->m_FreezeStartTick < m_Tick - SERVER_TICK_SPEED)
		{
			pChar->m_FreezeStartTick = m_Tick;
			pChar->m_FreezeEndTick = m_Tick + Seconds * SERVER_TICK_SPEED;
			return true;
		}
		return false;
	}

	// CCharacter::Unfreeze
	bool Unfreeze(CPlayer *pChar)
	{
		if(pChar->m_FreezeEndTick <= m_Tick)
			return false;
		if(pChar->m_Core.m_ActiveWeapon >= 0 && !pChar->m_Core.m_aWeapons[pChar->m_Core.m_ActiveWeapon].m_Got)
			pChar->m_Core.m_ActiveWeapon = WEAPON_GUN;
		pChar->m_FreezeStartTick = 0;
		pChar->m_FreezeEndTick = 0;
		pChar->m_FrozenLastTick = true;
		return true;
	}

	void AddVelocity(CPlayer *pChar, vec2 Addition)
	{
		pChar->m_Core.m_Vel = ClampVel(pChar->m_MoveRestrictions, pChar->m_Core.m_Vel + Addition);
	}

	void SetRawVelocity(CPlayer *pChar, vec2 Velocity)
	{
		pChar->m_Core.m_Vel = Velocity;
	}

	// CCharacter::GiveWeapon
	void GiveWeapon(CPlayer *pChar, int Weapon, bool Remove = false)
	{
		if(Weapon == WEAPON_NINJA)
		{
			if(Remove)
				RemoveNinja(pChar);
			else
				GiveNinja(pChar);
			return;
		}
		if(Remove)
		{
			if(pChar->m_Core.m_ActiveWeapon == Weapon)
				pChar->m_Core.m_ActiveWeapon = WEAPON_GUN;
		}
		else
		{
			pChar->m_Core.m_aWeapons[Weapon].m_Ammo = -1;
		}
		pChar->m_Core.m_aWeapons[Weapon].m_Got = !Remove;
	}

	// CCharacter::GiveNinja
	void GiveNinja(CPlayer *pChar)
	{
		pChar->m_Core.m_Ninja.m_ActivationTick = m_Tick;
		pChar->m_Core.m_aWeapons[WEAPON_NINJA].m_Got = true;
		pChar->m_Core.m_aWeapons[WEAPON_NINJA].m_Ammo = -1;
		if(pChar->m_Core.m_ActiveWeapon != WEAPON_NINJA)
			pChar->m_LastWeapon = pChar->m_Core.m_ActiveWeapon;
		pChar->m_Core.m_ActiveWeapon = WEAPON_NINJA;
	}

	// CCharacter::RemoveNinja
	void RemoveNinja(CPlayer *pChar)
	{
		pChar->m_Core.m_Ninja.m_ActivationDir = vec2(0.0f, 0.0f);
		pChar->m_Core.m_Ninja.m_ActivationTick = 0;
		pChar->m_Core.m_Ninja.m_CurrentMoveTime = 0;
		pChar->m_Core.m_Ninja.m_OldVelAmount = 0;
		pChar->m_Core.m_aWeapons[WEAPON_NINJA].m_Got = false;
		pChar->m_Core.m_aWeapons[WEAPON_NINJA].m_Ammo = 0;
		pChar->m_Core.m_ActiveWeapon = pChar->m_LastWeapon;
		SetWeapon(pChar, pChar->m_Core.m_ActiveWeapon);
	}

	// CCharacter::SetWeapon
	void SetWeapon(CPlayer *pChar, int Weapon)
	{
		if(Weapon == pChar->m_Core.m_ActiveWeapon)
			return;
		pChar->m_LastWeapon = pChar->m_Core.m_ActiveWeapon;
		pChar->m_QueuedWeapon = -1;
		pChar->m_Core.m_ActiveWeapon = Weapon;
		CreateSound(CharPos(pChar), SOUND_WEAPON_SWITCH, ClientId(pChar));
		if(pChar->m_Core.m_ActiveWeapon < 0 || pChar->m_Core.m_ActiveWeapon >= NUM_WEAPONS)
			pChar->m_Core.m_ActiveWeapon = 0;
	}

	bool HitDisabled(const CPlayer *pChar, int Weapon) const
	{
		switch(Weapon)
		{
		case WEAPON_HAMMER: return pChar->m_Core.m_HammerHitDisabled;
		case WEAPON_SHOTGUN: return pChar->m_Core.m_ShotgunHitDisabled;
		case WEAPON_GRENADE: return pChar->m_Core.m_GrenadeHitDisabled;
		case WEAPON_LASER: return pChar->m_Core.m_LaserHitDisabled;
		default: return false;
		}
	}

	// --- The events the entities produce. The recording has none of them,
	// they are rebuilt into the demo's snapshots.
	void CreateSound(vec2 Pos, int SoundId, int OwnerCid)
	{
		if(SoundId < 0 || !InRecordWindow() || (OwnerCid >= 0 && !IncludePlayer(OwnerCid)))
			return;
		m_vPendingEvents.push_back({NETEVENTTYPE_SOUNDWORLD, round_to_int(Pos.x), round_to_int(Pos.y), SoundId});
	}

	void CreateHammerHit(vec2 Pos, int OwnerCid)
	{
		if(!InRecordWindow() || (OwnerCid >= 0 && !IncludePlayer(OwnerCid)))
			return;
		m_vPendingEvents.push_back({NETEVENTTYPE_HAMMERHIT, round_to_int(Pos.x), round_to_int(Pos.y), 0});
	}

	// CGameContext::CreateDamageInd
	void CreateDamageInd(vec2 Pos, float Angle, int Amount, int OwnerCid)
	{
		if(!InRecordWindow() || (OwnerCid >= 0 && !IncludePlayer(OwnerCid)))
			return;
		const float a = 3 * pi / 2 + Angle;
		const float s = a - pi / 3;
		const float e = a + pi / 3;
		for(int i = 0; i < Amount; i++)
		{
			const float f = mix(s, e, (i + 1) / (float)(Amount + 1));
			m_vPendingEvents.push_back({NETEVENTTYPE_DAMAGEIND, round_to_int(Pos.x), round_to_int(Pos.y), (int)(f * 256.0f)});
		}
	}

	// CGameContext::CreateExplosion, which is a push for everyone in range
	void CreateExplosion(vec2 Pos, int Owner, int Weapon, bool NoDamage, int ActivatedTeam)
	{
		if(InRecordWindow() && (Owner < 0 || IncludePlayer(Owner)))
			m_vPendingEvents.push_back({NETEVENTTYPE_EXPLOSION, round_to_int(Pos.x), round_to_int(Pos.y), 0});

		CPlayer *apChars[MAX_CLIENTS];
		const float Radius = 135.0f;
		const float InnerRadius = 48.0f;
		const int Num = FindCharacters(Pos, Radius, apChars, MAX_CLIENTS);
		bool aTeamHit[NUM_DDRACE_TEAMS];
		std::fill(std::begin(aTeamHit), std::end(aTeamHit), false);
		CPlayer *pOwnerChar = GetPlayerChar(Owner);
		for(int i = 0; i < Num; i++)
		{
			CPlayer *pChar = apChars[i];
			const int Cid = ClientId(pChar);
			const vec2 Diff = CharPos(pChar) - Pos;
			vec2 ForceDir(0.0f, 1.0f);
			float l = length(Diff);
			if(l)
				ForceDir = normalize(Diff);
			l = 1 - std::clamp((l - InnerRadius) / (Radius - InnerRadius), 0.0f, 1.0f);
			const float Strength = Tuning(Owner < 0 ? 0 : m_aPlayers[Owner].m_TuneZone).m_ExplosionStrength;
			const float Dmg = Strength * l;
			if(!(int)Dmg)
				continue;
			if(!((pOwnerChar ? !HitDisabled(pOwnerChar, WEAPON_GRENADE) : m_Config.m_SvHit != 0) || NoDamage || Owner == Cid))
				continue;
			if(Owner != -1 && !CanCollide(Cid, Owner))
				continue;
			if(Owner == -1 && ActivatedTeam != -1 && Team(Cid) != ActivatedTeam)
				continue;
			// Explode at most once per team
			if((pOwnerChar ? HitDisabled(pOwnerChar, WEAPON_GRENADE) : m_Config.m_SvHit == 0) || NoDamage)
			{
				const int PlayerTeam = Team(Cid);
				if(PlayerTeam == TEAM_SUPER || aTeamHit[PlayerTeam])
					continue;
				aTeamHit[PlayerTeam] = true;
			}
			AddVelocity(pChar, ForceDir * Dmg * 2);
		}
	}

	// --- Snapshot, the modern entity objects a current client gets. A demo
	// counts as the latest client version, the same way the server's own
	// demos do.
	bool CanSnapCharacter(int Cid) const { return m_aPlayers[Cid].m_Alive && IncludePlayer(Cid); }

	// An entity is only sent to clients near it. The replay follows the run's
	// players, so an entity is kept while it is within the show distance of
	// one of them, which is what they received while playing.
	bool NetworkClipped(vec2 Pos, float Margin = 0.0f) const
	{
		if(Pos.x < m_ViewerBoxMin.x - SHOW_DISTANCE.x - Margin || Pos.x > m_ViewerBoxMax.x + SHOW_DISTANCE.x + Margin ||
			Pos.y < m_ViewerBoxMin.y - SHOW_DISTANCE.y - Margin || Pos.y > m_ViewerBoxMax.y + SHOW_DISTANCE.y + Margin)
			return true;
		for(const vec2 &ViewPos : m_vViewerPositions)
		{
			const vec2 Delta = ViewPos - Pos;
			if(absolute(Delta.x) <= SHOW_DISTANCE.x + Margin && absolute(Delta.y) <= SHOW_DISTANCE.y + Margin)
				return false;
		}
		return true;
	}

	bool NetworkClippedLine(vec2 StartPos, vec2 EndPos) const
	{
		for(const vec2 &ViewPos : m_vViewerPositions)
		{
			vec2 ClosestPoint;
			const vec2 Delta = closest_point_on_line(StartPos, EndPos, ViewPos, ClosestPoint) ? ViewPos - ClosestPoint : ViewPos - StartPos;
			const float ClipDistance = std::max(SHOW_DISTANCE.x, SHOW_DISTANCE.y);
			if(absolute(Delta.x) <= ClipDistance && absolute(Delta.y) <= ClipDistance)
				return false;
		}
		return true;
	}

	// CGameContext::SnapLaserObject
	void SnapLaserObject(int SnapId, vec2 To, vec2 From, int StartTick, int Owner, int LaserType, int Subtype, int SwitchNumber)
	{
		if(SnapId < 0 || m_pSnapBuilder == nullptr)
			return;
		CNetObj_DDNetLaser *pLaser = (CNetObj_DDNetLaser *)m_pSnapBuilder->NewItemRaw(NETOBJTYPE_DDNETLASER, SnapId, sizeof(CNetObj_DDNetLaser));
		if(pLaser == nullptr)
		{
			OnSnapItemDropped();
			return;
		}
		pLaser->m_ToX = (int)To.x;
		pLaser->m_ToY = (int)To.y;
		pLaser->m_FromX = (int)From.x;
		pLaser->m_FromY = (int)From.y;
		pLaser->m_StartTick = StartTick;
		pLaser->m_Owner = Owner;
		pLaser->m_Type = LaserType;
		pLaser->m_Subtype = Subtype;
		pLaser->m_SwitchNumber = SwitchNumber;
		pLaser->m_Flags = 0;
	}

	// CGameContext::SnapPickup
	void SnapPickupObject(int SnapId, vec2 Pos, int Type, int SubType, int SwitchNumber, int Flags)
	{
		if(SnapId < 0 || m_pSnapBuilder == nullptr)
			return;
		CNetObj_DDNetPickup *pPickup = (CNetObj_DDNetPickup *)m_pSnapBuilder->NewItemRaw(NETOBJTYPE_DDNETPICKUP, SnapId, sizeof(CNetObj_DDNetPickup));
		if(pPickup == nullptr)
		{
			OnSnapItemDropped();
			return;
		}
		pPickup->m_X = (int)Pos.x;
		pPickup->m_Y = (int)Pos.y;
		pPickup->m_Type = Type;
		pPickup->m_Subtype = SubType;
		pPickup->m_SwitchNumber = SwitchNumber;
		pPickup->m_Flags = Flags;
	}

	// CProjectile::NetInfo
	void SnapProjectileObject(int SnapId, vec2 Pos, vec2 Vel, int Type, int StartTick, int Owner, int SwitchNumber, int TuneZone, int Flags)
	{
		if(SnapId < 0 || m_pSnapBuilder == nullptr)
			return;
		CNetObj_DDNetProjectile *pProjectile = (CNetObj_DDNetProjectile *)m_pSnapBuilder->NewItemRaw(NETOBJTYPE_DDNETPROJECTILE, SnapId, sizeof(CNetObj_DDNetProjectile));
		if(pProjectile == nullptr)
		{
			OnSnapItemDropped();
			return;
		}
		pProjectile->m_X = round_to_int(Pos.x * 100.0f);
		pProjectile->m_Y = round_to_int(Pos.y * 100.0f);
		pProjectile->m_VelX = round_to_int(Vel.x);
		pProjectile->m_VelY = round_to_int(Vel.y);
		pProjectile->m_Type = Type;
		pProjectile->m_StartTick = StartTick;
		pProjectile->m_Owner = Owner;
		pProjectile->m_SwitchNumber = SwitchNumber;
		pProjectile->m_TuneZone = TuneZone;
		pProjectile->m_Flags = Flags;
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
			CPlayer *pTargetPlayer = &pTarget->m_aPlayers[Cid];
			str_copy(pTargetPlayer->m_aName, Player.m_aName);
			str_copy(pTargetPlayer->m_aClan, Player.m_aClan);
			pTargetPlayer->m_Country = Player.m_Country;
			str_copy(pTargetPlayer->m_aSkin, Player.m_aSkin);
			pTargetPlayer->m_UseCustomColor = Player.m_UseCustomColor;
			pTargetPlayer->m_ColorBody = Player.m_ColorBody;
			pTargetPlayer->m_ColorFeet = Player.m_ColorFeet;
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
			// A recording covers one map of one server, a skip of more than a
			// day of ticks is a corrupt file, not an idle server
			if(pUnpacker->Error() || Dt < 0 || Dt > 24 * 60 * 60 * SERVER_TICK_SPEED)
				return false;
			FlushTick();
			// A tick is only skipped because no tee moved, but the world keeps
			// running: entities tick, projectiles travel and lasers bounce.
			// Skipping them would leave a projectile standing still for the
			// length of the gap and then jump, and the client would
			// interpolate the gap, which plays hooks back in slow motion.
			for(int i = 0; i < Dt && !m_Done; i++)
			{
				m_Tick++;
				m_TickDirty = true;
				FlushTick();
			}
			m_Tick += 1;
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
			pPlayer->m_TileScanX = X;
			pPlayer->m_TileScanY = Y;
			pPlayer->m_PrevTick = -1;
			pPlayer->m_TeleNumber = 0;
			pPlayer->m_TeleCheck = false;
			pPlayer->m_TeleTick = -1;
			pPlayer->m_FreezeEndTick = 0;
			pPlayer->m_FreezeStartTick = 0;
			pPlayer->m_DeepFrozen = false;
			pPlayer->m_InFreezeTile = false;
			ResetWeapons(*pPlayer);
			pPlayer->m_SpawnOrder = m_NextSpawnOrder++;
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
			// The server records every input it receives, but only the FIRST
			// one of a tick reaches CCharacter::m_Input, which is what drives
			// movement and the hook (CGameContext::OnClientPredictedInput is
			// called once per tick, OnClientPredictedEarlyInput on every
			// input). Firing uses them all, so it keeps reading m_Input.
			if(!pPlayer->m_HasTickInput)
			{
				pPlayer->m_TickInput = pPlayer->m_Input;
				pPlayer->m_HasTickInput = true;
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
			str_copy(Fresh.m_aName, m_aPlayers[Cid].m_aName);
			str_copy(Fresh.m_aClan, m_aPlayers[Cid].m_aClan);
			Fresh.m_Country = m_aPlayers[Cid].m_Country;
			str_copy(Fresh.m_aSkin, m_aPlayers[Cid].m_aSkin);
			Fresh.m_UseCustomColor = m_aPlayers[Cid].m_UseCustomColor;
			Fresh.m_ColorBody = m_aPlayers[Cid].m_ColorBody;
			Fresh.m_ColorFeet = m_aPlayers[Cid].m_ColorFeet;
			Fresh.m_Connected = true;
			m_aPlayers[Cid] = Fresh;
			if(m_TeamsCore.Team(Cid) != 0)
				m_TeamsDirty = true;
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
			// The next client in the slot is not part of the run
			m_aFinisher[Cid] = false;
			break;
		}
		case TEEHISTORIAN_CONSOLE_COMMAND:
		{
			const int Cid = pUnpacker->GetInt();
			const int FlagMask = pUnpacker->GetInt();
			const char *pCommand = pUnpacker->GetString();
			char aChat[512];
			str_format(aChat, sizeof(aChat), "/%s", pCommand == nullptr ? "" : pCommand);
			const int NumArgs = pUnpacker->GetInt();
			if(pUnpacker->Error() || NumArgs < 0 || NumArgs > 128)
				return false;
			for(int i = 0; i < NumArgs; i++)
			{
				const char *pArg = pUnpacker->GetString();
				if(pArg != nullptr)
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
		m_Recorder.Stop(IDemoRecorder::EStopMode::KEEP_FILE);
		if(m_NumSnapshots == 0)
		{
			log_error(TOOL_NAME, "No ticks in the selected time range, recording covers %d:%02d:%02d hours",
				m_Tick / SERVER_TICK_SPEED / 3600, m_Tick / SERVER_TICK_SPEED / 60 % 60, m_Tick / SERVER_TICK_SPEED % 60);
			return false;
		}
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
		if(m_FilterTeam < 0)
			return true;
		// The team is dissolved in the tick it finishes in, but the demo runs
		// a few seconds past the finish, so the run keeps the players it had
		if(m_FinishLatched)
			return m_aFinisher[Cid];
		return m_TeamsCore.Team(Cid) == m_FilterTeam;
	}

	// The team a player is published with. Once the run's team is dissolved
	// its players keep it, otherwise the client would drop them out of
	// multi-view for the last seconds of the demo.
	int PublishedTeam(int Cid) const
	{
		if(!IncludePlayer(Cid))
			return TEAM_FLOCK;
		return m_FinishLatched ? m_FilterTeam : m_TeamsCore.Team(Cid);
	}

	// Whether the team the run belongs to still has a member. The server
	// dissolves it in the tick it finishes in.
	bool TeamAlive() const
	{
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			if(m_aPlayers[Cid].m_Connected && m_TeamsCore.Team(Cid) == m_FilterTeam)
				return true;
		}
		return false;
	}

	// The players the rank belongs to, a name cannot tell them apart
	void LatchRun()
	{
		m_vFinishCids = m_vTeamCids;
		if(m_FilterTeam <= 0)
			return;
		for(const int Cid : m_vFinishCids)
			m_aFinisher[Cid] = true;
		m_FinishLatched = true;
		m_TeamsDirty = true;
	}

	// Teehistorian writes a tick's team changes before the finish of the tick,
	// so a chunk that needs the team a player raced in reads this instead
	int TeamBeforeTick(int Cid) const
	{
		return m_TeamBeforeTickTick == m_Tick ? m_aTeamBeforeTick[Cid] : m_TeamsCore.Team(Cid);
	}

	void SaveTeamsBeforeTick()
	{
		if(m_TeamBeforeTickTick == m_Tick)
			return;
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
			m_aTeamBeforeTick[Cid] = m_TeamsCore.Team(Cid);
		m_TeamBeforeTickTick = m_Tick;
	}

	int FindPlayer(const char *pName) const
	{
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			if(m_aPlayers[Cid].m_Connected && str_comp(m_aPlayers[Cid].m_aName, pName) == 0)
				return Cid;
		}
		return -1;
	}

	// Recordings from before April 2024 have no finish events and the rank is
	// placed by its timestamp instead. A team is dissolved when it finishes,
	// so by then it is gone: remember the last tick its whole roster shared a
	// team and take the run's team and camera target from there.
	void UpdateRosterTeam()
	{
		if(m_pvRankNames == nullptr || m_pvRankNames->size() < 2 || m_RankExpectedTick < 0 || m_Tick > m_RankExpectedTick)
			return;
		int Team = -1;
		int FirstCid = -1;
		for(const char *pName : *m_pvRankNames)
		{
			const int Cid = FindPlayer(pName);
			if(Cid < 0 || m_TeamsCore.Team(Cid) == TEAM_FLOCK || (Team >= 0 && m_TeamsCore.Team(Cid) != Team))
				return;
			Team = m_TeamsCore.Team(Cid);
			if(FirstCid < 0)
				FirstCid = Cid;
		}
		m_ApproxCandidate = {m_RankExpectedTick, FirstCid, Team};
	}

	// CGameTeams::SendTeamsState. Without it every player looks like a member
	// of team 0 to the client, and multi-view, which frames the team of the
	// player it follows, falls back to whoever happens to be on screen.
	void SendTeamsState()
	{
		CPacker Packer;
		Packer.Reset();
		// An extended message is a zero, its uuid and then the payload
		Packer.AddInt(0);
		g_UuidManager.PackUuid(NETMSGTYPE_SV_TEAMSSTATE, &Packer);
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			Packer.AddInt(PublishedTeam(Cid));
		}
		QueueMessage(&Packer);
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
			if(!Unpacker.Error() && Cid >= 0 && Cid < MAX_CLIENTS && Team >= TEAM_FLOCK && Team <= TEAM_SUPER && m_TeamsCore.Team(Cid) != Team)
			{
				SaveTeamsBeforeTick();
				m_TeamsCore.Team(Cid, Team);
				m_TeamsDirty = true;
				UpdateRosterTeam();
			}
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
			// sv_rejoin_team_0 moves the team back to team 0 before the
			// finish is recorded, so its members are read from the teams as
			// they were when the tick began
			for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
			{
				if(m_aPlayers[Cid].m_Connected && TeamBeforeTick(Cid) == Team)
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
						if(m_aPlayers[Cid].m_Connected && TeamBeforeTick(Cid) == Team && str_comp(m_aPlayers[Cid].m_aName, pName) == 0)
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

	// Throw away what has been written and record from this tick on. Only the
	// snapshots of the discarded ticks are lost, everything a demo needs is
	// written per tick.
	void RestartRecording()
	{
		if(m_FirstTick < 0)
			return;
		m_Recorder.Stop(IDemoRecorder::EStopMode::REMOVE_FILE);
		m_FirstTick = -1;
		m_NumTicks = 0;
		m_NumSnapshots = 0;
		// The teams were written into the file that was just thrown away
		m_TeamsDirty = true;
		if(!StartDemo(m_aOutputPath, m_aMapName))
			m_Done = true;
	}

	// Where the teleporter the tee stepped on puts it. A checkpoint one uses
	// the outs of the last checkpoint that has any, CCharacter::HandleTiles.
	const std::vector<vec2> &TeleOutsOf(CPlayer &Player)
	{
		static const std::vector<vec2> s_vNone;
		if(Player.m_TeleNumber > 0)
			return m_Collision.TeleOuts(Player.m_TeleNumber - 1);
		if(!Player.m_TeleCheck)
			return s_vNone;
		for(int Checkpoint = Player.m_TeleCheckpoint - 1; Checkpoint >= 0; Checkpoint--)
		{
			if(!m_Collision.TeleCheckOuts(Checkpoint).empty())
				return m_Collision.TeleCheckOuts(Checkpoint);
		}
		return s_vNone;
	}

	// An instant respawn produces no join/leave chunks, only a position jump
	// onto a spawn point: reset the state a fresh character starts with
	void DetectPositionJumps()
	{
		for(auto &Player : m_aPlayers)
		{
			if(!Player.m_Alive || Player.m_PrevTick != m_Tick - 1)
				continue;
			const vec2 Pos(Player.m_X, Player.m_Y);
			// Further than a tee can move in one tick means it was placed
			// there: a teleporter, a rescue or a respawn
			if(distance(vec2(Player.m_PrevX, Player.m_PrevY), Pos) < 6 * 32)
				continue;
			// The server keeps scanning the tiles a tee walks over from where
			// it now is (m_PrevPos is assigned after the teleport), so the
			// line across the map must not be walked. It does start at the
			// teleporter's out tile though, and maps put an unfreeze tile
			// there to undo the freeze tile the teleporter sits on.
			vec2 ScanFrom = Pos;
			float Nearest = 6 * 32;
			// The tile is walked in the tick before the jump shows up, an
			// older teleporter is one the tee did not come through
			if(Player.m_TeleTick == m_Tick - 1)
			{
				for(const vec2 &Out : TeleOutsOf(Player))
				{
					const float Distance = distance(Out, Pos);
					if(Distance < Nearest)
					{
						Nearest = Distance;
						ScanFrom = Out;
					}
				}
			}
			Player.m_TeleNumber = 0;
			Player.m_TeleCheck = false;
			Player.m_TileScanX = ScanFrom.x;
			Player.m_TileScanY = ScanFrom.y;
			// Teleporting drops the hook and lets go of whoever held on
			Player.m_Core.SetHookedPlayer(-1);
			Player.m_Core.m_HookState = HOOK_RETRACTED;
			Player.m_Core.m_HookPos = Pos;
			for(auto &Other : m_aPlayers)
			{
				if(Other.m_Core.HookedPlayer() == &Player - m_aPlayers)
				{
					Other.m_Core.SetHookedPlayer(-1);
					Other.m_Core.m_HookState = HOOK_RETRACTED;
				}
			}
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
			// A kill right before the run would put the failed attempt in
			// front of it, so the demo starts after the last respawn instead.
			// The respawn only shows up once the seconds before the run are
			// being written, so the recording starts over.
			if(m_MarkerStartTick >= 0 && m_Tick < m_MarkerStartTick && IncludePlayer(&Player - m_aPlayers))
			{
				m_StartTick = std::max(m_StartTick, m_Tick + 1);
				RestartRecording();
			}
			Player.m_FreezeEndTick = 0;
			Player.m_FreezeStartTick = 0;
			Player.m_DeepFrozen = false;
			Player.m_InFreezeTile = false;
			ResetWeapons(Player);
			Player.m_SpawnOrder = m_NextSpawnOrder++;
			Player.m_Core.SetHookedPlayer(-1);
			Player.m_Core.m_HookState = HOOK_IDLE;
			SetSolo(&Player - m_aPlayers, false);
		}
	}

	// What a character starts with, CCharacter::Spawn and
	// IGameController::OnCharacterSpawn
	void ResetWeapons(CPlayer &Player)
	{
		Player.m_LastWeapon = WEAPON_HAMMER;
		Player.m_QueuedWeapon = -1;
		Player.m_ReloadTimer = 0;
		Player.m_NumObjectsHit = 0;
		Player.m_Core.m_Ninja = {};
		for(CCharacterCore::CWeaponStat &Weapon : Player.m_Core.m_aWeapons)
			Weapon = {};
		Player.m_Core.m_ActiveWeapon = WEAPON_GUN;
		GiveWeapon(&Player, WEAPON_HAMMER);
		GiveWeapon(&Player, WEAPON_GUN);
	}

	bool IsSwitchOpen(int Number, int Team) const
	{
		if(Number <= 0 || Number >= (int)m_WorldCore.m_vSwitchers.size() || Team < 0 || Team >= NUM_DDRACE_TEAMS)
			return true;
		return m_WorldCore.m_vSwitchers[Number].m_aStatus[Team];
	}

	// Timed switchers flip back on their own, CGameContext::OnTick does this
	// before the world runs
	void UpdateSwitchers()
	{
		for(SSwitchers &Switcher : m_WorldCore.m_vSwitchers)
		{
			for(int Team = 0; Team < NUM_DDRACE_TEAMS; Team++)
			{
				if(Switcher.m_aEndTick[Team] > m_Tick)
					continue;
				if(Switcher.m_aType[Team] == TILE_SWITCHTIMEDOPEN)
				{
					Switcher.m_aStatus[Team] = false;
					Switcher.m_aEndTick[Team] = 0;
					Switcher.m_aType[Team] = TILE_SWITCHCLOSE;
				}
				else if(Switcher.m_aType[Team] == TILE_SWITCHTIMEDCLOSE)
				{
					Switcher.m_aStatus[Team] = true;
					Switcher.m_aEndTick[Team] = 0;
					Switcher.m_aType[Team] = TILE_SWITCHOPEN;
				}
			}
		}
	}

	void SetSolo(int Cid, bool Solo)
	{
		m_TeamsCore.SetSolo(Cid, Solo);
		m_aPlayers[Cid].m_Core.m_Solo = Solo;
	}

	// CCharacter::IsSwitchActiveCb, which door tiles stop this player
	struct CSwitchActiveContext
	{
		const CConverter *m_pConverter;
		int m_Team;
	};
	static bool IsSwitchActiveCb(unsigned char Number, void *pUser)
	{
		const CSwitchActiveContext *pContext = (const CSwitchActiveContext *)pUser;
		const std::vector<SSwitchers> &vSwitchers = pContext->m_pConverter->m_WorldCore.m_vSwitchers;
		return !vSwitchers.empty() && pContext->m_Team != TEAM_SUPER && vSwitchers[Number].m_aStatus[pContext->m_Team];
	}

	// CCharacter::HandleTuneLayer, the zone a tee stands in decides the
	// tuning its weapons and its movement run with
	void UpdateTuneZones()
	{
		for(CPlayer &Player : m_aPlayers)
		{
			if(Player.m_Alive)
				Player.m_TuneZone = m_Collision.IsTune(m_Collision.GetMapIndex(CharPos(&Player)));
		}
	}

	// Everything one tile does to a player, mirroring CCharacter::HandleTiles
	void HandleTiles(CPlayer &Player, int Index)
	{
		const int Cid = &Player - m_aPlayers;
		if(Index < 0)
		{
			// The server still updates these two before it gives up on the
			// tile, and a stale move restriction clamps every later push
			const CSwitchActiveContext EmptyContext = {this, m_TeamsCore.Team(Cid)};
			Player.m_MoveRestrictions = m_Collision.GetMoveRestrictions(IsSwitchActiveCb, (void *)&EmptyContext, vec2(Player.m_X, Player.m_Y), 18.0f, Index);
			Player.m_LastRefillJumps = false;
			return;
		}
		const int Tile = m_Collision.GetTileIndex(Index);
		const int FrontTile = m_Collision.GetFrontTileIndex(Index);
		const int SwitchType = m_Collision.GetSwitchType(Index);
		const int SwitchNumber = m_Collision.GetSwitchNumber(Index);
		const int SwitchDelay = m_Collision.GetSwitchDelay(Index);
		const int Team = m_TeamsCore.Team(Cid);
		// A switch tile only acts while its switcher is on for the team, and
		// number 0 is always on
		const bool SwitchOn = SwitchNumber == 0 || IsSwitchOpen(SwitchNumber, Team);
		const CSwitchActiveContext Context = {this, Team};
		Player.m_MoveRestrictions = m_Collision.GetMoveRestrictions(IsSwitchActiveCb, (void *)&Context, vec2(Player.m_X, Player.m_Y), 18.0f, Index);

		// Which teleporter the tee stepped on, the tile scan continues from
		// its out tile once the recorded position jumps there
		const int TeleIn = m_Collision.IsTeleport(Index);
		const int Tele = TeleIn > 0 ? TeleIn : m_Collision.IsEvilTeleport(Index);
		const int TeleCheckpoint = m_Collision.IsTeleCheckpoint(Index);
		if(TeleCheckpoint > 0)
			Player.m_TeleCheckpoint = TeleCheckpoint;
		if(Tele > 0 || m_Collision.IsCheckTeleport(Index) || m_Collision.IsCheckEvilTeleport(Index))
		{
			Player.m_TeleNumber = Tele;
			Player.m_TeleCheck = Tele <= 0;
			Player.m_TeleTick = m_Tick;
		}

		if((Tile == TILE_FREEZE || FrontTile == TILE_FREEZE) && !Player.m_DeepFrozen)
			Freeze(&Player);
		else if((Tile == TILE_UNFREEZE || FrontTile == TILE_UNFREEZE) && !Player.m_DeepFrozen)
			Unfreeze(&Player);

		if(Team != TEAM_SUPER)
		{
			if((Tile == TILE_DFREEZE || FrontTile == TILE_DFREEZE) && !Player.m_DeepFrozen)
				Player.m_DeepFrozen = true;
			else if((Tile == TILE_DUNFREEZE || FrontTile == TILE_DUNFREEZE) && Player.m_DeepFrozen)
				Player.m_DeepFrozen = false;
		}

		// A walljump tile gives the air jump back, which is also what draws
		// the feet light again
		if((Tile == TILE_WALLJUMP || FrontTile == TILE_WALLJUMP) && Player.m_Core.m_Vel.y > 0 && Player.m_Core.m_Colliding && Player.m_Core.m_LeftWall)
		{
			Player.m_Core.m_LeftWall = false;
			Player.m_Core.m_JumpedTotal = Player.m_Core.m_Jumps >= 2 ? Player.m_Core.m_Jumps - 2 : 0;
			Player.m_Core.m_Jumped = 1;
		}

		// Live freeze keeps the tee moving but takes its controls away
		if(Team != TEAM_SUPER)
		{
			if(Tile == TILE_LFREEZE || FrontTile == TILE_LFREEZE)
				Player.m_Core.m_LiveFrozen = true;
			else if(Tile == TILE_LUNFREEZE || FrontTile == TILE_LUNFREEZE)
				Player.m_Core.m_LiveFrozen = false;
		}

		// A solo part decides who can hook and hit whom, so it decides which
		// hooks the replayed physics may attach to a player
		if(Tile == TILE_SOLO_ENABLE || FrontTile == TILE_SOLO_ENABLE)
			SetSolo(Cid, true);
		else if(Tile == TILE_SOLO_DISABLE || FrontTile == TILE_SOLO_DISABLE)
			SetSolo(Cid, false);

		if(Tile == TILE_HIT_DISABLE || FrontTile == TILE_HIT_DISABLE)
		{
			Player.m_Core.m_HammerHitDisabled = true;
			Player.m_Core.m_ShotgunHitDisabled = true;
			Player.m_Core.m_GrenadeHitDisabled = true;
			Player.m_Core.m_LaserHitDisabled = true;
		}
		else if(Tile == TILE_HIT_ENABLE || FrontTile == TILE_HIT_ENABLE)
		{
			Player.m_Core.m_HammerHitDisabled = false;
			Player.m_Core.m_ShotgunHitDisabled = false;
			Player.m_Core.m_GrenadeHitDisabled = false;
			Player.m_Core.m_LaserHitDisabled = false;
		}
		else if(SwitchType == TILE_HIT_ENABLE || SwitchType == TILE_HIT_DISABLE)
		{
			// The switch variant addresses one weapon, named by the delay
			const bool Disabled = SwitchType == TILE_HIT_DISABLE;
			switch(SwitchDelay)
			{
			case WEAPON_HAMMER: Player.m_Core.m_HammerHitDisabled = Disabled; break;
			case WEAPON_SHOTGUN: Player.m_Core.m_ShotgunHitDisabled = Disabled; break;
			case WEAPON_GRENADE: Player.m_Core.m_GrenadeHitDisabled = Disabled; break;
			case WEAPON_LASER: Player.m_Core.m_LaserHitDisabled = Disabled; break;
			}
		}

		if(Tile == TILE_NPC_DISABLE || FrontTile == TILE_NPC_DISABLE)
			Player.m_Core.m_CollisionDisabled = true;
		else if(Tile == TILE_NPC_ENABLE || FrontTile == TILE_NPC_ENABLE)
			Player.m_Core.m_CollisionDisabled = false;

		if(Tile == TILE_NPH_DISABLE || FrontTile == TILE_NPH_DISABLE)
			Player.m_Core.m_HookHitDisabled = true;
		else if(Tile == TILE_NPH_ENABLE || FrontTile == TILE_NPH_ENABLE)
			Player.m_Core.m_HookHitDisabled = false;

		// Endless hook, jetpack and the air jumps are persistent tile powerups,
		// Core.Reset() clears them on spawn
		if(Tile == TILE_EHOOK_ENABLE || FrontTile == TILE_EHOOK_ENABLE)
			Player.m_Core.m_EndlessHook = true;
		else if(Tile == TILE_EHOOK_DISABLE || FrontTile == TILE_EHOOK_DISABLE)
			Player.m_Core.m_EndlessHook = false;

		if(Tile == TILE_JETPACK_ENABLE || FrontTile == TILE_JETPACK_ENABLE)
			Player.m_Core.m_Jetpack = true;
		else if(Tile == TILE_JETPACK_DISABLE || FrontTile == TILE_JETPACK_DISABLE)
			Player.m_Core.m_Jetpack = false;

		// A tee that carries a teleporting weapon holds a different one
		if(Tile == TILE_TELE_GUN_ENABLE || FrontTile == TILE_TELE_GUN_ENABLE)
			Player.m_Core.m_HasTelegunGun = true;
		else if(Tile == TILE_TELE_GUN_DISABLE || FrontTile == TILE_TELE_GUN_DISABLE)
			Player.m_Core.m_HasTelegunGun = false;

		if(Tile == TILE_TELE_GRENADE_ENABLE || FrontTile == TILE_TELE_GRENADE_ENABLE)
			Player.m_Core.m_HasTelegunGrenade = true;
		else if(Tile == TILE_TELE_GRENADE_DISABLE || FrontTile == TILE_TELE_GRENADE_DISABLE)
			Player.m_Core.m_HasTelegunGrenade = false;

		if(Tile == TILE_TELE_LASER_ENABLE || FrontTile == TILE_TELE_LASER_ENABLE)
			Player.m_Core.m_HasTelegunLaser = true;
		else if(Tile == TILE_TELE_LASER_DISABLE || FrontTile == TILE_TELE_LASER_DISABLE)
			Player.m_Core.m_HasTelegunLaser = false;

		if(Tile == TILE_UNLIMITED_JUMPS_ENABLE || FrontTile == TILE_UNLIMITED_JUMPS_ENABLE)
			Player.m_Core.m_EndlessJump = true;
		else if(Tile == TILE_UNLIMITED_JUMPS_DISABLE || FrontTile == TILE_UNLIMITED_JUMPS_DISABLE)
			Player.m_Core.m_EndlessJump = false;

		// The switchers a tee opens and closes are per team, and they gate
		// doors, pickups and the switch layer's freeze tiles
		if(SwitchNumber > 0 && Team >= 0 && Team < TEAM_SUPER && (int)m_WorldCore.m_vSwitchers.size() > SwitchNumber &&
			(SwitchType == TILE_SWITCHOPEN || SwitchType == TILE_SWITCHTIMEDOPEN || SwitchType == TILE_SWITCHTIMEDCLOSE || SwitchType == TILE_SWITCHCLOSE))
		{
			SSwitchers &Switcher = m_WorldCore.m_vSwitchers[SwitchNumber];
			const bool Open = SwitchType == TILE_SWITCHOPEN || SwitchType == TILE_SWITCHTIMEDOPEN;
			const bool Timed = SwitchType == TILE_SWITCHTIMEDOPEN || SwitchType == TILE_SWITCHTIMEDCLOSE;
			Switcher.m_aStatus[Team] = Open;
			Switcher.m_aEndTick[Team] = Timed ? m_Tick + 1 + SwitchDelay * SERVER_TICK_SPEED : 0;
			Switcher.m_aType[Team] = SwitchType;
			Switcher.m_aLastUpdateTick[Team] = m_Tick;
		}

		// A stopper takes the air jumps back and holds the tee in place
		if(Player.m_Core.m_Vel.y > 0 && (Player.m_MoveRestrictions & CANTMOVE_DOWN))
		{
			Player.m_Core.m_Jumped = 0;
			Player.m_Core.m_JumpedTotal = 0;
		}
		Player.m_Core.m_Vel = ClampVel(Player.m_MoveRestrictions, Player.m_Core.m_Vel);

		if(SwitchOn && Team != TEAM_SUPER)
		{
			// The switch layer's freeze tiles, the freezing one for as many
			// seconds as its delay says
			if(SwitchType == TILE_FREEZE)
				Freeze(&Player, SwitchDelay);
			else if(SwitchType == TILE_DFREEZE)
				Player.m_DeepFrozen = true;
			else if(SwitchType == TILE_DUNFREEZE)
				Player.m_DeepFrozen = false;
			else if(SwitchType == TILE_LFREEZE)
				Player.m_Core.m_LiveFrozen = true;
			else if(SwitchType == TILE_LUNFREEZE)
				Player.m_Core.m_LiveFrozen = false;
		}

		if(SwitchType == TILE_JUMP)
		{
			// The switch delay carries the new jump count, 255 means ground jump only
			Player.m_Core.m_Jumps = SwitchDelay == 255 ? -1 : SwitchDelay;
		}

		// Refill jumps triggers on entering the tile, not every tick on it
		if(Tile == TILE_REFILL_JUMPS || FrontTile == TILE_REFILL_JUMPS)
		{
			if(!Player.m_LastRefillJumps)
			{
				Player.m_Core.m_JumpedTotal = 0;
				Player.m_Core.m_Jumped = 0;
				Player.m_LastRefillJumps = true;
			}
		}
		else
		{
			Player.m_LastRefillJumps = false;
		}
	}

	// The entity world of the tick, in the order CGameWorld runs it: an
	// entity created during a tick is not ticked in that tick, and one that
	// died is removed when every type has ticked.
	void TickEntities()
	{
		UpdateCharacterBox();
		for(auto &vpEntities : m_avpEntities)
		{
			for(int i = (int)vpEntities.size() - 1; i >= 0; i--)
			{
				if(!vpEntities[i]->MarkedForDestroy())
					vpEntities[i]->Tick();
			}
		}
		for(auto &vpEntities : m_avpEntities)
		{
			vpEntities.erase(std::remove_if(vpEntities.begin(), vpEntities.end(),
						 [](const std::unique_ptr<CReplayEntity> &pEntity) { return pEntity->MarkedForDestroy(); }),
				vpEntities.end());
		}
	}

	void SnapEntities(CSnapshotBuilder *pBuilder)
	{
		m_TickEndPositions = true;
		UpdateViewerBox();
		m_pSnapBuilder = pBuilder;
		for(auto &vpEntities : m_avpEntities)
		{
			for(int i = (int)vpEntities.size() - 1; i >= 0; i--)
			{
				vpEntities[i]->Snap();
			}
		}
		m_pSnapBuilder = nullptr;
		m_TickEndPositions = false;
	}

	// A snapshot holds at most 1024 items, the same limit the server's own
	// demos run into on entity-heavy maps
	void OnSnapItemDropped()
	{
		if(!m_SnapItemsDropped)
		{
			log_warn(TOOL_NAME, "snapshot full at tick %d, entities beyond the limit are not in the demo", m_Tick);
			m_SnapItemsDropped = true;
		}
	}

	// The characters of this tick and the box around them, so an entity far
	// from every player can skip the per-player checks. The order is the one
	// of the server's entity list, which inserts at the front: the character
	// that spawned last comes first, and it ticks first as well.
	void UpdateCharacterBox()
	{
		m_vAliveChars.clear();
		m_CharacterBoxMin = vec2(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
		m_CharacterBoxMax = vec2(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			const CPlayer &Player = m_aPlayers[Cid];
			if(!Player.m_Alive)
				continue;
			const vec2 Pos = CharPos(&Player);
			m_vAliveChars.push_back({Pos, Cid});
			m_CharacterBoxMin = vec2(std::min(m_CharacterBoxMin.x, Pos.x), std::min(m_CharacterBoxMin.y, Pos.y));
			m_CharacterBoxMax = vec2(std::max(m_CharacterBoxMax.x, Pos.x), std::max(m_CharacterBoxMax.y, Pos.y));
		}
		std::sort(m_vAliveChars.begin(), m_vAliveChars.end(), [this](const CCharacterRef &Left, const CCharacterRef &Right) {
			return m_aPlayers[Left.m_Cid].m_SpawnOrder > m_aPlayers[Right.m_Cid].m_SpawnOrder;
		});
	}

	// The same for the players the demo is recorded for, the ones an entity
	// has to be near to be in the snapshot
	void UpdateViewerBox()
	{
		m_vViewerPositions.clear();
		m_ViewerBoxMin = vec2(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
		m_ViewerBoxMax = vec2(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			if(!m_aPlayers[Cid].m_Alive || !IncludePlayer(Cid))
				continue;
			const vec2 Pos = CharPos(&m_aPlayers[Cid]);
			m_vViewerPositions.push_back(Pos);
			m_ViewerBoxMin = vec2(std::min(m_ViewerBoxMin.x, Pos.x), std::min(m_ViewerBoxMin.y, Pos.y));
			m_ViewerBoxMax = vec2(std::max(m_ViewerBoxMax.x, Pos.x), std::max(m_ViewerBoxMax.y, Pos.y));
		}
	}

	// CCharacter::DoWeaponSwitch
	void DoWeaponSwitch(CPlayer &Player)
	{
		// Switching waits for the weapon to be ready, and the ninja cannot be
		// put away at all
		if(Player.m_ReloadTimer != 0 || Player.m_QueuedWeapon == -1)
			return;
		if(Player.m_Core.m_aWeapons[WEAPON_NINJA].m_Got || !Player.m_Core.m_aWeapons[Player.m_QueuedWeapon].m_Got)
			return;
		SetWeapon(&Player, Player.m_QueuedWeapon);
	}

	// CCharacter::HandleWeaponSwitch, which weapon the player ends up holding
	void HandleWeaponSwitch(int Cid)
	{
		CPlayer &Player = m_aPlayers[Cid];
		int WantedWeapon = Player.m_QueuedWeapon == -1 ? Player.m_Core.m_ActiveWeapon : Player.m_QueuedWeapon;

		bool Anything = false;
		for(int Weapon = 0; Weapon < NUM_WEAPONS - 1; Weapon++)
			Anything = Anything || Player.m_Core.m_aWeapons[Weapon].m_Got;
		if(!Anything)
			return;

		// Mouse-wheel switching steps through the owned weapons. The input
		// fields are press+release counters wrapping at 64, count actual
		// presses like the server does.
		int Next = CountInput(Player.m_LastNextWeapon, Player.m_Input.m_NextWeapon).m_Presses;
		Player.m_LastNextWeapon = Player.m_Input.m_NextWeapon;
		int Prev = CountInput(Player.m_LastPrevWeapon, Player.m_Input.m_PrevWeapon).m_Presses;
		Player.m_LastPrevWeapon = Player.m_Input.m_PrevWeapon;
		if(Next < 128)
		{
			while(Next)
			{
				WantedWeapon = (WantedWeapon + 1) % NUM_WEAPONS;
				if(Player.m_Core.m_aWeapons[WantedWeapon].m_Got)
					Next--;
			}
		}
		if(Prev < 128)
		{
			while(Prev)
			{
				WantedWeapon = WantedWeapon - 1 < 0 ? NUM_WEAPONS - 1 : WantedWeapon - 1;
				if(Player.m_Core.m_aWeapons[WantedWeapon].m_Got)
					Prev--;
			}
		}
		// A weapon key overrides the wheel and only takes effect once the
		// player owns that weapon: the client keeps asking until then, it
		// only clears the field when the wheel is used
		if(Player.m_Input.m_WantedWeapon)
			WantedWeapon = Player.m_Input.m_WantedWeapon - 1;
		if(WantedWeapon >= 0 && WantedWeapon < NUM_WEAPONS && WantedWeapon != Player.m_Core.m_ActiveWeapon && Player.m_Core.m_aWeapons[WantedWeapon].m_Got)
			Player.m_QueuedWeapon = WantedWeapon;

		DoWeaponSwitch(Player);
	}

	// CCharacter::HandleNinja, without the dash: that one moves the tee
	// between the core tick and the move, so the simulation runs it
	void HandleNinja(CPlayer &Player)
	{
		if(Player.m_Core.m_ActiveWeapon != WEAPON_NINJA)
			return;

		if((m_Tick - Player.m_Core.m_Ninja.m_ActivationTick) > (NINJA_DURATION_MS * SERVER_TICK_SPEED / 1000))
		{
			RemoveNinja(&Player);
			return;
		}

		SetWeapon(&Player, WEAPON_NINJA);

		Player.m_Core.m_Ninja.m_CurrentMoveTime--;

		if(Player.m_Core.m_Ninja.m_CurrentMoveTime > 0)
		{
			const int Cid = &Player - m_aPlayers;
			if(GetSolo(Cid))
				return;

			CPlayer *apChars[MAX_CLIENTS];
			const float Radius = CCharacterCore::PhysicalSize() * 2.0f;
			const int Num = FindCharacters(CharPos(&Player), Radius, apChars, MAX_CLIENTS);
			for(int i = 0; i < Num; i++)
			{
				CPlayer *pChr = apChars[i];
				const int OtherCid = ClientId(pChr);
				if(pChr == &Player || Team(OtherCid) != Team(Cid) || GetSolo(OtherCid))
					continue;
				bool AlreadyHit = false;
				for(int j = 0; j < Player.m_NumObjectsHit; j++)
				{
					if(Player.m_aHitObjects[j] == OtherCid)
						AlreadyHit = true;
				}
				if(AlreadyHit)
					continue;
				if(distance(CharPos(pChr), CharPos(&Player)) > Radius)
					continue;

				CreateSound(CharPos(pChr), SOUND_NINJA_HIT, Cid);
				Player.m_aHitObjects[Player.m_NumObjectsHit++] = OtherCid;
				AddVelocity(pChr, vec2(0.0f, -10.0f));
			}
		}
	}

	// CCharacter::HandleJetpack, the push a jetpacking tee gets from firing.
	// It also runs between the core tick and the move, so the simulation
	// runs it with the input that drives the tick.
	void HandleJetpack(CPlayer &Player)
	{
		const int ActiveWeapon = Player.m_Core.m_ActiveWeapon;
		if(ActiveWeapon < 0)
			return;
		const CNetObj_PlayerInput &Input = Player.m_Core.m_Input;

		bool FullAuto = false;
		if(ActiveWeapon == WEAPON_GRENADE || ActiveWeapon == WEAPON_SHOTGUN || ActiveWeapon == WEAPON_LASER)
			FullAuto = true;
		if(Player.m_Core.m_Jetpack && ActiveWeapon == WEAPON_GUN)
			FullAuto = true;

		bool WillFire = CountInput(Player.m_PrevSimInput.m_Fire, Input.m_Fire).m_Presses != 0;
		if(FullAuto && (Input.m_Fire & 1) && Player.m_Core.m_aWeapons[ActiveWeapon].m_Ammo)
			WillFire = true;
		if(!WillFire)
			return;

		if(!Player.m_Core.m_aWeapons[ActiveWeapon].m_Ammo || IsFrozen(Player))
			return;

		if(ActiveWeapon == WEAPON_GUN && Player.m_Core.m_Jetpack)
		{
			const vec2 Direction = normalize(vec2(Input.m_TargetX, Input.m_TargetY));
			const float Strength = Tuning(Player.m_TuneZone).m_JetpackStrength;
			AddVelocity(&Player, Direction * -1.0f * (Strength / 100.0f / 6.11f));
		}
	}

	// The dash of a ninja, which moves the tee itself, and the jetpack push:
	// the server runs both after the core ticked and before it moves
	void HandleSimulatedWeapons(CPlayer &Player)
	{
		HandleJetpack(Player);
		if(Player.m_Core.m_ActiveWeapon != WEAPON_NINJA)
			return;
		if(Player.m_Core.m_Ninja.m_CurrentMoveTime == 0)
		{
			Player.m_Core.m_Vel = Player.m_Core.m_Ninja.m_ActivationDir * Player.m_Core.m_Ninja.m_OldVelAmount;
		}
		else if(Player.m_Core.m_Ninja.m_CurrentMoveTime > 0)
		{
			Player.m_Core.m_Vel = Player.m_Core.m_Ninja.m_ActivationDir * NINJA_VELOCITY;
			const vec2 GroundElasticity = vec2(Player.m_Core.m_Tuning.m_GroundElasticityX, Player.m_Core.m_Tuning.m_GroundElasticityY);
			m_Collision.MoveBox(&Player.m_Core.m_Pos, &Player.m_Core.m_Vel, CCharacterCore::PhysicalSizeVec2(), GroundElasticity);
			// The velocity is reset so the client does not predict the dash
			Player.m_Core.m_Vel = vec2(0.0f, 0.0f);
		}
	}

	// CCharacter::FireWeapon. Everything a shot creates is an entity that is
	// ticked and snapped like the map's own.
	void FireWeapon(int Cid)
	{
		CPlayer &Player = m_aPlayers[Cid];
		if(Player.m_ReloadTimer != 0)
			return;

		DoWeaponSwitch(Player);
		const vec2 MouseTarget = vec2(Player.m_Input.m_TargetX, Player.m_Input.m_TargetY);
		const vec2 Direction = normalize(MouseTarget);
		const int ActiveWeapon = Player.m_Core.m_ActiveWeapon;

		// Holding the fire button keeps firing for the weapons the server
		// calls full auto, which includes the gun while a jetpack is on: that
		// is the stream of bullets a jetpacking tee leaves behind
		bool FullAuto = false;
		if(ActiveWeapon == WEAPON_GRENADE || ActiveWeapon == WEAPON_SHOTGUN || ActiveWeapon == WEAPON_LASER)
			FullAuto = true;
		if(Player.m_Core.m_Jetpack && ActiveWeapon == WEAPON_GUN)
			FullAuto = true;
		// Allow firing directly after coming out of freeze
		if(Player.m_FrozenLastTick)
			FullAuto = true;

		if(!m_Config.m_SvDeepfly && ActiveWeapon == WEAPON_HAMMER && Player.m_Core.m_DeepFrozen)
			return;

		bool WillFire = false;
		if(CountInput(Player.m_LastFire, Player.m_Input.m_Fire).m_Presses)
			WillFire = true;
		if(FullAuto && (Player.m_Input.m_Fire & 1) && ActiveWeapon >= 0 && Player.m_Core.m_aWeapons[ActiveWeapon].m_Ammo)
			WillFire = true;
		if(!WillFire)
			return;

		const vec2 Pos = CharPos(&Player);
		if(IsFrozen(Player))
		{
			// Firing in freeze screams instead, at most once a second
			if(Player.m_PainSoundTimer <= 0 && !(Player.m_LastFire & 1))
			{
				Player.m_PainSoundTimer = SERVER_TICK_SPEED;
				CreateSound(Pos, SOUND_PLAYER_PAIN_LONG, Cid);
			}
			return;
		}

		if(ActiveWeapon < 0 || !Player.m_Core.m_aWeapons[ActiveWeapon].m_Ammo)
			return;

		const vec2 ProjStartPos = Pos + Direction * CCharacterCore::PhysicalSize() * 0.75f;

		switch(ActiveWeapon)
		{
		case WEAPON_HAMMER:
		{
			CreateSound(Pos, SOUND_HAMMER_FIRE, Cid);
			if(Player.m_Core.m_HammerHitDisabled)
				break;

			CPlayer *apChars[MAX_CLIENTS];
			int Hits = 0;
			const int Num = FindCharacters(ProjStartPos, CCharacterCore::PhysicalSize() * 0.5f, apChars, MAX_CLIENTS);
			for(int i = 0; i < Num; i++)
			{
				CPlayer *pTarget = apChars[i];
				const int TargetCid = ClientId(pTarget);
				if(pTarget == &Player || !CanCollide(Cid, TargetCid))
					continue;

				const vec2 TargetPos = CharPos(pTarget);
				if(length(TargetPos - ProjStartPos) > 0.0f)
					CreateHammerHit(TargetPos - normalize(TargetPos - ProjStartPos) * CCharacterCore::PhysicalSize() * 0.5f, Cid);
				else
					CreateHammerHit(ProjStartPos, Cid);

				const vec2 Dir = length(TargetPos - Pos) > 0.0f ? normalize(TargetPos - Pos) : vec2(0.0f, -1.0f);
				const float Strength = Tuning(Player.m_TuneZone).m_HammerStrength;
				vec2 Temp = pTarget->m_Core.m_Vel + normalize(Dir + vec2(0.0f, -1.1f)) * 10.0f;
				Temp = ClampVel(pTarget->m_MoveRestrictions, Temp);
				Temp -= pTarget->m_Core.m_Vel;
				AddVelocity(pTarget, (vec2(0.0f, -1.0f) + Temp) * Strength);
				Unfreeze(pTarget);
				Hits++;
			}

			// If we hit anything, we have to wait for the reload
			if(Hits)
				Player.m_ReloadTimer = Tuning(Player.m_TuneZone).m_HammerHitFireDelay * TICK_SPEED / 1000;
			break;
		}

		case WEAPON_GUN:
		{
			// A ninjajetpack tee swings instead of shooting, that setting is
			// not in the recording, so everyone shoots
			const int Lifetime = (int)(TICK_SPEED * Tuning(Player.m_TuneZone).m_GunLifetime);
			new CReplayProjectile(this, WEAPON_GUN, Cid, ProjStartPos, Direction, Lifetime, false, false, -1, MouseTarget, LAYER_GAME, 0);
			CreateSound(Pos, SOUND_GUN_FIRE, Cid);
			break;
		}

		case WEAPON_SHOTGUN:
		{
			new CReplayLaser(this, Pos, Direction, Tuning(Player.m_TuneZone).m_LaserReach, Cid, WEAPON_SHOTGUN);
			CreateSound(Pos, SOUND_SHOTGUN_FIRE, Cid);
			break;
		}

		case WEAPON_GRENADE:
		{
			const int Lifetime = (int)(TICK_SPEED * Tuning(Player.m_TuneZone).m_GrenadeLifetime);
			new CReplayProjectile(this, WEAPON_GRENADE, Cid, ProjStartPos, Direction, Lifetime, false, true, SOUND_GRENADE_EXPLODE, MouseTarget, LAYER_GAME, 0);
			CreateSound(Pos, SOUND_GRENADE_FIRE, Cid);
			break;
		}

		case WEAPON_LASER:
		{
			new CReplayLaser(this, Pos, Direction, Tuning(Player.m_TuneZone).m_LaserReach, Cid, WEAPON_LASER);
			CreateSound(Pos, SOUND_LASER_FIRE, Cid);
			break;
		}

		case WEAPON_NINJA:
		{
			Player.m_NumObjectsHit = 0;
			Player.m_Core.m_Ninja.m_ActivationDir = Direction;
			Player.m_Core.m_Ninja.m_CurrentMoveTime = NINJA_MOVETIME_MS * SERVER_TICK_SPEED / 1000;
			Player.m_Core.m_Ninja.m_OldVelAmount = std::clamp(length(Player.m_Core.m_Vel), 0.0f, 6000.0f);
			CreateSound(Pos, SOUND_NINJA_FIRE, Cid);
			break;
		}
		}

		Player.m_AttackTick = m_Tick;

		if(!Player.m_ReloadTimer && ActiveWeapon != -1)
			Player.m_ReloadTimer = Tuning(Player.m_TuneZone).GetWeaponFireDelay(ActiveWeapon) * TICK_SPEED;
	}

	// CCharacter::HandleWeapons. The server switches and fires a weapon for
	// every input it receives, at the end of the tick the input arrived in,
	// with the tee at the position that tick recorded: that is the input this
	// tick's chunks carry, so the weapon phase runs on those positions.
	void HandleWeapons()
	{
		m_TickEndPositions = true;
		UpdateCharacterBox();
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			CPlayer &Player = m_aPlayers[Cid];
			if(!Player.m_Alive)
				continue;
			HandleWeaponSwitch(Cid);
			if(Player.m_PainSoundTimer > 0)
				Player.m_PainSoundTimer--;
			if(Player.m_ReloadTimer)
				Player.m_ReloadTimer--;
			else
				FireWeapon(Cid);
			// DDRacePostCoreTick clears this right after the weapons ran
			Player.m_FrozenLastTick = false;
			Player.m_LastFire = Player.m_Input.m_Fire;
		}
		m_TickEndPositions = false;
	}

	void UpdateFreeze()
	{
		// Scan passes run without a map
		if(m_Layers.GameLayer() == nullptr)
			return;
		for(auto &Player : m_aPlayers)
		{
			if(!Player.m_Alive)
				continue;
			// A deep frozen tee is frozen again every tick, so leaving deep
			// freeze still leaves a freeze running (CCharacter::DDRacePostCoreTick)
			if(Player.m_DeepFrozen)
				Freeze(&Player);
			const vec2 Pos(Player.m_X, Player.m_Y);
			// The server runs the tile handler for every tile between the
			// previous and the current position (the anti-skip pass in
			// CCharacter::DDRacePostCoreTick), so a fast player cannot fly
			// through an unfreeze tile between two recorded positions
			const std::vector<int> vIndices = m_Collision.GetMapIndices(vec2(Player.m_TileScanX, Player.m_TileScanY), Pos);
			if(vIndices.empty())
			{
				HandleTiles(Player, m_Collision.GetMapIndex(Pos));
			}
			else
			{
				for(const int Index : vIndices)
					HandleTiles(Player, Index);
			}
			Player.m_TileScanX = Player.m_X;
			Player.m_TileScanY = Player.m_Y;

			// Whether the tee is standing in freeze right now, for the
			// snapshot flag (CCharacter::DDRaceTick)
			const int Here = m_Collision.GetPureMapIndex(Pos);
			const int aTiles[] = {m_Collision.GetTileIndex(Here), m_Collision.GetFrontTileIndex(Here), m_Collision.GetSwitchType(Here)};
			Player.m_InFreezeTile = false;
			for(const int Tile : aTiles)
			{
				if(Tile == TILE_FREEZE || Tile == TILE_DFREEZE || Tile == TILE_LFREEZE || Tile == TILE_DEATH)
				{
					Player.m_InFreezeTile = true;
					break;
				}
			}
		}
	}

	// The tick a freeze ends on, the server lets the tee fire right away
	void UpdateFrozenLastTick()
	{
		for(auto &Player : m_aPlayers)
		{
			if(Player.m_Alive && Player.m_FreezeEndTick > 0 && Player.m_FreezeEndTick == m_Tick && !Player.m_DeepFrozen)
				Player.m_FrozenLastTick = true;
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
				// Replay the server's tick: this tick's input applied to the
				// previous tick's recorded position, so hooks launch from the
				// same spot and fly along the same line as they did
				if(SimTicks == 1 && Player.m_PrevTick == m_Tick - 1)
					Player.m_Core.m_Pos = vec2(Player.m_PrevX, Player.m_PrevY);
				else
					Player.m_Core.m_Pos = vec2(Player.m_X, Player.m_Y);
			}
			// The order the world ticks its characters in decides who wins a
			// hook or a push, and it is the order of its entity list
			for(const CCharacterRef &Char : m_vAliveChars)
			{
				CPlayer &Player = m_aPlayers[Char.m_Cid];
				if(!Player.m_Alive)
					continue;
				// The zone the tee stands in decides its tuning, the server
				// copies it into the core before every tick (HandleTuneLayer)
				const int TuneZone = m_Collision.IsTune(m_Collision.GetMapIndex(Player.m_Core.m_Pos));
				Player.m_Core.m_Tuning = TuneZone == 0 ? m_Tuning : m_aTuneZones[TuneZone];
				Player.m_Core.m_Input = Player.m_SimInput;
				if(IsFrozen(Player))
				{
					// Frozen tees cannot move, jump or hook
					Player.m_Core.m_Input.m_Direction = 0;
					Player.m_Core.m_Input.m_Jump = 0;
					Player.m_Core.m_Input.m_Hook = 0;
				}
				// GRAB debug: the exact geometry and gates the flying-hook
				// player check inside Tick() will use this tick
				if(m_Tick >= m_DebugStartTick && m_Tick <= m_DebugEndTick && Player.m_Core.m_Input.m_Hook)
				{
					const int Cid = &Player - &m_aPlayers[0];
					const CCharacterCore &C = Player.m_Core;
					const vec2 Dir = C.m_HookState == HOOK_FLYING ? C.m_HookDir : normalize(vec2(C.m_Input.m_TargetX, C.m_Input.m_TargetY));
					const vec2 Start = C.m_HookState == HOOK_FLYING ? C.m_HookPos : C.m_Pos + Dir * C.PhysicalSize() * 1.5f;
					vec2 End = Start + Dir * (float)C.m_Tuning.m_HookFireSpeed;
					// Same clamp as Tick(): the player test only covers the segment up to the first wall
					vec2 Clamped = End;
					int TeleNr = 0;
					const int Hit = m_Collision.IntersectLineTeleHook(Start, End, &Clamped, nullptr, &TeleNr);
					log_info(TOOL_NAME, "GRAB tick=%d cid=%d hookstate=%d hooked=%d pos=%.2f,%.2f seg=%.1f,%.1f->%.1f,%.1f wallhit=%d clamped=%.1f,%.1f playerhooking=%.2f hookhitdisabled=%d solo=%d newhook=%d endlesshook=%d hooktick=%d",
						m_Tick, Cid, C.m_HookState, C.HookedPlayer(), C.m_Pos.x, C.m_Pos.y, Start.x, Start.y, End.x, End.y, Hit, Clamped.x, Clamped.y, (float)C.m_Tuning.m_PlayerHooking, C.m_HookHitDisabled, C.m_Solo, C.m_NewHook, C.m_EndlessHook, C.m_HookTick);
					for(int i = 0; i < (int)std::size(m_aPlayers); i++)
					{
						const auto &Other = m_aPlayers[i];
						if(i == Cid || !Other.m_Alive)
							continue;
						vec2 Closest;
						const bool OnSeg = closest_point_on_line(Start, Clamped, Other.m_Core.m_Pos, Closest);
						log_info(TOOL_NAME, "GRAB    other cid=%d pos=%.2f,%.2f onseg=%d distToSeg=%.2f radius=%.1f cancollide=%d team_me=%d team_other=%d solo=%d",
							i, Other.m_Core.m_Pos.x, Other.m_Core.m_Pos.y, OnSeg, OnSeg ? distance(Other.m_Core.m_Pos, Closest) : -1.0f, C.PhysicalSize() + 2.0f,
							m_TeamsCore.CanCollide(i, Cid), m_TeamsCore.Team(Cid), m_TeamsCore.Team(i), Other.m_Core.m_Solo);
					}
				}
				Player.m_Core.Tick(true);
				HandleSimulatedWeapons(Player);
				Player.m_PrevSimInput = Player.m_Core.m_Input;
				// The server never lets a hook on a player time out while
				// endless hook is active (CCharacter::DDRacePostCoreTick)
				if(Player.m_Core.m_EndlessHook)
					Player.m_Core.m_HookTick = 0;
				// Same place decides whether the feet are drawn light or dark
				if(Player.m_Core.m_Jumps <= 0)
					Player.m_Core.m_Jumped |= 2;
				else if(Player.m_Core.m_Jumps == 1 && Player.m_Core.m_Jumped > 0)
					Player.m_Core.m_Jumped |= 2;
				else if(Player.m_Core.m_JumpedTotal < Player.m_Core.m_Jumps - 1 && Player.m_Core.m_Jumped > 1)
					Player.m_Core.m_Jumped = 1;
				if(Player.m_Core.m_EndlessJump && Player.m_Core.m_Jumped > 1)
					Player.m_Core.m_Jumped = 1;
			}
			for(const CCharacterRef &Char : m_vAliveChars)
			{
				CPlayer &Player = m_aPlayers[Char.m_Cid];
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
			// How far the replayed physics drifted from the recording this
			// tick, before it is snapped back: the measure of how exactly the
			// simulation matches the server
			if(SimTicks == 1 && Player.m_PrevTick == m_Tick - 1)
			{
				const float Err = distance(Player.m_Core.m_Pos, RecordedPos);
				m_ErrSum += Err;
				m_ErrCount++;
				m_ErrMax = std::max(m_ErrMax, (double)Err);
				if(Err > 4.0f)
					m_ErrOver4++;
			}
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
			// The gates the real attach passes (CCharacterCore::Tick): a map
			// or a tile that forbids hooking players never lets one through
			if(Core.m_HookHitDisabled || Core.m_Tuning.m_PlayerHooking == 0)
				continue;
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
				From = Core.m_Pos + TargetDirection * Core.PhysicalSize() * 1.5f;
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
				if(distance(Other.m_Core.m_Pos, ClosestPoint) < Core.PhysicalSize() * 1.5f && (ClosestCid == -1 || TargetDistance < ClosestDistance))
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

	bool InRecordWindow() const { return m_Tick >= m_StartTick && m_Tick <= m_EndTick; }

	// One row per included alive player: the recorded position, the physics
	// state of the guided simulation and the raw input that the following
	// tick is simulated with.
	void WriteDatasetTick()
	{
		if(m_DatasetFile == nullptr)
			return;
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			const CPlayer &Player = m_aPlayers[Cid];
			if(!Player.m_Alive || !IncludePlayer(Cid))
				continue;
			const CNetObj_PlayerInput &Input = Player.m_Input;
			char aRow[512];
			str_format(aRow, sizeof(aRow),
				"{\"t\":%d,\"cid\":%d,\"team\":%d,\"x\":%d,\"y\":%d,\"vx\":%.4f,\"vy\":%.4f,"
				"\"hook_state\":%d,\"hook_x\":%.2f,\"hook_y\":%.2f,\"hooked\":%d,\"jumped\":%d,\"weapon\":%d,"
				"\"in_dir\":%d,\"in_tx\":%d,\"in_ty\":%d,\"in_jump\":%d,\"in_fire\":%d,\"in_hook\":%d,"
				"\"in_flags\":%d,\"in_wanted\":%d,\"in_next\":%d,\"in_prev\":%d}\n",
				m_Tick, Cid, m_TeamsCore.Team(Cid), Player.m_X, Player.m_Y, Player.m_Core.m_Vel.x, Player.m_Core.m_Vel.y,
				Player.m_Core.m_HookState, Player.m_Core.m_HookPos.x, Player.m_Core.m_HookPos.y, Player.m_Core.HookedPlayer(),
				Player.m_Core.m_Jumped, Player.m_Core.m_ActiveWeapon,
				Input.m_Direction, Input.m_TargetX, Input.m_TargetY, Input.m_Jump, Input.m_Fire, Input.m_Hook,
				Input.m_PlayerFlags, Input.m_WantedWeapon, Input.m_NextWeapon, Input.m_PrevWeapon);
			io_write(m_DatasetFile, aRow, str_length(aRow));
			m_NumDatasetRows++;
		}
	}

	void FlushTick()
	{
		if(!m_TickDirty)
			return;

		// Recordings from before September 2021 have no team chunks either,
		// there the run is whoever holds the rank's names
		if(m_pvRankNames != nullptr && m_ApproxCandidate.m_Cid < 0 && m_RankExpectedTick >= 0 && m_Tick >= m_RankExpectedTick)
		{
			for(const char *pName : *m_pvRankNames)
			{
				const int Cid = FindPlayer(pName);
				if(Cid >= 0)
					m_vApproxCids.push_back(Cid);
			}
			// Every name has to be there, a run with a member missing is
			// shown without that tee
			if(m_vApproxCids.size() == m_pvRankNames->size())
				m_ApproxCandidate = {m_RankExpectedTick, m_vApproxCids[0], m_TeamsCore.Team(m_vApproxCids[0])};
			else
				m_vApproxCids.clear();
		}

		DetectPositionJumps();

		if(m_Tick >= m_DebugStartTick && m_Tick <= m_DebugEndTick)
		{
			for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
			{
				const CPlayer &Player = m_aPlayers[Cid];
				if(!Player.m_Alive)
					continue;
				log_info(TOOL_NAME, "tick=%d cid=%d '%s' pos=%d,%d dir=%d jump=%d hook=%d target=%d,%d frozen=%d(intile=%d deep=%d end=%d) hookstate=%d hooked=%d weapon=%d wanted=%d next=%d prev=%d fire=%d",
					m_Tick, Cid, Player.m_aName, Player.m_X, Player.m_Y,
					Player.m_Input.m_Direction, Player.m_Input.m_Jump, Player.m_Input.m_Hook,
					Player.m_Input.m_TargetX, Player.m_Input.m_TargetY,
					IsFrozen(Player), Player.m_InFreezeTile, Player.m_DeepFrozen, Player.m_FreezeEndTick,
					Player.m_Core.m_HookState, Player.m_Core.HookedPlayer(),
					Player.m_Core.m_ActiveWeapon, Player.m_Input.m_WantedWeapon, Player.m_Input.m_NextWeapon, Player.m_Input.m_PrevWeapon, Player.m_Input.m_Fire);
			}
		}

		if(m_Tick > m_EndTick)
		{
			m_Done = true;
			m_TickDirty = false;
			m_vTickMessages.clear();
			return;
		}
		// A scan pass runs without a map, it only looks for the finish chunks
		const bool HasWorld = m_Layers.GameLayer() != nullptr;
		// The world ticks its entities before the characters, and a character
		// switches and fires its weapon before its core moves
		if(HasWorld)
		{
			TickEntities();
			UpdateTuneZones();
			UpdateFrozenLastTick();
			// The ninja timer runs in the character's tick, before the core,
			// the rest of the weapons after it
			for(CPlayer &Player : m_aPlayers)
			{
				if(Player.m_Alive)
					HandleNinja(Player);
			}
		}

		const bool Record = m_Tick >= m_StartTick;
		if(Record)
		{
			// An input is recorded in the tick section it ARRIVES in, but the
			// server applies it in the next tick: CServer::Run calls
			// OnClientPredictedEarlyInput (where teehistorian records) before
			// m_CurrentGameTick++, while BeginTick for the new tick only runs
			// inside the following OnTick. So the input of section T drives
			// tick T+1. Measured on real runs: mean simulation error against
			// the recording drops from 1.22 to 1.07 px on Kobra 4 and from
			// 2.02 to 1.73 px on Multeasymap, with 39 % and 30 % fewer ticks
			// off by more than 4 px. T2D_INPUT_DELAY=0 replays the old way.
			const char *pDelayOverride = getenv("T2D_INPUT_DELAY");
			const bool InputDelay = pDelayOverride != nullptr ? str_comp(pDelayOverride, "0") != 0 : m_InputAppliedNextTick;
			for(auto &Player : m_aPlayers)
			{
				// A tick can carry several recorded inputs, the server moves
				// and hooks with the first of them
				const CNetObj_PlayerInput &TickInput = Player.m_HasTickInput ? Player.m_TickInput : Player.m_Input;
				Player.m_SimInput = InputDelay ? Player.m_DelayedInput : TickInput;
				Player.m_DelayedInput = TickInput;
			}
			Simulate();
		}
		// The weapons run after the core has ticked, the same way the server
		// fires them from the inputs that arrived during the tick, with the
		// tee already standing where this tick recorded it
		if(HasWorld)
			HandleWeapons();
		if(Record)
		{
			WriteDatasetTick();
			// The team is dissolved in the tick it finishes in, so the run
			// keeps the players it had one tick earlier. This has to happen
			// before the snapshot of the tick is built, or that snapshot
			// holds nobody at all, and it cannot wait for the finish marker:
			// a rank placed by its timestamp only reaches it seconds later.
			if(m_FilterTeam > 0 && !m_FinishLatched && m_MarkerStartTick < 0 && !m_vTeamCids.empty() && !TeamAlive())
				LatchRun();
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
				if(!m_FinishLatched)
					LatchRun();
			}
			if(m_TeamsDirty)
			{
				SendTeamsState();
				m_TeamsDirty = false;
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
				Player.m_PrevX = Player.m_X;
				Player.m_PrevY = Player.m_Y;
				Player.m_PrevTick = m_Tick;
				Player.m_HasTickInput = false;
			}
		}
		m_MaxPlayersSeen = std::max(m_MaxPlayersSeen, NumPlayers);
		if(Record && m_MarkerFinishTick >= 0 && !m_FinishLatched)
		{
			m_vTeamCids.clear();
			for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
			{
				if(m_aPlayers[Cid].m_Alive && IncludePlayer(Cid))
					m_vTeamCids.push_back(Cid);
			}
		}
		// The server handles tiles after weapons and movement, so freeze from
		// a tile entered this tick takes effect next tick: a hammer fired
		// while entering freeze still lands and unfreezes its target
		UpdateFreeze();
		// CGameContext::OnTick expires the timed switchers once the world and
		// its characters have run
		UpdateSwitchers();

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
		// One builder for the whole demo, never a fresh one per tick: a builder
		// numbers the extended item types in the order it first met them and
		// Init() puts every type it knows back in that order, which keeps the
		// numbering the same from one snapshot to the next. A new builder per
		// tick numbers them by first use in that tick, so the number of a
		// type shifts whenever a laser or a projectile appears, and the demo
		// deltas, keyed by that number, pair the wrong items: positions drift
		// by tens of pixels until the next keyframe and a freeze end becomes a
		// tick hours away.
		CSnapshotBuilder &Builder = m_SnapshotBuilder;
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
					pCharacter->m_Weapon = pPlayer->m_Core.m_ActiveWeapon;
					// The ninja bar of the hud is drawn from the ammo count
					if(pPlayer->m_Core.m_ActiveWeapon == WEAPON_NINJA)
						pCharacter->m_AmmoCount = pPlayer->m_Core.m_Ninja.m_ActivationTick + NINJA_DURATION_MS * SERVER_TICK_SPEED / 1000;
					pCharacter->m_Emote = Frozen ? EMOTE_PAIN : EMOTE_NORMAL;
					pCharacter->m_AttackTick = pPlayer->m_AttackTick;
				}

				CNetObj_DDNetCharacter *pDDNetCharacter = (CNetObj_DDNetCharacter *)Builder.NewItemRaw(NETOBJTYPE_DDNETCHARACTER, Cid, sizeof(CNetObj_DDNetCharacter));
				if(pDDNetCharacter)
				{
					mem_zero(pDDNetCharacter, sizeof(*pDDNetCharacter));
					// The weapon flags say which weapons the player owns, not
					// which one is drawn in its hands
					if(pPlayer->m_Core.m_aWeapons[WEAPON_HAMMER].m_Got)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_HAMMER;
					if(pPlayer->m_Core.m_aWeapons[WEAPON_GUN].m_Got)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_GUN;
					if(pPlayer->m_Core.m_aWeapons[WEAPON_SHOTGUN].m_Got)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_SHOTGUN;
					if(pPlayer->m_Core.m_aWeapons[WEAPON_GRENADE].m_Got)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_GRENADE;
					if(pPlayer->m_Core.m_aWeapons[WEAPON_LASER].m_Got)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_LASER;
					if(pPlayer->m_Core.m_ActiveWeapon == WEAPON_NINJA)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_WEAPON_NINJA;
					if(pPlayer->m_Core.m_Solo)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_SOLO;
					if(pPlayer->m_Core.m_LiveFrozen)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_MOVEMENTS_DISABLED;
					if(pPlayer->m_Core.m_CollisionDisabled)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_COLLISION_DISABLED;
					if(pPlayer->m_Core.m_HookHitDisabled)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_HOOK_HIT_DISABLED;
					if(pPlayer->m_Core.m_HammerHitDisabled)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_HAMMER_HIT_DISABLED;
					if(pPlayer->m_Core.m_ShotgunHitDisabled)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_SHOTGUN_HIT_DISABLED;
					if(pPlayer->m_Core.m_GrenadeHitDisabled)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_GRENADE_HIT_DISABLED;
					if(pPlayer->m_Core.m_LaserHitDisabled)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_LASER_HIT_DISABLED;
					if(pPlayer->m_InFreezeTile)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_IN_FREEZE;
					if(pPlayer->m_Core.m_EndlessHook)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_ENDLESS_HOOK;
					if(pPlayer->m_Core.m_Jetpack)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_JETPACK;
					if(pPlayer->m_Core.m_EndlessJump)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_ENDLESS_JUMP;
					if(pPlayer->m_Core.m_HasTelegunGun)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_TELEGUN_GUN;
					if(pPlayer->m_Core.m_HasTelegunGrenade)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_TELEGUN_GRENADE;
					if(pPlayer->m_Core.m_HasTelegunLaser)
						pDDNetCharacter->m_Flags |= CHARACTERFLAG_TELEGUN_LASER;
					pDDNetCharacter->m_FreezeEnd = pPlayer->m_DeepFrozen ? -1 : (Frozen ? pPlayer->m_FreezeEndTick : 0);
					pDDNetCharacter->m_FreezeStart = Frozen ? pPlayer->m_FreezeStartTick : 0;
					pDDNetCharacter->m_StrongWeakId = StrongWeakId(Cid);
					pDDNetCharacter->m_Jumps = pPlayer->m_Core.m_Jumps;
					pDDNetCharacter->m_JumpedTotal = pPlayer->m_Core.m_JumpedTotal;
					pDDNetCharacter->m_NinjaActivationTick = pPlayer->m_Core.m_Ninja.m_ActivationTick;
					pDDNetCharacter->m_TargetX = pPlayer->m_Input.m_TargetX;
					pDDNetCharacter->m_TargetY = pPlayer->m_Input.m_TargetY;
					pDDNetCharacter->m_TuneZoneOverride = TuneZone::OVERRIDE_NONE;
				}
			}
		}

		SnapEntities(&Builder);

		for(const CPendingEvent &Event : m_vPendingEvents)
		{
			if(Event.m_Type == NETEVENTTYPE_SOUNDWORLD)
			{
				CNetEvent_SoundWorld *pEvent = (CNetEvent_SoundWorld *)Builder.NewItemRaw(NETEVENTTYPE_SOUNDWORLD, m_NextEventId, sizeof(CNetEvent_SoundWorld));
				if(pEvent)
				{
					pEvent->m_X = Event.m_X;
					pEvent->m_Y = Event.m_Y;
					pEvent->m_SoundId = Event.m_Data;
				}
			}
			else if(Event.m_Type == NETEVENTTYPE_DAMAGEIND)
			{
				CNetEvent_DamageInd *pEvent = (CNetEvent_DamageInd *)Builder.NewItemRaw(NETEVENTTYPE_DAMAGEIND, m_NextEventId, sizeof(CNetEvent_DamageInd));
				if(pEvent)
				{
					pEvent->m_X = Event.m_X;
					pEvent->m_Y = Event.m_Y;
					pEvent->m_Angle = Event.m_Data;
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

		// CGameContext::SnapSwitchers. Without it the client keeps every
		// switcher closed, so the doors and lasers a run opens stay in the way
		const int SwitchTeam = SnapTeam();
		if(!m_WorldCore.m_vSwitchers.empty() && SwitchTeam != TEAM_SUPER)
		{
			CNetObj_SwitchState *pSwitchState = (CNetObj_SwitchState *)Builder.NewItemRaw(NETOBJTYPE_SWITCHSTATE, SwitchTeam, sizeof(CNetObj_SwitchState));
			if(pSwitchState)
			{
				mem_zero(pSwitchState, sizeof(*pSwitchState));
				pSwitchState->m_HighestSwitchNumber = std::clamp((int)m_WorldCore.m_vSwitchers.size() - 1, 0, 255);
				// The end ticks of up to four switchers that are about to
				// toggle back, the earliest first
				std::vector<std::pair<int, int>> vEndTicks;
				for(int Number = 0; Number <= pSwitchState->m_HighestSwitchNumber; Number++)
				{
					const SSwitchers &Switcher = m_WorldCore.m_vSwitchers[Number];
					pSwitchState->m_aStatus[Number / 32] |= (int)Switcher.m_aStatus[SwitchTeam] << (Number % 32);
					const int EndTick = Switcher.m_aEndTick[SwitchTeam];
					if(EndTick > 0 && EndTick < m_Tick + 3 * SERVER_TICK_SPEED && Switcher.m_aLastUpdateTick[SwitchTeam] < m_Tick)
						vEndTicks.emplace_back(EndTick, Number);
				}
				std::sort(vEndTicks.begin(), vEndTicks.end());
				const size_t NumTimedSwitchers = std::min(vEndTicks.size(), std::size(pSwitchState->m_aEndTicks));
				for(size_t i = 0; i < NumTimedSwitchers; i++)
				{
					pSwitchState->m_aSwitchNumbers[i] = vEndTicks[i].second;
					pSwitchState->m_aEndTicks[i] = vEndTicks[i].first;
				}
			}
		}

		CSnapshotBuffer Buffer;
		const int SnapshotSize = Builder.Finish(&Buffer);
		m_Recorder.RecordSnapshot(m_Tick, Buffer.AsSnapshot(), SnapshotSize);
		m_NumSnapshots++;
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			if(m_aPlayers[Cid].m_Alive && m_aPlayers[Cid].m_Core.m_HookState == HOOK_GRABBED && m_aPlayers[Cid].m_Core.HookedPlayer() >= 0)
			{
				m_NumPlayerHookTicks++;
				break;
			}
		}
		for(int Cid = 0; Cid < MAX_CLIENTS; Cid++)
		{
			if(m_aPlayers[Cid].m_Alive && IsFrozen(m_aPlayers[Cid]))
			{
				m_NumFrozenTicks++;
				break;
			}
		}
	}
};

// ---------------------------------------------------------------------------
// Entity implementations, see the declarations above CConverter
// ---------------------------------------------------------------------------

CReplayEntity::CReplayEntity(CConverter *pConverter, EType EntityType, vec2 Pos, int Layer, int Number) :
	m_pConverter(pConverter),
	m_EntityType(EntityType),
	m_Id(pConverter->NewSnapId()),
	m_Pos(Pos),
	m_Layer(Layer),
	m_Number(Number)
{
}

CReplayEntity::~CReplayEntity()
{
	m_pConverter->FreeSnapId(m_Id);
}

CCollision *CReplayEntity::Collision() const { return m_pConverter->Collision(); }
std::vector<SSwitchers> &CReplayEntity::Switchers() const { return m_pConverter->Switchers(); }
int CReplayEntity::ServerTick() const { return m_pConverter->ServerTick(); }
const CTuningParams &CReplayEntity::Tuning(int Zone) const { return m_pConverter->Tuning(Zone); }

CReplayPickup::CReplayPickup(CConverter *pConverter, int Type, int SubType, vec2 Pos, int Layer, int Number, int Flags) :
	CReplayEntity(pConverter, PICKUP, Pos, Layer, Number),
	m_Type(Type),
	m_Subtype(SubType),
	m_Flags(Flags)
{
	m_pConverter->InsertEntity(this);
}

void CReplayPickup::Move()
{
	if(ServerTick() % (int)(TICK_SPEED * 0.15f) == 0)
	{
		Collision()->MoverSpeed(m_Pos.x, m_Pos.y, &m_Core);
		m_Pos += m_Core;
	}
}

void CReplayPickup::Tick()
{
	Move();

	CConverter::CPlayer *apChars[MAX_CLIENTS];
	const int Num = m_pConverter->FindCharacters(m_Pos, ms_PhysicsRadius + ms_CollisionExtraSize, apChars, MAX_CLIENTS);
	for(int i = 0; i < Num; i++)
	{
		CConverter::CPlayer *pChr = apChars[i];
		const int Cid = m_pConverter->ClientId(pChr);
		const int Team = m_pConverter->Team(Cid);
		if(m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[Team])
			continue;
		bool Sound = false;
		switch(m_Type)
		{
		case POWERUP_FREEZE:
			if(m_pConverter->Freeze(pChr))
				m_pConverter->CreateSound(m_Pos, SOUND_PICKUP_HEALTH, Cid);
			break;

		case POWERUP_ARMOR:
			if(Team == TEAM_SUPER)
				continue;
			for(int Weapon = WEAPON_SHOTGUN; Weapon < NUM_WEAPONS; Weapon++)
			{
				if(pChr->m_Core.m_aWeapons[Weapon].m_Got)
				{
					pChr->m_Core.m_aWeapons[Weapon].m_Got = false;
					pChr->m_Core.m_aWeapons[Weapon].m_Ammo = 0;
					Sound = true;
				}
			}
			pChr->m_Core.m_Ninja.m_ActivationDir = vec2(0, 0);
			pChr->m_Core.m_Ninja.m_ActivationTick = -500;
			pChr->m_Core.m_Ninja.m_CurrentMoveTime = 0;
			if(Sound)
			{
				pChr->m_LastWeapon = WEAPON_GUN;
				m_pConverter->CreateSound(m_Pos, SOUND_PICKUP_ARMOR, Cid);
			}
			if(pChr->m_Core.m_ActiveWeapon >= WEAPON_SHOTGUN)
				pChr->m_Core.m_ActiveWeapon = WEAPON_HAMMER;
			break;

		case POWERUP_ARMOR_SHOTGUN:
		case POWERUP_ARMOR_GRENADE:
		case POWERUP_ARMOR_LASER:
		{
			if(Team == TEAM_SUPER)
				continue;
			const int Weapon = m_Type == POWERUP_ARMOR_SHOTGUN ? WEAPON_SHOTGUN : (m_Type == POWERUP_ARMOR_GRENADE ? WEAPON_GRENADE : WEAPON_LASER);
			if(pChr->m_Core.m_aWeapons[Weapon].m_Got)
			{
				pChr->m_Core.m_aWeapons[Weapon].m_Got = false;
				pChr->m_Core.m_aWeapons[Weapon].m_Ammo = 0;
				pChr->m_LastWeapon = WEAPON_GUN;
				m_pConverter->CreateSound(m_Pos, SOUND_PICKUP_ARMOR, Cid);
			}
			if(pChr->m_Core.m_ActiveWeapon == Weapon)
				pChr->m_Core.m_ActiveWeapon = WEAPON_HAMMER;
			break;
		}

		case POWERUP_ARMOR_NINJA:
			if(Team == TEAM_SUPER)
				continue;
			pChr->m_Core.m_Ninja.m_ActivationDir = vec2(0, 0);
			pChr->m_Core.m_Ninja.m_ActivationTick = -500;
			pChr->m_Core.m_Ninja.m_CurrentMoveTime = 0;
			break;

		case POWERUP_WEAPON:
			if(m_Subtype >= 0 && m_Subtype < NUM_WEAPONS && (!pChr->m_Core.m_aWeapons[m_Subtype].m_Got || pChr->m_Core.m_aWeapons[m_Subtype].m_Ammo != -1))
			{
				m_pConverter->GiveWeapon(pChr, m_Subtype);
				if(m_Subtype == WEAPON_GRENADE)
					m_pConverter->CreateSound(m_Pos, SOUND_PICKUP_GRENADE, Cid);
				else if(m_Subtype == WEAPON_SHOTGUN || m_Subtype == WEAPON_LASER)
					m_pConverter->CreateSound(m_Pos, SOUND_PICKUP_SHOTGUN, Cid);
			}
			break;

		case POWERUP_NINJA:
			m_pConverter->GiveNinja(pChr);
			break;

		default:
			break;
		}
	}
}

void CReplayPickup::Snap()
{
	if(m_pConverter->NetworkClipped(m_Pos))
		return;
	m_pConverter->SnapPickupObject(m_Id, m_Pos, m_Type, m_Subtype, m_Number, m_Flags);
}

CReplayDoor::CReplayDoor(CConverter *pConverter, vec2 Pos, float Rotation, int Length, int Number) :
	CReplayEntity(pConverter, LASER, Pos, LAYER_GAME, Number),
	m_Length(Length)
{
	m_Direction = vec2(std::sin(Rotation), std::cos(Rotation));
	const vec2 To = Pos + normalize(m_Direction) * m_Length;
	Collision()->IntersectNoLaser(Pos, To, &m_To, nullptr);
	ResetCollision();
	m_pConverter->InsertEntity(this);
}

void CReplayDoor::ResetCollision()
{
	if(Collision()->GetTile(m_Pos.x, m_Pos.y) || Collision()->GetFrontTile(m_Pos.x, m_Pos.y))
		return;

	for(int i = 0; i < m_Length - 1; i++)
	{
		const vec2 CurrentPos = m_Pos + m_Direction * i;
		if(Collision()->CheckPoint(CurrentPos))
			break;
		Collision()->SetDoorCollisionAt(CurrentPos.x, CurrentPos.y, TILE_STOPA, 0, m_Number);
	}
}

void CReplayDoor::Snap()
{
	if(m_pConverter->NetworkClipped(m_Pos) && m_pConverter->NetworkClipped(m_To))
		return;
	m_pConverter->SnapLaserObject(m_Id, m_Pos, m_To, -1, -1, LASERTYPE_DOOR, 0, m_Number);
}

CReplayLight::CReplayLight(CConverter *pConverter, vec2 Pos, float Rotation, int Length, int Layer, int Number) :
	CReplayEntity(pConverter, LASER, Pos, Layer, Number),
	m_Rotation(Rotation),
	m_EvalTick(pConverter->ServerTick()),
	m_Tick((int)(TICK_SPEED * 0.15f)),
	m_Length(Length)
{
	m_pConverter->InsertEntity(this);
	Step();
}

bool CReplayLight::HitCharacter()
{
	if(!m_pConverter->CharactersNear(m_Pos, m_Pos, m_Length))
		return false;
	const std::vector<CConverter::CPlayer *> vpHitCharacters = m_pConverter->IntersectedCharacters(m_Pos, To(), 0.0f, nullptr);
	if(vpHitCharacters.empty())
		return false;
	for(CConverter::CPlayer *pChr : vpHitCharacters)
	{
		if(m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[m_pConverter->Team(m_pConverter->ClientId(pChr))])
			continue;
		m_pConverter->Freeze(pChr);
	}
	return true;
}

void CReplayLight::Move()
{
	if(m_Speed != 0)
	{
		if((m_CurveLength >= m_Length && m_Speed > 0) || (m_CurveLength <= 0 && m_Speed < 0))
			m_Speed = -m_Speed;
		m_CurveLength += m_Speed * m_Tick + m_LengthL;
		m_LengthL = 0;
		if(m_CurveLength > m_Length)
		{
			m_LengthL = m_CurveLength - m_Length;
			m_CurveLength = m_Length;
		}
		else if(m_CurveLength < 0)
		{
			m_LengthL = 0 + m_CurveLength;
			m_CurveLength = 0;
		}
	}

	m_Rotation += m_AngularSpeed * m_Tick;
	if(m_Rotation > pi * 2)
		m_Rotation -= pi * 2;
	else if(m_Rotation < 0)
		m_Rotation += pi * 2;
}

void CReplayLight::Step()
{
	Move();
	m_ToDirty = true;
}

vec2 CReplayLight::To()
{
	if(m_ToDirty)
	{
		const vec2 Direction = vec2(std::sin(m_Rotation), std::cos(m_Rotation));
		const vec2 NextPosition = m_Pos + normalize(Direction) * m_CurveLength;
		Collision()->IntersectNoLaser(m_Pos, NextPosition, &m_To, nullptr);
		m_ToDirty = false;
	}
	return m_To;
}

void CReplayLight::Tick()
{
	if(ServerTick() % (int)(TICK_SPEED * 0.15f) == 0)
	{
		m_EvalTick = ServerTick();
		Collision()->MoverSpeed(m_Pos.x, m_Pos.y, &m_Core);
		m_Pos += m_Core;
		Step();
	}

	HitCharacter();
}

void CReplayLight::Snap()
{
	// The beam ends at most its length away, so a light farther than that is
	// clipped without tracing where it ends
	if(m_pConverter->NetworkClipped(m_Pos, m_Length))
		return;
	if(m_pConverter->NetworkClipped(m_Pos) && m_pConverter->NetworkClipped(To()))
		return;

	// A light is drawn at its full length while it is on for the team the
	// replay follows, and as a dot while it is off
	vec2 From = m_Pos;
	if(m_Layer == LAYER_SWITCH && m_Number > 0)
	{
		if(Switchers()[m_Number].m_aStatus[m_pConverter->SnapTeam()])
			From = To();
	}
	else
	{
		From = To();
	}
	m_pConverter->SnapLaserObject(m_Id, m_Pos, From, -1, -1, LASERTYPE_FREEZE, 0, m_Number);
}

CReplayDraggerBeam::CReplayDraggerBeam(CConverter *pConverter, CReplayDragger *pDragger, vec2 Pos, float Strength,
	bool IgnoreWalls, int ForClientId, int Layer, int Number) :
	CReplayEntity(pConverter, LASER, Pos, Layer, Number),
	m_pDragger(pDragger),
	m_Strength(Strength),
	m_IgnoreWalls(IgnoreWalls),
	m_ForClientId(ForClientId),
	m_EvalTick(pConverter->ServerTick()),
	m_Active(true)
{
	m_pConverter->InsertEntity(this);
}

void CReplayDraggerBeam::Deactivate()
{
	Reset();
	m_Active = false;
	m_pDragger->RemoveDraggerBeam(m_ForClientId);
}

void CReplayDraggerBeam::Tick()
{
	if(!m_Active)
		return;

	CConverter::CPlayer *pTarget = m_pConverter->GetPlayerChar(m_ForClientId);
	if(!pTarget)
	{
		Deactivate();
		return;
	}

	// The dragger only looks for players every 150 ms, so a beam can stay
	// alive up to 6 ticks after its switcher was closed
	if(ServerTick() % (int)(TICK_SPEED * 0.15f) == 0)
	{
		if(m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[m_pConverter->Team(m_ForClientId)])
		{
			Deactivate();
			return;
		}
	}

	if(distance(m_pConverter->CharPos(pTarget), m_Pos) >= m_pConverter->Config().m_SvDraggerRange ||
		(m_IgnoreWalls ?
				Collision()->IntersectNoLaserNoWalls(m_Pos, m_pConverter->CharPos(pTarget), nullptr, nullptr) :
				Collision()->IntersectNoLaser(m_Pos, m_pConverter->CharPos(pTarget), nullptr, nullptr)))
	{
		Deactivate();
		return;
	}
	// In the center of the dragger a tee does not experience speed-up
	else if(distance(m_pConverter->CharPos(pTarget), m_Pos) > 28)
	{
		m_pConverter->AddVelocity(pTarget, normalize(m_Pos - m_pConverter->CharPos(pTarget)) * m_Strength);
	}
}

void CReplayDraggerBeam::Snap()
{
	if(!m_Active)
		return;

	CConverter::CPlayer *pTarget = m_pConverter->GetPlayerChar(m_ForClientId);
	if(!pTarget || !m_pConverter->CanSnapCharacter(m_ForClientId))
		return;

	const vec2 TargetPos = m_pConverter->CharPos(pTarget);
	if(distance(TargetPos, m_Pos) >= m_pConverter->Config().m_SvDraggerRange || m_pConverter->NetworkClippedLine(m_Pos, TargetPos))
		return;

	const int Subtype = (m_IgnoreWalls ? 1 : 0) | (std::clamp(round_to_int(m_Strength - 1.0f), 0, 2) << 1);
	// The dragger lends its id to the beam of the team the replay follows, so
	// the client sees one entity moving instead of two
	const int SnapId = m_pDragger->WillDraggerBeamUseDraggerId(m_ForClientId, m_pConverter->SnapCid()) ? m_pDragger->SnapId() : m_Id;
	m_pConverter->SnapLaserObject(SnapId, TargetPos, m_Pos, -1, m_ForClientId, LASERTYPE_DRAGGER, Subtype, m_Number);
}

CReplayDragger::CReplayDragger(CConverter *pConverter, vec2 Pos, float Strength, bool IgnoreWalls, int Layer, int Number) :
	CReplayEntity(pConverter, LASER, Pos, Layer, Number),
	m_Strength(Strength),
	m_IgnoreWalls(IgnoreWalls),
	m_EvalTick(pConverter->ServerTick())
{
	for(int &TargetId : m_aTargetIdInTeam)
	{
		TargetId = -1;
	}
	m_pConverter->InsertEntity(this);
}

void CReplayDragger::Tick()
{
	if(ServerTick() % (int)(TICK_SPEED * 0.15f) == 0)
	{
		m_EvalTick = ServerTick();
		Collision()->MoverSpeed(m_Pos.x, m_Pos.y, &m_Core);
		m_Pos += m_Core;

		for(CReplayDraggerBeam *pDraggerBeam : m_apDraggerBeam)
		{
			if(pDraggerBeam != nullptr)
				pDraggerBeam->SetPos(m_Pos);
		}

		LookForPlayersToDrag();
	}
}

void CReplayDragger::LookForPlayersToDrag()
{
	CConverter::CPlayer *apPlayersInRange[MAX_CLIENTS];
	const int NumPlayersInRange = m_pConverter->FindCharacters(m_Pos,
		m_pConverter->Config().m_SvDraggerRange - CCharacterCore::PhysicalSize(), apPlayersInRange, MAX_CLIENTS);

	// The closest player (within range) in a team is selected as the target
	int aClosestTargetIdInTeam[MAX_CLIENTS];
	bool aCanStillBeTeamTarget[MAX_CLIENTS];
	bool aIsTarget[MAX_CLIENTS];
	int aMinDistInTeam[MAX_CLIENTS];
	std::fill(std::begin(aCanStillBeTeamTarget), std::end(aCanStillBeTeamTarget), false);
	std::fill(std::begin(aMinDistInTeam), std::end(aMinDistInTeam), 0);
	std::fill(std::begin(aIsTarget), std::end(aIsTarget), false);
	std::fill(std::begin(aClosestTargetIdInTeam), std::end(aClosestTargetIdInTeam), -1);

	for(int i = 0; i < NumPlayersInRange; i++)
	{
		CConverter::CPlayer *pTarget = apPlayersInRange[i];
		const int TargetClientId = m_pConverter->ClientId(pTarget);
		const int TargetTeam = m_pConverter->Team(TargetClientId);

		if(TargetTeam == TEAM_SUPER)
			continue;
		if(m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[TargetTeam])
			continue;

		const bool IsReachable =
			m_IgnoreWalls ?
				!Collision()->IntersectNoLaserNoWalls(m_Pos, m_pConverter->CharPos(pTarget), nullptr, nullptr) :
				!Collision()->IntersectNoLaser(m_Pos, m_pConverter->CharPos(pTarget), nullptr, nullptr);
		if(!IsReachable)
			continue;

		// Solo players are dragged independently from the rest of the team
		if(m_pConverter->GetSolo(TargetClientId))
		{
			aIsTarget[TargetClientId] = true;
		}
		else
		{
			const int Distance = distance(m_pConverter->CharPos(pTarget), m_Pos);
			if(aMinDistInTeam[TargetTeam] == 0 || aMinDistInTeam[TargetTeam] > Distance)
			{
				aMinDistInTeam[TargetTeam] = Distance;
				aClosestTargetIdInTeam[TargetTeam] = TargetClientId;
			}
			aCanStillBeTeamTarget[TargetClientId] = true;
		}
	}

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if((m_aTargetIdInTeam[i] != -1 && !aCanStillBeTeamTarget[m_aTargetIdInTeam[i]]) || m_aTargetIdInTeam[i] == -1)
		{
			m_aTargetIdInTeam[i] = aClosestTargetIdInTeam[i];
		}
		if(m_aTargetIdInTeam[i] != -1)
		{
			aIsTarget[m_aTargetIdInTeam[i]] = true;
		}
	}

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(aIsTarget[i] && m_apDraggerBeam[i] == nullptr)
		{
			m_apDraggerBeam[i] = new CReplayDraggerBeam(m_pConverter, this, m_Pos, m_Strength, m_IgnoreWalls, i, m_Layer, m_Number);
			// A new entity is not reached by the tick that created it, so the
			// beam is ticked by hand to keep the old game logic
			m_apDraggerBeam[i]->Tick();
		}
		else if(!aIsTarget[i] && m_apDraggerBeam[i] != nullptr)
		{
			m_apDraggerBeam[i]->Deactivate();
		}
	}
}

std::optional<int> CReplayDragger::DraggerBeamUsingDraggerId(int SnappingClientId)
{
	// At most one dragger beam uses the dragger id for a given snapping
	// client, in which case the dragger itself must not be snapped
	if(m_pConverter->GetPlayerChar(SnappingClientId) == nullptr)
		return std::nullopt;

	const int SnapTeam = m_pConverter->Team(SnappingClientId);
	if(SnapTeam >= MAX_CLIENTS)
		return std::nullopt;

	const int TargetClientId = m_pConverter->GetSolo(SnappingClientId) || m_aTargetIdInTeam[SnapTeam] < 0 ?
					   SnappingClientId :
					   m_aTargetIdInTeam[SnapTeam];
	if(m_apDraggerBeam[TargetClientId] == nullptr)
		return std::nullopt;

	if(m_pConverter->GetPlayerChar(TargetClientId) == nullptr || m_pConverter->Team(TargetClientId) != SnapTeam)
		return std::nullopt;

	return TargetClientId;
}

bool CReplayDragger::WillDraggerBeamUseDraggerId(int TargetClientId, int SnappingClientId)
{
	return DraggerBeamUsingDraggerId(SnappingClientId) == TargetClientId;
}

void CReplayDragger::Snap()
{
	if(m_pConverter->NetworkClipped(m_Pos))
		return;

	// Send the dragger in its resting position if the replay does not
	// otherwise see a dragger beam of its own team
	if(DraggerBeamUsingDraggerId(m_pConverter->SnapCid()).has_value())
		return;

	const int Subtype = (m_IgnoreWalls ? 1 : 0) | (std::clamp(round_to_int(m_Strength - 1.0f), 0, 2) << 1);
	m_pConverter->SnapLaserObject(m_Id, m_Pos, m_Pos, -1, -1, LASERTYPE_DRAGGER, Subtype, m_Number);
}

CReplayPlasma::CReplayPlasma(CConverter *pConverter, vec2 Pos, vec2 Dir, bool Freeze, bool Explosive, int ForClientId) :
	CReplayEntity(pConverter, LASER, Pos, LAYER_GAME, 0),
	m_Core(Dir),
	m_Freeze(Freeze),
	m_Explosive(Explosive),
	m_ForClientId(ForClientId),
	m_EvalTick(pConverter->ServerTick()),
	m_LifeTime((int)(TICK_SPEED * 1.5f))
{
	m_pConverter->InsertEntity(this);
}

void CReplayPlasma::Tick()
{
	if(m_LifeTime == 0)
	{
		Reset();
		return;
	}
	if(m_pConverter->GetPlayerChar(m_ForClientId) == nullptr)
	{
		Reset();
		return;
	}
	m_LifeTime--;
	Move();
	HitCharacter();
	// Plasma bullets may explode twice if they would hit both a player and an
	// obstacle in the next move step
	HitObstacle();
}

void CReplayPlasma::Move()
{
	static const float PLASMA_ACCEL = 1.1f;
	m_Pos += m_Core;
	m_Core *= PLASMA_ACCEL;
}

bool CReplayPlasma::HitCharacter()
{
	vec2 IntersectPos;
	CConverter::CPlayer *pHitPlayer = m_pConverter->IntersectCharacter(m_Pos, m_Pos + m_Core, 0.0f, IntersectPos, nullptr, m_ForClientId);
	if(!pHitPlayer)
		return false;
	if(m_pConverter->Team(m_pConverter->ClientId(pHitPlayer)) == TEAM_SUPER)
		return false;

	if(m_Freeze)
		m_pConverter->Freeze(pHitPlayer);
	else
		m_pConverter->Unfreeze(pHitPlayer);
	if(m_Explosive)
	{
		// Plasma turrets are very precise weapons, only one tee gets speed
		// from them
		m_pConverter->CreateExplosion(m_Pos, m_ForClientId, WEAPON_GRENADE, true, m_pConverter->Team(m_ForClientId));
	}
	Reset();
	return true;
}

bool CReplayPlasma::HitObstacle()
{
	if(Collision()->IntersectNoLaser(m_Pos, m_Pos + m_Core, nullptr, nullptr))
	{
		if(m_Explosive)
			m_pConverter->CreateExplosion(m_Pos, m_ForClientId, WEAPON_GRENADE, true, m_pConverter->Team(m_ForClientId));
		Reset();
		return true;
	}
	return false;
}

void CReplayPlasma::Snap()
{
	if(!m_pConverter->CanSnapCharacter(m_ForClientId) || m_pConverter->NetworkClipped(m_Pos))
		return;
	const int Subtype = (m_Explosive ? 1 : 0) | (m_Freeze ? 2 : 0);
	m_pConverter->SnapLaserObject(m_Id, m_Pos, m_Pos, m_EvalTick, m_ForClientId, LASERTYPE_PLASMA, Subtype, m_Number);
}

CReplayGun::CReplayGun(CConverter *pConverter, vec2 Pos, bool Freeze, bool Explosive, int Layer, int Number) :
	CReplayEntity(pConverter, LASER, Pos, Layer, Number),
	m_Freeze(Freeze),
	m_Explosive(Explosive),
	m_EvalTick(pConverter->ServerTick())
{
	m_pConverter->InsertEntity(this);
}

void CReplayGun::Tick()
{
	if(ServerTick() % (int)(TICK_SPEED * 0.15f) == 0)
	{
		m_EvalTick = ServerTick();
		Collision()->MoverSpeed(m_Pos.x, m_Pos.y, &m_Core);
		m_Pos += m_Core;
	}
	if(m_pConverter->Config().m_SvPlasmaPerSec > 0)
		Fire();
}

void CReplayGun::Fire()
{
	if(!m_pConverter->CharactersNear(m_Pos, m_Pos, m_pConverter->Config().m_SvPlasmaRange))
		return;

	CConverter::CPlayer *apPlayersInRange[MAX_CLIENTS];
	const int NumPlayersInRange = m_pConverter->FindCharacters(m_Pos, m_pConverter->Config().m_SvPlasmaRange, apPlayersInRange, MAX_CLIENTS);

	// The closest player (within range) in a team is selected as the target
	int aTargetIdInTeam[MAX_CLIENTS];
	bool aIsTarget[MAX_CLIENTS];
	int aMinDistInTeam[MAX_CLIENTS];
	std::fill(std::begin(aMinDistInTeam), std::end(aMinDistInTeam), 0);
	std::fill(std::begin(aIsTarget), std::end(aIsTarget), false);
	std::fill(std::begin(aTargetIdInTeam), std::end(aTargetIdInTeam), -1);

	for(int i = 0; i < NumPlayersInRange; i++)
	{
		CConverter::CPlayer *pTarget = apPlayersInRange[i];
		const int TargetClientId = m_pConverter->ClientId(pTarget);
		const int TargetTeam = m_pConverter->Team(TargetClientId);
		if(TargetTeam == TEAM_SUPER)
			continue;
		if(m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[TargetTeam])
			continue;

		// Turrets can only shoot at a speed of sv_plasma_per_sec
		const bool TargetIsSolo = m_pConverter->GetSolo(TargetClientId);
		if((TargetIsSolo &&
			   m_aLastFireSolo[TargetClientId] + SERVER_TICK_SPEED / m_pConverter->Config().m_SvPlasmaPerSec > ServerTick()) ||
			(!TargetIsSolo &&
				m_aLastFireTeam[TargetTeam] + SERVER_TICK_SPEED / m_pConverter->Config().m_SvPlasmaPerSec > ServerTick()))
		{
			continue;
		}

		if(Collision()->IntersectLine(m_Pos, m_pConverter->CharPos(pTarget), nullptr, nullptr))
			continue;

		// Turrets fire on solo players regardless of the rest of the team
		if(TargetIsSolo)
		{
			aIsTarget[TargetClientId] = true;
			m_aLastFireSolo[TargetClientId] = ServerTick();
		}
		else
		{
			const int Distance = distance(m_pConverter->CharPos(pTarget), m_Pos);
			if(aMinDistInTeam[TargetTeam] == 0 || aMinDistInTeam[TargetTeam] > Distance)
			{
				aMinDistInTeam[TargetTeam] = Distance;
				aTargetIdInTeam[TargetTeam] = TargetClientId;
			}
		}
	}

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(aTargetIdInTeam[i] != -1)
		{
			aIsTarget[aTargetIdInTeam[i]] = true;
			m_aLastFireTeam[i] = ServerTick();
		}
	}

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(aIsTarget[i])
		{
			CConverter::CPlayer *pTarget = m_pConverter->GetPlayerChar(i);
			new CReplayPlasma(m_pConverter, m_Pos, normalize(m_pConverter->CharPos(pTarget) - m_Pos), m_Freeze, m_Explosive, i);
		}
	}
}

void CReplayGun::Snap()
{
	if(m_pConverter->NetworkClipped(m_Pos))
		return;
	const int Subtype = (m_Explosive ? 1 : 0) | (m_Freeze ? 2 : 0);
	m_pConverter->SnapLaserObject(m_Id, m_Pos, m_Pos, -1, -1, LASERTYPE_GUN, Subtype, m_Number);
}

CReplayProjectile::CReplayProjectile(CConverter *pConverter, int Type, int Owner, vec2 Pos, vec2 Dir, int Span,
	bool Freeze, bool Explosive, int SoundImpact, vec2 InitDir, int Layer, int Number) :
	CReplayEntity(pConverter, PROJECTILE, Pos, Layer, Number),
	m_Direction(Dir),
	m_InitDir(InitDir),
	m_LifeSpan(Span),
	m_Owner(Owner),
	m_Type(Type),
	m_SoundImpact(SoundImpact),
	m_StartTick(pConverter->ServerTick()),
	m_Explosive(Explosive),
	m_Freeze(Freeze)
{
	m_TuneZone = Collision()->IsTune(Collision()->GetMapIndex(m_Pos));
	CConverter::CPlayer *pOwnerChar = m_pConverter->GetPlayerChar(m_Owner);
	m_DdraceTeam = m_Owner == -1 ? 0 : m_pConverter->Team(m_Owner);
	m_IsSolo = pOwnerChar != nullptr && pOwnerChar->m_Core.m_Solo;
	m_pConverter->InsertEntity(this);
}

vec2 CReplayProjectile::GetPos(float Time) const
{
	float Curvature = 0.0f;
	float Speed = 0.0f;
	const CTuningParams &TuningParams = Tuning(m_TuneZone);

	switch(m_Type)
	{
	case WEAPON_GRENADE:
		Curvature = TuningParams.m_GrenadeCurvature;
		Speed = TuningParams.m_GrenadeSpeed;
		break;
	case WEAPON_SHOTGUN:
		Curvature = TuningParams.m_ShotgunCurvature;
		Speed = TuningParams.m_ShotgunSpeed;
		break;
	case WEAPON_GUN:
		Curvature = TuningParams.m_GunCurvature;
		Speed = TuningParams.m_GunSpeed;
		break;
	}

	return CalcPos(m_Pos, m_Direction, Curvature, Speed, Time);
}

void CReplayProjectile::Tick()
{
	const float Pt = (ServerTick() - m_StartTick - 1) / (float)SERVER_TICK_SPEED;
	const float Ct = (ServerTick() - m_StartTick) / (float)SERVER_TICK_SPEED;
	const vec2 PrevPos = GetPos(Pt);
	const vec2 CurPos = GetPos(Ct);
	vec2 ColPos;
	vec2 NewPos;
	const int Collide = Collision()->IntersectLine(PrevPos, CurPos, &ColPos, &NewPos);

	CConverter::CPlayer *pOwnerChar = m_Owner >= 0 ? m_pConverter->GetPlayerChar(m_Owner) : nullptr;
	CConverter::CPlayer *pTargetChr = nullptr;
	if(pOwnerChar ? !m_pConverter->HitDisabled(pOwnerChar, WEAPON_GRENADE) : m_pConverter->Config().m_SvHit != 0)
		pTargetChr = m_pConverter->IntersectCharacter(PrevPos, ColPos, m_Freeze ? 1.0f : 6.0f, ColPos, pOwnerChar, m_Owner);

	if(m_LifeSpan > -1)
		m_LifeSpan--;

	bool IsWeaponCollide = false;
	if(pOwnerChar && pTargetChr && !m_pConverter->CanCollide(m_pConverter->ClientId(pTargetChr), m_Owner))
		IsWeaponCollide = true;
	if(!pOwnerChar && m_Owner >= 0 && (m_Type != WEAPON_GRENADE || m_pConverter->Config().m_SvDestroyBulletsOnDeath))
	{
		Reset();
		return;
	}

	if(((pTargetChr && (pOwnerChar ? !m_pConverter->HitDisabled(pOwnerChar, WEAPON_GRENADE) : m_pConverter->Config().m_SvHit != 0 || m_Owner == -1 || pTargetChr == pOwnerChar)) || Collide) && !IsWeaponCollide)
	{
		if(m_Explosive && (!pTargetChr || (!m_Freeze || (m_Type == WEAPON_SHOTGUN && Collide))))
		{
			m_pConverter->CreateExplosion(ColPos, m_Owner, m_Type, m_Owner == -1,
				!pTargetChr ? -1 : m_pConverter->Team(m_pConverter->ClientId(pTargetChr)));
			m_pConverter->CreateSound(ColPos, m_SoundImpact, m_Owner);
		}
		else if(m_Freeze)
		{
			CConverter::CPlayer *apChars[MAX_CLIENTS];
			const int Num = m_pConverter->FindCharacters(CurPos, 1.0f, apChars, MAX_CLIENTS);
			for(int i = 0; i < Num; i++)
			{
				if(m_Layer != LAYER_SWITCH || (m_Number > 0 && Switchers()[m_Number].m_aStatus[m_pConverter->Team(m_pConverter->ClientId(apChars[i]))]))
					m_pConverter->Freeze(apChars[i]);
			}
		}

		if(Collide && m_Bouncing != 0)
		{
			m_StartTick = ServerTick();
			m_Pos = NewPos + (-(m_Direction * 4));
			if(m_Bouncing == 1)
				m_Direction.x = -m_Direction.x;
			else if(m_Bouncing == 2)
				m_Direction.y = -m_Direction.y;
			if(absolute(m_Direction.x) < 1e-6f)
				m_Direction.x = 0;
			if(absolute(m_Direction.y) < 1e-6f)
				m_Direction.y = 0;
			m_Pos += m_Direction;
		}
		else if(m_Type == WEAPON_GUN)
		{
			m_pConverter->CreateDamageInd(CurPos, -std::atan2(m_Direction.x, m_Direction.y), 10, m_Owner);
			Reset();
			return;
		}
		else if(!m_Freeze)
		{
			Reset();
			return;
		}
	}

	if(m_LifeSpan == -1)
	{
		if(m_Explosive)
		{
			m_pConverter->CreateExplosion(ColPos, m_Owner, m_Type, m_Owner == -1,
				!pOwnerChar ? -1 : m_pConverter->Team(m_Owner));
			m_pConverter->CreateSound(ColPos, m_SoundImpact, m_Owner);
		}
		Reset();
		return;
	}

	// Teleporting weapons. The server picks a random exit of the teleporter
	// with its own prng, whose state cannot be reconstructed from the
	// recording, so the first exit is used.
	const int Index = Collision()->GetIndex(PrevPos, CurPos);
	const int Teleporter = m_pConverter->Config().m_SvOldTeleportWeapons ? Collision()->IsTeleport(Index) : Collision()->IsTeleportWeapon(Index);
	if(Teleporter && !Collision()->TeleOuts(Teleporter - 1).empty())
	{
		m_Pos = Collision()->TeleOuts(Teleporter - 1)[0];
		m_StartTick = ServerTick();
	}
}

void CReplayProjectile::Snap()
{
	const float Ct = (ServerTick() - m_StartTick) / (float)SERVER_TICK_SPEED;
	if(m_pConverter->NetworkClipped(GetPos(Ct)))
		return;
	if(m_Owner >= 0 && !m_pConverter->CanSnapCharacter(m_Owner))
		return;

	int Flags = 0;
	if(m_Bouncing & 1)
		Flags |= PROJECTILEFLAG_BOUNCE_HORIZONTAL;
	if(m_Bouncing & 2)
		Flags |= PROJECTILEFLAG_BOUNCE_VERTICAL;
	if(m_Explosive)
		Flags |= PROJECTILEFLAG_EXPLOSIVE;
	if(m_Freeze)
		Flags |= PROJECTILEFLAG_FREEZE;

	vec2 Vel = m_Direction * 1e6f;
	if(m_Owner >= 0)
	{
		Vel = m_InitDir;
		Flags |= PROJECTILEFLAG_NORMALIZE_VEL;
	}
	m_pConverter->SnapProjectileObject(m_Id, m_Pos, Vel, m_Type, m_StartTick, m_Owner, m_Number, m_TuneZone, Flags);
}

CReplayLaser::CReplayLaser(CConverter *pConverter, vec2 Pos, vec2 Direction, float StartEnergy, int Owner, int Type) :
	CReplayEntity(pConverter, LASER, Pos, LAYER_GAME, 0),
	m_From(Pos),
	m_Dir(Direction),
	m_PrevPos(Pos),
	m_Energy(StartEnergy),
	m_Owner(Owner),
	m_Type(Type)
{
	m_TuneZone = Collision()->IsTune(Collision()->GetMapIndex(m_Pos));
	m_pConverter->InsertEntity(this);
	DoBounce();
}

bool CReplayLaser::HitCharacter(vec2 From, vec2 To)
{
	static const vec2 StackedLaserShotgunBugSpeed = vec2(-2147483648.0f, -2147483648.0f);
	vec2 At;
	CConverter::CPlayer *pOwnerChar = m_pConverter->GetPlayerChar(m_Owner);
	const bool DontHitSelf = m_pConverter->Config().m_SvOldLaser || (m_Bounces == 0 && !m_WasTele);

	CConverter::CPlayer *pHit;
	if(pOwnerChar ? !m_pConverter->HitDisabled(pOwnerChar, m_Type) : m_pConverter->Config().m_SvHit != 0)
		pHit = m_pConverter->IntersectCharacter(m_Pos, To, 0.0f, At, DontHitSelf ? pOwnerChar : nullptr, m_Owner);
	else
		pHit = m_pConverter->IntersectCharacter(m_Pos, To, 0.0f, At, DontHitSelf ? pOwnerChar : nullptr, m_Owner, pOwnerChar);

	if(!pHit)
		return false;

	m_From = From;
	m_Pos = At;
	m_Energy = -1;
	if(m_Type == WEAPON_SHOTGUN)
	{
		const float Strength = Tuning(m_TuneZone).m_ShotgunStrength;
		const vec2 &HitPos = m_pConverter->CharPos(pHit);
		if(!m_pConverter->Config().m_SvOldLaser)
		{
			if(m_PrevPos != HitPos)
				m_pConverter->AddVelocity(pHit, normalize(m_PrevPos - HitPos) * Strength);
			else
				m_pConverter->SetRawVelocity(pHit, StackedLaserShotgunBugSpeed);
		}
		else if(pOwnerChar)
		{
			if(m_pConverter->CharPos(pOwnerChar) != HitPos)
				m_pConverter->AddVelocity(pHit, normalize(m_pConverter->CharPos(pOwnerChar) - HitPos) * Strength);
			else
				m_pConverter->SetRawVelocity(pHit, StackedLaserShotgunBugSpeed);
		}
	}
	else if(m_Type == WEAPON_LASER)
	{
		m_pConverter->Unfreeze(pHit);
	}
	return true;
}

void CReplayLaser::DoBounce()
{
	m_EvalTick = ServerTick();

	if(m_Energy < 0)
	{
		Reset();
		return;
	}
	m_PrevPos = m_Pos;
	vec2 Coltile;

	if(m_WasTele)
	{
		m_PrevPos = m_TelePos;
		m_Pos = m_TelePos;
		m_TelePos = vec2(0.0f, 0.0f);
	}

	vec2 To = m_Pos + m_Dir * m_Energy;
	int TeleNr = 0;
	const int Res = Collision()->IntersectLineTeleWeapon(m_Pos, To, &Coltile, &To, &TeleNr);

	if(Res)
	{
		if(!HitCharacter(m_Pos, To))
		{
			// intersected
			m_From = m_Pos;
			m_Pos = To;

			vec2 TempPos = m_Pos;
			vec2 TempDir = m_Dir * 4.0f;

			int Tile = 0;
			if(Res == -1)
			{
				Tile = Collision()->GetTile(round_to_int(Coltile.x), round_to_int(Coltile.y));
				Collision()->SetCollisionAt(round_to_int(Coltile.x), round_to_int(Coltile.y), TILE_SOLID);
			}
			Collision()->MovePoint(&TempPos, &TempDir, 1.0f, nullptr);
			if(Res == -1)
			{
				Collision()->SetCollisionAt(round_to_int(Coltile.x), round_to_int(Coltile.y), Tile);
			}
			m_Pos = TempPos;
			m_Dir = normalize(TempDir);

			const float Distance = distance(m_From, m_Pos);
			// Prevent infinite bounces
			if(Distance == 0.0f && m_ZeroEnergyBounceInLastTick)
				m_Energy = -1;
			else
				m_Energy -= Distance + Tuning(m_TuneZone).m_LaserBounceCost;
			m_ZeroEnergyBounceInLastTick = Distance == 0.0f;

			// The server picks a random exit of the teleporter with its own
			// prng, whose state cannot be reconstructed, so the first one is
			// used
			if(Res == TILE_TELEINWEAPON && !Collision()->TeleOuts(TeleNr - 1).empty())
			{
				m_TelePos = Collision()->TeleOuts(TeleNr - 1)[0];
				m_WasTele = true;
			}
			else
			{
				m_Bounces++;
				m_WasTele = false;
			}

			if(m_Bounces > Tuning(m_TuneZone).m_LaserBounceNum)
				m_Energy = -1;

			m_pConverter->CreateSound(m_Pos, SOUND_LASER_BOUNCE, m_Owner);
		}
	}
	else if(!HitCharacter(m_Pos, To))
	{
		m_From = m_Pos;
		m_Pos = To;
		m_Energy = -1;
	}
}

void CReplayLaser::Tick()
{
	const float Delay = Tuning(m_TuneZone).m_LaserBounceDelay;
	if((ServerTick() - m_EvalTick) > (TICK_SPEED * Delay / 1000.0f))
		DoBounce();
}

void CReplayLaser::Snap()
{
	if(m_pConverter->NetworkClipped(m_Pos) && m_pConverter->NetworkClipped(m_From))
		return;
	if(m_Owner >= 0 && !m_pConverter->CanSnapCharacter(m_Owner))
		return;
	const int LaserType = m_Type == WEAPON_LASER ? LASERTYPE_RIFLE : (m_Type == WEAPON_SHOTGUN ? LASERTYPE_SHOTGUN : -1);
	m_pConverter->SnapLaserObject(m_Id, m_Pos, m_From, m_EvalTick, m_Owner, LaserType, 0, m_Number);
}

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

	std::unique_ptr<CInputSource> m_pSource;
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
			const unsigned Read = m_pSource->Read(m_vWindow.data() + m_Fill, m_vWindow.size() - m_Fill);
			if(Read == 0)
			{
				m_EndOfFile = true;
				break;
			}
			m_Fill += Read;
		}
	}

public:
	// Opens the input (a file path or, in the web build, an http(s) URL) and
	// validates the magic bytes. Returns the parsed json header on success,
	// which the caller frees.
	json_value *Open(const char *pPath)
	{
		if(str_startswith(pPath, "http://") != nullptr || str_startswith(pPath, "https://") != nullptr)
		{
#if defined(CONF_PLATFORM_EMSCRIPTEN)
			auto pHttpSource = std::make_unique<CHttpInputSource>();
			if(!pHttpSource->Open(pPath))
			{
				return nullptr;
			}
			m_pSource = std::move(pHttpSource);
#else
			log_error(TOOL_NAME, "Streaming from a URL is only supported in the web build");
			return nullptr;
#endif
		}
		else
		{
			IOHANDLE InputFile = io_open(pPath, IOFLAG_READ);
			if(!InputFile)
			{
				log_error(TOOL_NAME, "Failed to open '%s'", pPath);
				return nullptr;
			}
			m_pSource = std::make_unique<CFileInputSource>(InputFile);
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
		m_pSource = nullptr;
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

	// Optional per-tick dataset output, extracted from anywhere on the
	// command line before the positional arguments are parsed. Everything
	// after --rank is a player name, and a player can be called --dataset.
	int Limit = argc - 1;
	for(int i = 1; i < argc; i++)
	{
		if(str_comp(argv[i], "--rank") == 0)
		{
			Limit = i;
			break;
		}
	}
	const char *pDatasetPath = nullptr;
	for(int i = 1; i < Limit; i++)
	{
		if(str_comp(argv[i], "--dataset") == 0)
		{
			pDatasetPath = argv[i + 1];
			for(int j = i; j + 2 < argc; j++)
			{
				argv[j] = argv[j + 2];
			}
			argc -= 2;
			break;
		}
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
	int DebugStartSeconds = -1;
	int DebugEndSeconds = -1;
	if(ArgIndex + 2 < argc && str_comp(argv[ArgIndex], "--debug") == 0)
	{
		DebugStartSeconds = ParseTimeSeconds(argv[ArgIndex + 1]);
		DebugEndSeconds = ParseTimeSeconds(argv[ArgIndex + 2]);
		ArgIndex += 3;
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
		log_error(TOOL_NAME, "--dataset <out.jsonl> additionally writes per-tick state and input rows");
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
	std::vector<int> vRunCids;
	if(RankMode)
	{
		// First pass: find the rank's finish event. Without simulation and
		// demo output this only parses the stream. Allow for the server tick
		// falling behind wall-clock time during long sessions.
		constexpr int SCAN_SLACK_TICKS = 30 * 60 * SERVER_TICK_SPEED;
		CConverter Scanner(pStorage.get(), pSnapshotDelta.get());
		NameScanner.CopyPlayerIdentitiesTo(&Scanner);
		Scanner.ScanForRank(&vRankNames, RankTimeTicks, RankExpectedTick,
			RankExpectedTick < 0 ? std::numeric_limits<int>::max() : RankExpectedTick + SCAN_SLACK_TICKS);
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
			vRunCids = Scanner.ApproxRunCids();
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
	Converter.ApplyConfig(json_object_get(pHeader, "config"));
	// The server builds its entities once the settings are in, and the map's
	// own tiles override some of them again
	Converter.CreateAllEntities();
	const json_value &VersionMinor = (*pHeader)["version_minor"];
	const int MinorVersion = VersionMinor.type == json_string ? str_toint(VersionMinor) : 0;
	const bool InputAppliedNextTick = MinorVersion < 24;
	Converter.SetInputAppliedNextTick(InputAppliedNextTick);
	log_info(TOOL_NAME, "teehistorian version_minor=%d, inputs apply %s", MinorVersion, InputAppliedNextTick ? "one tick after the section they are stored in" : "in their own section");

	Converter.SetTickRange(DemoStartTick, DemoEndTick);
	if(DebugStartSeconds >= 0)
	{
		Converter.SetDebugRange(DebugStartSeconds * SERVER_TICK_SPEED, DebugEndSeconds * SERVER_TICK_SPEED);
	}
	if(RankMode)
	{
		if(RankTarget.m_Team == TEAM_FLOCK && vRunCids.size() >= 2)
			Converter.SetRunCids(vRunCids);
		else
			Converter.SetTeamFilter(RankTarget.m_Team);
		Converter.SetSnapCid(RankTarget.m_Cid);
		Converter.SetRankMarkers(RankTarget.m_FinishTick - RankTimeTicks, RankTarget.m_FinishTick);
	}

	if(!Converter.StartDemo(argv[3], aMapName))
	{
		json_value_free(pHeader);
		return -1;
	}

	IOHANDLE DatasetFile = nullptr;
	if(pDatasetPath != nullptr)
	{
		DatasetFile = io_open(pDatasetPath, IOFLAG_WRITE);
		if(!DatasetFile)
		{
			log_error(TOOL_NAME, "Failed to open dataset output '%s'", pDatasetPath);
			json_value_free(pHeader);
			return -1;
		}
		Converter.SetDatasetOutput(DatasetFile);
	}

	Reader.ParseChunks(&Converter);

	const bool Success = Converter.Finish();
	if(DatasetFile != nullptr)
	{
		io_close(DatasetFile);
	}

	json_value_free(pHeader);
	if(Success && RankMode)
	{
		// Machine-readable result for the archive server
		if(Converter.ErrCount() > 0)
			log_info(TOOL_NAME, "sim-vs-recording: mean %.3f px over %d ticks, max %.1f px, %d ticks over 4 px",
				Converter.ErrMean(), Converter.ErrCount(), Converter.ErrMax(), Converter.ErrOver4());
		// A solo rank belongs to one player. Its demo keeps the rest of team
		// 0 around it, and those are not who the rank is for.
		const std::vector<int> vSolo = {RankTarget.m_Cid};
		char aFinishCids[256] = "";
		for(const int Cid : vRankNames.size() == 1 ? vSolo : Converter.FinishCids())
		{
			char aOne[16];
			str_format(aOne, sizeof(aOne), "%s%d", aFinishCids[0] == '\0' ? "" : ",", Cid);
			str_append(aFinishCids, aOne);
		}
		printf("{\"cid\":%d,\"team\":%d,\"finish_cids\":[%s],\"demo_start_tick\":%d,\"run_start_tick\":%d,\"finish_tick\":%d,\"dataset_rows\":%d}\n",
			RankTarget.m_Cid, RankTarget.m_Team, aFinishCids, Converter.FirstTick() >= 0 ? Converter.FirstTick() : DemoStartTick,
			RankTarget.m_FinishTick - RankTimeTicks, RankTarget.m_FinishTick, Converter.NumDatasetRows());
	}
	return Success ? 0 : -1;
}
