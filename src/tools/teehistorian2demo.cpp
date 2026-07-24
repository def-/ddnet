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
#include <game/teamscore.h>
#include <game/version.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#if defined(CONF_PLATFORM_EMSCRIPTEN)
#include <emscripten/emscripten.h>
#endif

static const char *TOOL_NAME = "teehistorian2demo";

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

		CCharacterCore m_Core;
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

	// stats
	int m_NumChunks = 0;
	int m_NumTicks = 0;
	int m_NumSnapshots = 0;
	int m_NumChatMessages = 0;
	int m_MaxPlayersSeen = 0;

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
			pPlayer->m_Core.Init(&m_WorldCore, &m_Collision, &m_TeamsCore);
			pPlayer->m_Core.Reset();
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
			m_aPlayers[Cid] = CPlayer();
			m_aPlayers[Cid].m_Connected = true;
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
		log_info(TOOL_NAME, "Wrote %d snapshots covering %d ticks (%d:%02d min), %d chat messages, %d players seen",
			m_NumSnapshots, m_NumTicks,
			m_NumTicks / SERVER_TICK_SPEED / 60, m_NumTicks / SERVER_TICK_SPEED % 60,
			m_NumChatMessages, m_MaxPlayersSeen);
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
				m_aPlayers[Cid].m_Score = -TimeTicks / SERVER_TICK_SPEED;
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
			break;
		}
		default:
			break;
		}
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
				Player.m_Core.m_Input = Player.m_SimInput;
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
	}

	void FlushTick()
	{
		if(!m_TickDirty)
			return;

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
			for(const auto &vMessage : m_vTickMessages)
			{
				m_Recorder.RecordMessage(vMessage.data(), vMessage.size());
			}
		}
		m_vTickMessages.clear();

		int NumPlayers = 0;
		for(auto &Player : m_aPlayers)
		{
			Player.m_SimInput = Player.m_Input;
			if(Player.m_Alive)
			{
				NumPlayers++;
				if(Player.m_Input.m_WantedWeapon > 0)
					Player.m_Weapon = std::clamp(Player.m_Input.m_WantedWeapon - 1, (int)WEAPON_HAMMER, (int)WEAPON_NINJA);
				if(Player.m_Input.m_Fire != Player.m_LastFire)
				{
					if(Player.m_Input.m_Fire & 1)
						Player.m_AttackTick = m_Tick;
					Player.m_LastFire = Player.m_Input.m_Fire;
				}
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
			if(!pPlayer->m_Connected)
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
					pCharacter->m_Emote = EMOTE_NORMAL;
					pCharacter->m_AttackTick = pPlayer->m_AttackTick;
				}
			}
		}

		CSnapshotBuffer Buffer;
		const int SnapshotSize = Builder.Finish(&Buffer);
		m_Recorder.RecordSnapshot(m_Tick, Buffer.AsSnapshot(), SnapshotSize);
		m_NumSnapshots++;
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

	if(argc < 4 || argc > 6)
	{
		log_error(TOOL_NAME, "Usage: %s <input.teehistorian> <map.map> <output.demo> [start] [end]", TOOL_NAME);
		log_error(TOOL_NAME, "start/end limit the converted time range, given as seconds, M:SS or H:MM:SS");
		return -1;
	}
	int StartSeconds = 0;
	int EndSeconds = -1;
	if(argc >= 5)
	{
		StartSeconds = ParseTimeSeconds(argv[4]);
		if(StartSeconds < 0)
		{
			log_error(TOOL_NAME, "Invalid start time '%s'", argv[4]);
			return -1;
		}
	}
	if(argc >= 6)
	{
		EndSeconds = ParseTimeSeconds(argv[5]);
		if(EndSeconds < 0 || EndSeconds <= StartSeconds)
		{
			log_error(TOOL_NAME, "Invalid end time '%s'", argv[5]);
			return -1;
		}
	}

	std::unique_ptr<CInputSource> pSource;
	if(str_startswith(argv[1], "http://") != nullptr || str_startswith(argv[1], "https://") != nullptr)
	{
#if defined(CONF_PLATFORM_EMSCRIPTEN)
		auto pHttpSource = std::make_unique<CHttpInputSource>();
		if(!pHttpSource->Open(argv[1]))
		{
			return -1;
		}
		pSource = std::move(pHttpSource);
#else
		log_error(TOOL_NAME, "Streaming from a URL is only supported in the web build");
		return -1;
#endif
	}
	else
	{
		IOHANDLE InputFile = io_open(argv[1], IOFLAG_READ);
		if(!InputFile)
		{
			log_error(TOOL_NAME, "Failed to open '%s'", argv[1]);
			return -1;
		}
		pSource = std::make_unique<CFileInputSource>(InputFile);
	}

	// Sliding window over the input. The input is only read sequentially,
	// teehistorian files can be larger than memory. Single chunks are limited
	// by the 64 KiB packer buffer on the writing side, so as long as the
	// window is refilled before it runs lower than that, chunks never appear
	// truncated before the actual end of the file.
	constexpr size_t WINDOW_SIZE = 16 * 1024 * 1024;
	constexpr size_t REFILL_THRESHOLD = 256 * 1024;
	std::vector<unsigned char> vWindow(WINDOW_SIZE);
	size_t Fill = 0;
	size_t Pos = 0;
	bool EndOfFile = false;

	const auto &&Refill = [&]() {
		memmove(vWindow.data(), vWindow.data() + Pos, Fill - Pos);
		Fill -= Pos;
		Pos = 0;
		while(Fill < vWindow.size())
		{
			const unsigned Read = pSource->Read(vWindow.data() + Fill, vWindow.size() - Fill);
			if(Read == 0)
			{
				EndOfFile = true;
				break;
			}
			Fill += Read;
		}
	};
	Refill();

	// Magic bytes, then the json header terminated by a null byte, both must
	// fit into the initial window.
	static const CUuid TEEHISTORIAN_UUID = CalculateUuid("teehistorian@ddnet.tw");
	if(Fill < sizeof(CUuid) + 1 || mem_comp(vWindow.data(), &TEEHISTORIAN_UUID, sizeof(CUuid)) != 0)
	{
		log_error(TOOL_NAME, "'%s' is not a teehistorian file", argv[1]);
		return -1;
	}
	const unsigned char *pHeaderEnd = (const unsigned char *)memchr(vWindow.data() + sizeof(CUuid), 0, Fill - sizeof(CUuid));
	if(pHeaderEnd == nullptr)
	{
		log_error(TOOL_NAME, "Missing or unterminated teehistorian header");
		return -1;
	}

	json_value *pHeader = JsonParse((const char *)vWindow.data() + sizeof(CUuid), pHeaderEnd - (vWindow.data() + sizeof(CUuid)));
	if(pHeader == nullptr)
	{
		log_error(TOOL_NAME, "Failed to parse teehistorian header");
		return -1;
	}
	Pos = pHeaderEnd - vWindow.data() + 1;

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

	std::unique_ptr<CSnapshotDelta> pSnapshotDelta = CreateSnapshotDelta();
	CConverter Converter(pStorage.get(), pSnapshotDelta.get());
	if(!Converter.LoadMap(argv[2], MapSha256.type == json_string ? (const char *)MapSha256 : nullptr))
	{
		json_value_free(pHeader);
		return -1;
	}

	Converter.ApplyTuning(json_object_get(pHeader, "tuning"));

	if(StartSeconds > 0 || EndSeconds >= 0)
	{
		Converter.SetTickRange(StartSeconds * SERVER_TICK_SPEED,
			EndSeconds < 0 ? std::numeric_limits<int>::max() : EndSeconds * SERVER_TICK_SPEED);
	}

	if(!Converter.StartDemo(argv[3], aMapName))
	{
		json_value_free(pHeader);
		return -1;
	}

	CUnpacker Unpacker;
	Unpacker.Reset(vWindow.data() + Pos, Fill - Pos);
	while(true)
	{
		if(!EndOfFile && Fill - Pos < REFILL_THRESHOLD)
		{
			Refill();
			Unpacker.Reset(vWindow.data(), Fill);
		}
		if(!Converter.ParseChunk(&Unpacker))
			break;
		Pos = Fill - Unpacker.RemainingSize();
	}
	pSource = nullptr;

	const bool Success = Converter.Finish();

	json_value_free(pHeader);
	return Success ? 0 : -1;
}
