// Shared in-process DDNet server fixture for the libFuzzer targets that need a real
// CServer / CGameContext (fz_gamemsg, fz_serverpkt).
//
// Construction follows src/test/gameworld_test.cpp, which is the project's own recipe for
// standing one up. Everything that deviates from it is load-bearing and commented.
//
// This is a header with definitions on purpose: each fuzz target is a single translation
// unit linked into its own binary, so there is nothing to collide with.
#ifndef FUZZ_FZ_SERVER_FIXTURE_H
#define FUZZ_FZ_SERVER_FIXTURE_H

#include <base/bytes.h>
#include <base/dbg.h>
#include <base/fs.h>
#include <base/io.h>
#include <base/logger.h>
#include <base/mem.h>
#include <base/net.h>
#include <base/str.h>

#include <engine/engine.h>
#include <engine/http.h>
#include <engine/kernel.h>
#include <engine/server/antibot.h>
#include <engine/server/databases/connection_pool.h>
#include <engine/server/server.h>
#include <engine/shared/config.h>
#include <engine/shared/network.h>
#include <engine/storage.h>

#include <game/prng.h>
#include <game/server/gamecontext.h>
#include <game/version.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>

// UBSan takes its options from the environment, so a suppression would have to be repeated
// by the campaign and by every manual replay, and a replay that forgets it reports a finding
// the campaign deliberately ignores. A default travels with the build instead. The
// environment still wins for every option it names, which is how run.sh sets halt_on_error.
extern "C" const char *__ubsan_default_options()
{
	return "suppressions=" FZ_UBSAN_SUPP;
}

namespace fzserver
{

	inline CServer *g_pServer = nullptr;
	inline CGameContext *g_pGameServer = nullptr;

	// Two slots the targets send as: one 0.6 client and one 0.7 client, so the translation
	// layer is exercised on alternating inputs.
	//
	// TWO, and raising it does not help - measured, not assumed. Two 900 s campaigns run side by
	// side from the same corpus, four slots against two, ended at 26874 edges against 27013 at
	// half the executions per second: every extra tee costs a character to simulate, a snapshot to
	// build and a server-info cache to fill on every single input, and buys nothing back.
	//
	// The reason is worth writing down, because the same idea will suggest itself again. The team
	// machinery is not gated on how many tees exist, it is gated on a client sending /team, /lock,
	// /practice and /save in that order, and idle tees do not make that sequence more likely.
	//
	// Which protocol each slot claims is decided per target, not here: the two grew opposite
	// conventions (fz_serverpkt's slot 0 is the 0.7 one, fz_gamemsg's slot 1 is), and changing
	// either would silently move every path an existing corpus had learned to reach.
	enum
	{
		CLIENT_SIX = 0,
		CLIENT_SEVEN = 1,
		NUM_FUZZ_CLIENTS = 2
	};

	// Route the server's logging to /dev/null. It logs several lines per message, which
	// otherwise dominates the runtime and buries libFuzzer's own output. Set FZ_SERVER_LOG=1
	// to get it back while triaging a finding - without it an assert message is invisible.
	inline void SetupLogging()
	{
		if(getenv("FZ_SERVER_LOG") != nullptr)
		{
			log_set_global_logger_default();
			return;
		}
		static std::unique_ptr<ILogger> s_pNullLogger = log_logger_file(io_open("/dev/null", IOFLAG_WRITE));
		log_set_global_logger(s_pNullLogger.get());
	}

	// Give a slot the address a client has.
	//
	// These targets accept no real connection, so CNetServer::ClientAddr hands back the slot's
	// zeroed peer address, whose type is NETTYPE_INVALID. Both consequences are bad: everything
	// keyed on the address collapses onto one key shared by every slot, and net_addr_str asserts
	// on the unknown type, so an ordinary `unmute` or a chat mute aborts the process as if the
	// server had a bug.
	//
	// A client slot with no connection is a state the server has its own name for, the debug
	// dummy, and CServer::ClientAddr returns m_DebugDummyAddr for one. Give it the address
	// UpdateDebugDummies would (server.cpp:3096-3111), except derived from the client id rather
	// than random: a random address would make an artifact that mutes a slot not reproduce.
	inline void SetClientAddr(int ClientId)
	{
		CServer::CClient &Client = g_pServer->m_aClients[ClientId];
		Client.m_DebugDummy = true;

		NETADDR &Addr = Client.m_DebugDummyAddr;
		mem_zero(&Addr, sizeof(Addr));
		// See https://en.wikipedia.org/wiki/Unique_local_address
		Addr.type = NETTYPE_IPV6;
		Addr.ip[0] = 0xfd;
		Addr.ip[6] = 0xc0;
		Addr.ip[7] = 0xde;
		uint_to_bytes_be(&Addr.ip[12], ClientId);
		Addr.port = 1024 + ClientId;
		net_addr_str(&Addr, Client.m_aDebugDummyAddrString.data(), Client.m_aDebugDummyAddrString.size(), true);
		net_addr_str(&Addr, Client.m_aDebugDummyAddrStringNoPort.data(), Client.m_aDebugDummyAddrStringNoPort.size(), false);
	}

	// The server's own accept-time initialiser, plus the address. Order matters: the callback is
	// what clears m_DebugDummy.
	inline void NewClient(int ClientId, bool Sixup)
	{
		CServer::NewClientCallback(ClientId, g_pServer, Sixup);
		SetClientAddr(ClientId);
	}

	// Rebuild the server-info caches, which CServer::Run does through UpdateServerInfo before it
	// ever serves a client.
	//
	// Not optional, and not merely for coverage: CServer::GetServerInfoSixup starts with
	// `m_aSixupServerInfoCache[SendClients].m_vCache.front()`, and OnNetMsgEnterGame calls it for
	// every 0.7 client. With the cache empty that is front() on an empty vector - a
	// _GLIBCXX_DEBUG abort inside the ENTERGAME handler that has nothing to do with the message
	// and would be reported as a crash. A real server cannot be in that state.
	//
	// The two loops are lifted verbatim from UpdateServerInfo (server.cpp:2835-2841).
	// UpdateServerInfo itself cannot be called: it opens with UpdateRegisterServerInfo, which
	// ends in `m_pRegister->OnNewInfo(...)`, and these targets never build a register
	// (CServer::Run does), so m_pRegister is null.
	//
	// The slots are debug dummies for their address and IncludedInServerInfo() excludes those, a
	// browser rule rather than a packing one, so clear the flag while the builders run. Otherwise
	// they never see a name, clan, country or score, which is the client-controlled half of this
	// surface and the only reason it is worth driving.
	inline void CacheServerInfos()
	{
		bool aWasDebugDummy[MAX_CLIENTS];
		for(int i = 0; i < MAX_CLIENTS; i++)
		{
			aWasDebugDummy[i] = g_pServer->m_aClients[i].m_DebugDummy;
			g_pServer->m_aClients[i].m_DebugDummy = false;
		}

		for(int Type = 0; Type < 3; Type++)
			for(int SendClients = 0; SendClients < 2; SendClients++)
				g_pServer->CacheServerInfo(&g_pServer->m_aServerInfoCache[Type * 2 + SendClients], Type, SendClients != 0);
		for(int SendClients = 0; SendClients < 2; SendClients++)
			g_pServer->CacheServerInfoSixup(&g_pServer->m_aSixupServerInfoCache[SendClients], SendClients != 0, MAX_CLIENTS);

		for(int i = 0; i < MAX_CLIENTS; i++)
			g_pServer->m_aClients[i].m_DebugDummy = aWasDebugDummy[i];
	}

	// Put back the state an artifact cannot carry, so a finding reproduces from its own input.
	//
	// A libFuzzer artifact is a single input, but this server is not stateless, and two pieces of
	// state used to survive from one input to the next: g_Config, which an input can rewrite
	// through rcon, and a client's rcon session, which lives on in a slot that stays INGAME. The
	// campaign found crashes behind both and then could not reproduce them from the artifact,
	// which is the whole value of finding them. Neither reset invents a state: it is a server on
	// its configured settings and a peer that has not authenticated yet.
	inline void ResetPerInput()
	{
		static const CConfig s_PristineConfig = g_Config;
		g_Config = s_PristineConfig;

		// The rcon KEYS, not just the config string. Restoring g_Config alone is not enough and
		// the gap silently ends rcon coverage for the rest of the process: an authed client that
		// runs `sv_rcon_password x` reaches ConchainRconPasswordChangeGeneric
		// (server.cpp:4324-4350), which rewrites the admin key through UpdateKey/AddDefaultKey,
		// or removes it outright for an empty argument. g_Config.m_SvRconPassword is then rolled
		// back to "a" here while the key stays whatever the input made it, so every later
		// OnNetMsgRconAuth fails and OnNetMsgRconCmd, UpdateClientRconCommands and the maplist go
		// with it. `auth_add` and `auth_remove` do the same. All three are in the dictionary, so
		// a campaign trips this early and then fuzzes a server it can never log in to.
		//
		// Measured: one input carrying `sv_rcon_password zzz`, then an input that authenticates
		// with the fixture's own password, logs 1 successful auth instead of 2.
		//
		// The copy is safe despite CKey::m_pRole being a raw CRconRole* into m_Roles
		// (authmanager.h:62): s_PristineAuth is a function-local static, so it outlives every
		// input, and std::unordered_map nodes keep their addresses, so the restored keys point at
		// roles that stay put and read back the same name and rank.
		static const CAuthManager s_PristineAuth = g_pServer->m_AuthManager;
		g_pServer->m_AuthManager = s_PristineAuth;

		for(auto &Client : g_pServer->m_aClients)
		{
			// What NewClientCallback does to the session (server.cpp:1198, :1200). Not
			// m_AuthTries: the kick and ban for too many rcon tries need it to reach
			// sv_rcon_max_tries, which is more tries than the 16 records of one input.
			Client.m_AuthKey = -1;
			Client.m_AuthHidden = false;
		}
	}

	// Remove a score database left behind by a fuzz process that has exited.
	//
	// libFuzzer leaves through _Exit, so atexit never runs and a process cannot clean up after
	// itself. Sweeping at startup instead is race free: a pid that no longer exists cannot still
	// be writing. Nothing did this before, and one campaign left 70952 of these files, 13 GB, in
	// the real DDNet user directory, because -fork starts a fresh process for every job.
	inline int RemoveStaleScoreDb(const char *pName, int IsDir, int DirType, void *pUser)
	{
		int Pid;
		if(IsDir || sscanf(pName, "fuzz_score_%d.sqlite", &Pid) != 1 || Pid == (int)getpid() || kill(Pid, 0) == 0)
			return 0;

		char aPath[IO_MAX_PATH_LENGTH];
		str_format(aPath, sizeof(aPath), "%s/%s", (const char *)pUser, pName);
		(void)fs_remove(aPath);
		str_append(aPath, "-journal");
		(void)fs_remove(aPath);
		return 0;
	}

	// Build the whole server stack. Must be called from LLVMFuzzerInitialize: CreateStorage
	// asserts unless it is handed a non-empty argument list, and libFuzzer's argv is the only
	// one available.
	//
	// The working directory must contain data/ - storage resolves $DATADIR relative to the CWD
	// and LoadMap needs data/maps/<map>.map. From anywhere else this aborts on the LoadMap
	// assert below rather than failing quietly.
	inline void Init(int *pArgc, char ***pArgv, bool WithSqlite)
	{
		SetupLogging();

		g_pServer = CreateServer();

		// The way main.cpp does it (main.cpp:112), because CServer::ConchainLoglevel calls
		// m_pFileLogger->SetFilter() with no null check: a server without a file logger is a
		// state main.cpp cannot produce, and leaving it null turned the rcon command `loglevel`
		// into a null dereference the fixture manufactured. The stdout logger stays null, which
		// is what --silent does and what this fixture wants anyway. The future logger is not part
		// of the global logger set below, so it never receives a message to buffer.
		g_pServer->SetLoggers(std::make_shared<CFutureLogger>(), nullptr);

		IKernel *pKernel = IKernel::Create();
		pKernel->RegisterInterface(g_pServer);

		IEngine *pEngine = CreateTestEngine(GAME_NAME);
		pKernel->RegisterInterface(pEngine);

		IStorage *pStorage = CreateStorage(IStorage::EInitializationType::SERVER, *pArgc, (const char **)*pArgv);
		dbg_assert(pStorage != nullptr, "failed to create storage");
		pKernel->RegisterInterface(pStorage, false);

		IConsole *pConsole = CreateConsole(CFGFLAG_SERVER | CFGFLAG_ECON).release();
		pKernel->RegisterInterface(pConsole);

		IConfigManager *pConfigManager = CreateConfigManager();
		pKernel->RegisterInterface(pConfigManager);

		IEngineHttp *pEngineHttp = CreateEngineHttp();
		pKernel->RegisterInterface(pEngineHttp);
		pKernel->RegisterInterface(static_cast<IHttp *>(pEngineHttp), false);

		IEngineAntibot *pEngineAntibot = CreateEngineAntibot();
		pKernel->RegisterInterface(pEngineAntibot);
		pKernel->RegisterInterface(static_cast<IAntibot *>(pEngineAntibot), false);

		IGameServer *pGameServer = CreateGameServer();
		pKernel->RegisterInterface(pGameServer);
		g_pGameServer = (CGameContext *)pGameServer;

		pEngine->Init();
		pConsole->Init();
		pConfigManager->Init();

		g_pServer->RegisterCommands();
		dbg_assert(g_pServer->LoadMap("coverage") != 0, "failed to load map 'coverage'");
		g_pServer->m_RunServer = CServer::RUNNING;

		// What CServer::Run does once the startup config has been read (server.cpp:3262), and
		// the fixture never calls Run. A console starts with m_StoreCommands set, which QUEUES
		// every CFGFLAG_STORE command instead of running it: `ban`, `unban`, `unban_all`,
		// `ban_range`, `bans_save`, `change_map`, `random_map`, `record` and `reset` were all
		// dead for the campaign, and the queue is never drained either, so it grew by an 8 KB
		// entry per command until libFuzzer reported an out of memory. A real server drains it
		// before the first client can connect, so this is the state a peer meets.
		g_pServer->Console()->StoreCommands(false);
		// Install a known rcon password the way a server config does: through the console, so
		// the sv_rcon_password CHAIN runs. This is load-bearing in a way that is easy to get
		// wrong - I did, first time. Assigning g_Config.m_SvRconPassword directly bypasses the
		// chain, and CAuthManager::Init only calls AddDefaultKey inside its
		// `!g_Config.m_SvRconPassword[0]` branch, so a directly-assigned password leaves the key
		// list EMPTY: DefaultKey(ADMIN) returns -1 and every CheckKey fails. Leaving it unset is
		// no better - Init then generates a random 6-character password and installs that, and
		// CheckKey is an MD5-with-salt compare that libFuzzer's string interceptors cannot
		// solve. Either way rcon authentication never succeeds and NETMSG_RCON_CMD never
		// executes a command - which is the surface that produced the C2 finding.
		//
		// Must run after RegisterCommands (which registers the chain) and before
		// AuthManager::Init (so Init sees a configured password and does not generate one).
		g_pServer->Console()->ExecuteLine("sv_rcon_password a", IConsole::CLIENT_ID_NO_GAME, false);
		// The other two roles, the way a server config installs them, so AUTHED_MOD and
		// AUTHED_HELPER are reachable at all: OnNetMsgRconAuth tries the moderator and helper
		// default keys in turn (server.cpp:2140-2143) and both are absent with only an admin
		// password, so every access check below admin level was decided by a branch that could
		// never be taken.
		g_pServer->Console()->ExecuteLine("sv_rcon_mod_password m", IConsole::CLIENT_ID_NO_GAME, false);
		g_pServer->Console()->ExecuteLine("sv_rcon_helper_password h", IConsole::CLIENT_ID_NO_GAME, false);
		g_pServer->m_AuthManager.Init();

		// A score backend, so CScoreWorker and the rank/name SQL string building are reachable
		// from client input instead of sitting cold. CServer::Run normally does this; we never
		// call Run, so register it here - and it must happen before OnInit, which constructs
		// CScore against the pool.
		//
		// The filename carries the pid because campaign workers run concurrently and would
		// otherwise contend on, and corrupt, one database file.
		if(WithSqlite)
		{
			char aFile[64];
			str_format(aFile, sizeof(aFile), "fuzz_score_%d.sqlite", (int)getpid());
			char aFullPath[IO_MAX_PATH_LENGTH];
			pStorage->GetCompletePath(IStorage::TYPE_SAVE, aFile, aFullPath, sizeof(aFullPath));
			g_pServer->DbPool()->RegisterSqliteDatabase(CDbConnectionPool::READ, aFullPath);
			g_pServer->DbPool()->RegisterSqliteDatabase(CDbConnectionPool::WRITE, aFullPath);

			// libFuzzer leaves through _Exit, so atexit never runs and this process cannot
			// clean up after itself. Sweep what earlier ones left instead, which is race free:
			// a pid that no longer exists cannot still be writing its database. Nothing did
			// this before, and one campaign left 70952 of these files, 13 GB, in the real
			// DDNet user directory, because -fork starts a fresh process for every job.
			char aSaveDir[IO_MAX_PATH_LENGTH];
			pStorage->GetCompletePath(IStorage::TYPE_SAVE, "", aSaveDir, sizeof(aSaveDir));
			fs_listdir(aSaveDir, RemoveStaleScoreDb, IStorage::TYPE_SAVE, aSaveDir);
		}

		const int PersistentSize = pGameServer->PersistentClientDataSize();
		for(auto &Client : g_pServer->m_aClients)
		{
			Client.m_HasPersistentData = false;
			Client.m_pPersistentData = malloc(PersistentSize);
		}
		g_pServer->m_pPersistentData = malloc(pGameServer->PersistentDataSize());

		pEngineHttp->Init(std::chrono::seconds{2});

		// gameworld_test never sends, so it leaves the net server closed. These targets do send,
		// on nearly every input, and DoSnapshot sends too. Opening it on loopback gives the
		// queued responses a valid socket; the peer addresses stay zeroed, so net_udp_send
		// rejects them and nothing leaves the machine.
		//
		// ORDER MATTERS, and getting it wrong fails silently: CNetServer::Open starts with
		//     this->~CNetServer(); new(this) CNetServer{};
		// (network_server.cpp), so anything set beforehand is erased. Callbacks installed before
		// Open leave m_pfnDelClient null, CNetServer::Drop becomes a no-op, and no client can
		// ever be dropped - which silently deletes the whole disconnect path (DelClientCallback,
		// OnClientDrop, ~CPlayer, entity teardown) from every campaign. The real server opens
		// first and installs callbacks after (server.cpp:3205 then :3230); do the same.
		NETADDR BindAddr;
		mem_zero(&BindAddr, sizeof(BindAddr));
		BindAddr.type = NETTYPE_ALL;
		BindAddr.port = 0;
		// MaxClientsPerIp must be >= 1: Open assigns it unclamped and TryAcceptClient rejects
		// on `NumClientsWithAddr(Addr) + 1 > m_MaxClientsPerIp`, so 0 refuses every client.
		g_pServer->m_NetServer.Open(BindAddr, &g_pServer->m_ServerBan, MAX_CLIENTS, 4);

		g_pServer->m_NetServer.SetCallbacks(
			CServer::NewClientCallback,
			CServer::NewClientNoAuthCallback,
			CServer::ClientRejoinCallback,
			CServer::DelClientCallback, g_pServer);

		// Spam protection is the single biggest damper on this target's feedback signal. Chat
		// commands are rate-limited to one per second, four "/" commands in a second are dropped
		// outright, and exceeding a score threshold mutes the client for sv_spam_mute_duration
		// - measured in WALL-CLOCK seconds, not ticks, so a 60 second mute is millions of fuzz
		// executions with the entire chat and vote surface dead. Worse, mutes are keyed on the
		// peer address with the port zeroed, and every fixture slot shares an all-zero address,
		// so muting one mutes them all. Both are ordinary server settings whose documented
		// "off" value is 0.
		g_pServer->Antibot()->Init();
		pGameServer->OnInit(nullptr);

		// Make the world's randomness deterministic, so a finding reproduces from its own input.
		//
		// CGameContext::OnInit seeds its CPrng from secure_random_fill, i.e. differently in every
		// process, and CWorldCore::RandomOr0 is what chooses which teleporter exit a tee arrives
		// at. An artifact whose crash depends on that choice therefore does not reproduce, which
		// is the whole value of having found it - the same shape as the g_Config and rcon-session
		// state the reset below exists for. Measured before this, on a target with no database
		// thread to blame: three identical replays of one corpus covered 380, 380 and 374
		// functions. The game context keeps its own seeded CPrng for its description string; only
		// the world's pointer, which is the one gameplay reads, is redirected here.
		static CPrng s_Prng;
		static uint64_t s_aSeed[2] = {0x5eed5eed5eed5eedull, 0xf0cacc1af0cacc1aull};
		s_Prng.Seed(s_aSeed);
		g_pGameServer->m_World.m_Core.m_pPrng = &s_Prng;

		// AFTER OnInit, deliberately. OnInit executes sv_reset_file and then the map's embedded
		// settings, either of which would silently overwrite these - the same "set before it
		// gets wiped" shape as the SetCallbacks/Open bug. Today neither file sets them, but
		// that is one map away from being untrue.
		g_Config.m_SvSpamprotection = 0;
		g_Config.m_SvSpamMuteDuration = 0;
		// Default 0. With it off, /pause degrades to a no-op and the whole paused tick-mode
		// (CCharacter::Pause, the TickPaused family, force_pause) is unreachable.
		g_Config.m_SvPauseable = 1;

		// The vote system was ENTIRELY dead, and sv_join_vote_delay is why. Its default is 300
		// SECONDS (config_variables.h:285); CPlayer's constructor turns that into
		// `m_FirstVoteTick = Now + 300 * TickSpeed` (player.cpp:140), i.e. tick 15000, and
		// CGameContext::RateLimitPlayerVote returns true for every call before it
		// (gamecontext.cpp:5310). These targets advance roughly seventeen ticks per input, so
		// the whole surface stays shut for the first ~900 inputs of every process - and a
		// re-entering client resets its own clock, so a target that drops and re-adds clients
		// can stay shut forever.
		//
		// Measured over the campaign's 57281-entry fz_gamemsg corpus, before this line:
		// CGameContext::CallVote 0/57, StartVote 0/35, SendVoteSet 0/142, SendVoteStatus 0/53,
		// EndVote 0/5, AbortVoteKickOnDisconnect 0/22, ForceVote 0/8, and
		// OnCallVoteNetMessage stuck at 28/587 - not one vote had ever been started.
		//
		// Both are ordinary server settings with a documented minimum of 0, in the same class as
		// the spam-protection knobs above; plenty of real servers run them at 0.
		g_Config.m_SvJoinVoteDelay = 0;
		// Default 3 seconds between any two votes by the same player. Harmless on a real server,
		// but it gates the second and every later vote of an input sequence, which is exactly
		// where the interesting state (a vote already running, a victim already leaving) is.
		g_Config.m_SvVoteDelay = 0;
		// Default 25 seconds, and this one is WALL CLOCK, not ticks: StartVote sets
		// m_VoteCloseTime = time_get() + time_freq() * sv_vote_time (gamecontext.cpp:986), so no
		// amount of ticking closes it. While it is set, OnCallVoteNetMessage returns early,
		// AttemptJoinTeam refuses the creator through IsRunningKickOrSpecVote and OnKillNetMessage
		// refuses it too - so one unresolved vote shuts that whole surface for twenty-five seconds
		// of REAL time, i.e. for thousands of inputs. 1 is the documented minimum.
		g_Config.m_SvVoteTime = 1;
		// Default 3 seconds, checked in AttemptJoinTeam (ddracechat.cpp), so the second and every
		// later /team of a sequence is refused - which is most of what a team sequence is.
		g_Config.m_SvTeamChangeDelay = 0;
		// Default 1 second per player, and CScore::RateLimitPlayer stamps the clock BEFORE the
		// callee's own preconditions are checked (score.cpp:58-67), so a /save that bails still
		// costs the following /load its turn. Every score query in one input otherwise needs 50
		// ticks of separation, which is most of the tick budget.
		g_Config.m_SvSqlQueriesDelay = 0;

		CacheServerInfos();

		// Skip static teardown. These targets deliberately never run the server's shutdown
		// sequence (OnShutdown, Econ/Fifo, DbPool), so at exit the async logger joins an
		// already-joined thread and ASan aborts - recorded as a crash on the empty input, which
		// stops the campaign for a reason unrelated to the target. Registering this last means
		// it runs first at exit, before any destructor gets the chance.
		atexit([]() { _Exit(0); });
	}

	// CServer advances its own tick inside Run(), which these targets never call, and
	// m_CurrentGameTick is protected in IServer with no setter. Reach it without casting the
	// object to a type it is not: forming a pointer-to-member through a derived class is
	// allowed for a protected base member, and applying that pointer to the real object is then
	// ordinary, well-defined member access.
	struct CTickAccess : public CServer
	{
		static int CServer::*Member()
		{
			return static_cast<int CServer::*>(&CTickAccess::m_CurrentGameTick);
		}
	};

	// Advance one tick the way CServer::Run does, so state settles and time-dependent code
	// (spam protection, vote timers, respawn delays) actually progresses between messages.
	//
	// "The way CServer::Run does" now means all four steps of server.cpp:3362-3409, not just
	// the two that are obvious. Run brackets the tick increment with the queued per-client
	// inputs:
	//
	//     for each INGAME client: OnClientPredictedEarlyInput(c, input for Tick()+1, else null)
	//     m_CurrentGameTick++
	//     for each INGAME client: OnClientPredictedInput(c, input for Tick(), else null)
	//     OnTick()
	//
	// Leaving those two loops out did not merely skip two functions: NETMSG_INPUT's whole
	// reason for existing is to fill m_aInputs, and those loops are the ONLY readers. Without
	// them a client's input never reached its character at all, so every character ran on a
	// zeroed CNetObj_PlayerInput forever and the largest attacker-driven surface in the game
	// server sat at its entry block. Measured on the campaign's fz_serverpkt corpus, before
	// this: OnClientPredictedEarlyInput 0/59, OnClientPredictedInput 0/33,
	// CCharacter::OnDirectInput 0/50, CCharacter::OnPredictedInput 0/33,
	// CCharacter::HandleWeaponSwitch 0/100, FireWeapon 23/624, HandleNinja 3/385,
	// HandleTiles 168/1912.
	//
	// The null argument is not a shortcut: Run passes nullptr for a client that has no input
	// queued for this tick, and CGameContext handles that case explicitly
	// (gamecontext.cpp:1597, :1616). Passing it faithfully keeps both arms live.
	inline void AdvanceTick()
	{
		for(int i = 0; i < MAX_CLIENTS; i++)
		{
			if(g_pServer->m_aClients[i].m_State != CServer::CClient::STATE_INGAME)
				continue;
			const int *pData = nullptr;
			for(auto &Input : g_pServer->m_aClients[i].m_aInputs)
			{
				if(Input.m_GameTick == g_pServer->Tick() + 1)
				{
					pData = Input.m_aData;
					break;
				}
			}
			g_pGameServer->OnClientPredictedEarlyInput(i, pData);
		}

		++(g_pServer->*CTickAccess::Member());

		for(int i = 0; i < MAX_CLIENTS; i++)
		{
			if(g_pServer->m_aClients[i].m_State != CServer::CClient::STATE_INGAME)
				continue;
			const int *pData = nullptr;
			for(auto &Input : g_pServer->m_aClients[i].m_aInputs)
			{
				if(Input.m_GameTick == g_pServer->Tick())
				{
					pData = Input.m_aData;
					break;
				}
			}
			g_pGameServer->OnClientPredictedInput(i, pData);
		}

		g_pGameServer->OnTick();

		// Hand back the freed snapshot ids on the tick count rather than on the wall clock.
		// CSnapIdPool keeps a freed id for five seconds of real time before reusing it, and a
		// target runs thousands of ticks per real second, so the pool empties within seconds and
		// SnapNewId returns nullopt for the rest of the process. Every entity then snaps without
		// an id, which kills the snapshot surface these targets exist to cover, and it also made
		// a crash in CCharacter look like the campaign had found something a real server reaches.
		// Batched rather than recycled per tick, so a freed id is still not reused immediately.
		static int s_TicksSinceIdRecycle = 0;
		if(++s_TicksSinceIdRecycle >= 5 * SERVER_TICK_SPEED)
		{
			s_TicksSinceIdRecycle = 0;
			g_pServer->m_IdPool.TimeoutIds();
		}
	}

	// Advance up to Want ticks, spending no more than Budget, and report how many ran.
	//
	// The drivers used to advance at most one tick per record, so a whole input covered at most
	// seventeen ticks, a third of a second of game time. Nearly everything the server gates on
	// time is longer than that: a respawn is 25 ticks, a freeze 150, a vote 1250. Those
	// boundaries could therefore only be crossed BETWEEN inputs, where libFuzzer credits the
	// coverage to whichever input happened to be running when the timer fired rather than to the
	// one that set it, and the feedback signal that would steer the fuzzer towards them is lost.
	// It shows in the coverage: CGameContext::OnTick sat at 13% of its edges and
	// OnCallVoteNetMessage at 8%.
	//
	// A budget rather than a free count, because ticking is the expensive part of an input and
	// an unbounded count would trade the whole campaign's throughput for it.
	inline int AdvanceTicks(int Want, int Budget)
	{
		const int Ticks = std::clamp(Want, 0, Budget);
		for(int i = 0; i < Ticks; i++)
			AdvanceTick();
		return Ticks;
	}

	inline int CurrentTick()
	{
		return g_pServer->Tick();
	}

	// Bring a slot fully in game through the normal path, so m_apPlayers[ClientId] exists.
	//
	// NewClientCallback FIRST. It is the server's own accept-time initialiser and the only
	// caller of CClient::Reset(), which is the only place several per-client fields are ever
	// initialised - m_CurrentInput among them. CServer is allocated with a user-provided
	// constructor and CServer::Init does not cover those fields, so assigning m_State directly
	// (as this used to) left m_CurrentInput holding heap garbage. NETMSG_INPUT then indexes
	// m_aInputs[m_CurrentInput] BEFORE the `%= 200`, which is an out-of-bounds write that no
	// real peer can cause - a manufactured crash - and it made every artifact depend on
	// whatever was in that memory, so findings would not reproduce.
	inline void EnterGame(int ClientId, bool WantSixup)
	{
		// Only claim sixup if the server would actually accept a 0.7 client. LoadMap sets
		// sv_sixup to 0 and logs "disabling 0.7 compatibility" when maps7/<map>.map is absent,
		// and then m_apCurrentMapData[MAP_TYPE_SIXUP] is null - so a forced sixup slot makes
		// SendMapData hand a null pointer to mem_copy on the first NETMSG_REQUEST_MAP_DATA.
		// That is a crash no peer can cause, because without the 0.7 map no 0.7 peer exists.
		// To actually cover the 0.7 half, put a converted map at data/maps7/<map>.map.
		const bool Sixup = WantSixup && g_Config.m_SvSixup != 0;
		if(WantSixup && !Sixup)
		{
			// Loudly, because the previous behaviour was to downgrade in silence: both server
			// targets ran with two 0.6 clients while claiming to cover the 0.7 translation
			// layer, and nothing said otherwise. LoadMap only reads and hashes the 0.7 map, it
			// never parses it, so any valid 0.7 map file at that path is enough.
			static bool s_Warned = false;
			if(!s_Warned)
			{
				s_Warned = true;
				fprintf(stderr,
					"WARNING: sv_sixup is off (no data/maps7/<map>.map), so the 0.7 half of this\n"
					"         target - PreProcessMsg's translation, MsgFromSixup, the sixup\n"
					"         RCON_AUTH/ENTERGAME branches - is NOT being covered.\n");
			}
		}

		NewClient(ClientId, Sixup);
		g_pServer->m_aClients[ClientId].m_State = CServer::CClient::STATE_INGAME;
		g_pServer->m_aClients[ClientId].m_Sixup = Sixup;
		g_pServer->m_aClients[ClientId].m_HasPersistentData = false;

		// DoSnapshot refuses to serve a non-sixup client whose reported DDNet version is below
		// VERSION_DDNET_OLD, and Reset() leaves it at VERSION_NONE (-1). Without this the 0.6
		// slot is never snapped AT ALL - which silently made the snapshot oracle a sixup-only
		// test - and CGameContext::OnTick kicks the slot every tick once its grace period
		// expires. A real client reports this in Cl_IsDDNetLegacy, so setting it is faithful.
		g_pServer->SetClientDDNetVersion(ClientId, VERSION_DDNET_UPDATER_FIXED);

		// Name the client BEFORE it enters, the way a real one does: the name arrives in
		// Cl_StartInfo before ENTERGAME, so OnClientEnter and everything it drives (the join
		// message, the controller, the player mapping) sees it. A real INGAME client always has
		// a name - empty ones are rejected and auto-renamed - and CGameContext::FindClientIdByName
		// is an exact str_comp against ClientName(), so leaving them empty kept the entire
		// unquoted branch of Whisper dead.
		char aName[16];
		str_format(aName, sizeof(aName), "fuzz%d", ClientId);
		g_pServer->SetClientName(ClientId, aName);

		g_pGameServer->OnClientConnected(ClientId, nullptr);
		g_pGameServer->OnClientEnter(ClientId);
	}

} // namespace fzserver

#endif // FUZZ_FZ_SERVER_FIXTURE_H
