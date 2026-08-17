// libFuzzer target: CConsole::ExecuteLine — the chat-command parser.
//
// Every "/cmd arg arg" a player types in chat reaches the server as a plain string and is
// handed to the same console that runs rcon commands, with the flag mask narrowed to
// CFGFLAG_CHAT. The tokenizer, the quote/escape handling and ParseArgs' range checks are
// therefore attacker-controlled on every public server, from an unauthenticated client.
//
// The specific reason this target exists: ParseArgs validates numeric ranges on the token
// it extracted, but the quoted-string branch (console.cpp, `if(*pStr == '"')`) rewrites the
// buffer in place while unescaping, and the value handed to GetInteger is whatever survived
// that rewrite. A quoted INT_MIN is the known shape that defeats the range check.
//
// Why dummy commands rather than the real ones: registering the real chat commands needs a
// live CGameContext (world, players, score backend). The bug class we are after lives in
// the console's own tokenizer and in IResult's accessors, which are entirely independent of
// what the callback then does. So we register commands whose FORMAT STRINGS mirror the real
// ones and whose callbacks only read the parsed arguments back out.
//
// Reachability: direct. Any connected client can send any chat line. A crash here needs no
// reachability argument beyond "the server has chat enabled".
//
// What this target CAN prove, because none of it depends on which callback is behind the
// command: the tokenizer in ExecuteLineStroked (quote/escape state machine, ';' splitting,
// '#' comments, the "mc;" prefix), ParseStart's copy into m_aStringStorage, ParseArgs for
// every specifier DDNet actually registers, IResult's accessors (GetString/GetInteger/
// GetFloat/GetColor/GetVictim), and the special-victim fan-out.
//
// What it CANNOT prove, and why:
//   * anything about the real command BODIES. A bug inside ConWhisper or ConSave is not
//     visible here at all; that is the end-to-end harness' job.
//   * the format shapes DDNet does not register. In particular the 'c' (colour) specifier
//     at console.cpp:295 has no registered command anywhere in the tree - a grep for
//     Register(..., "...c[...") returns nothing - so ParseArgs' colour arm and
//     PARSEARGS_INVALID_COLOR are dead code, and a '?' appearing BEFORE a 'v' (which is
//     what makes ParseArgs default a missing victim to "me", console.cpp:215) does not
//     occur either: all eleven v[ commands start with the 'v'. Registering fake shapes
//     would light those up but could only ever produce findings against code no client can
//     reach, so they are left dark deliberately.
//   * the branches gated on flags no chat command carries. CFGFLAG_STORE (console.cpp:609,
//     which is the ONLY thing that constructs CResult's pointer-rebasing copy constructor)
//     is server/rcon-only AND m_StoreCommands is true only while the server is loading its
//     config (server.cpp:3325-3327); CMDFLAG_TEST is CFGFLAG_SERVER-only. The "Access
//     denied" arm (console.cpp:674) cannot fire either: CServer::CanClientUseCommand
//     returns true for every CFGFLAG_CHAT command before it looks at rcon auth
//     (server.cpp:3682).
//   * CLIENT_ID_GAME / CLIENT_ID_NO_GAME. Those are map/config-file pseudo-clients; a chat
//     line always carries a real slot, which is why byte 0 is reduced modulo MAX_CLIENTS.
//
// A mirror of CGameContext::ClientsForVictim is installed so "me" and "all" resolve the way
// they do on a server; without it the console rejects every special victim and half the
// dispatch stays dark. An earlier attempt at this reported a -fsanitize=function
// type-mismatch on the indirect call from console.cpp. That was not a target defect and not
// a UBSan quirk: the harness was built without _GLIBCXX_DEBUG while the library was built
// with it, so std::vector was std::__debug::vector on one side of the call and not the
// other - genuinely different types. build.sh now mirrors the project's own defines.
//
// Input encoding:
//   byte 0 : client id (clamped to a legal slot)
//   byte 1 : bit 0 -> interpret semicolons
//   rest   : the chat line, which must be valid UTF-8 (see below)
#include <base/dbg.h>
#include <base/mem.h>
#include <base/str.h>

#include <engine/console.h>
#include <engine/shared/config.h>
#include <engine/shared/network.h>
#include <engine/shared/protocol.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

// Harnesses linked against the game-server object set replace main.cpp, so anything
// main.cpp defined has to be stubbed.
bool IsInterrupted()
{
	return false;
}

// Mirror of CGameContext::ClientsForVictim, so the special-victim dispatch is exercised.
std::optional<std::vector<int>> ClientsForVictim(int ClientId, const char *pVictim, void *pUser)
{
	if(str_comp(pVictim, "me") == 0)
		return std::make_optional(std::vector<int>{ClientId});
	if(str_comp(pVictim, "all") == 0)
		return std::make_optional(std::vector<int>{0, 1, 2});
	return std::nullopt;
}

namespace
{

	// Mirror of CGameContext::CommandCallback (gamecontext.cpp:233), which the server installs
	// unconditionally in OnInit (gamecontext.cpp:4179). Without it the branch at console.cpp:621
	// is dark and the CFGFLAG_NONTEEHISTORIC test inside it never runs, even though on a live
	// server every chat command goes through here. The body reads the arguments back the way
	// CTeeHistorian::RecordConsoleCommand does (teehistorian.cpp:620-624) - GetString on every
	// index - so the recorder's view of the CResult is exercised, not just the callback's.
	void TeeHistorianCallback(int ClientId, int FlagMask, const char *pCmd, IConsole::IResult *pResult, void *pUser)
	{
		volatile char Sink = pCmd[0];
		const int Num = pResult->NumArguments();
		for(int i = 0; i < Num; i++)
			Sink ^= pResult->GetString(i)[0];
		(void)Sink;
		(void)ClientId;
		(void)FlagMask;
		(void)pUser;
	}

	// Read every argument back the way a real command callback does. Parsing a token is only
	// half the surface: GetInteger/GetFloat/GetColor convert it afterwards, and that conversion
	// is where an unvalidated token turns into a bad value.
	void FuzzCallback(IConsole::IResult *pResult, void *pUser)
	{
		// GetVictim() dbg_asserts unless the format string actually declared a 'v' and the
		// parser filled it in (console.cpp:1145). Real command callbacks only call it for
		// commands that declared one, so the harness has to honour the same contract -
		// otherwise the target reports our misuse instead of the parser's defects.
		const bool HasVictim = pUser != nullptr;
		const int Num = pResult->NumArguments();
		for(unsigned i = 0; i < (unsigned)Num; i++)
		{
			volatile int Int = pResult->GetInteger(i);
			volatile float Float = pResult->GetFloat(i);
			const char *pStr = pResult->GetString(i);
			volatile char Ch = pStr != nullptr ? pStr[0] : '\0';
			ColorHSLA Color = pResult->GetColor(i, 0.0f);
			volatile float Hue = Color.h;
			(void)Int;
			(void)Float;
			(void)Ch;
			(void)Hue;
		}
		if(HasVictim)
		{
			volatile int Victim = pResult->GetVictim();
			(void)Victim;
		}
	}

	// Format strings copied in shape from the real DDNet chat commands, covering every
	// specifier the parser understands ('i' int, 'f' float, 's' token, 'r' rest-of-line,
	// 'v' victim, '?' optional) and the combinations that appear in ddracechat.cpp.
	const struct
	{
		const char *m_pName;
		const char *m_pParams;
		bool m_Victim; // declares a 'v' specifier, so GetVictim() is legal in the callback
		int m_ExtraFlags;
	} s_aCommands[] = {
		{"fz_none", "", false, 0},
		{"fz_int", "i[a]", false, 0},
		{"fz_int_opt", "?i[a]", false, 0},
		{"fz_int_int", "i[a]i[b]", false, 0},
		{"fz_float", "f[a]", false, 0},
		{"fz_str", "s[name]", false, 0},
		{"fz_rest", "r[text]", false, 0},
		{"fz_victim", "v[id]", true, 0},
		{"fz_victim_reason", "v[id]?r[reason]", true, 0},
		{"fz_mixed", "s[name]?i[points]?r[reason]", false, 0},
		{"fz_many", "i[a]?i[b]?i[c]?i[d]", false, 0},
		// "?s[...] i[...]" is the shape of /emote (gamecontext.cpp:4059). Once a '?' is seen
		// Optional stays true for the REST of the format string, so the trailing required-looking
		// 'i' is optional too - a quirk no other registered shape exposes.
		{"fz_opt_then_req", "?s[a]i[b]", false, 0},
		// The whisper family carries CFGFLAG_NONTEEHISTORIC (gamecontext.cpp:4065-4068), which is
		// the false side of the teehistorian test at console.cpp:621.
		{"fz_nonhistoric", "s[name]r[message]", false, CFGFLAG_NONTEEHISTORIC},
	};

	IConsole *Console()
	{
		static std::unique_ptr<IConsole> s_pConsole = []() {
			auto pConsole = CreateConsole(CFGFLAG_SERVER | CFGFLAG_CHAT);
			pConsole->SetGetVictimsCommandCallback(ClientsForVictim, nullptr);
			pConsole->SetTeeHistorianCommandCallback(TeeHistorianCallback, nullptr);
			for(const auto &Command : s_aCommands)
			{
				pConsole->Register(Command.m_pName, Command.m_pParams,
					CFGFLAG_SERVER | CFGFLAG_CHAT | Command.m_ExtraFlags, FuzzCallback,
					Command.m_Victim ? (void *)&Command : nullptr, "fuzz");
			}
			return pConsole;
		}();
		return s_pConsole.get();
	}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 3)
		return 0;

	// A chat line arrives inside one net message, so it cannot be longer than a chunk.
	// Without this the fuzzer spends its time on lengths no client can send.
	if(Size > NET_MAX_CHUNK_SIZE)
		Size = NET_MAX_CHUNK_SIZE;

	const int ClientId = (int)(pData[0] % MAX_CLIENTS);
	const bool Semicolons = (pData[1] & 1) != 0;
	pData += 2;
	Size -= 2;

	// ExecuteLine takes a C string; give it an exactly-sized NUL-terminated heap block so a
	// read one byte past the end is a hard ASan error rather than landing in slack space.
	char *pLine = (char *)malloc(Size + 1);
	if(pLine == nullptr)
		return 0;
	memcpy(pLine, pData, Size);
	pLine[Size] = '\0';

	// Only valid UTF-8 can reach ExecuteLine from the network: CUnpacker::GetString runs
	// str_utf8_check and sets m_Error, and every caller (e.g. the NETMSG_RCON_CMD handler,
	// server.cpp:1935) bails on it. Without this gate the target reports crashes that no
	// client can trigger - it found exactly one, an invalid-UTF-8 victim token that
	// str_copy truncates to "", which is unreachable over the wire and only matters for
	// operator-controlled config files.
	if(!str_utf8_check(pLine))
	{
		free(pLine);
		return 0;
	}

	// ExecuteLineFlag narrows the mask to CFGFLAG_CHAT for the duration of the call, which
	// is exactly what the server does for chat. It also keeps the console's own built-ins
	// (exec, echo, ...) out of reach, so we never fire a command that would need the
	// storage/config interfaces this harness deliberately does not construct.
	Console()->ExecuteLineFlag(pLine, CFGFLAG_CHAT, ClientId, Semicolons);

	free(pLine);
	return 0;
}
