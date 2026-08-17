// libFuzzer target: CSaveTee::FromString / CSaveTeam::FromString — the save-code parser.
//
// Three findings were reported against this parser in earlier review rounds and NONE was
// ever executed, because reaching it requires a row in the shared score database and round
// 1 concluded there is no client-side injection path:
//
//   * crafted Savegame row -> heap overflow
//   * wild m_HookedPlayer index
//   * ~25 MB out-of-bounds read via m_TuneZone
//   * m_LastTimeCp parsed without the range check every sibling field has (save.cpp:414),
//     then used to index m_aBestTimeCp[25] / m_aCurrentTimeCp[25] guarded only by > -1
//
// Component fuzzing settles whether the parser is actually broken, cheaply and in seconds.
// The *reachability* question is then a separate triage step, per the agreed protocol:
//   - if it crashes and a crafted savegame is client-reachable -> e2e reproducer -> FINDINGS
//   - if it crashes but the DB is not client-injectable -> record the invariant and dismiss
//     the category, rather than pretending it is a live bug
//
// Input: byte 0 selects CSaveTee (even) or CSaveTeam (odd); byte 1 is the member count
// passed to CSaveTee::FromString (deliberately allowed to go negative); the rest is the
// NUL-terminated save string.
#include <base/dbg.h>
#include <base/mem.h>

#include <engine/shared/protocol.h>

#include <game/server/save.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

// Harnesses linked against the full game-server object set replace main.cpp, so anything
// main.cpp defined has to be stubbed. Any future game-server harness (chat handlers, the
// 0.7 translation) needs this same stub.
bool IsInterrupted()
{
	return false;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 4)
		return 0;

	const bool Team = (pData[0] & 1) != 0;
	// Range-reduced rather than a raw signed byte. FromString rejects any tee whose
	// m_HookedPlayer is >= MembersCount, and for a NEGATIVE count that test rejects
	// EVERYTHING (-1 >= -1), so half of byte 1's values turned the whole CSaveTee half of
	// the input space into a guaranteed reject after a full 115-conversion sscanf. Keep
	// exactly one negative value so the rejection path stays covered.
	const int Members = (int)(pData[1] % (SERVER_MAX_CLIENTS + 2)) - 1;
	pData += 2;
	Size -= 2;

	// The parser takes a C string; give it an exactly-sized NUL-terminated heap buffer so
	// ASan redzones catch a read one byte past the end.
	char *pStr = (char *)malloc(Size + 1);
	if(!pStr)
		return 0;
	memcpy(pStr, pData, Size);
	pStr[Size] = '\0';

	// No global logger is installed here, so the parser's own log_error/dbg_msg output is
	// dropped and a failed parse is indistinguishable from a successful one. Set
	// FZ_SAVETEE_LOG=1 to see which it was - essential when checking whether a seed corpus
	// actually gets past the sscanf field count.
	static const bool s_Verbose = getenv("FZ_SAVETEE_LOG") != nullptr;
	const bool Verbose = s_Verbose;
	if(Team)
	{
		CSaveTeam SaveTeam;
		// NOTE the opposite conventions: CSaveTeam::FromString returns an int where 0 means
		// SUCCESS (it returns 1 or Num+1 on failure), while CSaveTee::FromString below
		// returns a bool where true means success. Easy to misread when triaging.
		const int Err = SaveTeam.FromString(pStr);
		if(Verbose)
			fprintf(stderr, "[fz] CSaveTeam::FromString -> %d (%s)\n", Err, Err == 0 ? "parsed" : "rejected");
	}
	else
	{
		CSaveTee SaveTee;
		const bool Ok = SaveTee.FromString(pStr, Members);
		if(Verbose)
			fprintf(stderr, "[fz] CSaveTee::FromString(members=%d) -> %d (%s)\n",
				Members, (int)Ok, Ok ? "parsed" : "rejected");
	}

	free(pStr);
	return 0;
}
