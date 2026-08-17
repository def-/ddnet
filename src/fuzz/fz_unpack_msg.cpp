// libFuzzer target: CNetObjHandler::SecureUnpackMsg — the single funnel every client
// message passes through, for both 0.6 and 0.7.
//
// This is the most directly attacker-reachable parser in the server: anything a client
// sends as a game message arrives here, and the generated unpacker is what enforces the
// NetIntRange / string rules that the rest of the server then trusts.
//
// Input encoding, kept as thin as possible:
//   byte 0 : protocol select (bit 0) — 0 = 0.6, 1 = 0.7
//   byte 1-2 : message id, little-endian (taken modulo the id space, so the fuzzer does
//              not waste its time on ids that cannot exist)
//   rest   : the raw message body handed to CUnpacker
//
// Reachability note for triage: a crash here is e2e-reachable by definition — a client can
// send any message id with any body. That is exactly why this target is worth having: no
// reachability argument is needed for anything it finds.
//
// MEASUREMENT NOTE (why the stats block below exists): edge coverage cannot tell you
// whether this target is doing its job. A fully-successful unpack and one that fails its
// LAST range check share every basic block up to the `if(m_pMsgFailedOn) return nullptr`
// at generated/protocol.cpp:1915, so libFuzzer sees no new edge for getting a message
// right and has no gradient toward valid messages. The only honest metric is how many of
// the selectable message ids ever return NON-NULL. Set FZ_MSG_STATS=1 to have the harness
// count that per id and print it at exit.
#include <base/dbg.h>
#include <base/mem.h>

#include <engine/shared/network.h>
#include <engine/shared/packer.h>
#include <engine/shared/protocol.h>

#include <generated/protocol.h>
#include <generated/protocol7.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace
{

	// NUM_NETMSGTYPES counts only the NON-extended messages. The 24 NetMessageEx entries -
	// Cl_ShowDistance, Cl_ShowOthers, Cl_CameraInfo, Cl_EnableSpectatorCount, Sv_SaveCode,
	// Sv_PreInput and friends - live in a separate range starting at OFFSET_NETMSGTYPE_UUID
	// (>= OFFSET_UUID = 1 << 16), so a plain modulo can never produce one and their unpackers
	// were unreachable. They ARE client-reachable: CUnpacker/UnpackMessageId maps a
	// client-supplied UUID onto these ids and CGameContext::SecureUnpackMsg passes it straight
	// through. Select across both ranges so the fuzzer can choose either.
	const int NUM_EX_6 = (int)OFFSET_MAPITEMTYPE_UUID - (int)OFFSET_NETMSGTYPE_UUID;
	const int NUM_SEL_6 = (int)NUM_NETMSGTYPES + NUM_EX_6;
	const int NUM_SEL_7 = (int)protocol7::NUM_NETMSGTYPES;
	const int NUM_SEL = NUM_SEL_6 + NUM_SEL_7;

	// Per-selector attempt / success counters. Only touched when FZ_MSG_STATS is set, so the
	// fuzzing loop pays one predictable branch and nothing else.
	bool g_Stats = false;
	long g_aTry[NUM_SEL];
	long g_aOk[NUM_SEL];

	void PrintStats()
	{
		CNetObjHandler Handler6;
		protocol7::CNetObjHandler Handler7;
		int NumOk = 0;
		for(int i = 0; i < NUM_SEL; i++)
		{
			const bool Six = i < NUM_SEL_6;
			const int Sel = Six ? i : i - NUM_SEL_6;
			const int MsgId = !Six ? Sel :
						 (Sel < (int)NUM_NETMSGTYPES ? Sel : (int)OFFSET_NETMSGTYPE_UUID + (Sel - (int)NUM_NETMSGTYPES));
			const char *pName = Six ? Handler6.GetMsgName(MsgId) : Handler7.GetMsgName(MsgId);
			if(g_aOk[i] > 0)
				NumOk++;
			fprintf(stderr, "MSGSTAT %s %-3d %-34s try=%ld ok=%ld%s\n",
				Six ? "6" : "7", Sel, pName, g_aTry[i], g_aOk[i], g_aOk[i] > 0 ? "" : "   <-- NEVER");
		}
		fprintf(stderr, "MSGSTAT SUMMARY: %d/%d ids ever unpacked successfully\n", NumOk, NUM_SEL);
	}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	static bool s_Init = false;
	if(!s_Init)
	{
		s_Init = true;
		g_Stats = getenv("FZ_MSG_STATS") != nullptr;
		if(g_Stats)
			atexit(PrintStats);
	}

	if(Size < 4)
		return 0;

	const bool Sixup = (pData[0] & 1) != 0;
	int MsgId = (int)(pData[1] | (pData[2] << 8));
	pData += 3;
	Size -= 3;

	// A client cannot send more than one chunk's worth in a single message.
	if(Size > NET_MAX_CHUNK_SIZE)
		Size = NET_MAX_CHUNK_SIZE;

	// CUnpacker::GetString sanitises strings IN PLACE, so the unpacker writes into the
	// buffer it is given. libFuzzer's input is const, so give it a private copy - otherwise
	// the harness trips "overwrites-const-input" and reports itself rather than the target.
	static unsigned char s_aBuf[NET_MAX_CHUNK_SIZE];
	mem_copy(s_aBuf, pData, Size);

	CUnpacker Unpacker;
	Unpacker.Reset(s_aBuf, (int)Size);

	static CNetObjHandler s_Handler6;
	static protocol7::CNetObjHandler s_Handler7;

	int StatIndex;
	void *pRawMsg;
	if(Sixup)
	{
		// 0.7 has no extended messages (datasrc/seven/network.py defines none), so the plain
		// range is the whole space here.
		MsgId %= NUM_SEL_7;
		StatIndex = NUM_SEL_6 + MsgId;
		pRawMsg = s_Handler7.SecureUnpackMsg(MsgId, &Unpacker);
	}
	else
	{
		const int Sel = MsgId % NUM_SEL_6;
		MsgId = Sel < (int)NUM_NETMSGTYPES ? Sel : (int)OFFSET_NETMSGTYPE_UUID + (Sel - (int)NUM_NETMSGTYPES);
		StatIndex = Sel;
		pRawMsg = s_Handler6.SecureUnpackMsg(MsgId, &Unpacker);
	}

	// Touch the result the way CGameContext::OnMessage would, so a bad pointer or a short
	// struct is caught here rather than silently ignored.
	if(pRawMsg != nullptr && !Unpacker.Error())
	{
		volatile unsigned char Sink = 0;
		Sink ^= *(const unsigned char *)pRawMsg;
		(void)Sink;
	}

	if(g_Stats)
	{
		g_aTry[StatIndex]++;
		if(pRawMsg != nullptr)
			g_aOk[StatIndex]++;
	}
	return 0;
}
