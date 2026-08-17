// libFuzzer target: CGameContext::OnMessage — the whole server-side game message dispatch,
// against a REAL CGameContext, driven with multi-message sequences and checked by the real
// snapshot path.
//
// Two things live here and nowhere else:
//
//   1. CGameContext::PreProcessMsg - the 0.7 -> 0.6 message translation. It
//      reinterpret_casts 0.7 messages onto one shared `static char s_aRawMsg[1024]`, per
//      branch, with no placement new (the code says so itself: "Should probably use a
//      placement new to start the lifetime of the object to avoid future weirdness"). A 0.7
//      Cl_Command is rewritten into a synthetic Cl_Say via str_format of two unbounded
//      network strings and relabelled NETMSGTYPE_CL_SAY, so it reaches the chat path
//      without having passed the 0.6 NetStringStrict validation its sibling did.
//
//   2. CGameContext::Whisper - a second hand-rolled quoted/escaped argument parser,
//      distinct from CConsole::ExecuteLine. It compacts in place into the live network
//      buffer, its error path writes before it returns, and it temporarily NUL-truncates
//      and restores that buffer around a name lookup. Reached by any Cl_Say starting "/w ".
//
// Why SEQUENCES rather than one message per input: almost everything interesting in a game
// server is a state machine. /team then /practice then /save, vote call then vote, pause,
// kill, timeout - none of it is reachable one message at a time, because each message is
// evaluated against the state the previous ones left behind. One input therefore encodes a
// sequence of messages, with the tick advanced between them so time-dependent code (spam
// protection, vote timers, respawn delays) actually progresses.
//
// Why the SNAPSHOT ORACLE: a crash-only harness over OnMessage can only see faults inside
// the handler. But the damage a hostile message does is usually *stored* - a bad position,
// a dangling id, an inconsistent team - and surfaces later, when the server serialises that
// state for other clients. So after each sequence the target runs the real
// CServer::DoSnapshot, which is exactly where C1 manifested: it builds every client's
// snapshot and deltas it against a previous one through CSnapshotDelta::CreateDelta. The
// acked tick that selects the delta base comes from the fuzz input, just as it comes from
// the client on the wire - CServer::ProcessClientPacket assigns m_LastAckedSnapshot straight
// from NETMSG_INPUT with no check that the server ever sent that tick.
//
// Reachability: direct. Every message arrives from a client that finished the handshake;
// ProcessClientPacket gates OnMessage only on state >= STATE_READY, and PreProcessMsg runs
// before even the ClientIngame check. No rcon auth.
//
// Input encoding - a sequence of records, then a one-byte trailer:
//   per record: u8 client_and_flags, u8 msg_id_lo, u8 msg_id_hi, u16 body_len (LE), body
//               bit 0 of client_and_flags picks the slot, bit 1 requests a tick afterwards
//               and bits 3-7 say how many further ticks, drawn from a per-input budget.
//               Bits 0-1 keep the meaning they had before the budget existed, so an existing
//               corpus still means what it did
//   trailer   : selects the acked tick offset used as the snapshot delta base
#include "fz_server_fixture.h"

#include <base/dbg.h>
#include <base/mem.h>

#include <engine/shared/network.h>
#include <engine/shared/packer.h>
#include <engine/shared/protocol.h>

#include <generated/protocol.h>
#include <generated/protocol7.h>

#include <cstddef>
#include <cstdint>

bool IsInterrupted()
{
	return false;
}

namespace
{

	// Run the real snapshot path over whatever state the messages left behind.
	//
	// AckOffset chooses the delta base: 0 means "no base, send in full", anything else acks a
	// tick relative to the current one - including ticks the server never sent, which is what a
	// hostile client does and what made C1 reachable.
	void SnapshotOracle(uint8_t AckOffset)
	{
		for(int i = 0; i < fzserver::NUM_FUZZ_CLIENTS; i++)
		{
			// The offset is forced EVEN because DoSnapshot only stores a snapshot on global
			// ticks (every second tick by default), so an odd offset can never name a stored
			// tick and the delta base is always empty. Measured: ack=1 leaves DiffItem at 0/20,
			// ack=2 reaches 11/20.
			const int Offset = 2 * (int)AckOffset;
			fzserver::g_pServer->m_aClients[i].m_LastAckedSnapshot =
				AckOffset == 0 ? -1 : fzserver::CurrentTick() - Offset;
			// Unconditionally FULL. ProcessClientPacket sets both together when a client acks;
			// leaving it at SNAPRATE_INIT gets the client served once every ten ticks, and a
			// baseless delta downgrades it to RECOVER (once every fifty).
			fzserver::g_pServer->m_aClients[i].m_SnapRate = CServer::CClient::SNAPRATE_FULL;
		}
		fzserver::g_pServer->DoSnapshot();
	}

	class CReader
	{
	public:
		CReader(const uint8_t *pData, size_t Size) :
			m_pData(pData), m_Size(Size), m_Pos(0) {}

		bool Done() const { return m_Pos >= m_Size; }
		size_t Left() const { return m_Pos < m_Size ? m_Size - m_Pos : 0; }
		uint8_t U8() { return m_Pos < m_Size ? m_pData[m_Pos++] : 0; }
		const uint8_t *Bytes(size_t n)
		{
			const uint8_t *p = m_pData + m_Pos;
			m_Pos += n;
			return p;
		}

	private:
		const uint8_t *m_pData;
		size_t m_Size;
		size_t m_Pos;
	};

} // namespace

extern "C" int LLVMFuzzerInitialize(int *pArgc, char ***pArgv)
{
	fzserver::Init(pArgc, pArgv, /*WithSqlite=*/true);
	fzserver::EnterGame(fzserver::CLIENT_SIX, false);
	fzserver::EnterGame(fzserver::CLIENT_SEVEN, true);
	return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 6)
		return 0;

	// Re-arm any slot that was dropped by the previous input. Now that the fixture installs
	// its net callbacks AFTER Open (so CNetServer::Drop actually fires), a client really can
	// leave - e.g. gamecontext kicks any client whose reported DDNet version is too old.
	// Sending a message to a slot whose CPlayer is gone is harness misuse, not a finding:
	// CServer::ProcessClientPacket only reaches OnMessage for a connected slot. Re-entering
	// models the reconnect a real peer performs, and keeps the connect/drop churn live.
	for(int i = 0; i < fzserver::NUM_FUZZ_CLIENTS; i++)
	{
		if(fzserver::g_pGameServer->m_apPlayers[i] == nullptr)
			fzserver::EnterGame(i, i == fzserver::CLIENT_SEVEN);
	}

	// The trailer picks the delta base for the snapshot oracle below.
	const uint8_t AckOffset = pData[Size - 1];
	CReader Reader(pData, Size - 1);

	// Bound the sequence: without a cap a single input can spend seconds inside the world,
	// and libFuzzer's per-input timeout would then report a hang that is really just a long
	// script rather than a defect. The tick budget is the same bound for time rather than
	// messages, see fzserver::AdvanceTicks for why a record may now spend more than one.
	const int MaxRecords = 16;
	int TickBudget = 64;
	for(int Record = 0; Record < MaxRecords && !Reader.Done(); Record++)
	{
		const uint8_t Flags = Reader.U8();
		const int ClientId = Flags & 1;
		int MsgId = (int)Reader.U8();
		MsgId |= (int)Reader.U8() << 8;
		// 16-bit. The target's own headline case - a 0.7 Cl_Command rewritten into a
		// synthetic Cl_Say via str_format into a 1024-byte static buffer - needs ~1000 bytes
		// of strings to reach its truncation boundary, which a u8 length could never supply.
		size_t BodyLen = (size_t)Reader.U8() | ((size_t)Reader.U8() << 8);
		if(BodyLen > Reader.Left())
			BodyLen = Reader.Left();

		// The unpacker sanitises strings in place, so it must not be handed libFuzzer's
		// const input. This also matches the server, where the buffer is the packet it just
		// received and handlers such as Whisper write into it.
		static unsigned char s_aBuf[NET_MAX_CHUNK_SIZE];
		if(BodyLen > sizeof(s_aBuf))
			BodyLen = sizeof(s_aBuf);
		mem_copy(s_aBuf, Reader.Bytes(BodyLen), BodyLen);

		CUnpacker Unpacker;
		Unpacker.Reset(s_aBuf, (int)BodyLen);

		// Take the id modulo the right protocol's space, so the fuzzer spends its budget on
		// ids that can exist rather than on the dispatch's default branch.
		if(fzserver::g_pServer->m_aClients[ClientId].m_Sixup)
		{
			MsgId %= protocol7::NUM_NETMSGTYPES;
		}
		else
		{
			// Select across the plain range AND the UUID-extended one. The extended game
			// messages (Cl_ShowOthers, Cl_ShowDistance, Cl_CameraInfo,
			// Cl_EnableSpectatorCount) live at ids >= OFFSET_UUID, so a plain modulo could
			// never produce them and their handlers were dead - including the one that
			// writes attacker-supplied zoom into CPlayer::m_CameraInfo.
			const int NumEx = OFFSET_MAPITEMTYPE_UUID - OFFSET_NETMSGTYPE_UUID;
			const int Sel = MsgId % (NUM_NETMSGTYPES + NumEx);
			MsgId = Sel < NUM_NETMSGTYPES ? Sel : OFFSET_NETMSGTYPE_UUID + (Sel - NUM_NETMSGTYPES);
		}

		// Re-check per record, not just per input: a record can advance the tick, and a tick
		// can drop a client (the version kick fires from OnTick). Delivering the NEXT record
		// to a slot whose CPlayer is gone null-derefs in OnStartInfoNetMessage, which is
		// dispatched outside the ClientIngame guard - a crash no real peer can cause,
		// because the server only reaches OnMessage for a connected slot.
		if(fzserver::g_pGameServer->m_apPlayers[ClientId] == nullptr)
			fzserver::EnterGame(ClientId, ClientId == fzserver::CLIENT_SEVEN);

		fzserver::g_pGameServer->OnMessage(MsgId, &Unpacker, ClientId);

		if((Flags & 2) != 0)
			TickBudget -= fzserver::AdvanceTicks(1 + (Flags >> 3), TickBudget);
	}

	// Always tick once before snapping: a snapshot of a world that has not stepped since the
	// messages arrived would miss anything the world applies on tick.
	fzserver::AdvanceTick();
	SnapshotOracle(AckOffset);
	return 0;
}
