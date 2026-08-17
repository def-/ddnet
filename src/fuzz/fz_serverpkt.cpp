// libFuzzer target: CServer::ProcessClientPacket — the server's own system-message layer.
//
// This is the gap between the other two server harnesses: fz_netserver stops at
// CNetServer::Recv, and fz_gamemsg starts at CGameContext::OnMessage. Everything in
// between is the server's own protocol handling, and it is where authentication and session
// state live:
//
//   NETMSG_INFO             the PASSWORD CHECK, on a pre-auth string from an unknown peer
//   NETMSG_RCON_AUTH        rcon login name and password
//   NETMSG_RCON_CMD         the command string that produced the C2 finding
//   NETMSG_REQUEST_MAP_DATA a client-chosen chunk index feeding map-download offset maths
//   NETMSG_INPUT            m_LastAckedSnapshot, assigned straight from the client with no
//                           check that the server ever sent that tick - the delta base that
//                           made C1 reachable
//   NETMSG_READY / ENTERGAME the state transitions themselves
//
// The last two are why this target is worth having beyond mere coverage. C1 was a *state
// confusion* bug, not a malformed-bytes bug, and this function is where a client's session
// state is advanced. Driving it lets the fuzzer discover message orders that a well-behaved
// client never produces.
//
// Two slots are prepared so both halves are reachable from the first input: one client left
// in STATE_AUTH, so the password and handshake paths are live, and one brought fully in
// game through the normal path, so the post-auth handlers are live. States are advanced by
// the messages themselves rather than being assigned from the fuzz input - a client cannot
// set the server's idea of its state, and forcing arbitrary combinations would manufacture
// crashes that no real peer can cause.
//
// Input encoding - a sequence of records:
//   u8 slot_and_flags : bit 0 picks the slot, bit 1 marks the chunk non-vital, bit 2 requests
//                       a tick afterwards and bits 4-7 say how many further ticks, drawn from
//                       a per-input budget. Bits 0-2 keep the meaning they had before the
//                       budget existed, so an existing corpus still means what it meant.
//   u8 msg_id, u8 sys : message id and whether it is a system message
//   u16 body_len, body: the packed message body (little-endian length)
#include "fz_server_fixture.h"

#include <base/dbg.h>
#include <base/mem.h>

#include <engine/shared/network.h>
#include <engine/shared/packer.h>
#include <engine/shared/protocol.h>

#include <cstddef>
#include <cstdint>

bool IsInterrupted()
{
	return false;
}

namespace
{

	// The slot that stays pre-auth, so NETMSG_INFO's password check stays reachable.
	constexpr int CLIENT_AUTH = fzserver::CLIENT_SEVEN;

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

	// One slot fully in game, so the post-handshake handlers are reachable immediately.
	// Sixup, because otherwise the entire 0.7 half is dead: MsgFromSixup's id remapping,
	// PreProcessMsg's translation branch and its shared 1024-byte s_aRawMsg, the IsSixup
	// branches of RCON_AUTH and ENTERGAME, and RepackMsg's sixup path. A 0.7 client is a
	// state a real peer causes - TryAcceptClient takes Sixup, and sv_sixup defaults to 1.
	fzserver::EnterGame(fzserver::CLIENT_SIX, true);

	// One slot left pre-auth, initialised the way the server itself does it.
	// NewClientCallback calls CClient::Reset(), which is the ONLY place m_CurrentInput,
	// m_SnapRate, m_LastAckedSnapshot and m_NextMapChunk are set - CServer::Init does not
	// touch them and they have no member initialisers. Assigning m_State directly (as this
	// used to) left them holding heap garbage, and NETMSG_INPUT indexes
	// m_aInputs[m_CurrentInput] BEFORE the modulo. It leaves the slot in PREAUTH, which is
	// what a freshly accepted connection really is, and is the only state OnNetMsgClientVer
	// accepts.
	fzserver::NewClient(CLIENT_AUTH, false);

	// Give the server a password, otherwise the NETMSG_INFO branch short-circuits and the
	// comparison never runs.
	str_copy(g_Config.m_Password, "fuzz", sizeof(g_Config.m_Password));
	return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 5)
		return 0;

	// Re-arm before EVERY packet, not once per input: a record can drop the client and the
	// next record in the same sequence would then hit an empty slot. With the fixture's callbacks now
	// installed after Open, "Wrong version" / "Wrong password" / "Too many rcon tries"
	// genuinely free the slot. Delivering a further packet to an empty slot is harness
	// misuse - CServer::GetAuthedState asserts on exactly that, and CNetServer::Recv would
	// never hand such a packet up on a real server. Re-arming is what a fresh peer causes.
	const auto ReArmSlots = []() {
		if(fzserver::g_pServer->m_aClients[fzserver::CLIENT_SIX].m_State == CServer::CClient::STATE_EMPTY)
			fzserver::EnterGame(fzserver::CLIENT_SIX, true);
		if(fzserver::g_pServer->m_aClients[CLIENT_AUTH].m_State == CServer::CClient::STATE_EMPTY)
		{
			// NewClient runs the server's own accept-time initialiser, so this
			// reproduces the PREAUTH state a real accepted connection starts in.
			fzserver::NewClient(CLIENT_AUTH, false);
		}
	};
	ReArmSlots();
	fzserver::ResetPerInput();

	CReader Reader(pData, Size);

	// Bounded, for the same reason as fz_gamemsg: an unbounded script turns into a reported
	// hang rather than a finding.
	const int MaxRecords = 16;
	int TickBudget = 64;
	for(int Record = 0; Record < MaxRecords && !Reader.Done(); Record++)
	{
		const uint8_t Flags = Reader.U8();
		const int ClientId = (Flags & 1) != 0 ? CLIENT_AUTH : fzserver::CLIENT_SIX;
		const int MsgId = Reader.U8();
		const bool Sys = (Reader.U8() & 1) != 0;
		// 16-bit: a real chunk carries up to NET_MAX_CHUNK_SIZE (1023) bytes, and the rcon
		// line limit is 512, so a u8 cap could never reach any string-truncation boundary.
		size_t BodyLen = (size_t)Reader.U8() | ((size_t)Reader.U8() << 8);
		if(BodyLen > Reader.Left())
			BodyLen = Reader.Left();

		// Frame it the way the wire does: a varint of (id << 1) | sys, then the body. Doing
		// this rather than feeding raw bytes is what gets the fuzzer past UnpackMessageId
		// and into the per-message handlers, which is the point of the target.
		static CPacker s_Packer;
		s_Packer.Reset();
		s_Packer.AddInt((MsgId << 1) | (Sys ? 1 : 0));
		if(BodyLen > 0)
			s_Packer.AddRaw(Reader.Bytes(BodyLen), (int)BodyLen);
		if(s_Packer.Error())
			continue;

		// ProcessClientPacket reads through a CUnpacker over this buffer and the handlers
		// sanitise strings in place, so it must be writable and privately owned.
		static unsigned char s_aBuf[NET_MAX_PACKETSIZE];
		int PacketSize = s_Packer.Size();
		if(PacketSize > (int)sizeof(s_aBuf))
			PacketSize = (int)sizeof(s_aBuf);
		mem_copy(s_aBuf, s_Packer.Data(), PacketSize);

		CNetChunk Chunk;
		mem_zero(&Chunk, sizeof(Chunk));
		Chunk.m_ClientId = ClientId;
		Chunk.m_pData = s_aBuf;
		Chunk.m_DataSize = PacketSize;
		// Most messages are refused unless the chunk was vital; let the fuzzer clear the
		// flag sometimes so that refusal path is covered too.
		Chunk.m_Flags = (Flags & 2) != 0 ? 0 : NET_CHUNKFLAG_VITAL;

		ReArmSlots();
		fzserver::g_pServer->ProcessClientPacket(&Chunk);

		if((Flags & 4) != 0)
			TickBudget -= fzserver::AdvanceTicks(1 + (Flags >> 4), TickBudget);
	}

	// Snapshot afterwards for the same reason fz_gamemsg does: NETMSG_INPUT sets the delta
	// base, and the only place that choice is acted on is DoSnapshot.
	// SNAPRATE_FULL first: Reset() leaves clients in SNAPRATE_INIT, which DoSnapshot serves
	// only every tenth tick, and a baseless delta then downgrades them to RECOVER (every
	// fiftieth). Without this the snapshot path runs on a small fraction of inputs.
	fzserver::AdvanceTick();
	for(int i = 0; i < fzserver::NUM_FUZZ_CLIENTS; i++)
		fzserver::g_pServer->m_aClients[i].m_SnapRate = CServer::CClient::SNAPRATE_FULL;
	fzserver::g_pServer->DoSnapshot();

	// The two things CServer::Run does immediately after DoSnapshot, on the same tick
	// (server.cpp:3419-3421). They are the tail of the rcon handshake: OnNetMsgRconAuth only
	// sets m_pRconCmdToSend, and this loop is what actually walks the command list and emits
	// it, so without it a successful rcon login stopped halfway. Measured before this:
	// UpdateClientRconCommands 0/33, SendRconCmdAdd 0/16, SendRconCmdGroupEnd 0/3,
	// UpdateClientMaplistEntries 0/56, SendMaplistGroupStart 0/5, SendMaplistGroupEnd 0/3.
	//
	// Run picks ONE client per tick, `Tick() % MAX_CLIENTS`, so on a real server each slot
	// gets a turn every 128 ticks. Doing that literally here would give each of the two
	// fixture slots a turn every 128 inputs, and since UpdateClientRconCommands emits only
	// MAX_RCONCMD_SEND commands per call out of a list several hundred long, finishing one
	// client's list would take tens of thousands of inputs. Calling it for both slots every
	// input runs exactly the same code with the same arguments, just at the rate a
	// fully-populated server would.
	for(int i = 0; i < fzserver::NUM_FUZZ_CLIENTS; i++)
	{
		fzserver::g_pServer->UpdateClientRconCommands(i);
		fzserver::g_pServer->UpdateClientMaplistEntries(i);
	}

	// Rebuild the server-info caches, which is the other thing the tick loop does when
	// anything about a client changed (`if(m_ServerInfoNeedsUpdate) UpdateServerInfo(...)`,
	// server.cpp:3439). Almost every message this target sends is one of those changes: a
	// name, a clan, a country, a score, a client joining or leaving. The builders read all
	// of it back out and pack it into a length-limited browser response, so they are a
	// genuine attacker-influenced surface (CacheServerInfo alone is 265 edges,
	// CacheServerInfoSixup 208) rather than bookkeeping.
	fzserver::CacheServerInfos();
	return 0;
}
