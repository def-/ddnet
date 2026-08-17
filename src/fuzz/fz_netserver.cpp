// libFuzzer target: CNetServer::Recv — the whole packet ingress path, unauthenticated.
//
// This is the only surface on the server that needs NO state at all: no token, no
// handshake, no password. CNetServer::Recv is what every UDP datagram hits first, and from
// there it fans out to OnPreConnMsg, OnConnCtrlMsg, OnTokenCtrlMsg, OnSixupCtrlMsg,
// TryAcceptClient and the connection state machine.
//
// The specific thing that motivated this target is in OnPreConnMsg
// (network_server.cpp, the sv_vanilla_antispoof branch, which is default-on):
//
//   CNetChunkHeader h;
//   unsigned char *pData = Packet.m_aChunkData;
//   pData = h.Unpack(pData);          // reads up to 3 bytes with no m_DataSize check
//   CUnpacker Unpacker;
//   Unpacker.Reset(pData, h.m_Size);  // h.m_Size is 10 attacker bits, never compared
//                                     // against Packet.m_DataSize
//
// Its siblings do check: OnConnCtrlMsg requires `m_DataSize == 1 + sizeof(SECURITY_TOKEN)`
// and OnSixupCtrlMsg requires `m_DataSize >= 1 + sizeof(SECURITY_TOKEN)`. Validation
// applied on one path and not on its sibling is the shape that produced the console
// finding, so it is worth driving rather than reasoning about.
//
// Rather than reach the private OnPreConnMsg directly, the harness does what an attacker
// does: it sends real datagrams to a real CNetServer bound on loopback and lets Recv()
// dispatch them. That costs a syscall pair per iteration but exercises the true ingress
// path with real connection state, including the parts that only run on a second or third
// packet from the same address.
//
// THE SERVER'S OWN TICK IS PART OF THE TARGET. CServer::PumpNetwork (server.cpp:2867) is
// the only caller of CNetServer::Recv on a real server, and it does three things around it
// every single tick:
//
//     m_NetServer.Update();          // server.cpp:2871
//     m_NetServer.BeginFlushBatch(); // server.cpp:2875
//     ... Recv() loop, ProcessClientPacket, which replies via m_NetServer.Send() ...
//     m_NetServer.EndFlushBatch();   // server.cpp:2970
//
// An earlier version of this harness called only Recv(). That was not merely incomplete, it
// was self-limiting: CNetServer::Update is the ONLY thing that turns an ERROR'd slot back
// into an OFFLINE one (Update -> Drop -> Disconnect -> Reset), and TryAcceptClient only
// ever hands out OFFLINE slots. Every client CLOSE, token mismatch or timeout therefore
// burned a slot permanently, and after 128 of them the server was full forever and every
// accept path in the target was dead for the rest of the process. Measured on the
// campaign's own 393-entry corpus: CNetServer::Update 0/32, CNetConnection::Update 0/57,
// Drop 0/9, Disconnect 0/15, CNetBan::Update 0/80, Send 0/83, QueueChunkEx 0/64, Flush 0/6,
// AckChunks 0/10, ResendChunk 0/21, CNetChunkHeader::Pack 0/29, EndFlushBatch 0/24 - all of
// it structurally unreachable, not merely undiscovered.
//
// Input: the raw datagram, verbatim, minus a one-byte prologue. That byte selects which
// source socket it is sent from, so the fuzzer can build per-address state (slots, bans,
// token challenges) instead of always looking like a brand-new peer, and whether the
// source's token fixup is applied.
#include <base/dbg.h>
#include <base/mem.h>
#include <base/net.h>

#include <engine/shared/config.h>
#include <engine/shared/netban.h>
#include <engine/shared/network.h>
#include <engine/shared/packer.h>
#include <engine/shared/protocol.h>
#include <engine/shared/protocol7.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>

// Harnesses linked against the server object set replace main.cpp.
bool IsInterrupted()
{
	return false;
}

namespace
{

	CNetServer *g_pNetServer = nullptr;
	CNetBan *g_pNetBan = nullptr;
	NETADDR g_ServerAddr;

	// How each source socket is meant to be known to the server. The whole point of having
	// several is that the accept paths differ, and each one leaves the connection in a
	// different state that the packets after it then have to satisfy.
	enum
	{
		// Accepted through sv_vanilla_antispoof, i.e. with NET_SECURITY_TOKEN_UNSUPPORTED.
		// CNetConnection::Feed then skips its per-packet token check entirely, so raw fuzz
		// datagrams reach the chunk layer with no fixup at all - the fuzzer keeps every byte.
		KIND_VANILLA,
		// Accepted through the DDNet token handshake, i.e. with a real per-process token. Feed
		// takes its `m_SecurityToken != UNKNOWN && != UNSUPPORTED` branch and requires the token
		// as the last four bytes of every packet, exactly as CNetBase::SendPacket appends it on
		// the client. That branch is unreachable from a KIND_VANILLA slot.
		KIND_DDNET_TOKEN,
		// Accepted through the 0.7 handshake. Feed takes `m_Sixup && SecurityToken != m_Token`
		// and the token lives in the packet HEADER (bytes 3..6) rather than the tail, and
		// UnpackPacket parses a 7-byte header and a 6-bit chunk-size split.
		KIND_SIXUP,
		// Never accepted, and re-armed to stay that way. TryAcceptClient's rejection arms and
		// the three pre-connection handlers are only reachable from an address with no slot.
		KIND_FREE,
		NUM_SOURCE_KINDS
	};

	enum
	{
		NUM_SOURCES = 5
	};

	struct CSource
	{
		NETSOCKET m_Socket;
		NETADDR m_Addr; // this socket's own address, as the server sees it (fixed port)
		int m_Kind;
	};
	CSource g_aSources[NUM_SOURCES];

	// Slot bookkeeping, maintained from the net callbacks rather than guessed. CNetServer
	// exposes no "is slot i online", and knowing whether a source still owns a slot is what
	// makes re-arming possible - see MaintainSources.
	bool g_aSlotUsed[NET_MAX_CLIENTS];
	NETADDR g_aSlotAddr[NET_MAX_CLIENTS];

	void MarkSlot(int ClientId)
	{
		g_aSlotUsed[ClientId] = true;
		g_aSlotAddr[ClientId] = *g_pNetServer->ClientAddr(ClientId);
	}

	int NewClient(int ClientId, void *pUser, bool Sixup)
	{
		MarkSlot(ClientId);
		return 0;
	}
	int NewClientNoAuth(int ClientId, void *pUser)
	{
		MarkSlot(ClientId);
		return 0;
	}
	int ClientRejoin(int ClientId, void *pUser)
	{
		MarkSlot(ClientId);
		return 0;
	}
	int DelClient(int ClientId, const char *pReason, void *pUser)
	{
		g_aSlotUsed[ClientId] = false;
		return 0;
	}

	bool SourceHasSlot(const CSource &Source)
	{
		for(int i = 0; i < NET_MAX_CLIENTS; i++)
		{
			if(g_aSlotUsed[i] && net_addr_comp(&g_aSlotAddr[i], &Source.m_Addr) == 0)
				return true;
		}
		return false;
	}

	// Drain whatever the server has to say. Also used after each handshake step, because the
	// server's reply is what the next step is supposed to be answering.
	void DrainServer(int MaxChunks)
	{
		CNetChunk Chunk;
		mem_zero(&Chunk, sizeof(Chunk));
		SECURITY_TOKEN ResponseToken = 0;
		int Guard = 0;
		while(g_pNetServer->Recv(&Chunk, &ResponseToken) && ++Guard < MaxChunks)
			;
	}

	void SendFrom(const CSource &Source, const void *pData, int Size)
	{
		net_udp_send(Source.m_Socket, &g_ServerAddr, pData, Size);
		// Wait for the datagram to actually land before the caller drains. Without this the
		// send and the processing straddle iterations: libFuzzer attributes the coverage of
		// input N to input N + 1, the feedback signal is nonsense and the corpus never grows
		// (measured: 79 edges and a 2-entry corpus after 10M executions).
		net_socket_read_wait(g_pNetServer->Socket(), std::chrono::milliseconds(1));
	}

	// --- the three handshakes -------------------------------------------------------------
	//
	// All three use CNetServer's PUBLIC GetToken/GetVanillaToken rather than poking at private
	// state. That is legitimate: a real client learns exactly these values from the server's
	// own reply (CONNECTACCEPT for the DDNet and 0.7 handshakes, the NETMSG_SNAPEMPTY gametick
	// for the vanilla one). Replaying the reply here rather than parsing it off the wire is a
	// shortcut in plumbing, not in reachability.
	//
	// It is also the difference between a target and a facade. The seed is re-rolled with
	// secure_random_fill on every CNetServer::Open, so libFuzzer cracking the 32-bit compare
	// (measured: ~1.2M executions) buys nothing - the corpus entry that did it is dead in the
	// next process, and any crash artifact found past an accept would not reproduce.
	//
	// GetToken hashes only 20 bytes of the NETADDR - `sha256_update(&Sha256, &Addr, 20); //
	// omit port, bad idea!` (network_server.cpp:147) - so a port-zero address yields the same
	// token as any of the source ports, and the harness never needs to know its own port to
	// compute one.

	SECURITY_TOKEN TokenForSources()
	{
		NETADDR Addr;
		mem_zero(&Addr, sizeof(Addr));
		Addr.type = NETTYPE_IPV4;
		Addr.ip[0] = 127;
		Addr.ip[3] = 1;
		return g_pNetServer->GetToken(Addr);
	}

	SECURITY_TOKEN VanillaTokenForSources()
	{
		NETADDR Addr;
		mem_zero(&Addr, sizeof(Addr));
		Addr.type = NETTYPE_IPV4;
		Addr.ip[0] = 127;
		Addr.ip[3] = 1;
		return g_pNetServer->GetVanillaToken(Addr);
	}

	void HandshakeVanilla(const CSource &Source)
	{
		// A non-control 0.6 packet carrying one chunk: NETMSG_INPUT followed by the vanilla
		// token, which is what OnPreConnMsg's `else if(!IsCtrl && ...)` arm parses. Accepts with
		// NET_SECURITY_TOKEN_UNSUPPORTED and VanillaAuth=true.
		CPacker Payload;
		Payload.Reset();
		Payload.AddInt((NETMSG_INPUT << 1) | 1); // system message
		Payload.AddInt(VanillaTokenForSources());

		unsigned char aPacket[NET_MAX_PACKETSIZE];
		const int ChunkSize = Payload.Size();
		aPacket[0] = 0; // flags: not control, not compressed
		aPacket[1] = 0; // ack
		aPacket[2] = 1; // one chunk
		// Non-vital chunk header, 0.6 split of 4.
		aPacket[3] = (unsigned char)((ChunkSize >> 4) & 0x3f);
		aPacket[4] = (unsigned char)(ChunkSize & 0x0f);
		mem_copy(aPacket + 5, Payload.Data(), ChunkSize);
		SendFrom(Source, aPacket, 5 + ChunkSize);
		DrainServer(32);
	}

	void HandshakeDDNetToken(const CSource &Source)
	{
		const SECURITY_TOKEN Token = TokenForSources();

		// Step 1: a DDNet CONNECT - control packet, SECURITY_TOKEN_MAGIC then any token. This is
		// what IsDDNetControlMsg recognises, which is what routes it to OnTokenCtrlMsg instead
		// of OnPreConnMsg. The server answers CONNECTACCEPT carrying GetToken(Addr).
		unsigned char aConnect[NET_MAX_PACKETSIZE];
		aConnect[0] = NET_PACKETFLAG_CONTROL << 2;
		aConnect[1] = 0; // ack
		aConnect[2] = 0; // control packets carry zero chunks (IsValidConnectionOrientedPacket)
		aConnect[3] = NET_CTRLMSG_CONNECT;
		mem_copy(aConnect + 4, SECURITY_TOKEN_MAGIC, sizeof(SECURITY_TOKEN_MAGIC));
		WriteSecurityToken(aConnect + 4 + sizeof(SECURITY_TOKEN_MAGIC), NET_SECURITY_TOKEN_UNKNOWN);
		SendFrom(Source, aConnect, 4 + sizeof(SECURITY_TOKEN_MAGIC) + sizeof(SECURITY_TOKEN));
		DrainServer(32);

		// Step 2: echo the token back in a CTRLMSG_ACCEPT. OnTokenCtrlMsg compares it against
		// GetToken(Addr) and calls TryAcceptClient(Addr, Token) on a match.
		unsigned char aAccept[NET_MAX_PACKETSIZE];
		aAccept[0] = NET_PACKETFLAG_CONTROL << 2;
		aAccept[1] = 0;
		aAccept[2] = 0;
		aAccept[3] = NET_CTRLMSG_ACCEPT;
		WriteSecurityToken(aAccept + 4, Token);
		SendFrom(Source, aAccept, 4 + sizeof(SECURITY_TOKEN));
		DrainServer(32);
	}

	void HandshakeSixup(const CSource &Source)
	{
		const SECURITY_TOKEN Token = TokenForSources();

		// A 0.7 control packet. Flags bit 0 does double duty in UnpackPacket: it is what makes
		// `if(m_Flags & NET_PACKETFLAG_UNUSED) Sixup = true` fire, and after
		// PacketFlags_SevenToSix it is also protocol7's CONTROL bit. Header is 7 bytes with the
		// security token at 3..6, so DataStart is 7 rather than 3.
		unsigned char aPacket[NET_MAX_PACKETSIZE];
		aPacket[0] = 1 << 2; // flags = protocol7 CONTROL, ack high bits zero
		aPacket[1] = 0; // ack
		aPacket[2] = 0; // zero chunks
		WriteSecurityToken(aPacket + 3, Token);
		// OnSixupCtrlMsg needs m_DataSize >= 1 + sizeof(SECURITY_TOKEN) and reads the client's
		// own response token out of bytes 1..4 of the chunk data.
		aPacket[7] = protocol7::NET_CTRLMSG_CONNECT;
		WriteSecurityToken(aPacket + 8, Token);
		SendFrom(Source, aPacket, 8 + sizeof(SECURITY_TOKEN));
		DrainServer(32);
	}

	void Handshake(const CSource &Source)
	{
		switch(Source.m_Kind)
		{
		case KIND_VANILLA: HandshakeVanilla(Source); break;
		case KIND_DDNET_TOKEN: HandshakeDDNetToken(Source); break;
		case KIND_SIXUP: HandshakeSixup(Source); break;
		default: break;
		}
	}

	// Keep the source population at its intended shape.
	//
	// Both halves matter, and both fix a way the old harness decayed to nothing over a long
	// run - silently, which is the worst kind:
	//
	//   * Re-arm any managed source that has lost its slot. Now that Update() actually drops
	//     connections, slots really do go away - on a client CLOSE, on a timeout, on a "too
	//     weak connection". Reconnecting is what a real peer does when the server drops it.
	//   * Drop any slot belonging to the FREE source. If that address is ever accepted, then
	//     EVERY source has a slot, and OnPreConnMsg / OnTokenCtrlMsg / OnSixupCtrlMsg - the
	//     three handlers this target exists for, all reached only from `Slot == -1` - go dark
	//     for the rest of the process. Dropping a client is an ordinary server action
	//     (server.cpp calls m_NetServer.Drop for kicks, wrong passwords, a full server and
	//     redirects), so this is not a state a real deployment cannot be in.
	//
	// Free first, then re-arm: with sv_max_clients_per_ip set to the managed count, the free
	// source is normally refused outright, and doing it in this order means a transiently
	// missing managed slot cannot be stolen.
	void MaintainSources()
	{
		for(int i = 0; i < NET_MAX_CLIENTS; i++)
		{
			if(!g_aSlotUsed[i])
				continue;
			for(const auto &Source : g_aSources)
			{
				if(Source.m_Kind == KIND_FREE && net_addr_comp(&g_aSlotAddr[i], &Source.m_Addr) == 0)
				{
					g_pNetServer->Drop(i, "fuzz: keeping this address unconnected");
					break;
				}
			}
		}

		for(auto &Source : g_aSources)
		{
			if(Source.m_Kind == KIND_FREE)
				continue;
			if(!SourceHasSlot(Source))
				Handshake(Source);
		}
	}

} // namespace

extern "C" int LLVMFuzzerInitialize(int *pArgc, char ***pArgv)
{
	net_init();

	g_pNetBan = new CNetBan();
	// UnbanAll, not because anything is banned, but because it is the only reachable call
	// that zeroes the pools. CNetBan::CBanPool has no constructor and no member initialisers
	// (netban.h:141-145: m_aapHashList, m_aBans, m_pFirstFree, m_pFirstUsed, m_CountUsed),
	// and Reset() is called from exactly two places, CNetBan::Init (netban.cpp:280) and
	// UnbanAll (netban.cpp:173). Init needs an IConsole and an IStorage this target has no
	// reason to build. Without either, CNetServer::Recv reads uninitialised memory on EVERY
	// datagram: network_server.cpp:636 calls IsBanned, which indexes m_aapHashList and
	// dereferences whatever it finds as a CBan* (netban.cpp:395). Fresh pages read as zero
	// today, which is why nothing has crashed, but it is undefined, MSan would fire on the
	// first input, and any crash reported through IsBanned would be the harness's own.
	g_pNetBan->UnbanAll();
	g_pNetServer = new CNetServer();

	NETADDR BindAddr;
	mem_zero(&BindAddr, sizeof(BindAddr));
	BindAddr.type = NETTYPE_IPV4;
	BindAddr.ip[0] = 127;
	BindAddr.ip[3] = 1;
	// A fixed port, not 0: CNetServer::Address() reports the address it was *given*, not the
	// one the OS actually bound, so with port 0 every datagram would be sent to port 0 and
	// silently dropped. That failure is quiet - the harness still runs, it just never
	// reaches the server (measured: 79 edges, a 2-entry corpus, and a full 2 ms wait per
	// iteration because nothing ever arrived).
	bool Opened = false;
	int ServerPort = 0;
	for(int Port = 41337; Port < 41437 && !Opened; Port++)
	{
		BindAddr.port = Port;
		ServerPort = Port;
		// MaxClientsPerIp of 0 rejects EVERY client: Open assigns it unclamped and
		// TryAcceptClient tests `NumClientsWithAddr(Addr) + 1 > m_MaxClientsPerIp`. With 0
		// no slot was ever accepted, so the entire connected-client half of Recv - the
		// connection state machine, the chunk unpacker, OnConnCtrlMsg - was unreachable.
		// It must also be EXACTLY the number of sources this harness keeps connected: they
		// are all on 127.0.0.1 and NumClientsWithAddr compares without the port, so a
		// smaller value makes the last handshake fail forever, and a larger one lets the
		// KIND_FREE source be accepted and stops TryAcceptClient's per-IP rejection arm from
		// ever running.
		Opened = g_pNetServer->Open(BindAddr, g_pNetBan, MAX_CLIENTS, NUM_SOURCES - 1);
	}
	dbg_assert(Opened, "failed to open net server on any port in 41337-41436");

	// This harness builds no IConfigManager, so g_Config is a zero-initialised global: every
	// sv_* default is 0 rather than its documented value. That silently disabled the exact
	// branch this target exists for - OnPreConnMsg's vanilla-antispoof path is guarded by
	// `g_Config.m_SvVanillaAntiSpoof`, which defaults to 1 but was 0 here - and it disabled
	// 0.7 connections entirely. Set the handful that gate the ingress paths.
	g_Config.m_SvVanillaAntiSpoof = 1;
	g_Config.m_SvSixup = 1;
	g_Config.m_SvVanConnPerSecond = 10;
	// All five fuzz sources are loopback, and the connection limiter compares addresses
	// without the port, so they count as one IP. A non-zero window would throttle the
	// fuzzer to a handful of connects; 0 is the legal minimum and still runs both halves.
	g_Config.m_SvConnlimit = 5;
	g_Config.m_SvConnlimitTime = 0;
	// conn_timeout and conn_timeout_protection are read by CNetConnection::Update, which
	// this harness now calls. Their documented defaults are 100 and 1000
	// (config_variables.h:637-638) and both are minimum-5; left at the zero-initialised 0
	// every connection times out on the first Update and the target would spend its whole
	// life re-handshaking instead of fuzzing.
	g_Config.m_ConnTimeout = 100;
	g_Config.m_ConnTimeoutProtection = 1000;

	g_pNetServer->SetCallbacks(NewClient, NewClientNoAuth, ClientRejoin, DelClient, nullptr);

	g_ServerAddr = BindAddr;

	// Several source addresses, so the fuzzer can occupy more than one slot and reach the
	// per-address logic (slot reuse, MaxClientsPerIp, ban checks, token challenges).
	//
	// FIXED ports, unlike before. The harness has to be able to tell whether a given source
	// still owns a slot (MaintainSources), and the only public way to ask is to compare
	// against CNetServer::ClientAddr - which needs the source's own address, port included.
	static const int s_aKinds[NUM_SOURCES] = {
		KIND_VANILLA, KIND_VANILLA, KIND_DDNET_TOKEN, KIND_SIXUP, KIND_FREE};
	int NextPort = ServerPort + 1;
	for(int i = 0; i < NUM_SOURCES; i++)
	{
		g_aSources[i].m_Kind = s_aKinds[i];
		bool Bound = false;
		for(; NextPort < ServerPort + 200 && !Bound; NextPort++)
		{
			NETADDR SrcAddr;
			mem_zero(&SrcAddr, sizeof(SrcAddr));
			SrcAddr.type = NETTYPE_IPV4;
			SrcAddr.ip[0] = 127;
			SrcAddr.ip[3] = 1;
			SrcAddr.port = NextPort;
			g_aSources[i].m_Socket = net_udp_create(SrcAddr);
			if(g_aSources[i].m_Socket)
			{
				g_aSources[i].m_Addr = SrcAddr;
				Bound = true;
			}
		}
		dbg_assert(Bound, "failed to bind fuzz source socket");
	}

	// Bring every non-FREE source up front, so the post-accept surface is live from the
	// first input rather than only after the fuzzer has rediscovered the handshakes.
	MaintainSources();

	// Skip static teardown; see fz_gamemsg for why.
	atexit([]() { _Exit(0); });
	return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 2)
		return 0;

	// A datagram larger than this cannot arrive: net_udp_recv reads into a
	// NET_MAX_PACKETSIZE buffer, so anything longer is truncated by the network stack and
	// fuzzing it would only find overflows the server can never see. Reserve room for the
	// four token bytes the DDNet-token fixup appends.
	if(Size > NET_MAX_PACKETSIZE - sizeof(SECURITY_TOKEN))
		Size = NET_MAX_PACKETSIZE - sizeof(SECURITY_TOKEN);

	const uint8_t Prologue = pData[0];
	const CSource &Source = g_aSources[Prologue % NUM_SOURCES];
	// Bit 5 suppresses the fixup, so the token-MISMATCH arms of Feed and OnSixupCtrlMsg stay
	// reachable. Without it the harness would only ever produce well-formed tokens and the
	// rejection branches would be as dead as the accept branches were before.
	const bool ApplyFixup = (Prologue & 0x20) == 0;

	static unsigned char s_aDatagram[NET_MAX_PACKETSIZE];
	int Len = (int)Size - 1;
	mem_copy(s_aDatagram, pData + 1, Len);

	// Token fixups. Each one writes the token exactly where the corresponding real client
	// puts it, and nowhere else; the rest of the datagram stays under the fuzzer's control.
	if(ApplyFixup)
	{
		const SECURITY_TOKEN Token = TokenForSources();
		if(Source.m_Kind == KIND_DDNET_TOKEN)
		{
			// CNetBase::SendPacket appends the token after the payload for every
			// non-connless packet on a token-carrying connection; CNetConnection::Feed
			// strips it from the tail before anything else.
			WriteSecurityToken(s_aDatagram + Len, Token);
			Len += sizeof(SECURITY_TOKEN);
		}
		else if(Source.m_Kind == KIND_SIXUP && Len >= 7)
		{
			// 0.7 puts it in the header instead, at bytes 3..6.
			WriteSecurityToken(s_aDatagram + 3, Token);
		}

		// A placeholder any source can use to say "the token goes here". The token is
		// re-rolled per process, so a corpus entry holding a literal one is worthless on the
		// next run - the same argument that justifies the handshakes above. This is the only
		// way the fuzzer can express the token INSIDE a chunk body, which is what
		// OnConnCtrlMsg's rejoin arm and OnTokenCtrlMsg's accept arm compare against, and a
		// real client legitimately knows the value because the server sent it.
		static const unsigned char PLACEHOLDER[4] = {0xC0, 0xDE, 0x70, 0x4B};
		for(int i = 0; i + (int)sizeof(PLACEHOLDER) <= Len; i++)
		{
			if(mem_comp(s_aDatagram + i, PLACEHOLDER, sizeof(PLACEHOLDER)) == 0)
			{
				WriteSecurityToken(s_aDatagram + i, Token);
				i += sizeof(PLACEHOLDER) - 1;
			}
		}
	}

	// Mirror CServer::PumpNetwork (server.cpp:2867) exactly: Update first, then the batch,
	// then the receive loop, then EndFlushBatch.
	g_pNetServer->Update();
	MaintainSources();
	g_pNetServer->BeginFlushBatch();

	SendFrom(Source, s_aDatagram, Len);

	// Drain everything the server makes of it. Recv() loops internally over the packet's
	// chunks, so one datagram can yield several chunks or none.
	// Zeroed: OnSixupCtrlMsg's short TOKEN-request branch returns 1 having set only
	// m_Flags/m_ClientId/m_Address/m_DataSize, leaving m_pData untouched. Reading an
	// indeterminate pointer is UB and would be a hard MSan report.
	CNetChunk Chunk;
	mem_zero(&Chunk, sizeof(Chunk));
	SECURITY_TOKEN ResponseToken = 0;
	int Guard = 0;
	while(g_pNetServer->Recv(&Chunk, &ResponseToken) && ++Guard < 64)
	{
		if(Chunk.m_pData != nullptr && Chunk.m_DataSize > 0)
		{
			volatile unsigned char Sink = 0;
			for(int i = 0; i < Chunk.m_DataSize; i++)
				Sink ^= ((const unsigned char *)Chunk.m_pData)[i];
			(void)Sink;
		}

		// Answer the client, the way the server does for essentially every message it
		// receives (INPUTTIMING, PING_REPLY, the rcon and snapshot traffic - server.cpp:952,
		// :984, :1006, :2645 all funnel into m_NetServer.Send). This is the only way to
		// reach the OUTBOUND half of a connection: QueueChunk/QueueChunkEx, the resend ring
		// buffer, CNetChunkHeader::Pack, Flush, and - once outbound vital chunks exist - the
		// AckChunks/IsSeqInBackroom path that the client's own ack field drives.
		if(Chunk.m_ClientId >= 0 && Chunk.m_ClientId < g_pNetServer->MaxClients())
		{
			CNetChunk Reply;
			mem_zero(&Reply, sizeof(Reply));
			Reply.m_ClientId = Chunk.m_ClientId;
			Reply.m_pData = Chunk.m_pData;
			// AssertSizeSanity aborts above NET_MAX_CHUNK_SIZE; a chunk arriving from the
			// wire can be at most that, but clamp rather than rely on it.
			Reply.m_DataSize = std::min(Chunk.m_DataSize, (int)NET_MAX_CHUNK_SIZE);
			// Mirror the incoming chunk's vital flag, and always request a flush so the
			// batch has something to do. Non-vital replies keep the non-buffering arm of
			// QueueChunkEx alive.
			Reply.m_Flags = NETSENDFLAG_FLUSH;
			if(Chunk.m_Flags & NET_CHUNKFLAG_VITAL)
				Reply.m_Flags |= NETSENDFLAG_VITAL;
			if(Reply.m_DataSize > 0)
				g_pNetServer->Send(&Reply);
		}
	}

	g_pNetServer->EndFlushBatch();

	return 0;
}
