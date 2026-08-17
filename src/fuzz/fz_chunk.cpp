// libFuzzer target: CPacketChunkUnpacker::UnpackNextChunk — the chunk layer.
//
// This sits directly downstream of fz_unpack_packet: it consumes the CNetPacketConstruct
// that UnpackPacket produces, and it runs on every connection-oriented packet from an
// accepted client, BEFORE CServer::ProcessClientPacket and therefore before any message-id
// dispatch, password check or rcon auth. The only gate is the address-derived security
// token, which an attacker obtains legitimately by handshaking.
//
// Why it is worth fuzzing (network.cpp):
//
//   * The chunk-skip loop carries a literal "// TODO: add checking here so we don't read
//     too far" and walks `pData = SkippedHeader.Unpack(pData, HeaderSplit); pData +=
//     SkippedHeader.m_Size;` with no comparison against pEnd inside the loop.
//   * CNetChunkHeader::Unpack reads pData[0], pData[1] and - for vital chunks - pData[2]
//     BEFORE the caller's only bounds check (`if(pData + Header.m_Size > pEnd)`).
//   * m_NumChunks is attacker-controlled up to 255 (UnpackPacket takes it straight from
//     byte 2 of the packet) and is never validated against m_DataSize.
//   * The 0.6 and 0.7 header encodings decode sizes into different domains through the same
//     struct: with Split = 6, m_Size reaches (0x3f << 6) | 0x3f = 4095, which is larger
//     than both NET_MAX_CHUNK_SIZE (1023) and m_aChunkData (1397 bytes). Two type systems
//     meeting on one buffer is the seam that produced the snapshot-delta bug.
//
// DETECTION LIMIT, stated plainly because it decides what this target can prove: inside
// CNetPacketConstruct, m_aChunkData is followed by m_aExtraData, and m_Data is the last
// member of the unpacker with several bytes of tail padding after it. An over-read of the
// chunk buffer therefore lands in a sibling member of the same object, which ASan cannot
// see - and when m_DataSize is below the array size the walk never leaves m_aChunkData at
// all. So this target proves the *logic* of the skip walk, not its memory safety; the
// ~11 bytes of slack are only reachable at the maximum packet size. Real 0.6 connections
// are stricter still: CNetConnection::Feed strips 4 token bytes off m_DataSize before the
// unpacker sees the packet.
//
// Input encoding:
//   byte 0 : bit 0   -> 0.7 (HeaderSplit 6) instead of 0.6 (4)
//            bit 1   -> SetUnknownSeq(), which lets vital chunks through the sequence check
//            bit 2-4 -> how many chunks to preload into the resend buffer that
//                       ResumeConnection copies (0..7)
//            bit 5-7 -> drop the client after this many chunks (0 = never), which is the
//                       CNetServer::Recv path that calls CPacketChunkUnpacker::Reset()
//   byte 1 : number of chunks claimed by the packet header
//   byte 2-3 : the connection's INCOMING sequence (m_Ack), little-endian
//   rest   : the chunk stream
//
// Spare BITS of byte 0 rather than new bytes on purpose: every byte added at the front
// reinterprets the whole existing corpus, and the corpus is the expensive part.
#include <base/dbg.h>
#include <base/mem.h>

#include <engine/shared/config.h>
#include <engine/shared/network.h>

#include <cstddef>
#include <cstdint>
#include <new>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 4)
		return 0;

	const bool Sixup = (pData[0] & 1) != 0;
	const bool UnknownSeq = (pData[0] & 2) != 0;
	const int NumResend = (pData[0] >> 2) & 7;
	const int DropAfter = (pData[0] >> 5) & 7;
	const int NumChunks = pData[1];
	// Two bytes: NET_MAX_SEQUENCE is 1 << 10, so a single byte cannot express three
	// quarters of the sequence space and `% NET_MAX_SEQUENCE` on it was a no-op.
	const int Ack = (pData[2] | (pData[3] << 8)) % NET_MAX_SEQUENCE;
	pData += 4;
	Size -= 4;

	auto *pPacket = new(std::nothrow) CNetPacketConstruct;
	auto *pConn = new(std::nothrow) CNetConnection;
	auto *pUnpacker = new(std::nothrow) CPacketChunkUnpacker;
	if(pPacket == nullptr || pConn == nullptr || pUnpacker == nullptr)
	{
		delete pPacket;
		delete pConn;
		delete pUnpacker;
		return 0;
	}

	// FeedPacket asserts on its preconditions, so satisfy them rather than feeding them:
	// an assert here would report harness misuse instead of a defect in the target.
	mem_zero(pPacket, sizeof(*pPacket));
	// No RESEND flag: UnpackNextChunk never reads m_Flags (FeedPacket only asserts that
	// CONNLESS/CONTROL are clear), so setting it changed nothing under test.
	pPacket->m_Flags = 0;
	pPacket->m_Ack = 0;
	pPacket->m_NumChunks = NumChunks > 0 ? NumChunks : 1;
	if(Size > sizeof(pPacket->m_aChunkData))
		Size = sizeof(pPacket->m_aChunkData);
	pPacket->m_DataSize = Size > 0 ? (int)Size : 1;
	mem_copy(pPacket->m_aChunkData, pData, Size);

	NETADDR Addr;
	mem_zero(&Addr, sizeof(Addr));
	Addr.type = NETTYPE_IPV4;
	Addr.port = 1;

	// Init takes the socket only to store it; nothing in the paths below sends, and
	// SignalResend just ORs a flag onto the connection's own construct buffer.
	pConn->Init(nullptr, false);

	// ResumeConnection, not DirectInit + SetSequence. SetSequence sets m_Sequence, the
	// OUTGOING counter, which nothing under test reads; UnpackNextChunk compares against
	// m_Ack, the INCOMING one. With DirectInit->Reset() zeroing m_Ack, the only chunk
	// sequence that could ever be accepted was 1, and the fuzzer's sequence byte was inert.
	// ResumeConnection is the only public route that sets m_Ack, and it also sets the peer
	// address, the token, m_Sixup and ONLINE state.
	// ResumeConnection's other half is a resend-buffer COPY (network_conn.cpp:583-591), and
	// with an empty buffer that loop never ran - 4 of its 22 edges were all this target
	// reached. In production the buffer is non-empty exactly when it matters: the only
	// caller is CNetServer::TryAcceptClient's slot takeover (network_server.cpp:775), which
	// moves a timed-out client's still-unacked chunks onto its new slot. Note the copy is
	// `mem_copy(pResend, pFirst, sizeof(CNetChunkResend) + pFirst->m_DataSize)` with NO
	// check on Allocate's result, unlike QueueChunkEx (network_conn.cpp:177-192) which does
	// check - so this is worth actually executing rather than assuming.
	static CStaticRingBuffer<CNetChunkResend, NET_CONN_BUFFERSIZE> s_ResendSrc;
	s_ResendSrc.Init();
	for(int i = 0; i < NumResend; i++)
	{
		// A vital chunk is capped at NET_MAX_CHUNK_SIZE by QueueChunk's own callers, so
		// anything larger could not be in a real resend buffer.
		const int DataSize = 1 + (int)((unsigned)(Ack * 2654435761u + (unsigned)i * 40503u) % NET_MAX_CHUNK_SIZE);
		CNetChunkResend *pResend = s_ResendSrc.Allocate((int)sizeof(CNetChunkResend) + DataSize);
		if(pResend == nullptr) // ring full - the harness must check even though ResumeConnection does not
			break;
		pResend->m_Sequence = (Ack + i) % NET_MAX_SEQUENCE;
		pResend->m_Flags = NET_CHUNKFLAG_VITAL;
		pResend->m_DataSize = DataSize;
		pResend->m_pData = (unsigned char *)(pResend + 1);
		pResend->m_FirstSendTime = 0;
		pResend->m_LastSendTime = 0;
		mem_zero(pResend->m_pData, DataSize);
	}
	pConn->ResumeConnection(&Addr, /*Sequence=*/0, Ack, /*SecurityToken=*/0, &s_ResendSrc, Sixup);

	if(UnknownSeq)
	{
		// Otherwise every vital chunk is dropped by the anti-spoof sequence check and the
		// interesting half of the loop is never reached.
		pConn->SetUnknownSeq();
	}

	CNetChunk Chunk;
	// CNetServer::Recv calls UnpackNextChunk at the TOP of its loop, before any packet has
	// been fed (network_server.cpp:610), so the !m_Valid early return (network.cpp:58-61) is
	// a normal production entry - it was the only way this target could not reach it.
	(void)pUnpacker->UnpackNextChunk(&Chunk);

	pUnpacker->FeedPacket(Addr, *pPacket, pConn, 0);

	int Unpacked = 0;
	while(pUnpacker->UnpackNextChunk(&Chunk))
	{
		// Read the chunk back the way CServer::ProcessClientPacket would, so a chunk that
		// points outside the buffer faults here rather than being silently ignored.
		volatile unsigned char Sink = 0;
		for(int i = 0; i < Chunk.m_DataSize; i++)
			Sink ^= ((const unsigned char *)Chunk.m_pData)[i];
		(void)Sink;

		// The caller may drop the client in response to a chunk; CNetServer::Recv then
		// abandons the rest of the packet with Reset() (network_server.cpp:615-622) and
		// keeps looping, which lands on the !m_Valid return above. Nothing else in the
		// server calls Reset, so without this the function was unreachable (0/1 edges).
		if(DropAfter != 0 && ++Unpacked >= DropAfter)
			pUnpacker->Reset();
	}

	delete pPacket;
	delete pConn;
	delete pUnpacker;
	return 0;
}
