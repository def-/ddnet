// libFuzzer target: CNetBase::UnpackPacket — the first thing the server does with a
// datagram from an unauthenticated peer. Covers header parsing, the size/flag checks and
// the huffman decompression path.
//
// Deliberately stateless: the chunk layer below this (CPacketChunkUnpacker) needs a live
// CNetConnection, and CPacketChunkUnpacker::FeedPacket has dbg_asserts on its
// preconditions, so feeding it raw fuzz output would abort on harness misuse rather than
// on a real defect. The chunk/connection layer is covered by the stateful session fuzzer
// (tools/fuzz/fuzz_session.py) instead.
//
// Byte 0 selects 0.6 vs 0.7 interpretation; the rest is the datagram.
#include <base/dbg.h>
#include <base/mem.h>

#include <engine/shared/network.h>

#include <cstddef>
#include <cstdint>

static bool s_Init = false;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(!s_Init)
	{
		CNetBase::Init();
		s_Init = true;
	}
	if(Size < 2)
		return 0;

	const bool WantSixup = (pData[0] & 1) != 0;
	pData += 1;
	Size -= 1;

	// net_udp_recv truncates at NET_MAX_PACKETSIZE, so anything longer is unreachable in
	// production and would only produce findings the network stack makes impossible.
	if(Size > NET_MAX_PACKETSIZE)
		Size = NET_MAX_PACKETSIZE;

	// UnpackPacket writes through its buffer argument; give it a private copy so ASan can
	// see an overflow of exactly the buffer the real caller supplies.
	unsigned char aBuffer[NET_MAX_PACKETSIZE];
	mem_copy(aBuffer, pData, Size);

	CNetPacketConstruct Packet;
	bool Sixup = WantSixup;
	SECURITY_TOKEN SecurityToken = 0;
	SECURITY_TOKEN ResponseToken = 0;

	if(CNetBase::UnpackPacket(aBuffer, (int)Size, &Packet, Sixup, &SecurityToken, &ResponseToken) != 0)
		return 0;

	(void)CNetBase::IsValidConnectionOrientedPacket(&Packet);

	// Touch the decoded payload so a bogus m_DataSize is caught by ASan rather than
	// silently ignored.
	if(Packet.m_DataSize > 0)
	{
		volatile unsigned char Sink = 0;
		for(int i = 0; i < Packet.m_DataSize; i++)
			Sink ^= Packet.m_aChunkData[i];
		(void)Sink;
	}
	return 0;
}
