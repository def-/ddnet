#include <base/mem.h>
#include <base/net.h>
#include <base/secure.h>

#include <engine/shared/network.h>

#include <gtest/gtest.h>

#include <chrono>

using namespace std::chrono_literals;

TEST(Net, Ipv4AndIpv6Work)
{
	NETADDR Bindaddr = {};
	NETSOCKET Socket1;
	NETSOCKET Socket2;

	Bindaddr.type = NETTYPE_IPV4 | NETTYPE_IPV6;
	Socket2 = net_udp_create(Bindaddr);
	do
	{
		Bindaddr.port = secure_rand_below(65535 - 1024) + 1024;
	} while(!(Socket1 = net_udp_create(Bindaddr)));

	NETADDR LocalhostV4;
	NETADDR LocalhostV6;
	NETADDR TargetV4;
	NETADDR TargetV6;
	ASSERT_FALSE(net_addr_from_str(&LocalhostV4, "127.0.0.1"));
	ASSERT_FALSE(net_addr_from_str(&LocalhostV6, "[::1]"));
	TargetV4 = LocalhostV4;
	TargetV6 = LocalhostV6;
	TargetV4.port = Bindaddr.port;
	TargetV6.port = Bindaddr.port;

	NETADDR Addr;
	unsigned char *pData;

	EXPECT_EQ(net_udp_send(Socket2, &TargetV4, "abc", 3), 3);

	EXPECT_EQ(net_socket_read_wait(Socket1, 10s), 1);
	ASSERT_EQ(net_udp_recv(Socket1, &Addr, &pData), 3);
	Addr.port = 0;
	EXPECT_EQ(Addr, LocalhostV4);
	EXPECT_EQ(mem_comp(pData, "abc", 3), 0);

	EXPECT_EQ(net_udp_send(Socket2, &TargetV6, "def", 3), 3);

	EXPECT_EQ(net_socket_read_wait(Socket1, 10s), 1);
	ASSERT_EQ(net_udp_recv(Socket1, &Addr, &pData), 3);
	Addr.port = 0;
	EXPECT_EQ(Addr, LocalhostV6);
	EXPECT_EQ(mem_comp(pData, "def", 3), 0);

	net_udp_close(Socket1);
	net_udp_close(Socket2);
}

// A packet that is received with the maximum packet size must fit into the
// packet construct without overflowing its data buffer.
TEST(Net, UnpackMaxSizePacket)
{
	CNetBase::Init();

	unsigned char aPacket[NET_MAX_PACKETSIZE];
	mem_zero(aPacket, sizeof(aPacket));
	aPacket[0] = 0; // no flags, in particular not compressed
	aPacket[1] = 0; // ack
	aPacket[2] = 1; // one chunk

	CNetPacketConstruct Packet;
	bool Sixup = false;
	SECURITY_TOKEN SecurityToken;
	ASSERT_EQ(CNetBase::UnpackPacket(aPacket, sizeof(aPacket), &Packet, Sixup, &SecurityToken), 0);
	EXPECT_LE(Packet.m_DataSize, (int)sizeof(Packet.m_aChunkData));
}

// Sending a packet with the maximum amount of chunk data must not exceed the
// maximum packet size, also when the security token is appended to the data.
TEST(Net, SendMaxSizePacket)
{
	CNetBase::Init();

	NETADDR BindAddr = {};
	BindAddr.type = NETTYPE_IPV4;
	NETSOCKET Receiver;
	do
	{
		BindAddr.port = secure_rand_below(65535 - 1024) + 1024;
	} while(!(Receiver = net_udp_create(BindAddr)));
	const int ReceiverPort = BindAddr.port;
	BindAddr.port = 0;
	NETSOCKET Sender = net_udp_create(BindAddr);
	ASSERT_NE(Sender, nullptr);

	NETADDR Target;
	ASSERT_FALSE(net_addr_from_str(&Target, "127.0.0.1"));
	Target.port = ReceiverPort;

	CNetPacketConstruct Packet;
	mem_zero(&Packet, sizeof(Packet));
	Packet.m_NumChunks = 1;
	Packet.m_DataSize = NET_MAX_CHUNKDATASIZE;
	// incompressible data, so that the uncompressed data is sent
	secure_random_fill(Packet.m_aChunkData, Packet.m_DataSize);

	CNetBase::SendPacket(Sender, &Target, &Packet, 0x12345678);
	EXPECT_LE(Packet.m_DataSize, (int)sizeof(Packet.m_aChunkData));

	NETADDR From;
	unsigned char *pData;
	ASSERT_EQ(net_socket_read_wait(Receiver, 10s), 1);
	const int Size = net_udp_recv(Receiver, &From, &pData);
	EXPECT_GT(Size, 0);
	EXPECT_LE(Size, NET_MAX_PACKETSIZE);

	net_udp_close(Sender);
	net_udp_close(Receiver);
}
