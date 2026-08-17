// libFuzzer target: the 0.7 <-> 0.6 translation layer.
//
// DDNet speaks two protocols on one socket, and every 0.7 client's data is rewritten into
// 0.6 shapes before the game code sees it. That rewrite is a natural place for bugs: it is
// the one part of the server where an attacker picks which of two different type systems
// their bytes are interpreted under. C1 came from exactly that dual-protocol seam.
//
// Three pieces are standalone enough to fuzz in isolation:
//
//   1. CTeeInfo::ToSixup / FromSixup — converts between the 0.6 skin representation (one
//      name + two colours) and the 0.7 one (six part names + six colours + six flags), and
//      does arithmetic on client-supplied colour ints in ColorHSLA::UnclampLighting/Pack.
//   2. CTeeInfo's six-part-array constructor, which is the real entry point: the 0.7
//      Cl_StartInfo and Cl_SkinChange handlers build a CTeeInfo straight out of
//      pMsg->m_apSkinPartNames (gamecontext.cpp:2124 and :2143) and only then call
//      FromSixup. Those names are unpacker-owned strings of unbounded length, and the
//      constructor str_copy's each into a char[protocol7::MAX_SKIN_LENGTH], so the
//      truncation (and str_utf8_fix_truncation under it) is on the live path.
//   3. The translation tables and the hand-written glue: the generated Msg_/Obj_
//      SixToSeven/SevenToSix lookups, plus protocolglue.cpp's PacketFlags_*, PlayerFlags_*,
//      GameFlags_ClampToSix and PickupType_SevenToSix.
//
// (An earlier version of this comment claimed the two skin arrays had DIFFERENT lengths.
// They do not: protocol.h:111 and protocol7.h:73 both define MAX_SKIN_LENGTH = 24. The
// interesting size mismatch is between the wire string, which is bounded only by the chunk,
// and the 24-byte field it is copied into.)
//
// Deliberately NOT here:
//   * CGameContext::PreProcessMsg — needs a real CGameContext, so it stays with the
//     end-to-end harness.
//   * PickupType_SixToSeven over arbitrary ints. Its default arms are
//     dbg_assert_failed("invalid type %d"), i.e. a deliberate abort, and its only caller
//     (CGameContext::SnapPickup, gamecontext.cpp:604) passes the type of a pickup ENTITY,
//     which comes from the map, not from a client. Feeding it fuzzer ints would report an
//     assert that no client can reach. It is driven over its valid domain only, which is
//     enough to catch a table/enum regression without manufacturing a fake finding.
//
// Reachability: direct. A 0.7 client sends its skin parts in CNetMsg_Cl_StartInfo and
// CNetMsg_Cl_SkinChange, entirely attacker-controlled, and the message ids go through the
// tables on every packet.
//
// What the wire can actually deliver: every one of those names arrives through
// CUnpacker::GetString(SANITIZE_CC | SKIP_START_WHITESPACES) (protocol7.cpp:1000-1005).
// That rejects invalid UTF-8 outright (packer.cpp:192), replaces control characters with
// spaces, and skips leading whitespace. CReader::String below applies the same three rules,
// so a crash found here cannot be an artefact of bytes no client could send.
//
// Input encoding (the prefix is unchanged from the first version of this target, so the
// existing corpus keeps its meaning):
//   byte 0 : bit 0 -> start from the 0.6 side (ToSixup) or the 0.7 side (FromSixup)
//            bit 1 -> run the conversion twice, to catch state that does not settle
//            bit 2 -> build the CTeeInfo through the six-part-array constructor (the real
//                     0.7 path) instead of assigning the fields directly
//   byte 1 : bitmask of m_aUseCustomColors, plus bit 7 for m_UseCustomColor
//   then   : 2 x int32 (0.6 body/feet colour), 6 x int32 (0.7 part colours)
//   rest   : NUL-separated skin names, first for 0.6 then six for 0.7
#include <base/dbg.h>
#include <base/mem.h>
#include <base/str.h>

#include <engine/shared/protocol.h>
#include <engine/shared/protocol7.h>
#include <engine/shared/protocolglue.h>

#include <generated/protocol.h>
#include <generated/protocol7.h>
#include <generated/protocolglue.h>

#include <game/server/teeinfo.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

// Harnesses linked against the game-server object set replace main.cpp.
bool IsInterrupted()
{
	return false;
}

namespace
{

	// The longest name the 0.7 unpacker will hand over is bounded only by the chunk size, and
	// the whole point of the array constructor is that IT does the truncation. Keep a buffer
	// big enough that the harness is never the thing that truncates first.
	enum
	{
		MAX_WIRE_STRING = 256
	};

	class CReader
	{
	public:
		CReader(const uint8_t *pData, size_t Size) :
			m_pData(pData), m_Size(Size), m_Pos(0) {}

		uint8_t U8() { return m_Pos < m_Size ? m_pData[m_Pos++] : 0; }
		int32_t I32()
		{
			uint32_t v = U8();
			v |= (uint32_t)U8() << 8;
			v |= (uint32_t)U8() << 16;
			v |= (uint32_t)U8() << 24;
			return (int32_t)v;
		}

		// Read a NUL-terminated run of input and put it through the same three filters
		// CUnpacker::GetString(SANITIZE_CC | SKIP_START_WHITESPACES) applies, so pDst holds
		// exactly what the message handler would have been given. Invalid UTF-8 yields "",
		// which is what GetString returns once it sets m_Error (packer.cpp:192-196).
		// pDst must have room for MAX_WIRE_STRING bytes.
		void WireString(char *pDst)
		{
			char aTmp[MAX_WIRE_STRING];
			int i = 0;
			while(m_Pos < m_Size && i < (int)sizeof(aTmp) - 1)
			{
				const char c = (char)m_pData[m_Pos++];
				if(c == '\0')
					break;
				aTmp[i++] = c;
			}
			aTmp[i] = '\0';
			if(!str_utf8_check(aTmp))
			{
				pDst[0] = '\0';
				return;
			}
			str_sanitize_cc(aTmp);
			// str_utf8_skip_whitespaces returns a pointer INTO aTmp, so copy from there.
			str_copy(pDst, str_utf8_skip_whitespaces(aTmp), MAX_WIRE_STRING);
		}

	private:
		const uint8_t *m_pData;
		size_t m_Size;
		size_t m_Pos;
	};

	// The generated tables are bounds-guarded inline lookups and the protocolglue.cpp helpers
	// are pure bit/enum mappings; the point of driving them is to pin that down as a fact
	// rather than an assumption, and to catch it if a regeneration ever drops a guard.
	// Boundary values are included explicitly because a fuzzer reaches them slowly.
	void ExerciseTables(int Seed)
	{
		// Negate through unsigned: -Seed is undefined for INT_MIN, and tripping UBSan in the
		// harness would report the harness rather than the target.
		// 0..8 covers every protocol7::PICKUP_* arm of PickupType_SevenToSix (protocolglue.cpp:84)
		// plus its default; 20/21, 24/25, 39/40 and 55/56 are the last-valid/first-invalid index
		// of each of the four generated tables.
		const int aProbes[] = {Seed, (int)(0u - (unsigned)Seed), 0, 1, 2, 3, 4, 5, 6, 7, 8, -1,
			20, 21, 24, 25, 39, 40, 55, 56,
			std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
		volatile int Sink = 0;
		for(int Probe : aProbes)
		{
			Sink ^= Msg_SixToSeven(Probe);
			Sink ^= Msg_SevenToSix(Probe);
			Sink ^= Obj_SixToSeven(Probe);
			Sink ^= Obj_SevenToSix(Probe);

			// Hand-written glue. All of these are total functions over int: they test
			// individual flag bits or switch with a harmless default, so an arbitrary
			// attacker-supplied value is exactly what they are built to survive.
			// PacketFlags_* run on every sixup packet header (network.cpp:249 and :353) and
			// PlayerFlags_SevenToSix on every 0.7 input message (gamecontext.cpp:1567).
			Sink ^= PacketFlags_SixToSeven(Probe);
			Sink ^= PacketFlags_SevenToSix(Probe);
			Sink ^= PlayerFlags_SixToSeven(Probe);
			Sink ^= PlayerFlags_SevenToSix(Probe);
			Sink ^= GameFlags_ClampToSix(Probe);

			int Type6 = 0, SubType6 = 0;
			PickupType_SevenToSix(Probe, Type6, SubType6);
			Sink ^= Type6 ^ SubType6;
		}

		// PickupType_SixToSeven only over the pairs a pickup entity can actually hold - see
		// the header comment. Anything else is an intentional dbg_assert_failed.
		static const int s_aValidPickups[][2] = {
			{POWERUP_WEAPON, WEAPON_HAMMER}, {POWERUP_WEAPON, WEAPON_GUN},
			{POWERUP_WEAPON, WEAPON_SHOTGUN}, {POWERUP_WEAPON, WEAPON_GRENADE},
			{POWERUP_WEAPON, WEAPON_LASER}, {POWERUP_WEAPON, WEAPON_NINJA},
			{POWERUP_NINJA, 0}, {POWERUP_HEALTH, 0}, {POWERUP_ARMOR, 0},
			{POWERUP_ARMOR_SHOTGUN, 0}, {POWERUP_ARMOR_GRENADE, 0},
			{POWERUP_ARMOR_NINJA, 0}, {POWERUP_ARMOR_LASER, 0}};
		for(const auto &Pickup : s_aValidPickups)
			Sink ^= PickupType_SixToSeven(Pickup[0], Pickup[1]);

		(void)Sink;
	}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 8)
		return 0;

	CReader Reader(pData, Size);
	const uint8_t Mode = Reader.U8();
	const uint8_t ColorFlags = Reader.U8();

	const int ColorBody = Reader.I32();
	const int ColorFeet = Reader.I32();

	int aSkinPartColors[protocol7::NUM_SKINPARTS];
	int aUseCustomColors[protocol7::NUM_SKINPARTS];
	for(int i = 0; i < protocol7::NUM_SKINPARTS; i++)
	{
		aSkinPartColors[i] = Reader.I32();
		// On the wire this field is a plain GetInt (protocol7.cpp:1006-1011), so it is any
		// int at all, not a 0/1 flag - the constructor's int -> bool narrowing is part of
		// what is being tested. Derive a non-trivial value from the colour so the corpus'
		// existing bytes still decide the flag.
		aUseCustomColors[i] = ((ColorFlags >> i) & 1) ? aSkinPartColors[i] | 1 : 0;
	}

	char aSkinName[MAX_WIRE_STRING];
	char aaSkinPartNames[protocol7::NUM_SKINPARTS][MAX_WIRE_STRING];
	Reader.WireString(aSkinName);
	for(int i = 0; i < protocol7::NUM_SKINPARTS; i++)
		Reader.WireString(aaSkinPartNames[i]);

	CTeeInfo Info;
	if(Mode & 4)
	{
		// The real 0.7 path: hand the unbounded wire strings to the array constructor and
		// let IT do the str_copy into char[protocol7::MAX_SKIN_LENGTH].
		const char *apSkinPartNames[protocol7::NUM_SKINPARTS];
		for(int i = 0; i < protocol7::NUM_SKINPARTS; i++)
			apSkinPartNames[i] = aaSkinPartNames[i];
		Info = CTeeInfo(apSkinPartNames, aUseCustomColors, aSkinPartColors);
		// The constructor leaves the 0.6 half at its default member initialisers, exactly
		// as gamecontext.cpp:2124 does before calling FromSixup.
	}
	else
	{
		Info.m_UseCustomColor = (ColorFlags & 0x80) != 0;
		Info.m_ColorBody = ColorBody;
		Info.m_ColorFeet = ColorFeet;
		for(int i = 0; i < protocol7::NUM_SKINPARTS; i++)
		{
			Info.m_aUseCustomColors[i] = (ColorFlags >> i) & 1;
			Info.m_aSkinPartColors[i] = aSkinPartColors[i];
		}
		str_copy(Info.m_aSkinName, aSkinName, sizeof(Info.m_aSkinName));
		for(int i = 0; i < protocol7::NUM_SKINPARTS; i++)
			str_copy(Info.m_aaSkinPartNames[i], aaSkinPartNames[i], sizeof(Info.m_aaSkinPartNames[i]));
	}

	const int Rounds = (Mode & 2) ? 2 : 1;
	for(int r = 0; r < Rounds; r++)
	{
		if(Mode & 1)
			Info.ToSixup();
		else
			Info.FromSixup();
	}

	// Read every field back so a conversion that leaves a field unterminated shows up as an
	// over-read here instead of being carried silently into a snapshot later.
	volatile char Sink = 0;
	Sink ^= Info.m_aSkinName[0];
	Sink ^= (char)str_length(Info.m_aSkinName);
	for(int i = 0; i < protocol7::NUM_SKINPARTS; i++)
		Sink ^= (char)str_length(Info.m_aaSkinPartNames[i]);
	(void)Sink;

	ExerciseTables(Info.m_ColorBody);
	return 0;
}
