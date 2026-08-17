// libFuzzer target: CNameBans::IsBanned and the confusables/skeleton machinery under it.
//
// Reachability: direct and unauthenticated beyond the server password. Every Cl_StartInfo
// and Cl_ChangeInfo reaches CServer::SetClientNameImpl (server.cpp), whose first action is
// `m_NameBans.IsBanned(pNameRequest)`. The clan path (SetClientClanImpl) is the same shape.
//
// Why this is worth fuzzing (name_ban.cpp:67-83):
//
//   * Three fixed-size stack buffers whose sizes are derived by arithmetic on constants:
//       char aTrimmed[MAX_NAME_LENGTH]
//       int  aSkeleton[MAX_NAME_SKELETON_LENGTH]          // MAX_NAME_LENGTH * 4
//       int  aBuffer[MAX_NAME_SKELETON_LENGTH * 2 + 2]
//     The last is EXACTLY the minimum str_utf32_dist_buffer asserts on
//     (`dbg_assert(buf_len >= (a_len + 1) + (b_len + 1))`). Zero slack: any change to how
//     skeletons expand turns that assert into a live abort, and dbg_assert is active in
//     release builds.
//   * str_utf8_to_skeleton expands one codepoint into several via the confusables tables,
//     i.e. an input-driven expansion feeding a fixed-size destination.
//   * The banned names are themselves fuzzer-chosen here, because both sides of the
//     distance computation are variable-length and the assert bounds their SUM.
//
// A previous version also called str_utf8_to_skeleton directly and asserted the returned
// length fitted the buffer. That assert could never fire: the function's loop is bounded by
// buf_len and returns the loop counter, so it clamps by construction. It has been removed
// rather than left in looking like a check.
//
// Input encoding:
//   byte 0        : number of bans, 0-3
//   byte 1        : mode - bit 0 asks for the candidate to embed a ban name
//   per ban       : u8 distance, u8 is_substring, NUL-terminated name
//   remaining     : the candidate name to test
#include <base/dbg.h>
#include <base/mem.h>
#include <base/str.h>

#include <engine/server/name_ban.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

// Harnesses linked against the server object set replace main.cpp.
bool IsInterrupted()
{
	return false;
}

namespace
{

	class CReader
	{
	public:
		CReader(const uint8_t *pData, size_t Size) :
			m_pData(pData), m_Size(Size), m_Pos(0) {}

		uint8_t U8() { return m_Pos < m_Size ? m_pData[m_Pos++] : 0; }
		// Read a NUL-terminated run into a bounded scratch string.
		void String(char *pDst, size_t DstSize)
		{
			size_t i = 0;
			while(m_Pos < m_Size && i + 1 < DstSize)
			{
				const char c = (char)m_pData[m_Pos++];
				if(c == '\0')
					break;
				pDst[i++] = c;
			}
			pDst[i] = '\0';
		}
		size_t Left() const { return m_Pos < m_Size ? m_Size - m_Pos : 0; }
		const uint8_t *Rest() const { return m_pData + m_Pos; }

	private:
		const uint8_t *m_pData;
		size_t m_Size;
		size_t m_Pos;
	};

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 4)
		return 0;

	CReader Reader(pData, Size);
	const int NumBans = Reader.U8() % 4;
	const uint8_t Mode = Reader.U8();

	CNameBans Bans; // no constructor dependencies; InitConsole is optional
	char aaBanNames[4][MAX_NAME_LENGTH] = {};

	for(int i = 0; i < NumBans; i++)
	{
		const int Distance = (int)(int8_t)Reader.U8(); // deliberately allows negative
		const bool IsSubstring = (Reader.U8() & 1) != 0;
		// MAX_NAME_LENGTH, not 128: CNameBan's constructor str_copy's into a
		// char[MAX_NAME_LENGTH], so every byte past 15 was read out of the input and thrown
		// away - up to 336 wasted bytes across three bans, and a ban name with no interior
		// NUL swallowed input that should have been the candidate.
		char aName[MAX_NAME_LENGTH];
		Reader.String(aName, sizeof(aName));
		str_copy(aaBanNames[i], aName, sizeof(aaBanNames[i]));
		Bans.Ban(aName, "fuzz", Distance, IsSubstring);
	}

	// The candidate is the remaining bytes. Give it an exactly-sized NUL-terminated heap
	// block so a read past the end is a hard ASan error - IsBanned walks it with
	// str_utf8_skip_whitespaces and str_utf8_find_nocase, which are utf-8 decoders.
	const size_t CandidateLen = Reader.Left();
	char *pCandidate = (char *)malloc(CandidateLen + 1);
	if(pCandidate == nullptr)
		return 0;
	mem_copy(pCandidate, Reader.Rest(), CandidateLen);
	pCandidate[CandidateLen] = '\0';

	// Only valid UTF-8 can reach this on a real server: names arrive as NetStringStrict,
	// which CUnpacker::GetString rejects unless str_utf8_check passes, and control bytes are
	// already replaced by str_sanitize_cc. Without the gate, any crash found here has to be
	// re-checked by hand before it can be called a finding.
	if(!str_utf8_check(pCandidate))
	{
		free(pCandidate);
		return 0;
	}

	// Optionally build the candidate around one of the ban names. The substring branch
	// compares two independently-mutated regions of the input, which libFuzzer has no
	// gradient to correlate, so without this the IsSubstring half of IsBanned only ever
	// executed its failure path (or matched trivially on an empty needle).
	if(NumBans > 0 && (Mode & 1) != 0)
	{
		char aWrapped[256];
		str_format(aWrapped, sizeof(aWrapped), "x%sy", aaBanNames[Mode % NumBans]);
		Bans.IsBanned(aWrapped);
	}

	Bans.IsBanned(pCandidate);

	free(pCandidate);
	return 0;
}
