// libFuzzer target: sqlstr::EscapeLike and sqlstr::FuzzyString, the SQL LIKE helpers.
//
// EscapeLike is the reason to keep this target. It escapes the requesting player's NAME
// (scoreworker.cpp), so its input really is attacker-controlled on a live path, and it has
// a boundary its single call site hides: the loop is `while(DstPos + 2 < DstSize)`, so for
// a tiny DstSize it never runs, while the terminating `pDst[DstPos++] = '\0'` afterwards is
// unconditional. It gets an exactly sized destination below so that write is checked.
//
// FuzzyString is fuzzed alongside it, with one rule this harness got wrong once and must
// not get wrong again: THE BUFFER MUST CONTAIN A NUL WITHIN Size.
//
//   for(int i = 0; i < Size; i++)
//   {
//       if(!pString[i]) break;
//       pNewString[OutPos++] = pString[i];
//       if(pString[i] != '\\' && str_utf8_isstart(pString[i + 1]))
//           pNewString[OutPos++] = '%';
//   }
//
// The lookahead touches pString[Size] only at i == Size - 1, which the loop can only reach
// if pString[Size - 1] is non-zero. Both call sites (scoreworker.cpp:317 and :377) fill
// their char aFuzzyMap[128] with str_copy, which is `dst[0] = '\0'; strncat(dst, src,
// dst_size - 1)`, so the last byte is always '\0' and the break fires first. The read is in
// bounds for every input a caller can actually construct.
//
// An earlier version of this harness handed the function an exactly sized buffer that was
// deliberately NOT terminated, which broke that precondition and produced a one-byte
// heap-buffer-overflow report. It was filed as C3 and fixed upstream before Robyt3 pointed
// out in review that the guard above the lookahead already makes the read safe. The report
// was the harness's fault. Terminating the last byte, the way str_copy does, models the
// worst case a real caller can produce and keeps ASan's redzone right at pString[Size], so
// a genuine over-read would still be caught.
//
// Input: raw bytes, used verbatim as the buffer contents except for the forced terminator.
#include <base/dbg.h>
#include <base/mem.h>

#include <engine/server/sql_string_helpers.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

// Harnesses linked against the server object set replace main.cpp.
bool IsInterrupted()
{
	return false;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 1)
		return 0;

	// FuzzyString does `new char[Size * 4 - 1]`, so keep Size in the range a real caller
	// uses (a fixed array's sizeof) rather than fuzzing the allocation arithmetic itself.
	if(Size > 512)
		Size = 512;

	char *pStr = (char *)malloc(Size);
	if(pStr == nullptr)
		return 0;
	mem_copy(pStr, pData, Size);
	// str_copy always writes a terminator inside dst_size, so no caller can pass a buffer
	// without a NUL in [0, Size). Fuzzing that shape reports a bug the callers cannot have.
	pStr[Size - 1] = '\0';

	sqlstr::FuzzyString(pStr, (int)Size);
	free(pStr);

	// Exactly sized destination, so the unconditional terminating write is checked rather
	// than landing in slack.
	const int DstSize = pData[0] % 32;
	char *pSrc = (char *)malloc(Size + 1);
	char *pDst = DstSize > 0 ? (char *)malloc(DstSize) : nullptr;
	if(pSrc != nullptr)
	{
		mem_copy(pSrc, pData, Size);
		pSrc[Size] = '\0';
		if(pDst != nullptr)
			sqlstr::EscapeLike(pDst, pSrc, DstSize);
	}
	free(pSrc);
	free(pDst);
	return 0;
}
