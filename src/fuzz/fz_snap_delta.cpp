// libFuzzer target: CSnapshotDelta::CreateDelta / UnpackDelta.
//
// This models the one memory-safety bug round 3 found. CreateDelta (snapshot.cpp:341-355)
// matches items between two snapshots BY KEY ALONE and takes the item size from pTo, then
// DiffItem reads that many ints out of pFrom's item. UnpackDelta has the corresponding
// guard (snapshot.cpp:611, "return -207"); CreateDelta does not. A same-key/different-size
// pair therefore reads past pFrom's exact-sized allocation.
//
// In production that pair arises when a client slot's protocol flips (0.6 <-> 0.7) while
// its snapshot storage is retained, because the two protocols assign different structs to
// the same internal type number. Rather than re-enact that whole path, this harness builds
// both snapshots with the REAL CSnapshotBuilder from fuzzer-chosen (type, id, size) items
// - so both are structurally valid, exactly as the server's own builder guarantees, and
// any finding is about CreateDelta rather than about garbage input.
//
// Wire format of the fuzz input:
//   u16 num_from_items, u16 num_to_items, u8 modes
//        bit 0/1 = sixup per side
//        bit 2   = draw sizes independently, which re-enables the C1 shape
//        bit 3   = register static item sizes, the way CGameContext::OnInit does
//        bit 4   = allow UUID-extended item types (DDNetCharacter and friends)
//        bit 5   = keep reading past the end of the input by wrapping, so the item loop is
//                  bounded by NewItemRaw's MAX_SIZE check instead of by -max_len
//   then per item: u8 type, u8 id, [u8 size_in_ints], u8 diff_mask, u8 fill
//
// Keys are drawn from a SMALL pool on purpose. An earlier version drew type from
// [0, MAX_TYPE] and id from [0, MAX_ID], a 2^31 key space, so two snapshots of ~250 items
// shared a key with probability ~3e-5 - meaning CreateDelta's entire diff path
// (aPastIndices, DiffItem, the same-key size guard) and UnpackDelta's UndiffItem were
// essentially never executed, and any "N million executions clean" claim about them was
// unsupported. Item CONTENT is likewise derived from the key, with a fuzzer-chosen bitmask
// selecting which ints differ, so "same key, most ints unchanged" - the normal shape of a
// real snapshot update, and the only one that exercises UndiffItem's unchanged-int branch -
// is the default rather than something the fuzzer must stumble onto.
//
// STATIC ITEM SIZES (mode bit 3). CSnapshotDelta has a second wire encoding that this
// harness could not reach at all: when m_aItemSizes[Type] is non-zero, CreateDelta OMITS
// the size word (snapshot.cpp:366-367, 373) and UnpackDelta takes the size from the table
// instead of the stream (snapshot.cpp:587-588). The real server always populates that
// table - CGameContext::OnInit calls SnapSetStaticsize for every 0.6 object type and for
// the first 23 0.7 types (gamecontext.cpp:4188-4199) - so the size-omitting encoding is
// what a live server actually emits for the majority of its snapshot items, and it was
// 100% dark here. Mirroring that init requires the item's size to AGREE with the table,
// which is exactly the invariant the server's own SnapNewItem<T> maintains, so items of a
// statically-sized type take their size from the table and ignore the mismatch mode bit;
// the deliberate-mismatch shape stays available on every type that has no static size.
//
// UUID-EXTENDED TYPES (mode bit 4). NewItemRaw's Extended branch (snapshot.cpp:921-929),
// AddExtendedItemType and GetExtendedItemTypeIndex were likewise never executed, because
// the harness only ever asked for types 0..63. The server emits extended items constantly
// (NETOBJTYPE_DDNETCHARACTER and the other 17 entries between OFFSET_GAME_UUID and
// OFFSET_NETMSGTYPE_UUID), and each one makes the builder inject an extra NETOBJTYPE_EX
// item carrying the UUID and rewrite the stored type to MAX_TYPE-Index - a second type
// domain flowing through the same delta. Requesting them by their UUID id lets that
// happen for real instead of being simulated.
#include <base/dbg.h>
#include <base/mem.h>

#include <engine/shared/compression.h>
#include <engine/shared/snapshot.h>
#include <engine/shared/uuid_manager.h>

#include <generated/protocol.h>
#include <generated/protocol7.h>
#include <generated/protocolglue.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <new>

namespace
{

	// The UUID-extended object type ids, i.e. everything between the two enums in
	// generated/protocol.h. 18 entries at the time of writing.
	const int FIRST_EX_OBJ = (int)__NETOBJTYPE_UUID_HELPER + 1;
	const int NUM_EX_OBJ = (int)OFFSET_NETMSGTYPE_UUID - FIRST_EX_OBJ;

	// CSnapshotDelta::MAX_NETOBJSIZES is private; SetStaticsize asserts ItemType < it.
	const int MAX_STATIC_TYPES = 64;
	// gamecontext.cpp:4195 - 0.7 only registers the types that existed in the first 0.7
	// release, so that newer items do not break the delta against old clients.
	const int OLD_NUM_NETOBJTYPES7 = 23;

	short g_aStaticSize6[MAX_STATIC_TYPES];
	short g_aStaticSize7[MAX_STATIC_TYPES];

	// [0] no static sizes (the size-carrying encoding), [1] 0.6 sizes, [2] 0.7 sizes.
	// Three separate objects because m_aItemSizes is per-CSnapshotDelta state and the server
	// likewise keeps m_SnapshotDelta and m_SnapshotDeltaSixup apart (server.cpp:4567-4575).
	CSnapshotDelta *g_apDelta[3];

	// ---- optional statistics (FZ_SNAP_STATS=1) ---------------------------------------
	// Edge coverage cannot answer "is the round trip actually completing?" or "does
	// UndiffItem's unchanged-int branch run?", because those are properties of the DATA, not
	// of which blocks executed. Count them directly instead of guessing from a line report.
	bool g_Stats = false;
	// FZ_SNAP_STRICT=1 hands CreateDelta a destination of exactly CSnapshot::MAX_SIZE, which is
	// what CServer::DoSnapshot gives it (server.cpp:1110, on the STACK). That is the faithful
	// shape, but CreateDelta has no bound check at all and overruns it on a near-full snapshot,
	// so leaving it on would make every campaign die on the same input within seconds and
	// explore nothing else. The default is an oversized destination plus an explicit
	// DeltaSize > MAX_SIZE check, the same "count it, do not trap on it" treatment the known
	// -206/-207 case already gets. FZ_SNAP_STRICT=1 removes BOTH harness workarounds - the
	// oversized delta destination and the item-capacity accounting that compensates for
	// NewItemRaw not charging for the NETOBJTYPE_EX record it injects - so the target behaves
	// exactly like the server and both findings reproduce as hard crashes.
	bool g_Strict = false;
	struct
	{
		long Inputs, Built, DeltaZero, DeltaPos;
		long UnpackOk, UnpackSizeReject, UnpackOtherReject, RoundTripOk;
		long SharedKeys, SharedSameSize, DiffItemNonZero, Emitted, UndiffZeroBranch;
		long StaticSizedItems, ExtendedItems;
		// The header comment claims a near-full snapshot can push CreateDelta's output past the
		// fixed CSnapshot::MAX_SIZE stack buffer at server.cpp:1110. That claim is only worth
		// anything with a number next to it, so record how close the campaign actually gets.
		int MaxFrom, MaxTo, MaxDelta;
		long DeltaOverMaxSize;
	} g_S;

	void PrintStats()
	{
		fprintf(stderr,
			"SNAPSTAT inputs=%ld built=%ld (%.1f%%)\n"
			"SNAPSTAT delta: zero=%ld pos=%ld\n"
			"SNAPSTAT unpack: ok=%ld sizereject(-206/-207)=%ld otherreject=%ld roundtrip_ok=%ld\n"
			"SNAPSTAT diff: shared_keys=%ld same_size=%ld difftem_size>0=%ld emitted=%ld undiff_zero_branch=%ld\n"
			"SNAPSTAT items: static_sized=%ld extended=%ld\n"
			"SNAPSTAT max bytes: from=%d to=%d delta=%d (CSnapshot::MAX_SIZE=%d)\n"
			"SNAPSTAT delta over MAX_SIZE: %ld\n",
			g_S.Inputs, g_S.Built, g_S.Inputs ? 100.0 * g_S.Built / g_S.Inputs : 0.0,
			g_S.DeltaZero, g_S.DeltaPos,
			g_S.UnpackOk, g_S.UnpackSizeReject, g_S.UnpackOtherReject, g_S.RoundTripOk,
			g_S.SharedKeys, g_S.SharedSameSize, g_S.DiffItemNonZero, g_S.Emitted,
			g_S.UndiffZeroBranch,
			g_S.StaticSizedItems, g_S.ExtendedItems,
			g_S.MaxFrom, g_S.MaxTo, g_S.MaxDelta, (int)CSnapshot::MAX_SIZE,
			g_S.DeltaOverMaxSize);
	}

	class CReader
	{
	public:
		CReader(const uint8_t *pData, size_t Size) :
			m_pData(pData), m_Size(Size), m_Pos(0) {}

		bool Left(size_t n) const { return m_Pos + n <= m_Size; }
		// Past the end this WRAPS, numbering the lap into the value, instead of returning zeros.
		//
		// Returning zeros made the near-full snapshot - the only shape in which CreateDelta can
		// overrun the fixed CSnapshot::MAX_SIZE destination that server.cpp:1110 gives it -
		// structurally unreachable, and silently so. Each item costs 4-5 input bytes and
		// libFuzzer's default -max_len is 4096, so the item loop ran out of input after ~500
		// items per snapshot; worse, every item built after that decoded to the same
		// (type 0, id 0) key and was dropped as a duplicate, so the snapshot simply stopped
		// growing. Measured with the zero-filling reader, over the entire campaign corpus: the
		// largest delta produced was 20108 bytes against a 65536-byte buffer, 31% of the way.
		// Wrapping lets the item count be bounded by NewItemRaw's own MAX_SIZE check, which is
		// what bounds it on a real server.
		// Wrapping is a MODE, not the default: a wrapped input builds snapshots until
		// NewItemRaw's own MAX_SIZE check stops it, which is ~10x the work per execution and
		// ~20x fewer executions per second. Letting the fuzzer choose keeps the cheap
		// small-snapshot regime for finding logic bugs and the expensive full-snapshot regime
		// for finding size bugs, instead of trading one for the other.
		void SetWrap(bool Wrap) { m_Wrap = Wrap; }
		uint8_t U8()
		{
			if(m_Pos < m_Size)
				return m_pData[m_Pos++];
			if(m_Size == 0 || !m_Wrap)
				return 0;
			const size_t Off = m_Pos - m_Size;
			m_Pos++;
			return (uint8_t)(m_pData[Off % m_Size] + (uint8_t)(Off / m_Size));
		}
		uint16_t U16()
		{
			const uint16_t Lo = U8();
			return (uint16_t)(Lo | (U8() << 8));
		}
		const uint8_t *Bytes(size_t n)
		{
			if(!Left(n))
				return nullptr;
			const uint8_t *p = m_pData + m_Pos;
			m_Pos += n;
			return p;
		}

	private:
		const uint8_t *m_pData;
		size_t m_Size;
		size_t m_Pos;
		bool m_Wrap = false;
	};

	// Dump a snapshot's items. Only called on the trap paths, so it costs nothing while
	// fuzzing, but a bare "out=48 to=56" is not enough to tell a server defect from a harness
	// artefact - the item list is.
	void DumpSnap(const char *pName, const CSnapshot *pSnap)
	{
		fprintf(stderr, "  %s: %d items\n", pName, pSnap->NumItems());
		for(int i = 0; i < pSnap->NumItems(); i++)
		{
			const CSnapshotItem *pItem = pSnap->GetItem(i);
			fprintf(stderr, "    [%d] type=%d id=%d size=%d\n", i,
				pItem->Key() >> 16, pItem->Key() & 0xffff, pSnap->GetItemSize(i));
		}
	}

	// Semantic snapshot equality: same set of keys, each with the same size and payload.
	// Deliberately order-INSENSITIVE. CSnapshotBuilder::Finish does not sort, and UnpackDelta
	// rebuilds as [surviving pFrom items][delta items], so a perfectly correct result routinely
	// has a different item order than pTo. A memcmp oracle would fire on every reorder and
	// bury any real mismatch.
	bool SnapEqual(const CSnapshot *pA, const CSnapshot *pB)
	{
		if(pA->NumItems() != pB->NumItems())
			return false;
		for(int i = 0; i < pA->NumItems(); i++)
		{
			const CSnapshotItem *pItemA = pA->GetItem(i);
			const int SizeA = pA->GetItemSize(i);
			const int IndexB = pB->GetItemIndex(pItemA->Key());
			if(IndexB == -1 || pB->GetItemSize(IndexB) != SizeA)
				return false;
			if(SizeA > 0 && mem_comp(pItemA->Data(), pB->GetItem(IndexB)->Data(), SizeA) != 0)
				return false;
		}
		return true;
	}

	// What CreateDelta/DiffItem/UndiffItem will see, computed from the two snapshots rather
	// than from inside the functions under test (snapshot.cpp is read-only here).
	void CollectDiffStats(const CSnapshot *pFrom, const CSnapshot *pTo)
	{
		for(int i = 0; i < pTo->NumItems(); i++)
		{
			const CSnapshotItem *pToItem = pTo->GetItem(i);
			const int ToSize = pTo->GetItemSize(i);
			const int FromIndex = pFrom->GetItemIndex(pToItem->Key());
			if(FromIndex == -1)
				continue;
			g_S.SharedKeys++;
			if(pFrom->GetItemSize(FromIndex) != ToSize)
				continue;
			g_S.SharedSameSize++;
			if(ToSize <= 0)
				continue;
			// snapshot.cpp:369 - DiffItem runs here, with Size = ToSize/4 >= 1.
			g_S.DiffItemNonZero++;
			const int *pA = pFrom->GetItem(FromIndex)->Data();
			const int *pB = pToItem->Data();
			int Same = 0, Diff = 0;
			for(int k = 0; k < ToSize / (int)sizeof(int32_t); k++)
				((pA[k] == pB[k]) ? Same : Diff)++;
			if(Diff == 0)
				continue; // DiffItem returns 0, item is not emitted at all
			g_S.Emitted++;
			if(Same > 0)
				g_S.UndiffZeroBranch++; // snapshot.cpp:256 `if(*pDiff == 0)` taken
		}
	}

	// Build one structurally valid snapshot. Returns its size, or -1.
	// Byte cost the builder charges for one item: one offset word plus the item header plus
	// the payload (snapshot.cpp:914-915, 947-948).
	size_t ItemCost(int Size)
	{
		return sizeof(int) + sizeof(CSnapshotItem) + (size_t)Size;
	}

	int BuildSnapshot(CSnapshotBuilder &Builder, CSnapshotBuffer *pBuffer, CReader &Reader,
		int NumItems, bool Sixup, const short *pStaticSizes, bool AllowExtended,
		int *paRegisteredEx, int *pNumRegisteredEx)
	{
		// Requested extended (type, id) pairs already placed in THIS snapshot. The builder
		// rewrites an extended type to MAX_TYPE-Index before storing it, so
		// FindItemIndexByKey cannot be asked about the type we requested.
		int aSeenEx[CSnapshot::MAX_ITEMS];
		int NumSeenEx = 0;

		Builder.Init(Sixup);
		// Mirror of the builder's own capacity arithmetic, kept because the builder's is WRONG
		// for one case and asserting inside Finish would stall the campaign on it.
		//
		// NewItemRaw checks whether the requested item fits (snapshot.cpp:916) and only THEN
		// calls GetExtendedItemTypeIndex (snapshot.cpp:923), which for a type the builder has
		// not seen before injects an extra NETOBJTYPE_EX item carrying the UUID
		// (AddExtendedItemType, snapshot.cpp:848). That injected item is never counted by the
		// check that just passed, so an extended item added within ~24 bytes of the limit takes
		// the builder past CSnapshot::MAX_SIZE and Finish's
		// `dbg_assert(TotalSize <= MAX_SIZE, "Snapshot too large")` (snapshot.cpp:832) fires.
		// Reproduced from five separate campaign inputs; clearing only the extended-types mode
		// bit makes all five stop. Reported, not worked around in the engine - this accounting
		// just keeps the fuzzer moving past it.
		//
		// Init() re-adds every already-registered extended type while the builder is empty
		// (snapshot.cpp:772-775), so those are charged up front.
		size_t Used = sizeof(CSnapshot) + (size_t)*pNumRegisteredEx * ItemCost((int)sizeof(CUuid));
		for(int i = 0; i < NumItems; i++)
		{
			// In sixup mode NewItemRaw rewrites the type through Obj_SixToSeven, whose table has
			// 21 entries of which only 17 are non-negative; drawing from [0, MAX_TYPE] gave a
			// 0.05% acceptance rate, so a sixup snapshot was almost always empty and the
			// cross-protocol case this target exists for could not occur.
			const uint8_t RawType = Reader.U8();
			const int Id = Reader.U8() % 64;
			// The top of the type byte selects a UUID-extended type. Reusing the SAME byte
			// rather than adding a field keeps every existing corpus entry meaning what it
			// meant before; only inputs that set the mode bit change behaviour.
			const bool Extended = AllowExtended && RawType >= 208;
			const int Type = Extended ?
						 FIRST_EX_OBJ + (int)(RawType - 208) % NUM_EX_OBJ :
						 (int)(RawType % (Sixup ? 21 : 64));
			int StoredType = Type;
			if(!Extended)
			{
				StoredType = Sixup ? Obj_SixToSeven(Type) : Type;
				if(StoredType < 0) // untranslatable in 0.7 - NewItemRaw drops it without adding
					continue;
				if(Builder.FindItemIndexByKey((StoredType << 16) | Id))
					continue;
			}
			else
			{
				// The stored key is MAX_TYPE-Index and cannot collide with a plain item's
				// (plain types are <= 63), so deduping on the requested pair is enough.
				const int Want = (Type << 8) | Id;
				bool Dup = false;
				for(int k = 0; k < NumSeenEx; k++)
					if(aSeenEx[k] == Want)
						Dup = true;
				if(Dup)
					continue;
				if(NumSeenEx < (int)std::size(aSeenEx))
					aSeenEx[NumSeenEx++] = Want;
			}
			bool NeedsRegistration = false;
			if(Extended)
			{
				NeedsRegistration = true;
				for(int k = 0; k < *pNumRegisteredEx; k++)
					if(paRegisteredEx[k] == Type)
						NeedsRegistration = false;
				if(NeedsRegistration && *pNumRegisteredEx >= 63)
					continue; // GetExtendedItemTypeIndex asserts at MAX_EXTENDED_ITEM_TYPES
			}
			// Content and size are a function of the requested key, so two items with the same
			// key agree by construction. Extended types get their own key space bit so that a
			// (Type << 16) that has already wrapped past 2^32 cannot alias a plain type.
			const unsigned ContentKey = Extended ?
							    (0x80000000u | ((unsigned)(Type - FIRST_EX_OBJ) << 16) | (unsigned)Id) :
							    (unsigned)((StoredType << 16) | Id);

			// Size is derived from the KEY, not drawn independently, and never zero.
			//
			// Independently-drawn sizes made the oracle nearly decorative: a shared key is what
			// the diff path needs, but with 64 possible sizes a shared key matched in size only
			// 1 in 64 times, so almost every shared key instead aborted UnpackDelta at -206.
			// Measured on the live corpus: only 2.7% of inputs both completed a round trip AND
			// entered UndiffItem. Deriving size from the key makes a shared key imply a shared
			// size, and a minimum of 1 int stops zero-length items from making DiffItem a no-op
			// loop.
			//
			// The same-key/DIFFERENT-size case is the C1 shape, so it stays reachable - but on
			// purpose, behind a mode bit, rather than by accident on almost every input.
			const unsigned KeyHash = ContentKey * 2246822519u;
			// A statically sized type MUST use its table size: that is the invariant
			// SnapNewItem<T> gives the server, and it is what makes CreateDelta's omitted-size
			// encoding decodable at all. Breaking it here would make UnpackDelta read the wrong
			// length and the oracle would report a harness artefact as a server defect.
			const int StaticSize = (!Extended && pStaticSizes != nullptr && StoredType >= 0 &&
						       StoredType < MAX_STATIC_TYPES) ?
						       (int)pStaticSizes[StoredType] :
						       0;
			int SizeInInts;
			if(StaticSize > 0 && StaticSize % (int)sizeof(int32_t) != 0)
				continue; // NewItemRaw asserts on a non-multiple-of-4 size; never happens today
			if(StaticSize > 0)
				SizeInInts = StaticSize / (int)sizeof(int32_t);
			else
				SizeInInts = 1 + (int)(KeyHash % 63);
			const uint8_t Mask = Reader.U8();
			const uint8_t Fill = Reader.U8();
			// Content is a function of the KEY, so two items with the same key agree by
			// construction, and Mask selects which ints diverge. The previous scheme derived
			// every byte from one Fill value, which meant same-key items were either byte
			// identical (no update emitted at all) or differed in EVERY int - so the common
			// real case, and UndiffItem's `*pDiff == 0` branch, were unreachable.
			static int32_t s_aPayload[512];
			if(SizeInInts > (int)std::size(s_aPayload))
				continue;
			const int Size = SizeInInts * (int)sizeof(int32_t);
			for(int k = 0; k < SizeInInts; k++)
			{
				const int32_t Base = (int32_t)(ContentKey * 2654435761u + (unsigned)k * 40503u);
				s_aPayload[k] = Base + (((Mask >> (k & 7)) & 1) ? (int32_t)Fill + 1 : 0);
			}
			const uint8_t *pPayload = (const uint8_t *)s_aPayload;
			// Skip duplicate keys. The server never emits the same (type, id) twice in one
			// snapshot, and allowing it here would make the key-based oracle below report
			// mismatches that no real snapshot can produce.
			//
			// The key has to be the one the builder will actually STORE, not the one we asked
			// for: in sixup mode NewItemRaw rewrites the type through Obj_SixToSeven before
			// storing it (snapshot.cpp), so two different requested types can collapse onto one
			// stored key. Deduping on the requested key silently lets that through.
			const size_t Need = ItemCost(Size) +
					    (NeedsRegistration ? ItemCost((int)sizeof(CUuid)) : 0);
			if(!g_Strict && Used + Need > (size_t)CSnapshot::MAX_SIZE)
				break; // full, counting the record the builder forgets to count
			void *pItem = Builder.NewItemRaw(Type, Id, Size);
			if(pItem == nullptr) // builder full - legitimate, stop adding
				break;
			Used += Need;
			if(NeedsRegistration)
				paRegisteredEx[(*pNumRegisteredEx)++] = Type;
			if(Size > 0)
				mem_copy(pItem, pPayload, Size);
			if(g_Stats)
			{
				if(StaticSize > 0)
					g_S.StaticSizedItems++;
				if(Extended)
					g_S.ExtendedItems++;
			}
		}
		return Builder.Finish(pBuffer);
	}

	void InitOnce()
	{
		g_Stats = getenv("FZ_SNAP_STATS") != nullptr;
		g_Strict = getenv("FZ_SNAP_STRICT") != nullptr;
		if(g_Stats)
			atexit(PrintStats);

		// Exactly CGameContext::OnInit's table, gamecontext.cpp:4188-4199.
		static CNetObjHandler s_Handler6;
		static protocol7::CNetObjHandler s_Handler7;
		for(int i = 0; i < (int)NUM_NETOBJTYPES && i < MAX_STATIC_TYPES; i++)
			g_aStaticSize6[i] = (short)s_Handler6.GetObjSize(i);
		for(int i = 0; i < OLD_NUM_NETOBJTYPES7 && i < MAX_STATIC_TYPES; i++)
			g_aStaticSize7[i] = (short)s_Handler7.GetObjSize(i);

		for(int i = 0; i < 3; i++)
			g_apDelta[i] = new CSnapshotDelta();
		for(int i = 0; i < (int)NUM_NETOBJTYPES && i < MAX_STATIC_TYPES; i++)
			g_apDelta[1]->SetStaticsize(i, (size_t)g_aStaticSize6[i]);
		for(int i = 0; i < OLD_NUM_NETOBJTYPES7 && i < MAX_STATIC_TYPES; i++)
			g_apDelta[2]->SetStaticsize(i, (size_t)g_aStaticSize7[i]);
	}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	static bool s_Init = false;
	if(!s_Init)
	{
		s_Init = true;
		InitOnce();
	}
	if(Size < 8)
		return 0;

	CReader Reader(pData, Size);
	// Toward CSnapshot::MAX_ITEMS rather than 24, for the same reason as the payload cap:
	// the interesting failure needs a snapshot near the size limit, not a tiny one.
	const int NumFrom = (int)Reader.U16() % (CSnapshot::MAX_ITEMS + 1);
	const int NumTo = (int)Reader.U16() % (CSnapshot::MAX_ITEMS + 1);
	// The protocol mismatch is the point: let the fuzzer build the two snapshots under
	// different builder modes, which is what the real 0.6/0.7 slot-takeover produces.
	const uint8_t Modes = Reader.U8();
	// Bit 2 used to be a "draw each item's size independently" mode, which made the two
	// snapshots disagree on the size of a shared key. That is the C1 shape, and it is a
	// PRECONDITION VIOLATION rather than an input: CreateDelta matches items by key alone and
	// takes the size from pTo, so it is the caller's job never to delta across a size change,
	// and the server's job is done by CServer::SetTimedOut purging a reclaimed slot's
	// snapshots. Both snapshots a real caller holds come from one connection, so one protocol.
	//
	// With that fixed at the source, generating the violation only rediscovers a closed issue
	// - one round produced 3232 artifacts, every one of them the same over-read in DiffItem,
	// and nothing else in the target got any attention. The bit is left unread rather than
	// reused, so the byte keeps its position in the stream.
	const bool AllowExtended = (Modes & 16) != 0;
	Reader.SetWrap((Modes & 32) != 0);
	// A client slot has ONE protocol, so a delta object is either 0.6 or 0.7 - never both.
	// Follow the receiving ("to") side, which is what decides how the delta is encoded.
	const int DeltaSel = (Modes & 8) == 0 ? 0 : ((Modes & 2) ? 2 : 1);
	const short *pStaticSizes = DeltaSel == 0 ? nullptr : (DeltaSel == 1 ? g_aStaticSize6 : g_aStaticSize7);
	CSnapshotDelta &Delta = *g_apDelta[DeltaSel];

	if(g_Stats)
		g_S.Inputs++;

	// Heap-allocate so ASan redzones sit immediately around each snapshot, the way
	// CSnapshotStorage::Add's exact-size malloc does.
	auto *pFromBuf = new(std::nothrow) CSnapshotBuffer;
	auto *pToBuf = new(std::nothrow) CSnapshotBuffer;
	auto *pOutBuf = new(std::nothrow) CSnapshotBuffer;
	auto *pBuilder = new(std::nothrow) CSnapshotBuilder;
	if(!pFromBuf || !pToBuf || !pOutBuf || !pBuilder)
	{
		delete pFromBuf;
		delete pToBuf;
		delete pOutBuf;
		delete pBuilder;
		return 0;
	}

	// The builder is reused for both snapshots, exactly as CServer reuses m_SnapshotBuilder,
	// so its extended-type registrations carry from the first build into the second.
	int aRegisteredEx[64];
	int NumRegisteredEx = 0;
	const int FromSize = BuildSnapshot(*pBuilder, pFromBuf, Reader, NumFrom, (Modes & 1) != 0,
		pStaticSizes, AllowExtended, aRegisteredEx, &NumRegisteredEx);
	const int ToSize = BuildSnapshot(*pBuilder, pToBuf, Reader, NumTo, (Modes & 2) != 0,
		pStaticSizes, AllowExtended, aRegisteredEx, &NumRegisteredEx);

	if(FromSize > 0 && ToSize > 0 &&
		pFromBuf->AsSnapshot()->IsValid(FromSize) && pToBuf->AsSnapshot()->IsValid(ToSize))
	{
		// Copy each snapshot into an exact-sized heap block, so an over-read of even one
		// byte past the end is a hard ASan error instead of landing in slack space.
		auto *pFromExact = (CSnapshot *)malloc(FromSize);
		auto *pToExact = (CSnapshot *)malloc(ToSize);
		if(pFromExact && pToExact)
		{
			mem_copy(pFromExact, pFromBuf->AsSnapshot(), FromSize);
			mem_copy(pToExact, pToBuf->AsSnapshot(), ToSize);

			if(g_Stats)
			{
				g_S.Built++;
				if(FromSize > g_S.MaxFrom)
					g_S.MaxFrom = FromSize;
				if(ToSize > g_S.MaxTo)
					g_S.MaxTo = ToSize;
				CollectDiffStats(pFromExact, pToExact);
			}

			// Bound (worst case): 3 ints of CData header + 4 bytes per deleted key + 12 bytes
			// of item header per update item + the whole item payload, with NumItems capped
			// at CSnapshot::MAX_ITEMS and the payload at CSnapshot::MAX_SIZE.
			const size_t DeltaCap = g_Strict ? (size_t)CSnapshot::MAX_SIZE :
							   (size_t)CSnapshot::MAX_SIZE + 16u * CSnapshot::MAX_ITEMS + 64u;
			auto *pDeltaData = (char *)malloc(DeltaCap);
			if(pDeltaData)
			{
				const int DeltaSize = Delta.CreateDelta(pFromExact, pToExact, pDeltaData);
				if(g_Stats && DeltaSize > g_S.MaxDelta)
					g_S.MaxDelta = DeltaSize;
				if(DeltaSize > (int)CSnapshot::MAX_SIZE)
				{
					// On a real server this write already happened - into
					// `char aDeltaData[CSnapshot::MAX_SIZE]` on CServer::DoSnapshot's stack.
					static long s_Over = 0;
					if(s_Over == 0)
					{
						fprintf(stderr,
							"NOTE: CreateDelta wrote %d bytes into a %d-byte destination "
							"(from=%d to=%d) - server.cpp:1110 passes a stack buffer of that size\n",
							DeltaSize, (int)CSnapshot::MAX_SIZE, FromSize, ToSize);
						atexit([]() { fprintf(stderr, "NOTE: deltas larger than MAX_SIZE: %ld\n", s_Over); });
					}
					s_Over++;
					if(g_Stats)
						g_S.DeltaOverMaxSize++;
					free(pDeltaData);
					free(pFromExact);
					free(pToExact);
					delete pFromBuf;
					delete pToBuf;
					delete pOutBuf;
					delete pBuilder;
					return 0;
				}

				// ---- correctness oracle, not just a memory-safety check ------------
				// A crash-only harness sails straight past SILENT corruption, and C1 was
				// exactly that on a release build: a wrong delta produced without faulting.
				// So assert the round-trip contract the server actually relies on
				// (server.cpp:1113 "if(DeltaSize)"):
				//   DeltaSize == 0 -> "nothing changed", nothing is sent: pFrom MUST equal pTo
				//   DeltaSize  > 0 -> UnpackDelta must succeed and reproduce pTo
				if(DeltaSize == 0)
				{
					if(g_Stats)
						g_S.DeltaZero++;
					if(!SnapEqual(pFromExact, pToExact))
					{
						fprintf(stderr, "ORACLE: empty delta but from != to (from=%d to=%d)\n",
							FromSize, ToSize);
						__builtin_trap();
					}
				}
				else if(DeltaSize > 0)
				{
					if(g_Stats)
						g_S.DeltaPos++;
					// server.cpp:1118-1119 - what the server actually sends is the
					// varint-COMPRESSED delta, into a second MAX_SIZE stack buffer.
					// CVariableInt::Pack can turn a 4-byte int into 5 bytes, so this is the
					// step that decides whether a large delta fits at all, and it is the
					// only bounds check between CreateDelta and the wire.
					auto *pCompData = (char *)malloc(CSnapshot::MAX_SIZE);
					if(pCompData)
					{
						(void)CVariableInt::Compress(pDeltaData, DeltaSize, pCompData, CSnapshot::MAX_SIZE);
						free(pCompData);
					}
					const int Result = Delta.UnpackDelta(pFromExact, pOutBuf, pDeltaData, DeltaSize);
					if(Result < 0)
					{
						// -206/-207 are the decoder's same-key/different-size rejects. That is
						// the C1 shape: the fix in CreateDelta stops the over-read by emitting
						// the item in full, but UnpackDelta still refuses it because the key
						// exists in pFrom at another size. Known and written up in FINDINGS.md,
						// so count it rather than trapping - otherwise the campaign stalls here
						// and never reaches anything new.
						// -302, NewItemRaw failing during UnpackDelta's rebuild, is NOT exempt
						// any more. It needed a same-key/different-size pair, which the retired
						// mode was the only source of, so with sizes derived from the key it is
						// a real finding again and must trap rather than be counted.
						if(Result == -206 || Result == -207)
						{
							// Counted and reported, not just swallowed: without a number,
							// "no round-trip violations in N executions" cannot be
							// distinguished from "most executions bailed out here".
							static int s_SizeRejects = 0;
							if(s_SizeRejects == 0)
							{
								fprintf(stderr, "NOTE: first same-key/different-size reject (%d)\n", Result);
								atexit([]() { fprintf(stderr, "NOTE: same-key/different-size rejects: %d\n", s_SizeRejects); });
							}
							s_SizeRejects++;
							if(g_Stats)
								g_S.UnpackSizeReject++;
						}
						else
						{
							fprintf(stderr, "ORACLE: CreateDelta output rejected by UnpackDelta: %d\n", Result);
							DumpSnap("from", pFromExact);
							DumpSnap("to", pToExact);
							__builtin_trap();
						}
					}
					else if(!SnapEqual(pOutBuf->AsSnapshot(), pToExact))
					{
						fprintf(stderr, "ORACLE: round-trip mismatch (out=%d to=%d)\n", Result, ToSize);
						DumpSnap("from", pFromExact);
						DumpSnap("to", pToExact);
						DumpSnap("out", pOutBuf->AsSnapshot());
						__builtin_trap();
					}
					else if(g_Stats)
					{
						g_S.UnpackOk++;
						g_S.RoundTripOk++;
					}
				}
				free(pDeltaData);
			}
		}
		free(pFromExact);
		free(pToExact);
	}

	delete pFromBuf;
	delete pToBuf;
	delete pOutBuf;
	delete pBuilder;
	return 0;
}
