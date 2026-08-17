// libFuzzer target: the tile handlers - CCharacter::HandleTiles, HandleSkippableTiles,
// CGameControllerDDNet::HandleCharacterTiles and the collision lookups under them.
//
// Why a separate target rather than more of fz_serverpkt. Measured on that target's own
// corpus, HandleTiles sits at 552 of 1912 edges, and the reason is not that the input space
// is unexplored - it is that a tee has to physically TRAVEL to a tile before its handler
// runs. The fuzzer drives a tee through the network, sixteen records at a time, and no amount
// of packet mutation walks it across a map to a switch or a tune zone. Every other structural
// knob was tried against that: more client slots bought nothing, and a bigger tick budget
// bought 1%.
//
// So this target inverts it. Instead of moving the tee to the tiles, it writes the TILES
// under the tee, straight from the fuzz input, and steps the world. The tee never travels,
// the whole map layer stack is whatever the input says, and the handlers are reached in one
// tick instead of thousands.
//
// Three things make that legitimate rather than a way to manufacture crashes:
//
//   1. Every byte written here is a byte a real .map file can contain. The layers are the
//      map's own arrays, written in place, and the values are the same unsigned chars the
//      map format stores. A crash found here is a crash a hand-made map causes, and maps are
//      attacker-supplied: a server downloads them from a vote, a client from the server.
//
//   2. CCollision::Init is re-run after every repaint. That is what keeps the DERIVED state
//      consistent with the tiles - the teleporter target lists, the highest switch number,
//      the door tiles. Painting a tele-in whose number has no tele-out, without re-deriving,
//      would be a crash no map can cause. Init opens with Unload() and rebuilds all of it, so
//      re-running it is exactly what loading a different map does. CGameWorld::Init is called
//      with it, because the switcher vector is sized from m_HighestSwitchNumber and lives on
//      the world rather than on the collision.
//
//   3. The patch is saved and restored around every input. The map is process-wide state, so
//      without that an artifact would depend on every input that ran before it - the failure
//      mode that made 42 artifacts of an earlier campaign unreproducible.
//
// Input layout. Everything past the end reads as zero, so a short input is a valid short
// program and libFuzzer can grow it:
//
//   u8  ticks           1 + n % TICKS_MAX
//   u8  input bits      left, right, jump, hook, fire, aim x sign, aim y sign
//   i16 vel x, vel y    little-endian, in 1/256 units per tick
//   then the patch, PATCH_W * PATCH_H cells per layer, in this order:
//     game     2 bytes  index, flags
//     front    2 bytes  index, flags
//     tele     2 bytes  number, type
//     tune     2 bytes  number, type
//     switch   4 bytes  number, type, flags, delay
//     speedup  6 bytes  force, max speed, type, pad, angle lo, angle hi
#include "fz_server_fixture.h"

#include <base/mem.h>

#include <game/collision.h>
#include <game/mapitems.h>
#include <game/server/entities/character.h>
#include <game/server/gamecontext.h>
#include <game/server/player.h>

#include <cstddef>
#include <cstdint>

bool IsInterrupted()
{
	return false;
}

namespace
{

	// Five by five, and the tee starts in the middle. Wide enough that a tee moving at the
	// velocities this target can set crosses several cells within one input, so the handlers see
	// tile TRANSITIONS (which is what HandleTiles is mostly made of) rather than a tee sitting
	// still on one tile. Small enough that the whole patch is a few hundred bytes of input.
	constexpr int PATCH_W = 5;
	constexpr int PATCH_H = 5;
	constexpr int PATCH_TILES = PATCH_W * PATCH_H;

	// Where on the map the patch goes. Away from the spawn, so a tee that dies and respawns is
	// not immediately standing in fuzzer-written tiles again - that would blur which input caused
	// what.
	constexpr int PATCH_X = 8;
	constexpr int PATCH_Y = 8;

	// How long the world may run per input. Ticking is what this target spends its time on, by a
	// wide margin: with the tiles painted and CCollision::Init re-run, an input costs 33 ms, of
	// which the paint and the re-derive together are under 1 ms. The tile handlers fire on the
	// first tick after the tee is placed and again on each tile it crosses, so the tail of a long
	// run buys little - see the README for the measurement that picked this number.
	constexpr int TICKS_MAX = 8;

	class CReader
	{
	public:
		CReader(const uint8_t *pData, size_t Size) :
			m_pData(pData), m_Size(Size), m_Pos(0) {}

		// Past the end reads as zero rather than stopping, so every layer of the patch is always
		// written and a truncated input simply means "the rest is air".
		uint8_t U8()
		{
			return m_Pos < m_Size ? m_pData[m_Pos++] : 0;
		}
		int16_t I16()
		{
			const uint8_t Lo = U8();
			const uint8_t Hi = U8();
			return (int16_t)((uint16_t)Lo | ((uint16_t)Hi << 8));
		}

	private:
		const uint8_t *m_pData;
		size_t m_Size;
		size_t m_Pos;
	};

	// The map's own arrays, saved once and put back before every input.
	struct CSavedPatch
	{
		CTile m_aGame[PATCH_TILES];
		CTile m_aFront[PATCH_TILES];
		CTeleTile m_aTele[PATCH_TILES];
		CTuneTile m_aTune[PATCH_TILES];
		CSwitchTile m_aSwitch[PATCH_TILES];
		CSpeedupTile m_aSpeedup[PATCH_TILES];
		bool m_Saved = false;
	};
	CSavedPatch g_Saved;

	// The layers are handed out const because nothing in the game writes them. This target does,
	// and they are ordinary heap arrays owned by the map file, so the cast is a cast away of
	// constness that was never a promise about the storage. Keeping it in one place makes the
	// four call sites below read as what they are.
	template<typename T>
	T *Writable(const T *pLayer)
	{
		return const_cast<T *>(pLayer);
	}

	int PatchIndex(int Cell)
	{
		const CCollision *pCollision = fzserver::g_pGameServer->Collision();
		const int x = PATCH_X + Cell % PATCH_W;
		const int y = PATCH_Y + Cell / PATCH_W;
		return y * pCollision->GetWidth() + x;
	}

	bool PatchFits()
	{
		const CCollision *pCollision = fzserver::g_pGameServer->Collision();
		return PATCH_X + PATCH_W <= pCollision->GetWidth() && PATCH_Y + PATCH_H <= pCollision->GetHeight();
	}

	// Every layer the map carries. A layer the map does not have is null, and its section of the
	// input is read and discarded so the layout stays fixed either way.
	template<typename T, typename FRead>
	void PaintLayer(const T *pLayer, T *pSaved, CReader &Reader, FRead ReadOne)
	{
		for(int Cell = 0; Cell < PATCH_TILES; Cell++)
		{
			const T Value = ReadOne(Reader);
			if(pLayer == nullptr)
				continue;
			T *pWritable = Writable(pLayer);
			if(!g_Saved.m_Saved)
				pSaved[Cell] = pWritable[PatchIndex(Cell)];
			pWritable[PatchIndex(Cell)] = Value;
		}
	}

	template<typename T>
	void RestoreLayer(const T *pLayer, const T *pSaved)
	{
		if(pLayer == nullptr)
			return;
		T *pWritable = Writable(pLayer);
		for(int Cell = 0; Cell < PATCH_TILES; Cell++)
			pWritable[PatchIndex(Cell)] = pSaved[Cell];
	}

} // namespace

extern "C" int LLVMFuzzerInitialize(int *pArgc, char ***pArgv)
{
	// No sqlite: nothing here reaches the score backend, and the database worker thread makes
	// coverage non-deterministic, which is the thing that made an earlier round of
	// measurements on these harnesses worthless.
	fzserver::Init(pArgc, pArgv, /*WithSqlite=*/false);
	fzserver::EnterGame(fzserver::CLIENT_SIX, false);
	return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *pData, size_t Size)
{
	if(Size < 6 || !PatchFits())
		return 0;

	CGameContext *pGameServer = fzserver::g_pGameServer;
	CCollision *pCollision = pGameServer->Collision();

	if(pGameServer->m_apPlayers[fzserver::CLIENT_SIX] == nullptr)
		fzserver::EnterGame(fzserver::CLIENT_SIX, false);

	if(g_Saved.m_Saved)
	{
		RestoreLayer(pCollision->GameLayer(), g_Saved.m_aGame);
		RestoreLayer(pCollision->FrontLayer(), g_Saved.m_aFront);
		RestoreLayer(pCollision->TeleLayer(), g_Saved.m_aTele);
		RestoreLayer(pCollision->TuneLayer(), g_Saved.m_aTune);
		RestoreLayer(pCollision->SwitchLayer(), g_Saved.m_aSwitch);
		RestoreLayer(pCollision->SpeedupLayer(), g_Saved.m_aSpeedup);
	}

	CReader Reader(pData, Size);
	const int Ticks = 1 + Reader.U8() % TICKS_MAX;
	const uint8_t InputBits = Reader.U8();
	const float VelX = Reader.I16() / 256.0f;
	const float VelY = Reader.I16() / 256.0f;

	PaintLayer(pCollision->GameLayer(), g_Saved.m_aGame, Reader, [](CReader &r) {
		CTile Tile = {};
		Tile.m_Index = r.U8();
		Tile.m_Flags = r.U8();
		return Tile;
	});
	PaintLayer(pCollision->FrontLayer(), g_Saved.m_aFront, Reader, [](CReader &r) {
		CTile Tile = {};
		Tile.m_Index = r.U8();
		Tile.m_Flags = r.U8();
		return Tile;
	});
	PaintLayer(pCollision->TeleLayer(), g_Saved.m_aTele, Reader, [](CReader &r) {
		CTeleTile Tile = {};
		Tile.m_Number = r.U8();
		Tile.m_Type = r.U8();
		return Tile;
	});
	PaintLayer(pCollision->TuneLayer(), g_Saved.m_aTune, Reader, [](CReader &r) {
		CTuneTile Tile = {};
		Tile.m_Number = r.U8();
		Tile.m_Type = r.U8();
		return Tile;
	});
	PaintLayer(pCollision->SwitchLayer(), g_Saved.m_aSwitch, Reader, [](CReader &r) {
		CSwitchTile Tile = {};
		Tile.m_Number = r.U8();
		Tile.m_Type = r.U8();
		Tile.m_Flags = r.U8();
		Tile.m_Delay = r.U8();
		return Tile;
	});
	PaintLayer(pCollision->SpeedupLayer(), g_Saved.m_aSpeedup, Reader, [](CReader &r) {
		CSpeedupTile Tile = {};
		Tile.m_Force = r.U8();
		Tile.m_MaxSpeed = r.U8();
		Tile.m_Type = r.U8();
		r.U8(); // m_MustBe0, kept in the layout so the input maps onto the struct
		Tile.m_Angle = r.I16();
		return Tile;
	});
	g_Saved.m_Saved = true;

	// Re-derive everything the tiles imply, the way loading a map does. Without this a
	// tele-in can name a target that does not exist and a switch tile a number the switcher
	// vector was never sized for, and the crash that follows is the harness's, not the
	// server's.
	pCollision->Init(pGameServer->Layers());
	pGameServer->m_World.Init(pCollision, pGameServer->TuningList());

	// Rebuild the map's entities from the tiles that were just painted, which is the other
	// half of keeping derived state consistent and was missing here.
	//
	// CCollision::Init allocates a fresh zeroed door array (collision.cpp:74-75) and the only
	// thing that ever fills it is CDoor::ResetCollision, called from CDoor's constructor
	// (door.cpp:24). So re-running Init without rebuilding the entities left the map's doors
	// with no collision at all from the second input onward - the same tiles-versus-derived
	// desync this target re-runs Init to avoid, in the opposite direction.
	//
	// It is also the biggest thing this target was missing. Entities are built only by
	// CreateAllEntities (gamecontext.cpp:4358, gameworld.cpp:158), so until now painting a
	// pickup, door, dragger, plasma gun or light changed bytes that nothing read: the tee
	// could only ever hold the hammer and gun it spawns with (gamecontroller.cpp:541-542),
	// which left FireWeapon's shotgun, grenade and laser arms and all of HandleNinja
	// unreachable. Asking the world to reset is how a round restart does it, and it rebuilds
	// from exactly the tiles Init has just consumed, so every entity's m_Number stays inside
	// the switcher vector InitSwitchers just sized.
	pGameServer->m_World.m_ResetRequested = true;

	CCharacter *pChr = pGameServer->m_apPlayers[fzserver::CLIENT_SIX]->GetCharacter();
	if(pChr != nullptr)
	{
		// Middle of the patch, and the tee is placed rather than driven there: this target
		// exists because driving it is what does not work.
		const vec2 Pos((PATCH_X + PATCH_W / 2) * 32.0f + 16.0f, (PATCH_Y + PATCH_H / 2) * 32.0f + 16.0f);
		pChr->SetPosition(Pos);
		pChr->m_Pos = Pos;
		pChr->m_PrevPos = Pos;
		pChr->SetVelocity(vec2(VelX, VelY));

		// A tile handler that branches on where the tee is heading needs an input, not just a
		// position. The aim target is bounded the way OnClientPrepareInput bounds it.
		CNetObj_PlayerInput Input = {};
		Input.m_Direction = (InputBits & 1) != 0 ? -1 : ((InputBits & 2) != 0 ? 1 : 0);
		Input.m_Jump = (InputBits & 4) != 0;
		Input.m_Hook = (InputBits & 8) != 0;
		Input.m_Fire = (InputBits & 16) != 0;
		Input.m_TargetX = (InputBits & 32) != 0 ? -256 : 256;
		Input.m_TargetY = (InputBits & 64) != 0 ? -256 : 256;
		// The server's own two entry points, in the order CServer::Run calls them, rather
		// than assigning m_Input: OnDirectInput is what fires a weapon and OnPredictedInput
		// is what the next tick reads.
		pChr->OnDirectInput(&Input);
		pChr->OnPredictedInput(&Input);
	}

	fzserver::AdvanceTicks(Ticks, Ticks);
	return 0;
}
