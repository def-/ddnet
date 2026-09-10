#include "demo_scramble_noise.h"

#include <base/dbg.h>
#include <base/logger.h>
#include <base/math.h>
#include <base/mem.h>
#include <base/os.h>
#include <base/secure.h>
#include <base/str.h>
#include <base/vmath.h>

#include <engine/shared/demo.h>
#include <engine/shared/network.h>
#include <engine/shared/snapshot.h>
#include <engine/storage.h>

#include <generated/protocol.h>

#include <game/gamecore.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>

static const char *TOOL_NAME = "demo_scramble";

// A demo records the exact inputs of the run: the movement direction, the aim
// vector and the direction of every shot. Noising them costs a run its
// replayability while staying far below what a viewer can notice, a tile
// being 32 units and a full turn 2*pi.
//
// Positions are the exception, and the height more than the side: the client
// reads the ground under a tee from them (CheckPoint(m_X, m_Y + 16)), and a
// tee resting on the ground has only two units of margin there before it
// starts to flicker between the standing and the falling animation. Sideways
// that only matters on the very edge of a platform.
static constexpr float POSITION_AMPLITUDE_X = 2.0f;
static constexpr float POSITION_AMPLITUDE_Y = 1.0f;
static constexpr int POSITION_PERIOD = 40;
// Ticks a tee has to stand still before its published position is held, see
// ScrambleCharacter
static constexpr int STILL_TICKS = 3;
static constexpr float AIM_AMPLITUDE = 0.02f;
static constexpr int AIM_PERIOD = 25;
// Ticks a change of the movement direction may be held back by
static constexpr int DIRECTION_HOLD = 2;
// The same for the jump bits, and every second hook goes out a tick late
static constexpr int JUMP_HOLD = 1;
// Points that are not a tee, an explosion against a wall or a laser bouncing
// off one, are published with a noise of their own. Their true place would
// otherwise be there for the taking, and the noise the tees are published
// with follows from it. It moves as slowly as the one on the tees, so that
// many readings of the same point do not average it away.
// The velocity a tee is published with is the difference of the positions it
// is published with, see DerivedVelocity, so it says nothing they do not.

enum
{
	CHANNEL_X = 0,
	CHANNEL_Y,
	CHANNEL_AIM,
	CHANNEL_DIRECTION,
	CHANNEL_HOOK,
	CHANNEL_JUMP,
	CHANNEL_LOOSE_X,
	CHANNEL_LOOSE_Y,
	CHANNEL_LOOSE_AIM,
};

static vec2 Rotate(vec2 Vector, float Angle)
{
	const float Sin = std::sin(Angle);
	const float Cos = std::cos(Angle);
	return vec2(Vector.x * Cos - Vector.y * Sin, Vector.x * Sin + Vector.y * Cos);
}

class CScrambler
{
	CScrambleNoise m_Noise;
	// What the movement key of a player was last written as, and how many
	// ticks a change of it is still held back
	int m_aDirection[MAX_CLIENTS] = {};
	int m_aDirectionHold[MAX_CLIENTS] = {};
	// The hook of the tick before, to publish a hook a tick late, and the same
	// for the jump bits
	struct SHook
	{
		int m_State = HOOK_IDLE;
		int m_Tick = 0;
		int m_X = 0;
		int m_Y = 0;
		int m_Dx = 0;
		int m_Dy = 0;
		int m_HookedPlayer = -1;
	};
	SHook m_aPreviousHook[MAX_CLIENTS];
	bool m_aHookLate[MAX_CLIENTS] = {};
	ivec2 m_aPreviousPosition[MAX_CLIENTS] = {};
	// The recorded position of that tick, to tell a tee that stands still
	// from one whose velocity a collision zeroed while it still moved
	ivec2 m_aPreviousRecorded[MAX_CLIENTS] = {};
	int m_aPreviousTick[MAX_CLIENTS] = {};
	// The anchor a hook holds on to does not move, and one that is published
	// through a rotation about a moving tee would say what the rotation is
	ivec2 m_aAnchor[MAX_CLIENTS] = {};
	ivec2 m_aPublishedAnchor[MAX_CLIENTS] = {};
	int m_aJumped[MAX_CLIENTS] = {};
	int m_aJumpHold[MAX_CLIENTS] = {};
	int m_aJumpedTotal[MAX_CLIENTS] = {};
	int m_aJumpedTotalHold[MAX_CLIENTS] = {};
	// How many ticks in a row the recorded position of a tee has not moved,
	// per axis, see the pin in ScrambleCharacter
	ivec2 m_aStill[MAX_CLIENTS] = {};
	// The tick a hook left the tee. m_HookTick counts only while it holds on
	// to something, so it is not the tick the aim of the hook was taken at.
	int m_aHookFireTick[MAX_CLIENTS] = {};
	// Where the characters of the tick being scrambled were recorded and how
	// far their published position moved, see MoveEvent
	bool m_aTickPresent[MAX_CLIENTS] = {};
	ivec2 m_aTickRecorded[MAX_CLIENTS] = {};
	ivec2 m_aTickPrevious[MAX_CLIENTS] = {};
	ivec2 m_aTickDelta[MAX_CLIENTS] = {};
	CNetObj_Character *m_apTickCharacter[MAX_CLIENTS] = {};
	int m_TrackedPoints = 0;
	int m_LoosePoints = 0;

	vec2 PositionNoise(int ClientId, int Tick) const
	{
		return vec2(m_Noise.Noise(ClientId, CHANNEL_X, Tick, POSITION_PERIOD) * POSITION_AMPLITUDE_X, m_Noise.Noise(ClientId, CHANNEL_Y, Tick, POSITION_PERIOD) * POSITION_AMPLITUDE_Y);
	}

	float AimNoise(int ClientId, int Tick) const
	{
		return m_Noise.Noise(ClientId, CHANNEL_AIM, Tick, AIM_PERIOD) * AIM_AMPLITUDE;
	}

	// The velocity of a converted demo is the difference of two recorded
	// positions, so publishing the true one beside noised positions is a
	// second view of the same run: whichever of the noised velocities lies
	// closest to the published difference is the true one, and the trajectory
	// comes back exactly. It is taken from the published positions instead,
	// which says nothing a reader cannot already compute.
	//
	static int DerivedVelocity(int Published, int Previous)
	{
		return 256 * (Published - Previous);
	}

	// Publishes a value that changed only after a few ticks of the noise have
	// passed, so the tick a key went down in is not the tick it is written in.
	// pPublished holds what the last tick was written as, pHold how many ticks
	// the change still waits.
	int Held(int ClientId, int Channel, int Tick, int Limit, int Value, int *pPublished, int *pHold)
	{
		if(Value == *pPublished)
		{
			*pHold = -1;
		}
		else
		{
			if(*pHold < 0)
				*pHold = m_Noise.Hash(ClientId, Channel, Tick) % (Limit + 1);
			if(*pHold > 0)
			{
				(*pHold)--;
			}
			else
			{
				*pPublished = Value;
				*pHold = -1;
			}
		}
		return *pPublished;
	}

	void ScrambleCharacter(int ClientId, int Tick, CNetObj_Character *pCharacter)
	{
		const ivec2 Recorded = ivec2(pCharacter->m_X, pCharacter->m_Y);
		const vec2 Offset = PositionNoise(ClientId, Tick);
		pCharacter->m_X = round_to_int(Recorded.x + Offset.x);
		pCharacter->m_Y = round_to_int(Recorded.y + Offset.y);
		pCharacter->m_Angle += round_to_int(AimNoise(ClientId, Tick) * 256.0f);
		if(m_aPreviousTick[ClientId] == Tick - 1)
		{
			// A tee that stands still keeps the place it was published in,
			// or the published position would move while the tee does not and
			// the client would walk it on the spot. It takes a few ticks of
			// standing to count: pinning on the first one would say, tick by
			// tick, whether the recorded position stepped, and a tee that
			// moves slower than a unit per tick would be readable from that.
			// The recorded position decides, not the velocity: the tick a fall
			// lands in has a velocity of zero and a tee that moved, and
			// pinning it there would hold the tee in the air.
			m_aStill[ClientId].x = Recorded.x == m_aPreviousRecorded[ClientId].x ? m_aStill[ClientId].x + 1 : 0;
			m_aStill[ClientId].y = Recorded.y == m_aPreviousRecorded[ClientId].y ? m_aStill[ClientId].y + 1 : 0;
			if(m_aStill[ClientId].x >= STILL_TICKS)
				pCharacter->m_X = m_aPreviousPosition[ClientId].x;
			if(m_aStill[ClientId].y >= STILL_TICKS)
				pCharacter->m_Y = m_aPreviousPosition[ClientId].y;
			pCharacter->m_VelX = DerivedVelocity(pCharacter->m_X, m_aPreviousPosition[ClientId].x);
			pCharacter->m_VelY = DerivedVelocity(pCharacter->m_Y, m_aPreviousPosition[ClientId].y);
		}
		else
		{
			m_aStill[ClientId] = ivec2(0, 0);
			// A tee that was not in the last snapshot starts over: a hook from
			// its previous life would be drawn across the map for a tick, and
			// its recorded velocity has no published positions to come from
			pCharacter->m_VelX = 0;
			pCharacter->m_VelY = 0;
			m_aHookLate[ClientId] = false;
			m_aPreviousHook[ClientId] = SHook();
			m_aAnchor[ClientId] = ivec2(0, 0);
			m_aPublishedAnchor[ClientId] = ivec2(0, 0);
			m_aDirection[ClientId] = pCharacter->m_Direction;
			m_aDirectionHold[ClientId] = -1;
			m_aJumped[ClientId] = pCharacter->m_Jumped;
			m_aJumpHold[ClientId] = -1;
			m_aJumpedTotalHold[ClientId] = -1;
		}
		m_aPreviousRecorded[ClientId] = Recorded;
		m_aPreviousPosition[ClientId] = ivec2(pCharacter->m_X, pCharacter->m_Y);
		m_aPreviousTick[ClientId] = Tick;
	}

	// Everything of a character that needs the published position of the other
	// tees of the tick, so it runs once they all have one
	void ScrambleCharacterState(int ClientId, int Tick, CNetObj_Character *pCharacter)
	{
		const ivec2 Recorded = m_aTickRecorded[ClientId];
		const vec2 Offset = PositionNoise(ClientId, Tick);

		// The tick a hook goes out in is what a replay of a run is built on, so
		// every second hook is published a tick late. The whole hook moves
		// together, rope, direction and all, which is a hook that hangs 20 ms
		// behind the tee it belongs to and nothing a viewer can see.
		const SHook Hook = {pCharacter->m_HookState, pCharacter->m_HookTick, pCharacter->m_HookX,
			pCharacter->m_HookY, pCharacter->m_HookDx, pCharacter->m_HookDy, pCharacter->m_HookedPlayer};
		// m_HookTick only counts while a hook holds on to something, so the
		// hook is recognised by the tick it leaves the tee in
		if(pCharacter->m_HookState > HOOK_IDLE && m_aPreviousHook[ClientId].m_State <= HOOK_IDLE)
		{
			m_aHookLate[ClientId] = (m_Noise.Hash(ClientId, CHANNEL_HOOK, Tick) & 1) != 0;
			m_aHookFireTick[ClientId] = Tick;
		}
		if(m_aHookLate[ClientId])
		{
			const SHook &Late = m_aPreviousHook[ClientId];
			pCharacter->m_HookState = Late.m_State;
			pCharacter->m_HookTick = Late.m_Tick;
			pCharacter->m_HookX = Late.m_X;
			pCharacter->m_HookY = Late.m_Y;
			pCharacter->m_HookDx = Late.m_Dx;
			pCharacter->m_HookDy = Late.m_Dy;
			pCharacter->m_HookedPlayer = Late.m_HookedPlayer;
		}
		m_aPreviousHook[ClientId] = Hook;

		// The air jump the client draws is a bit of this, so a change of it is
		// held back the same way the movement key is
		pCharacter->m_Jumped = Held(ClientId, CHANNEL_JUMP, Tick, JUMP_HOLD, pCharacter->m_Jumped,
			&m_aJumped[ClientId], &m_aJumpHold[ClientId]);

		// A hook flies in the direction the tee aimed in when it was fired and
		// says so twice, as its own direction and as the path it took. Turning
		// the whole hook by the noise of the tick it started in keeps its shape
		// and follows the tee it hangs on.
		if(pCharacter->m_HookState > HOOK_IDLE && pCharacter->m_HookedPlayer >= 0 &&
			pCharacter->m_HookedPlayer < MAX_CLIENTS && m_aTickPresent[pCharacter->m_HookedPlayer])
		{
			// A hook that holds a tee sits on that tee (CCharacterCore::Tick),
			// so it is published where that tee is published. Anything else
			// would put the recorded position of the hooked tee in the demo.
			pCharacter->m_HookX = m_aTickRecorded[pCharacter->m_HookedPlayer].x + m_aTickDelta[pCharacter->m_HookedPlayer].x;
			pCharacter->m_HookY = m_aTickRecorded[pCharacter->m_HookedPlayer].y + m_aTickDelta[pCharacter->m_HookedPlayer].y;
			// The direction it flew out in is the aim of the tick it was fired
			// in, published there with a turn on it
			const vec2 Direction = Rotate(vec2(pCharacter->m_HookDx, pCharacter->m_HookDy), AimNoise(ClientId, m_aHookFireTick[ClientId]));
			pCharacter->m_HookDx = round_to_int(Direction.x);
			pCharacter->m_HookDy = round_to_int(Direction.y);
			m_aAnchor[ClientId] = ivec2(0, 0);
		}
		else if(pCharacter->m_HookState > HOOK_IDLE)
		{
			const ivec2 Anchor = ivec2(pCharacter->m_HookX, pCharacter->m_HookY);
			const vec2 Rope = vec2(Anchor.x - Recorded.x, Anchor.y - Recorded.y);
			// The same turn the aim of that tick was published with. A rope
			// is at most a few hundred units long, so its end moves by a
			// few units, and turning it by anything else would publish the
			// difference between the two as the noise of the aim.
			const float Angle = AimNoise(ClientId, m_aHookFireTick[ClientId]);
			// An anchor that has not moved keeps the place it was published in:
			// seen through a turn about a tee that moves, a fixed anchor would
			// wander, and how it wanders is the turn
			if(Anchor == m_aAnchor[ClientId])
			{
				pCharacter->m_HookX = m_aPublishedAnchor[ClientId].x;
				pCharacter->m_HookY = m_aPublishedAnchor[ClientId].y;
			}
			else
			{
				const vec2 Turned = vec2(Recorded.x, Recorded.y) + Offset + Rotate(Rope, Angle);
				pCharacter->m_HookX = round_to_int(Turned.x);
				pCharacter->m_HookY = round_to_int(Turned.y);
			}
			m_aAnchor[ClientId] = Anchor;
			m_aPublishedAnchor[ClientId] = ivec2(pCharacter->m_HookX, pCharacter->m_HookY);
			const vec2 Direction = Rotate(vec2(pCharacter->m_HookDx, pCharacter->m_HookDy), Angle);
			pCharacter->m_HookDx = round_to_int(Direction.x);
			pCharacter->m_HookDy = round_to_int(Direction.y);
		}
		else
		{
			// A hook that is not out sits on the tee (CCharacterCore::Tick),
			// so the recorded value here is the recorded position of the tee
			// and the difference to the published one is the whole noise
			pCharacter->m_HookX = pCharacter->m_X;
			pCharacter->m_HookY = pCharacter->m_Y;
			m_aAnchor[ClientId] = ivec2(0, 0);
		}

		// Every change of the movement key is held back by up to two ticks.
		// The moment a key goes down is what a replay of these inputs stands
		// on, and it is 40 ms of a tee walking that nobody sees.
		pCharacter->m_Direction = Held(ClientId, CHANNEL_DIRECTION, Tick, DIRECTION_HOLD, pCharacter->m_Direction,
			&m_aDirection[ClientId], &m_aDirectionHold[ClientId]);
	}

	void ScrambleDDNetCharacter(int ClientId, int Tick, CNetObj_DDNetCharacter *pCharacter, int Size)
	{
		// The counter goes up on the exact tick an air jump is used, which
		// would say when the jump bits of the character were really flipped
		if(Size >= (int)(offsetof(CNetObj_DDNetCharacter, m_JumpedTotal) + sizeof(int)))
		{
			pCharacter->m_JumpedTotal = Held(ClientId, CHANNEL_JUMP, Tick, JUMP_HOLD, pCharacter->m_JumpedTotal,
				&m_aJumpedTotal[ClientId], &m_aJumpedTotalHold[ClientId]);
		}
		// The target is optional, old servers did not send it
		if(Size < (int)(offsetof(CNetObj_DDNetCharacter, m_TargetY) + sizeof(int)))
			return;
		const vec2 Target = vec2(pCharacter->m_TargetX, pCharacter->m_TargetY);
		const float Length = length(Target);
		if(Length <= 0.0f)
			return;
		// A target the mouse holds on the tee itself is too short for a turn
		// of it to survive being rounded to whole units, which would publish
		// the true aim beside a turned angle. It is pushed out to where the
		// turn lands on another unit, half a tile from the tee.
		const vec2 Turned = Rotate(Target, AimNoise(ClientId, Tick)) * (std::max(Length, 32.0f) / Length);
		pCharacter->m_TargetX = round_to_int(Turned.x);
		pCharacter->m_TargetY = round_to_int(Turned.y);
		// The angle is a hundredth of the resolution the target has, so the
		// recorded angle plus the turn would say where the turn fell between
		// two of its steps. Take it from the published target instead, the
		// way the server takes it from the input (CCharacterCore::Tick).
		if(m_apTickCharacter[ClientId] != nullptr)
		{
			const float TmpAngle = std::atan2((float)pCharacter->m_TargetY, (float)pCharacter->m_TargetX);
			m_apTickCharacter[ClientId]->m_Angle = (int)((TmpAngle < -(pi / 2.0f) ? TmpAngle + 2.0f * pi : TmpAngle) * 256.0f);
		}
	}

	// The start of a projectile stays where it is for the whole flight, so its
	// noise is the one of the tick it was fired in
	void ScrambleProjectile(CNetObj_DDNetProjectile *pProjectile)
	{
		if(pProjectile->m_Owner < 0 || pProjectile->m_Owner >= MAX_CLIENTS)
			return;
		// A projectile starts at the hand of the tee, but a teleporter starts
		// one on a tile the map names, and the noise of the tee on top of that
		// would be there to read. It gets the noise of its own start instead,
		// which stays put for the whole flight. It is published a hundred
		// times finer than a tee, so it is rounded to what a tee has.
		const ivec2 Start = ivec2(round_to_int(pProjectile->m_X / 100.0f), round_to_int(pProjectile->m_Y / 100.0f));
		const ivec2 Moved = MovePoint(Start, pProjectile->m_StartTick);
		pProjectile->m_X = 100 * Moved.x;
		pProjectile->m_Y = 100 * Moved.y;
		const vec2 Velocity = Rotate(vec2(pProjectile->m_VelX, pProjectile->m_VelY), AimNoise(pProjectile->m_Owner, pProjectile->m_StartTick));
		pProjectile->m_VelX = round_to_int(Velocity.x);
		pProjectile->m_VelY = round_to_int(Velocity.y);
	}

	// A beam is drawn from where it started to where it ends, and both are
	// recorded positions. Each end is moved on its own, so the published beam
	// does not point exactly where the shot was aimed, and a point two beams
	// share, the place one bounces off a wall, keeps its own noise and stays
	// one point. Turning the beam instead would say what the turn is: a bounce
	// mirrors the direction, so the two published angles add up to twice it.
	// The beam of a dragger ends on a tee and follows that tee.
	void ScrambleLaser(CNetObj_DDNetLaser *pLaser, int Size, int Tick)
	{
		if(Size < (int)(offsetof(CNetObj_DDNetLaser, m_Type) + sizeof(int)))
			return;
		const ivec2 From = ivec2(pLaser->m_FromX, pLaser->m_FromY);
		const ivec2 To = ivec2(pLaser->m_ToX, pLaser->m_ToY);
		const ivec2 PublishedFrom = MovePoint(From, Tick);
		const ivec2 PublishedTo = MovePoint(To, Tick);
		pLaser->m_FromX = PublishedFrom.x;
		pLaser->m_FromY = PublishedFrom.y;
		pLaser->m_ToX = PublishedTo.x;
		pLaser->m_ToY = PublishedTo.y;
	}

public:
	const ivec2 *TickRecorded() const { return m_aTickRecorded; }
	const bool *TickPresent() const { return m_aTickPresent; }
	const ivec2 *TickDelta() const { return m_aTickDelta; }
	int TrackedPoints() const { return m_TrackedPoints; }
	int LoosePoints() const { return m_LoosePoints; }

	CScrambler(uint64_t KeyLow, uint64_t KeyHigh) :
		m_Noise(KeyLow, KeyHigh)
	{
		std::fill(std::begin(m_aDirectionHold), std::end(m_aDirectionHold), -1);
		std::fill(std::begin(m_aJumpHold), std::end(m_aJumpHold), -1);
		std::fill(std::begin(m_aJumpedTotalHold), std::end(m_aJumpedTotalHold), -1);
		std::fill(std::begin(m_aPreviousTick), std::end(m_aPreviousTick), std::numeric_limits<int>::min());
	}

	// The tee a recorded point belongs to, or -1. Everything the converter
	// writes at a character is written from the very position the character
	// of that tick was built from, so the two match exactly.
	int OwnerOf(ivec2 Point) const
	{
		for(int ClientId = 0; ClientId < MAX_CLIENTS; ClientId++)
		{
			if(m_aTickPresent[ClientId] && (Point == m_aTickRecorded[ClientId] || Point == m_aTickPrevious[ClientId]))
				return ClientId;
		}
		return -1;
	}

	// Moves a recorded point into the published world. A point that is a tee
	// moves with that tee, so that the two stay together and their difference
	// says nothing. Anything else gets a noise of its own, which keeps its
	// true place out of the demo without tying it to any tee's noise.
	// The noise of a loose point is keyed by the point itself, so that every
	// publication of one true point moves the same way: the ten damage stars
	// of one hit, an explosion and the sound of it, and the two sides of the
	// bounce a laser makes. Publishing one point twice through two noises
	// would let a reader average them away.
	static int PointSalt(ivec2 Point)
	{
		return Point.x * 65599 + Point.y;
	}

	ivec2 LooseOffset(ivec2 Point, int Tick) const
	{
		const int Salt = PointSalt(Point);
		return ivec2(round_to_int(m_Noise.Noise(Salt, CHANNEL_LOOSE_X, Tick, POSITION_PERIOD) * POSITION_AMPLITUDE_X),
			round_to_int(m_Noise.Noise(Salt, CHANNEL_LOOSE_Y, Tick, POSITION_PERIOD) * POSITION_AMPLITUDE_Y));
	}

	ivec2 MovePoint(ivec2 Point, int Tick)
	{
		const int Owner = OwnerOf(Point);
		if(Owner >= 0)
		{
			m_TrackedPoints++;
			return Point + m_aTickDelta[Owner];
		}
		m_LoosePoints++;
		return Point + LooseOffset(Point, Tick);
	}

	static bool IsEvent(int Type)
	{
		// A map sound is map geometry and says nothing about a tee
		return (Type >= NETEVENTTYPE_COMMON && Type <= NETEVENTTYPE_DAMAGEIND) ||
		       Type == NETEVENTTYPE_BIRTHDAY || Type == NETEVENTTYPE_FINISH;
	}

	void ScrambleEvent(int Type, CNetEvent_Common *pEvent, int Size, int Tick)
	{
		if(Size < (int)sizeof(CNetEvent_Common))
			return;
		const ivec2 Point = ivec2(pEvent->m_X, pEvent->m_Y);
		const ivec2 Moved = MovePoint(Point, Tick);
		pEvent->m_X = Moved.x;
		pEvent->m_Y = Moved.y;
		// The damage stars of a shot point the way it was fired, which is the
		// aim of that tick again and finer than the angle the tee is published
		// with. They are turned together, the fan they make is what is drawn.
		if(Type == NETEVENTTYPE_DAMAGEIND && Size >= (int)sizeof(CNetEvent_DamageInd))
		{
			CNetEvent_DamageInd *pDamage = (CNetEvent_DamageInd *)pEvent;
			pDamage->m_Angle += round_to_int(m_Noise.Noise(PointSalt(Point), CHANNEL_LOOSE_AIM, Tick, AIM_PERIOD) * AIM_AMPLITUDE * 256.0f);
		}
	}

	// Scrambles the snapshot in place
	void Scramble(CSnapshot *pSnapshot, int Tick)
	{
		// The characters first, the events of the tick are moved with them
		std::fill(std::begin(m_aTickPresent), std::end(m_aTickPresent), false);
		std::fill(std::begin(m_apTickCharacter), std::end(m_apTickCharacter), nullptr);
		for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
		{
			const CSnapshotItem *pItem = pSnapshot->GetItem(Index);
			const int ClientId = pItem->Id();
			if(pSnapshot->GetItemType(Index) != NETOBJTYPE_CHARACTER || ClientId >= MAX_CLIENTS ||
				pSnapshot->GetItemSize(Index) < (int)sizeof(CNetObj_Character))
			{
				continue;
			}
			CNetObj_Character *pCharacter = (CNetObj_Character *)const_cast<int *>(pItem->Data());
			const ivec2 Recorded = ivec2(pCharacter->m_X, pCharacter->m_Y);
			m_aTickPrevious[ClientId] = m_aPreviousTick[ClientId] == Tick - 1 ? m_aPreviousRecorded[ClientId] : Recorded;
			ScrambleCharacter(ClientId, Tick, pCharacter);
			m_aTickRecorded[ClientId] = Recorded;
			m_aTickDelta[ClientId] = ivec2(pCharacter->m_X - Recorded.x, pCharacter->m_Y - Recorded.y);
			m_aTickPresent[ClientId] = true;
			m_apTickCharacter[ClientId] = pCharacter;
		}
		// A hook may hold another tee, which has to be published before the
		// hook that ends on it can be
		for(int ClientId = 0; ClientId < MAX_CLIENTS; ClientId++)
		{
			if(m_aTickPresent[ClientId])
				ScrambleCharacterState(ClientId, Tick, m_apTickCharacter[ClientId]);
		}

		for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
		{
			const CSnapshotItem *pItem = pSnapshot->GetItem(Index);
			const int ItemSize = pSnapshot->GetItemSize(Index);
			// The item keeps its place, only its contents change
			void *pData = const_cast<int *>(pItem->Data());
			const int ClientId = pItem->Id();
			const int Type = pSnapshot->GetItemType(Index);
			switch(Type)
			{
			case NETOBJTYPE_DDNETCHARACTER:
				if(ClientId < MAX_CLIENTS)
					ScrambleDDNetCharacter(ClientId, Tick, (CNetObj_DDNetCharacter *)pData, ItemSize);
				break;
			case NETOBJTYPE_DDNETPROJECTILE:
				if(ItemSize >= (int)sizeof(CNetObj_DDNetProjectile))
					ScrambleProjectile((CNetObj_DDNetProjectile *)pData);
				break;
			case NETOBJTYPE_DDNETLASER:
				ScrambleLaser((CNetObj_DDNetLaser *)pData, ItemSize, Tick);
				break;
			}
			if(IsEvent(Type))
				ScrambleEvent(Type, (CNetEvent_Common *)pData, ItemSize, Tick);
		}
	}
};

// Counts, per snapshot item type and per field of it, how often the published
// value is the recorded one. A field that is never touched is a field whose
// true value the demo hands over, and every one of them has to be a field that
// says nothing about where a tee was, where it aimed or when it pressed a key.
class CFieldReport
{
	struct SField
	{
		int m_Same = 0;
		int m_Changed = 0;
	};
	std::map<int, std::vector<SField>> m_Types;
	std::map<std::pair<int, int>, int> m_Leaks;

public:
	// Any two fields next to each other that hold the position a tee was
	// recorded at are that tee's true position, whatever item they sit in and
	// whether or not this build knows what that item is. That is the one thing
	// no published demo may contain: the noise the tee is published with is
	// the difference of the two.
	void CheckPositions(const CSnapshot *pPublished, const ivec2 *pRecordedPositions, const bool *pPresent, const ivec2 *pDeltas)
	{
		for(int Index = 0; Index < pPublished->NumItems(); Index++)
		{
			const int Size = pPublished->GetItemSize(Index) / (int)sizeof(int);
			const int *pData = pPublished->GetItem(Index)->Data();
			for(int Field = 0; Field + 1 < Size; Field++)
			{
				const ivec2 Value = ivec2(pData[Field], pData[Field + 1]);
				bool Recorded = false;
				bool Published = false;
				for(int ClientId = 0; ClientId < MAX_CLIENTS; ClientId++)
				{
					if(!pPresent[ClientId])
						continue;
					// A tee whose noise rounded to nothing this tick is
					// published where it was recorded, and a field holding
					// that value says nothing that is not published anyway.
					// The same goes for a value that is where some tee is
					// published, two tees standing on each other included.
					Recorded = Recorded || (pDeltas[ClientId] != ivec2(0, 0) && Value == pRecordedPositions[ClientId]);
					Published = Published || Value == pRecordedPositions[ClientId] + pDeltas[ClientId];
				}
				if(Recorded && !Published)
					m_Leaks[std::make_pair(pPublished->GetItemType(Index), Field)]++;
			}
		}
	}

	void Compare(const CSnapshot *pRecorded, const CSnapshot *pPublished)
	{
		for(int Index = 0; Index < pPublished->NumItems(); Index++)
		{
			const int Size = pPublished->GetItemSize(Index) / (int)sizeof(int);
			const int *pAfter = pPublished->GetItem(Index)->Data();
			const int *pBefore = pRecorded->GetItem(Index)->Data();
			std::vector<SField> &vFields = m_Types[pPublished->GetItemType(Index)];
			vFields.resize(std::max((int)vFields.size(), Size));
			for(int Field = 0; Field < Size; Field++)
			{
				if(pAfter[Field] == pBefore[Field])
					vFields[Field].m_Same++;
				else
					vFields[Field].m_Changed++;
			}
		}
	}

	static const char *TypeName(int Type)
	{
		return Type >= OFFSET_UUID ? g_UuidManager.GetName(Type) : "core";
	}

	void Print() const
	{
		for(const auto &[TypeAndField, Count] : m_Leaks)
		{
			log_error(TOOL_NAME, "type %s field %d holds the recorded position of a tee %d times",
				TypeName(TypeAndField.first), TypeAndField.second, Count);
		}
		if(m_Leaks.empty())
			log_info(TOOL_NAME, "no published field holds the recorded position of a tee");
		for(const auto &[Type, vFields] : m_Types)
		{
			for(size_t Field = 0; Field < vFields.size(); Field++)
			{
				if(vFields[Field].m_Changed == 0 && vFields[Field].m_Same > 0)
					log_info(TOOL_NAME, "type %d (%s) field %d published as recorded %d times", Type, TypeName(Type), (int)Field, vFields[Field].m_Same);
				else if(vFields[Field].m_Changed > 0)
					log_info(TOOL_NAME, "type %d (%s) field %d noised %d of %d times", Type, TypeName(Type), (int)Field, vFields[Field].m_Changed, vFields[Field].m_Changed + vFields[Field].m_Same);
			}
		}
	}
};

// --dump <cid> <from> <to>: what the demo holds for one tee, tick by tick
static int g_DumpCid = -1;
static int g_DumpFrom = 0;
static int g_DumpTo = 0;
static void DumpCharacter(const CSnapshot *pSnapshot, int Tick)
{
	const CNetObj_Character *pChar = nullptr;
	const CNetObj_DDNetCharacter *pExt = nullptr;
	for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
	{
		const CSnapshotItem *pItem = pSnapshot->GetItem(Index);
		if(pItem->Id() != g_DumpCid)
			continue;
		if(pSnapshot->GetItemType(Index) == NETOBJTYPE_CHARACTER)
			pChar = (const CNetObj_Character *)pItem->Data();
		else if(pSnapshot->GetItemType(Index) == NETOBJTYPE_DDNETCHARACTER)
			pExt = (const CNetObj_DDNetCharacter *)pItem->Data();
	}
	// Structure of the whole snapshot: duplicate keys and the order the
	// extended types were mapped in, both of which the delta coder relies on
	{
		char aDups[256] = "";
		char aTypes[256] = "";
		int Dups = 0;
		for(int Index = 0; Index < pSnapshot->NumItems(); Index++)
		{
			const CSnapshotItem *pItem = pSnapshot->GetItem(Index);
			for(int Other = 0; Other < Index; Other++)
			{
				if(pSnapshot->GetItem(Other)->Key() == pItem->Key())
				{
					Dups++;
					if(str_length(aDups) < 200)
						str_format(aDups + str_length(aDups), sizeof(aDups) - str_length(aDups), "%d:%d/%d ", pItem->InternalType(), pItem->Id(), pSnapshot->GetItemType(Index));
					break;
				}
			}
			if(pItem->InternalType() == 0 && pItem->Id() >= CSnapshot::OFFSET_UUID_TYPE && str_length(aTypes) < 200)
				str_format(aTypes + str_length(aTypes), sizeof(aTypes) - str_length(aTypes), "%x ", ((const int *)pItem->Data())[0] & 0xffff);
		}
		log_info(TOOL_NAME, "STRUCT tick=%d items=%d dups=%d [%s] extypes=[%s]", Tick, pSnapshot->NumItems(), Dups, aDups, aTypes);
	}
	if(pChar == nullptr)
	{
		log_info(TOOL_NAME, "DUMP tick=%d cid=%d absent", Tick, g_DumpCid);
		return;
	}
	log_info(TOOL_NAME, "DUMP tick=%d cid=%d pos=%d,%d vel=%d,%d weapon=%d hookstate=%d hook=%d,%d hooked=%d freezeend=%d flags=0x%x jumps=%d items=%d bytes=%d",
		Tick, g_DumpCid, pChar->m_X, pChar->m_Y, pChar->m_VelX, pChar->m_VelY, pChar->m_Weapon, pChar->m_HookState,
		pChar->m_HookX, pChar->m_HookY, pChar->m_HookedPlayer,
		pExt ? pExt->m_FreezeEnd : -999, pExt ? pExt->m_Flags : 0, pExt ? pExt->m_Jumps : -999, pSnapshot->NumItems(), pSnapshot->DataSize());
}

class CScrambleListener : public CDemoPlayer::IListener
{
	CDemoPlayer *m_pDemoPlayer;
	CDemoRecorder *m_pDemoRecorder;
	CScrambler m_Scrambler;
	CSnapshotBuffer m_Snapshot;
	CSnapshotBuffer m_Recorded;
	CFieldReport *m_pReport;
	int m_FirstTick = -1;
	int m_LastTick = -1;

public:
	CScrambleListener(CDemoPlayer *pDemoPlayer, CDemoRecorder *pDemoRecorder, uint64_t KeyLow, uint64_t KeyHigh, CFieldReport *pReport) :
		m_pDemoPlayer(pDemoPlayer), m_pDemoRecorder(pDemoRecorder), m_Scrambler(KeyLow, KeyHigh), m_pReport(pReport)
	{
	}

	void OnDemoPlayerSnapshot(void *pData, int Size) override
	{
		// The player keeps this snapshot as the base of the next delta, so the
		// scrambling happens on a copy
		mem_copy(m_Snapshot.m_aData, pData, Size);
		const int Tick = m_pDemoPlayer->Info()->m_Info.m_CurrentTick;
		if(g_DumpCid >= 0 && Tick >= g_DumpFrom && Tick <= g_DumpTo)
			DumpCharacter((const CSnapshot *)pData, Tick);
		if(m_pReport != nullptr)
			mem_copy(m_Recorded.m_aData, pData, Size);
		m_Scrambler.Scramble(m_Snapshot.AsSnapshot(), Tick);
		if(m_pReport != nullptr)
		{
			m_pReport->Compare(m_Recorded.AsSnapshot(), m_Snapshot.AsSnapshot());
			m_pReport->CheckPositions(m_Snapshot.AsSnapshot(), m_Scrambler.TickRecorded(), m_Scrambler.TickPresent(), m_Scrambler.TickDelta());
		}
		m_pDemoRecorder->RecordSnapshot(Tick, m_Snapshot.m_aData, Size);
		if(m_FirstTick < 0)
			m_FirstTick = Tick;
		m_LastTick = Tick;
	}

	// The recorder asserts that a marker is inside what it wrote, and playback
	// of a broken demo can stop before the last one
	bool Recorded(int Tick) const { return m_FirstTick >= 0 && Tick >= m_FirstTick && Tick <= m_LastTick; }
	int TrackedPoints() const { return m_Scrambler.TrackedPoints(); }
	int LoosePoints() const { return m_Scrambler.LoosePoints(); }

	void OnDemoPlayerMessage(void *pData, int Size) override
	{
		m_pDemoRecorder->RecordMessage(pData, Size);
	}
};

static bool ScrambleDemo(const char *pInputPath, const char *pOutputPath, uint64_t KeyLow, uint64_t KeyHigh, CSnapshotDelta *pSnapshotDelta, CSnapshotDelta *pSnapshotDeltaSixup, IStorage *pStorage, CFieldReport *pReport)
{
	CDemoPlayer DemoPlayer(pSnapshotDelta, pSnapshotDeltaSixup, false);
	if(DemoPlayer.Load(pStorage, nullptr, pInputPath, IStorage::TYPE_ALL_OR_ABSOLUTE) == -1)
	{
		log_error(TOOL_NAME, "Demo file '%s' failed to load: %s", pInputPath, DemoPlayer.ErrorMessage());
		return false;
	}
	if(DemoPlayer.IsSixup())
	{
		log_error(TOOL_NAME, "0.7 demos are not supported");
		return false;
	}

	const CMapInfo *pMapInfo = DemoPlayer.GetMapInfo();
	const CDemoPlayer::CPlaybackInfo *pInfo = DemoPlayer.Info();
	if(!pMapInfo->m_Sha256.has_value())
	{
		log_error(TOOL_NAME, "Demo file '%s' has no map SHA256", pInputPath);
		return false;
	}

	CDemoRecorder DemoRecorder(pSnapshotDelta);
	unsigned char *pMapData = DemoPlayer.GetMapData(pStorage);
	const int Error = DemoRecorder.Start(pStorage, nullptr, pOutputPath, pInfo->m_Header.m_aNetversion, pMapInfo->m_aName,
		pMapInfo->m_Sha256.value(), pMapInfo->m_Crc, pInfo->m_Header.m_aType, pMapInfo->m_Size, pMapData, nullptr, nullptr, nullptr);
	free(pMapData);
	if(Error != 0)
	{
		log_error(TOOL_NAME, "Failed to start demo recorder for '%s'", pOutputPath);
		return false;
	}

	CScrambleListener Listener(&DemoPlayer, &DemoRecorder, KeyLow, KeyHigh, pReport);
	DemoPlayer.SetListener(&Listener);
	DemoPlayer.Play();
	while(DemoPlayer.IsPlaying())
	{
		DemoPlayer.Update(false);
		if(pInfo->m_Info.m_Paused)
			break;
	}

	for(int i = 0; i < pInfo->m_Info.m_NumTimelineMarkers; i++)
	{
		if(Listener.Recorded(pInfo->m_Info.m_aTimelineMarkers[i]))
			DemoRecorder.AddDemoMarker(pInfo->m_Info.m_aTimelineMarkers[i]);
	}

	DemoPlayer.Stop();
	DemoRecorder.Stop(IDemoRecorder::EStopMode::KEEP_FILE);
	log_info(TOOL_NAME, "%d points moved with a tee, %d noised on their own", Listener.TrackedPoints(), Listener.LoosePoints());
	if(pReport != nullptr)
		pReport->Print();
	return true;
}

static std::unique_ptr<CSnapshotDelta> CreateSnapshotDelta()
{
	std::unique_ptr<CSnapshotDelta> pResult = std::make_unique<CSnapshotDelta>();
	CNetObjHandler NetObjHandler;
	for(int i = 0; i < NUM_NETOBJTYPES; i++)
	{
		pResult->SetStaticsize(i, NetObjHandler.GetObjSize(i));
	}
	return pResult;
}

// The key as 32 hex digits, low half first
static bool ParseKey(const char *pKey, uint64_t *pOut)
{
	if(str_length(pKey) != 2 * (int)(2 * sizeof(uint64_t)))
		return false;
	pOut[0] = 0;
	pOut[1] = 0;
	for(int i = 0; i < 32; i++)
	{
		const char Digit = pKey[i];
		int Value;
		if(Digit >= '0' && Digit <= '9')
			Value = Digit - '0';
		else if(Digit >= 'a' && Digit <= 'f')
			Value = Digit - 'a' + 10;
		else
			return false;
		pOut[i / 16] = (pOut[i / 16] << 4) | (uint64_t)Value;
	}
	return true;
}

int main(int argc, const char *argv[])
{
	// Create storage before setting logger to avoid log messages from storage creation
	std::unique_ptr<IStorage> pStorage = CreateLocalStorage();

	CCmdlineFix CmdlineFix(&argc, &argv);
	log_set_global_logger_default();
	CNetBase::Init();

	if(!pStorage)
	{
		log_error(TOOL_NAME, "Error creating local storage");
		return -1;
	}

	if(argc > 6 && str_comp(argv[argc - 4], "--dump") == 0)
	{
		g_DumpCid = str_toint(argv[argc - 3]);
		g_DumpFrom = str_toint(argv[argc - 2]);
		g_DumpTo = str_toint(argv[argc - 1]);
		argc -= 4;
	}
	bool Report = false;
	if(argc > 3 && str_comp(argv[argc - 1], "--report") == 0)
	{
		Report = true;
		argc--;
	}
	if(argc != 3 && argc != 5)
	{
		log_error(TOOL_NAME, "Usage: %s <input.demo> <output.demo> [--key <32 hex digits>] [--report] [--dump <cid> <from tick> <to tick>]", TOOL_NAME);
		log_error(TOOL_NAME, "Noises the recorded run below the threshold of what is visible, so that");
		log_error(TOOL_NAME, "the demo cannot be turned into an input sequence that reproduces the run");
		log_error(TOOL_NAME, "--key is the 128 bit key of the noise, 128 random bits otherwise. Publishing");
		log_error(TOOL_NAME, "one run twice under two keys undoes both, the two noises average out, so a");
		log_error(TOOL_NAME, "demo that is scrambled again has to be scrambled under the key it had");
		log_error(TOOL_NAME, "--dump prints what the demo holds for one tee, tick by tick, as the client reads it");
		log_error(TOOL_NAME, "--report lists which field of which snapshot item the noise reached, so");
		log_error(TOOL_NAME, "that a field carrying a recorded position or aim cannot slip through");
		return -1;
	}
	uint64_t aKey[2];
	if(argc == 5)
	{
		if(str_comp(argv[3], "--key") != 0)
		{
			log_error(TOOL_NAME, "Unknown argument '%s'", argv[3]);
			return -1;
		}
		if(!ParseKey(argv[4], aKey))
		{
			log_error(TOOL_NAME, "The key has to be %d hex digits", (int)(2 * sizeof(aKey)));
			return -1;
		}
	}
	else
	{
		secure_random_fill(aKey, sizeof(aKey));
	}
	CFieldReport FieldReport;

	std::unique_ptr<CSnapshotDelta> pSnapshotDelta = CreateSnapshotDelta();
	std::unique_ptr<CSnapshotDelta> pSnapshotDeltaSixup = std::make_unique<CSnapshotDelta>();
	if(!ScrambleDemo(argv[1], argv[2], aKey[0], aKey[1], pSnapshotDelta.get(), pSnapshotDeltaSixup.get(), pStorage.get(), Report ? &FieldReport : nullptr))
	{
		return -1;
	}
	return 0;
}
