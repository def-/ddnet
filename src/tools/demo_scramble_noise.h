#ifndef TOOLS_DEMO_SCRAMBLE_NOISE_H
#define TOOLS_DEMO_SCRAMBLE_NOISE_H

#include <base/math.h>

#include <cstdint>

// The noise demo_scramble adds to a demo. Every value is drawn from a key, a
// player, a channel and a block of ticks, so a run can be noised in one pass
// and a rerun with the same key produces the same demo.
//
// The key is 128 bits because a demo publishes samples of this: the velocity
// of a converted demo is a whole number of units, which for a while made the
// noise added to it readable in the clear. Eight such samples were enough to
// find a 32 bit key by trying all of them in a second, and the key undoes
// every channel exactly. The velocity is quantized now, but a key that can be
// searched is a key that will be searched.
class CScrambleNoise
{
	uint64_t m_KeyLow;
	uint64_t m_KeyHigh;

	// The finalizer of splitmix64, which spreads every input bit over the
	// whole word. Combining alone does not: an exclusive-or leaves the high
	// bits of a counter untouched, and the noise is drawn from the high bits,
	// so neighbouring blocks and channels came out as the same number.
	static uint64_t Mix(uint64_t Value)
	{
		Value ^= Value >> 30;
		Value *= 0xbf58476d1ce4e5b9ull;
		Value ^= Value >> 27;
		Value *= 0x94d049bb133111ebull;
		Value ^= Value >> 31;
		return Value;
	}

public:
	CScrambleNoise(uint64_t KeyLow, uint64_t KeyHigh) :
		m_KeyLow(KeyLow), m_KeyHigh(KeyHigh)
	{
	}

	uint64_t Hash(int ClientId, int Channel, int Block) const
	{
		const uint64_t Input = ((uint64_t)(uint32_t)ClientId << 40) ^ ((uint64_t)(uint32_t)Channel << 32) ^ (uint32_t)Block;
		return Mix(Mix(Input + m_KeyLow) ^ m_KeyHigh);
	}

	// Random value in [-1, 1] that takes Period ticks to change, so that the
	// noise moves as slowly as the tee it is added to. A period of one tick is
	// noise without any smoothing.
	float Noise(int ClientId, int Channel, int Tick, int Period) const
	{
		// Rounding towards zero would put a negative tick in the block above
		// itself, and the smoothstep would then extrapolate outside [-1, 1]
		// Rounding towards zero would put a negative tick in the block above
		// itself, and the smoothstep would then extrapolate outside [-1, 1]
		const int Block = Tick >= 0 ? Tick / Period : -((-Tick + Period - 1) / Period);
		const float Fraction = (float)(Tick - Block * Period) / (float)Period;
		const float Start = Value(ClientId, Channel, Block);
		const float End = Value(ClientId, Channel, Block + 1);
		return mix(Start, End, Fraction * Fraction * (3.0f - 2.0f * Fraction));
	}

	float Value(int ClientId, int Channel, int Block) const
	{
		return (float)(Hash(ClientId, Channel, Block) >> 40) / (float)0xffffffu * 2.0f - 1.0f;
	}
};

#endif // TOOLS_DEMO_SCRAMBLE_NOISE_H
