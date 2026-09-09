#include <gtest/gtest.h>
#include <tools/demo_scramble_noise.h>

#include <cmath>

// A run of a few minutes at the periods demo_scramble uses spans this many
// blocks of noise
static constexpr int RUN_BLOCKS = 215;
static constexpr int FIRST_BLOCK = 4780;
static constexpr int SEEDS = 2000;

TEST(DemoScrambleNoise, InRange)
{
	for(uint32_t Seed = 1; Seed <= SEEDS; Seed++)
	{
		const CScrambleNoise Noise(Seed, Seed * 0x9e3779b97f4a7c15ull);
		for(int Block = FIRST_BLOCK; Block < FIRST_BLOCK + RUN_BLOCKS; Block++)
		{
			const float Value = Noise.Value(0, 0, Block);
			EXPECT_GE(Value, -1.0f);
			EXPECT_LE(Value, 1.0f);
		}
	}
}

TEST(DemoScrambleNoise, Deterministic)
{
	const CScrambleNoise First(12345, 6789);
	const CScrambleNoise Second(12345, 6789);
	EXPECT_EQ(First.Noise(3, 1, 77777, 40), Second.Noise(3, 1, 77777, 40));
	EXPECT_NE(First.Noise(3, 1, 77777, 40), CScrambleNoise(12346, 6789).Noise(3, 1, 77777, 40));
	EXPECT_NE(First.Noise(3, 1, 77777, 40), CScrambleNoise(12345, 6790).Noise(3, 1, 77777, 40));
}

// The whole point of the noise is that it moves over a run. A hash whose last
// step does not spread the block over the word returns the same number for
// every block, which makes the noise one constant per demo that anyone can
// measure once and subtract everywhere.
TEST(DemoScrambleNoise, MovesOverARun)
{
	float SmallestSpread = 2.0f;
	for(uint32_t Seed = 1; Seed <= SEEDS; Seed++)
	{
		const CScrambleNoise Noise(Seed, Seed * 0x9e3779b97f4a7c15ull);
		float Lowest = 1.0f;
		float Highest = -1.0f;
		for(int Block = FIRST_BLOCK; Block < FIRST_BLOCK + RUN_BLOCKS; Block++)
		{
			const float Value = Noise.Value(0, 0, Block);
			Lowest = std::min(Lowest, Value);
			Highest = std::max(Highest, Value);
		}
		SmallestSpread = std::min(SmallestSpread, Highest - Lowest);
	}
	EXPECT_GT(SmallestSpread, 1.5f);
}

// Recovering the noise of one channel must not hand over the others, so a
// velocity that can be read back exactly does not undo the aim
TEST(DemoScrambleNoise, ChannelsAreIndependent)
{
	int Equal = 0;
	double Correlation = 0.0;
	for(uint32_t Seed = 1; Seed <= SEEDS; Seed++)
	{
		const CScrambleNoise Noise(Seed, Seed * 0x9e3779b97f4a7c15ull);
		const float First = Noise.Value(0, 0, FIRST_BLOCK);
		const float Second = Noise.Value(0, 2, FIRST_BLOCK);
		Equal += First == Second ? 1 : 0;
		Correlation += First * Second;
	}
	EXPECT_EQ(Equal, 0);
	// Two independent draws from [-1, 1] average to zero, a channel that is a
	// copy of another would average to a third
	EXPECT_LT(std::fabs(Correlation / SEEDS), 0.05);
}

TEST(DemoScrambleNoise, PlayersAreIndependent)
{
	int Equal = 0;
	for(uint32_t Seed = 1; Seed <= SEEDS; Seed++)
	{
		const CScrambleNoise Noise(Seed, Seed * 0x9e3779b97f4a7c15ull);
		Equal += Noise.Value(0, 0, FIRST_BLOCK) == Noise.Value(1, 0, FIRST_BLOCK) ? 1 : 0;
	}
	EXPECT_EQ(Equal, 0);
}

// Neighbouring ticks of the smoothed noise stay close together, that is what
// keeps it under what a viewer can see
TEST(DemoScrambleNoise, SmoothWithinABlock)
{
	const CScrambleNoise Noise(4242, 2424);
	const int Period = 40;
	for(int Tick = 100000; Tick < 100000 + 3 * Period; Tick++)
	{
		const float Step = std::fabs(Noise.Noise(0, 0, Tick + 1, Period) - Noise.Noise(0, 0, Tick, Period));
		EXPECT_LT(Step, 4.0f / Period);
	}
}

// A hook that started before the demo did asks for the noise of a negative
// tick, where rounding towards zero would put the block above the tick and
// the interpolation would leave the range
TEST(DemoScrambleNoise, InRangeBeforeTickZero)
{
	const CScrambleNoise Noise(4242, 2424);
	for(int Tick = -200; Tick <= 200; Tick++)
	{
		const float Value = Noise.Noise(0, 0, Tick, 40);
		ASSERT_GE(Value, -1.0f);
		ASSERT_LE(Value, 1.0f);
	}
}
