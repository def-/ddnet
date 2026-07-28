/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef GAME_CLIENT_PREDICTION_PREDICTION_ENV_H
#define GAME_CLIENT_PREDICTION_PREDICTION_ENV_H

#include "gameworld.h"

#include <game/gameenv.h>

/*
	Class: CPredictionEnvironment
		Client side implementation of IGameEnvironment.

		Effects that the server would send to clients as net events are instead
		recorded in the world's predicted event list, so that CGameClient can
		suppress the real event when it arrives from the server. One environment
		belongs to each CGameWorld, because the events have to end up in the
		list of the world that is being simulated.
*/
class CPredictionEnvironment : public IGameEnvironment
{
public:
	explicit CPredictionEnvironment(CGameWorld *pGameWorld) :
		m_pGameWorld(pGameWorld)
	{
	}

	void CreateSound(vec2 Pos, int Sound, CClientMask Mask, int Id) override
	{
		m_pGameWorld->CreatePredictedSound(Pos, Sound, Id);
	}

	void CreateExplosionEvent(vec2 Pos, CClientMask Mask, int Id) override
	{
		m_pGameWorld->CreatePredictedExplosionEvent(Pos, Id);
	}

	void CreateDamageInd(vec2 Pos, float Angle, int Amount, CClientMask Mask, int Id) override
	{
		m_pGameWorld->CreatePredictedDamageIndEvent(Pos, Angle, Amount, Id);
	}

	void CreateHammerHit(vec2 Pos, CClientMask Mask, int Id) override
	{
		m_pGameWorld->CreatePredictedHammerHitEvent(Pos, Id);
	}

	// See IGameEnvironment: death effects are played straight from the snapshot.
	void CreateDeath(vec2 Pos, int ClientId, CClientMask Mask) override {}

	CClientMask ClientsMaskExcludeClientVersionAndHigher(int Version) const override
	{
		return CClientMask().set();
	}

private:
	CGameWorld *m_pGameWorld;
};

#endif
