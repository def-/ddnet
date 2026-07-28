/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef GAME_ENTITIES_LASER_H
#define GAME_ENTITIES_LASER_H

#include "entity.h"

#include <game/server/interactions.h>

class CLaserData;

class CLaser : public CEntity
{
	friend class CGameWorld;

public:
	CLaser(CGameWorld *pGameWorld, vec2 Pos, vec2 Direction, float StartEnergy, int Owner, int Type);

	void Reset() override;
	void Tick() override;
	void TickPaused() override;
	// Server only, defined in src/game/server/snap.cpp
	void Snap(int SnappingClient);
	void SwapClients(int Client1, int Client2) override;

	int GetOwnerId() const override { return m_Owner; }

	/*
		The interaction state records who may see and be hit by this laser as the
		owner connects, dies or leaves. It is server bookkeeping, so each side
		defines these for itself.
	*/
	void SyncInteractState();
	CClientMask BounceMask();
	bool CanHit(int ClientId);

	// Prediction only, defined in src/game/client/prediction/entities_predict.cpp.
	CLaser(CGameWorld *pGameWorld, int Id, CLaserData *pLaser);
	bool Match(CLaser *pLaser);
	CLaserData GetData() const;
	const vec2 &GetFrom() const { return m_From; }
	const int &GetOwner() const { return m_Owner; }
	const int &GetEvalTick() const { return m_EvalTick; }

protected:
	bool HitCharacter(vec2 From, vec2 To);
	void DoBounce();

private:
	vec2 m_From;
	vec2 m_Dir;
	vec2 m_TelePos;
	bool m_WasTele;
	float m_Energy;
	int m_Bounces;
	int m_EvalTick;
	int m_Owner;
	bool m_ZeroEnergyBounceInLastTick;
	CInteractions m_InteractState;

	// DDRace

	vec2 m_PrevPos;
	int m_Type;
	int m_TuneZone;
	bool m_TeleportCancelled;
	bool m_IsBlueTeleport;
	bool m_BelongsToPracticeTeam;
};

#endif
