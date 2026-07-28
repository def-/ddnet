/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef GAME_ENTITIES_PROJECTILE_H
#define GAME_ENTITIES_PROJECTILE_H

#include "entity.h"

class CProjectileData;

class CProjectile : public CEntity
{
	friend class CGameWorld;
	friend class CItems;

public:
	CProjectile(
		CGameWorld *pGameWorld,
		int Type,
		int Owner,
		vec2 Pos,
		vec2 Dir,
		int Span,
		bool Freeze,
		bool Explosive,
		int SoundImpact,
		vec2 InitDir,
		int Layer = 0,
		int Number = 0);

	vec2 GetPos(float Time);

	CNetObj_Projectile NetInfoVanilla() const;
	bool NetIsInfoLegacyCompatible() const;
	CNetObj_DDRaceProjectile NetInfoLegacy() const;
	CNetObj_DDNetProjectile NetInfo() const;

	void Reset() override;
	void Tick() override;
	void TickPaused() override;
	// Server only, defined in src/game/server/snap.cpp
	void Snap(int SnappingClient);
	void SwapClients(int Client1, int Client2) override;

private:
	vec2 m_Direction;
	int m_LifeSpan;
	int m_Owner;
	int m_Type;
	int m_SoundImpact;
	int m_StartTick;
	bool m_Explosive;

	// DDRace

	int m_Bouncing;
	bool m_Freeze;
	int m_TuneZone;
	bool m_BelongsToPracticeTeam;
	int m_DDRaceTeam;
	bool m_IsSolo;
	vec2 m_InitDir;

public:
	void SetBouncing(int Value);

	// Prediction only, defined in src/game/client/prediction/entities_predict.cpp.
	CProjectile(CGameWorld *pGameWorld, int Id, const CProjectileData *pProj);
	CProjectileData GetData() const;
	bool Match(CProjectile *pProj);
	const vec2 &GetDirection() const { return m_Direction; }
	const int &GetOwner() const { return m_Owner; }
	const int &GetStartTick() const { return m_StartTick; }

	bool CanCollide(int ClientId) override;
	int GetOwnerId() const override { return m_Owner; }
};

#endif
