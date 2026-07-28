/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef GAME_ENTITIES_PICKUP_H
#define GAME_ENTITIES_PICKUP_H

#include "entity.h"

class CPickupData;

class CPickup : public CEntity
{
public:
	static const int ms_CollisionExtraSize = 6;

	CPickup(CGameWorld *pGameWorld, int Type, int SubType, int Layer, int Number, int Flags);

	void Reset() override;
	void Tick() override;
	void TickPaused() override;
	// Server only, defined in src/game/server/snap.cpp
	void Snap(int SnappingClient);

	int Type() const { return m_Type; }
	int Subtype() const { return m_Subtype; }
	int Flags() const { return m_Flags; }

	// Prediction only, defined in src/game/client/prediction/entities_predict.cpp.
	CPickup(CGameWorld *pGameWorld, int Id, const CPickupData *pPickup);
	void FillInfo(CNetObj_Pickup *pPickup);
	bool Match(CPickup *pPickup);
	bool InDDNetTile() const { return m_IsCoreActive; }

private:
	int m_Type;
	int m_Subtype;
	int m_Flags;

	// DDRace

	void Move();
	vec2 m_Core;
	// Prediction only.
	bool m_IsCoreActive = false;
};

#endif
