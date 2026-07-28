/* (c) Shereef Marzouk. See "licence DDRace.txt" and the readme.txt in the root of the distribution for more information. */
#ifndef GAME_ENTITIES_DOOR_H
#define GAME_ENTITIES_DOOR_H

#include "entity.h"

class CGameWorld;

class CLaserData;

class CDoor : public CEntity
{
	vec2 m_To;
	// Prediction only: whether this door actually placed collision.
	bool m_Active = false;
	void ResetCollision();
	int m_Length;
	vec2 m_Direction;

public:
	CDoor(CGameWorld *pGameWorld, vec2 Pos, float Rotation, int Length,
		int Number);

	void Reset() override;

	// Prediction only, defined in src/game/client/prediction/entities_predict.cpp.
	CDoor(CGameWorld *pGameWorld, int Id, const CLaserData *pData);
	bool Match(const CDoor *pDoor) const;
	void Read(const CLaserData *pData);
	void Destroy() override;
	// Server only, defined in src/game/server/snap.cpp
	void Snap(int SnappingClient);
};

#endif // GAME_ENTITIES_DOOR_H
