/* (c) Shereef Marzouk. See "licence DDRace.txt" and the readme.txt in the root of the distribution for more information. */
#include "door.h"

#include <game/collision.h>
#include <game/gameenv.h>
#include <game/mapitems.h>

CDoor::CDoor(CGameWorld *pGameWorld, vec2 Pos, float Rotation, int Length,
	int Number) :
	CEntity(pGameWorld, EEntityClass::DOOR, true)
{
	m_Number = Number;
	m_Pos = Pos;
	m_Length = Length;
	m_Direction = vec2(std::sin(Rotation), std::cos(Rotation));
	vec2 To = Pos + normalize(m_Direction) * m_Length;

	Collision()->IntersectNoLaser(Pos, To, &this->m_To, nullptr);
	ResetCollision();
	GameWorld()->InsertEntity(this);
}

void CDoor::ResetCollision()
{
	if(Collision()->GetTile(m_Pos.x, m_Pos.y) || Collision()->GetFrontTile(m_Pos.x, m_Pos.y))
		return;

	for(int i = 0; i < m_Length - 1; i++)
	{
		vec2 CurrentPos = m_Pos + m_Direction * i;
		if(Collision()->CheckPoint(CurrentPos))
			break;
		else
			Collision()->SetDoorCollisionAt(CurrentPos.x, CurrentPos.y, TILE_STOPA, 0, m_Number);
	}
}

void CDoor::Reset()
{
	m_MarkedForDestroy = true;
}
