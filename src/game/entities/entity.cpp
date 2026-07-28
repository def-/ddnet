/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */

#include "entity.h"

#include <game/collision.h>
#include <game/gameenv.h>

//////////////////////////////////////////////////
// Entity
//////////////////////////////////////////////////
// Which entity list a class lives in. Several classes share ENTTYPE_LASER; that
// grouping decides tick order, so it is kept exactly as it was.
static int ObjTypeOf(EEntityClass EntityClass)
{
	switch(EntityClass)
	{
	case EEntityClass::CHARACTER:
		return CGameWorld::ENTTYPE_CHARACTER;
	case EEntityClass::PROJECTILE:
		return CGameWorld::ENTTYPE_PROJECTILE;
	case EEntityClass::PICKUP:
		return CGameWorld::ENTTYPE_PICKUP;
	case EEntityClass::LASER:
	case EEntityClass::DOOR:
	case EEntityClass::DRAGGER:
	case EEntityClass::DRAGGER_BEAM:
	case EEntityClass::PLASMA:
	case EEntityClass::LIGHT:
	case EEntityClass::GUN:
		return CGameWorld::ENTTYPE_LASER;
	}
	dbg_assert(false, "unhandled EEntityClass");
	dbg_break();
}

CEntity::CEntity(CGameWorld *pGameWorld, EEntityClass EntityClass, bool AllocateSnapId, vec2 Pos, int ProximityRadius)
{
	m_pGameWorld = pGameWorld;
	m_pCCollision = pGameWorld->Collision();

	m_EntityClass = EntityClass;
	m_ObjType = ObjTypeOf(EntityClass);
	m_Pos = Pos;
	m_ProximityRadius = ProximityRadius;

	m_MarkedForDestroy = false;
	if(AllocateSnapId)
		m_Id = Env()->AllocateSnapId();

	m_pPrevTypeEntity = nullptr;
	m_pNextTypeEntity = nullptr;
}

CEntity::~CEntity()
{
	GameWorld()->RemoveEntity(this);
	if(m_Id.has_value())
		Env()->FreeSnapId(m_Id.value());
}

bool CEntity::GameLayerClipped(vec2 CheckPos)
{
	return round_to_int(CheckPos.x) / 32 < -200 || round_to_int(CheckPos.x) / 32 > Collision()->GetWidth() + 200 ||
	       round_to_int(CheckPos.y) / 32 < -200 || round_to_int(CheckPos.y) / 32 > Collision()->GetHeight() + 200;
}

bool CEntity::GetNearestAirPos(vec2 Pos, vec2 PrevPos, vec2 *pOutPos)
{
	for(int k = 0; k < 16 && Collision()->CheckPoint(Pos); k++)
	{
		Pos -= normalize(PrevPos - Pos);
	}

	vec2 PosInBlock = vec2(round_to_int(Pos.x) % 32, round_to_int(Pos.y) % 32);
	vec2 BlockCenter = vec2(round_to_int(Pos.x), round_to_int(Pos.y)) - PosInBlock + vec2(16.0f, 16.0f);

	*pOutPos = vec2(BlockCenter.x + (PosInBlock.x < 16 ? -2.0f : 1.0f), Pos.y);
	if(!Collision()->TestBox(*pOutPos, CCharacterCore::PhysicalSizeVec2()))
		return true;

	*pOutPos = vec2(Pos.x, BlockCenter.y + (PosInBlock.y < 16 ? -2.0f : 1.0f));
	if(!Collision()->TestBox(*pOutPos, CCharacterCore::PhysicalSizeVec2()))
		return true;

	*pOutPos = vec2(BlockCenter.x + (PosInBlock.x < 16 ? -2.0f : 1.0f),
		BlockCenter.y + (PosInBlock.y < 16 ? -2.0f : 1.0f));
	return !Collision()->TestBox(*pOutPos, CCharacterCore::PhysicalSizeVec2());
}

bool CEntity::GetNearestAirPosPlayer(vec2 PlayerPos, vec2 *pOutPos)
{
	for(int Distance = 5; Distance >= -1; Distance--)
	{
		*pOutPos = vec2(PlayerPos.x, PlayerPos.y - Distance);
		if(!Collision()->TestBox(*pOutPos, CCharacterCore::PhysicalSizeVec2()))
		{
			return true;
		}
	}
	return false;
}
