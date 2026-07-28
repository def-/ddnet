/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */

#include "gameworld.h"

#include "entity.h"

#include <engine/shared/config.h>

#include <game/collision.h>
#include <game/mapbugs.h>
#include <game/mapitems.h>
#include <game/entities/character.h>
#include <game/gameenv.h>

#include <algorithm>
#include <utility>

//////////////////////////////////////////////////
// game world
//////////////////////////////////////////////////
CGameWorld::CGameWorld()
{
	m_pGameServer = nullptr;
	m_pConfig = nullptr;
	m_pServer = nullptr;
	m_pCollision = nullptr;
	m_pTuningList = nullptr;
	m_pMapBugs = nullptr;

	m_Paused = false;
	m_ResetRequested = false;
	for(auto &pFirstEntityType : m_apFirstEntityTypes)
		pFirstEntityType = nullptr;
	for(auto &pCharacter : m_apCharacters)
		pCharacter = nullptr;

	// prediction only
	m_GameTick = 0;
	m_LocalClientId = -1;
	m_IsValidCopy = false;
	m_pParent = nullptr;
	m_pChild = nullptr;
	mem_zero(&m_WorldConfig, sizeof(m_WorldConfig));
}

bool CGameWorld::EmulateBug(int Bug) const
{
	return m_pMapBugs != nullptr && m_pMapBugs->Contains(Bug);
}

void CGameWorld::Init(CCollision *pCollision, CTuningParams *pTuningList, const CMapBugs *pMapBugs)
{
	m_pCollision = pCollision;
	m_pTuningList = pTuningList;
	m_pMapBugs = pMapBugs;
}

CEntity *CGameWorld::FindLast(int Type)
{
	CEntity *pLast = FindFirst(Type);
	if(pLast)
		while(pLast->TypeNext())
			pLast = pLast->TypeNext();
	return pLast;
}

// Timed switches expire here for the prediction. The server does it in
// CGameContext::OnTick instead, at a different point in the tick, so the flag
// keeps it from happening twice.
void CGameWorld::ExpireSwitchers()
{
	for(auto &Switcher : Switchers())
	{
		for(int j = 0; j < NUM_DDRACE_TEAMS; ++j)
		{
			if(Switcher.m_aEndTick[j] <= GameTick() && Switcher.m_aType[j] == TILE_SWITCHTIMEDOPEN)
			{
				Switcher.m_aStatus[j] = false;
				Switcher.m_aEndTick[j] = 0;
				Switcher.m_aType[j] = TILE_SWITCHCLOSE;
			}
			else if(Switcher.m_aEndTick[j] <= GameTick() && Switcher.m_aType[j] == TILE_SWITCHTIMEDCLOSE)
			{
				Switcher.m_aStatus[j] = true;
				Switcher.m_aEndTick[j] = 0;
				Switcher.m_aType[j] = TILE_SWITCHOPEN;
			}
		}
	}
}

CGameWorld::~CGameWorld()
{
	// delete all entities
	for(auto &pFirstEntityType : m_apFirstEntityTypes)
		while(pFirstEntityType)
			delete pFirstEntityType; // NOLINT(clang-analyzer-cplusplus.NewDelete)
}

void CGameWorld::CreateExplosion(vec2 Pos, int Owner, int Weapon, bool NoDamage, int ActivatedTeam, CClientMask Mask, int Id)
{
	Env()->CreateExplosionEvent(Pos, Mask, Id);

	// deal damage
	CEntity *apEnts[MAX_CLIENTS];
	float Radius = 135.0f;
	float InnerRadius = 48.0f;
	int Num = FindEntities(Pos, Radius, apEnts, MAX_CLIENTS, CGameWorld::ENTTYPE_CHARACTER);
	CClientMask TeamMask = CClientMask().set();
	for(int i = 0; i < Num; i++)
	{
		auto *pChr = static_cast<CCharacter *>(apEnts[i]);
		vec2 Diff = pChr->m_Pos - Pos;
		vec2 ForceDir(0, 1);
		float l = length(Diff);
		if(l)
			ForceDir = normalize(Diff);
		l = 1 - std::clamp((l - InnerRadius) / (Radius - InnerRadius), 0.0f, 1.0f);
		const int TuneZone = ExplosionTuneZone(Owner);
		const float Strength = TuneZone == 0 ? GlobalTuning()->m_ExplosionStrength : GetTuning(TuneZone)->m_ExplosionStrength;

		float Dmg = Strength * l;
		if(!(int)Dmg)
			continue;

		if((GetCharacterById(Owner) ? !GetCharacterById(Owner)->GrenadeHitDisabled() : g_Config.m_SvHit) || NoDamage || Owner == pChr->GetCid())
		{
			if(Owner != -1 && pChr->IsAlive() && !pChr->CanCollide(Owner))
				continue;
			if(Owner == -1 && ActivatedTeam != -1 && pChr->IsAlive() && pChr->Team() != ActivatedTeam)
				continue;

			// Explode at most once per team
			int PlayerTeam = pChr->Team();
			if((GetCharacterById(Owner) ? GetCharacterById(Owner)->GrenadeHitDisabled() : !g_Config.m_SvHit) || NoDamage)
			{
				if(PlayerTeam == TEAM_SUPER)
					continue;
				if(!TeamMask.test(PlayerTeam))
					continue;
				TeamMask.reset(PlayerTeam);
			}

			pChr->TakeDamage(ForceDir * Dmg * 2, (int)Dmg, Owner, Weapon);
		}
	}
}

CEntity *CGameWorld::FindFirst(int Type)
{
	return Type < 0 || Type >= NUM_ENTTYPES ? nullptr : m_apFirstEntityTypes[Type];
}

int CGameWorld::FindEntities(vec2 Pos, float Radius, CEntity **ppEnts, int Max, int Type)
{
	if(Type < 0 || Type >= NUM_ENTTYPES)
		return 0;

	int Num = 0;
	for(CEntity *pEnt = m_apFirstEntityTypes[Type]; pEnt; pEnt = pEnt->m_pNextTypeEntity)
	{
		if(distance(pEnt->m_Pos, Pos) < Radius + pEnt->m_ProximityRadius)
		{
			if(ppEnts)
				ppEnts[Num] = pEnt;
			Num++;
			if(Num == Max)
				break;
		}
	}

	return Num;
}

void CGameWorld::InsertEntity(CEntity *pEnt, bool Last)
{
#ifdef CONF_DEBUG
	for(CEntity *pCur = m_apFirstEntityTypes[pEnt->m_ObjType]; pCur; pCur = pCur->m_pNextTypeEntity)
		dbg_assert(pCur != pEnt, "err");
#endif

	// insert it
	if(!Last)
	{
		if(m_apFirstEntityTypes[pEnt->m_ObjType])
			m_apFirstEntityTypes[pEnt->m_ObjType]->m_pPrevTypeEntity = pEnt;
		pEnt->m_pNextTypeEntity = m_apFirstEntityTypes[pEnt->m_ObjType];
		pEnt->m_pPrevTypeEntity = nullptr;
		m_apFirstEntityTypes[pEnt->m_ObjType] = pEnt;
	}
	else
	{
		// insert it at the end of the list
		CEntity *pLast = m_apFirstEntityTypes[pEnt->m_ObjType];
		if(pLast)
		{
			while(pLast->m_pNextTypeEntity)
				pLast = pLast->m_pNextTypeEntity;
			pLast->m_pNextTypeEntity = pEnt;
		}
		else
		{
			m_apFirstEntityTypes[pEnt->m_ObjType] = pEnt;
		}
		pEnt->m_pPrevTypeEntity = pLast;
		pEnt->m_pNextTypeEntity = nullptr;
	}
}

void CGameWorld::RemoveEntity(CEntity *pEnt)
{
	// not in the list
	if(!pEnt->m_pNextTypeEntity && !pEnt->m_pPrevTypeEntity && m_apFirstEntityTypes[pEnt->m_ObjType] != pEnt)
		return;

	// remove
	if(pEnt->m_pPrevTypeEntity)
		pEnt->m_pPrevTypeEntity->m_pNextTypeEntity = pEnt->m_pNextTypeEntity;
	else
		m_apFirstEntityTypes[pEnt->m_ObjType] = pEnt->m_pNextTypeEntity;
	if(pEnt->m_pNextTypeEntity)
		pEnt->m_pNextTypeEntity->m_pPrevTypeEntity = pEnt->m_pPrevTypeEntity;

	// keep list traversing valid
	if(m_pNextTraverseEntity == pEnt)
		m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;

	pEnt->m_pNextTypeEntity = nullptr;
	pEnt->m_pPrevTypeEntity = nullptr;
}

void CGameWorld::RemoveEntitiesFromPlayer(int PlayerId)
{
	RemoveEntitiesFromPlayers(&PlayerId, 1);
}

void CGameWorld::RemoveEntitiesFromPlayers(int PlayerIds[], int NumPlayers)
{
	for(auto *pEnt : m_apFirstEntityTypes)
	{
		for(; pEnt;)
		{
			m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
			for(int i = 0; i < NumPlayers; i++)
			{
				if(pEnt->GetOwnerId() == PlayerIds[i])
				{
					RemoveEntity(pEnt);
					pEnt->Destroy();
					break;
				}
			}
			pEnt = m_pNextTraverseEntity;
		}
	}
}

void CGameWorld::RemoveEntities()
{
	// destroy objects marked for destruction
	for(auto *pEnt : m_apFirstEntityTypes)
		for(; pEnt;)
		{
			m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
			if(pEnt->m_MarkedForDestroy)
			{
				RemoveEntity(pEnt);
				pEnt->Destroy();
			}
			pEnt = m_pNextTraverseEntity;
		}
}

void CGameWorld::Tick()
{
	if(m_ResetRequested)
		Reset();

	if(!m_Paused)
	{
		// update all objects
		for(int i = 0; i < NUM_ENTTYPES; i++)
		{
			// It's important to call PreTick() and Tick() after each other.
			// If we call PreTick() before, and Tick() after other entities have been processed, it causes physics changes such as a stronger shotgun or grenade.
			if(g_Config.m_SvNoWeakHook && i == ENTTYPE_CHARACTER)
			{
				auto *pEnt = m_apFirstEntityTypes[i];
				for(; pEnt;)
				{
					m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
					((CCharacter *)pEnt)->PreTick();
					pEnt = m_pNextTraverseEntity;
				}
			}

			auto *pEnt = m_apFirstEntityTypes[i];
			for(; pEnt;)
			{
				m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
				pEnt->Tick();
				pEnt = m_pNextTraverseEntity;
			}
		}

		for(auto *pEnt : m_apFirstEntityTypes)
			for(; pEnt;)
			{
				m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
				pEnt->TickDeferred();
				pEnt = m_pNextTraverseEntity;
			}
	}
	else
	{
		// update all objects
		for(auto *pEnt : m_apFirstEntityTypes)
			for(; pEnt;)
			{
				m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
				pEnt->TickPaused();
				pEnt = m_pNextTraverseEntity;
			}
	}

	RemoveEntities();

	if(m_ExpireSwitchersInTick)
		ExpireSwitchers();

	// find the characters' strong/weak id
	int StrongWeakId = 0;
	for(CCharacter *pChar = (CCharacter *)FindFirst(ENTTYPE_CHARACTER); pChar; pChar = (CCharacter *)pChar->TypeNext())
	{
		pChar->m_StrongWeakId = StrongWeakId;
		StrongWeakId++;
	}
}

void CGameWorld::SwapClients(int Client1, int Client2)
{
	// update all objects
	for(auto *pEnt : m_apFirstEntityTypes)
		for(; pEnt;)
		{
			m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
			pEnt->SwapClients(Client1, Client2);
			pEnt = m_pNextTraverseEntity;
		}
}

CCharacter *CGameWorld::IntersectCharacter(vec2 Pos0, vec2 Pos1, float Radius, vec2 &NewPos, const CCharacter *pNotThis, int CollideWith, const CCharacter *pThisOnly)
{
	return (CCharacter *)IntersectEntity(Pos0, Pos1, Radius, ENTTYPE_CHARACTER, NewPos, pNotThis, CollideWith, pThisOnly);
}

CEntity *CGameWorld::IntersectEntity(vec2 Pos0, vec2 Pos1, float Radius, int Type, vec2 &NewPos, const CEntity *pNotThis, int CollideWith, const CEntity *pThisOnly)
{
	float ClosestLen = distance(Pos0, Pos1) * 100.0f;
	CEntity *pClosest = nullptr;

	CEntity *pEntity = FindFirst(Type);
	for(; pEntity; pEntity = pEntity->TypeNext())
	{
		if(pEntity == pNotThis)
			continue;

		if(pThisOnly && pEntity != pThisOnly)
			continue;

		if(CollideWith != -1 && !pEntity->CanCollide(CollideWith))
			continue;

		vec2 IntersectPos;
		if(closest_point_on_line(Pos0, Pos1, pEntity->m_Pos, IntersectPos))
		{
			float Len = distance(pEntity->m_Pos, IntersectPos);
			if(Len < pEntity->m_ProximityRadius + Radius)
			{
				Len = distance(Pos0, IntersectPos);
				if(Len < ClosestLen)
				{
					NewPos = IntersectPos;
					ClosestLen = Len;
					pClosest = pEntity;
				}
			}
		}
	}

	return pClosest;
}

CCharacter *CGameWorld::ClosestCharacter(vec2 Pos, float Radius, const CEntity *pNotThis)
{
	// Find other players
	float ClosestRange = Radius * 2;
	CCharacter *pClosest = nullptr;

	CCharacter *p = (CCharacter *)FindFirst(ENTTYPE_CHARACTER);
	for(; p; p = (CCharacter *)p->TypeNext())
	{
		if(p == pNotThis)
			continue;

		float Len = distance(Pos, p->m_Pos);
		if(Len < p->m_ProximityRadius + Radius)
		{
			if(Len < ClosestRange)
			{
				ClosestRange = Len;
				pClosest = p;
			}
		}
	}

	return pClosest;
}

std::vector<CCharacter *> CGameWorld::IntersectedCharacters(vec2 Pos0, vec2 Pos1, float Radius, const CEntity *pNotThis)
{
	std::vector<CCharacter *> vpCharacters;
	CCharacter *pChr = (CCharacter *)FindFirst(CGameWorld::ENTTYPE_CHARACTER);
	for(; pChr; pChr = (CCharacter *)pChr->TypeNext())
	{
		if(pChr == pNotThis)
			continue;

		vec2 IntersectPos;
		if(closest_point_on_line(Pos0, Pos1, pChr->m_Pos, IntersectPos))
		{
			float Len = distance(pChr->m_Pos, IntersectPos);
			if(Len < pChr->m_ProximityRadius + Radius)
			{
				vpCharacters.push_back(pChr);
			}
		}
	}
	return vpCharacters;
}

void CGameWorld::ReleaseHooked(int ClientId)
{
	CCharacter *pChr = (CCharacter *)FindFirst(CGameWorld::ENTTYPE_CHARACTER);
	for(; pChr; pChr = (CCharacter *)pChr->TypeNext())
	{
		if(pChr->Core()->HookedPlayer() == ClientId && !pChr->IsSuper())
		{
			pChr->ReleaseHook();
		}
	}
}
