/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */

/*
	The parts of CGameWorld and CEntity that only exist on the server.

	The classes themselves are shared with the client's prediction, which has no
	CGameContext, no players and no snapshots. Everything here either reaches for
	one of those or answers a question that only makes sense on a server, so it
	is declared in the shared header and defined here; the prediction defines its
	own answers in src/game/client/prediction/.
*/

#include "gamecontext.h"
#include "gamecontroller.h"
#include "player.h"
#include "teams.h"

#include <game/collision.h>
#include <game/entities/entity.h>
#include <game/entities/gameworld.h>
#include <game/server/entities/character.h>

//////////////////////////////////////////////////
// CGameWorld
//////////////////////////////////////////////////

void CGameWorld::SetGameServer(CGameContext *pGameServer)
{
	m_pGameServer = pGameServer;
	m_pConfig = m_pGameServer->Config();
	m_pServer = m_pGameServer->Server();
}

IGameEnvironment *CGameWorld::Env()
{
	return m_pGameServer;
}

void CGameWorld::Init(CCollision *pCollision, CTuningParams *pTuningList)
{
	m_Core.InitSwitchers(pCollision->m_HighestSwitchNumber);
	m_pTuningList = pTuningList;
}

int CGameWorld::GameTick() const
{
	return m_pServer->Tick();
}

int CGameWorld::GameTickSpeed() const
{
	return m_pServer->TickSpeed();
}

CCollision *CGameWorld::Collision()
{
	return m_pGameServer->Collision();
}

CCharacter *CGameWorld::GetCharacterById(int ClientId)
{
	return m_pGameServer->GetPlayerChar(ClientId);
}

CTeamsCore *CGameWorld::TeamsCore()
{
	return &m_pGameServer->m_pController->Teams().m_Core;
}

void CGameWorld::Reset()
{
	// reset all entities
	for(auto *pEnt : m_apFirstEntityTypes)
		for(; pEnt;)
		{
			m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
			pEnt->Reset();
			pEnt = m_pNextTraverseEntity;
		}
	RemoveEntities();

	GameServer()->m_pController->OnReset();
	RemoveEntities();

	m_ResetRequested = false;

	GameServer()->CreateAllEntities(false);
}

//////////////////////////////////////////////////
// CEntity
//////////////////////////////////////////////////

bool CEntity::NetworkClipped(int SnappingClient) const
{
	return ::NetworkClipped(m_pGameWorld->GameServer(), SnappingClient, m_Pos);
}

bool CEntity::NetworkClipped(int SnappingClient, vec2 CheckPos) const
{
	return ::NetworkClipped(m_pGameWorld->GameServer(), SnappingClient, CheckPos);
}

bool CEntity::NetworkClippedLine(int SnappingClient, vec2 StartPos, vec2 EndPos) const
{
	return ::NetworkClippedLine(m_pGameWorld->GameServer(), SnappingClient, StartPos, EndPos);
}

bool NetworkClipped(const CGameContext *pGameServer, int SnappingClient, vec2 CheckPos)
{
	if(SnappingClient == SERVER_DEMO_CLIENT || pGameServer->m_apPlayers[SnappingClient]->m_ShowAll)
		return false;

	float dx = pGameServer->m_apPlayers[SnappingClient]->m_ViewPos.x - CheckPos.x;
	if(absolute(dx) > pGameServer->m_apPlayers[SnappingClient]->m_ShowDistance.x)
		return true;

	float dy = pGameServer->m_apPlayers[SnappingClient]->m_ViewPos.y - CheckPos.y;
	return absolute(dy) > pGameServer->m_apPlayers[SnappingClient]->m_ShowDistance.y;
}

bool NetworkClippedLine(const CGameContext *pGameServer, int SnappingClient, vec2 StartPos, vec2 EndPos)
{
	if(SnappingClient == SERVER_DEMO_CLIENT || pGameServer->m_apPlayers[SnappingClient]->m_ShowAll)
		return false;

	vec2 &ViewPos = pGameServer->m_apPlayers[SnappingClient]->m_ViewPos;
	vec2 &ShowDistance = pGameServer->m_apPlayers[SnappingClient]->m_ShowDistance;

	vec2 DistanceToLine, ClosestPoint;
	if(closest_point_on_line(StartPos, EndPos, ViewPos, ClosestPoint))
	{
		DistanceToLine = ViewPos - ClosestPoint;
	}
	else
	{
		// No line section was passed but two equal points
		DistanceToLine = ViewPos - StartPos;
	}
	float ClippDistance = std::max(ShowDistance.x, ShowDistance.y);
	return (absolute(DistanceToLine.x) > ClippDistance || absolute(DistanceToLine.y) > ClippDistance);
}
