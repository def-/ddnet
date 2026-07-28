/* (c) Shereef Marzouk. See "licence DDRace.txt" and the readme.txt in the root of the distribution for more information. */
#include "dragger.h"

#include <game/entities/character.h>
#include "dragger_beam.h"

#include <engine/server.h>
#include <engine/shared/config.h>

#include <game/mapitems.h>
#include <game/server/gamecontext.h>
#include <game/server/player.h>
#include <game/server/teams.h>

CDragger::CDragger(CGameWorld *pGameWorld, vec2 Pos, float Strength, bool IgnoreWalls, int Layer, int Number) :
	CEntity(pGameWorld, EEntityClass::DRAGGER, true)
{
	m_Core = vec2(0.0f, 0.0f);
	m_Pos = Pos;
	m_Strength = Strength;
	m_IgnoreWalls = IgnoreWalls;
	m_Layer = Layer;
	m_Number = Number;
	m_EvalTick = GameWorld()->GameTick();

	for(auto &TargetId : m_aTargetIdInTeam)
	{
		TargetId = -1;
	}
	std::fill(std::begin(m_apDraggerBeam), std::end(m_apDraggerBeam), nullptr);
	GameWorld()->InsertEntity(this);
}

void CDragger::Tick()
{
	if(GameWorld()->GameTick() % (int)(GameWorld()->GameTickSpeed() * 0.15f) == 0)
	{
		m_EvalTick = GameWorld()->GameTick();
		Collision()->MoverSpeed(m_Pos.x, m_Pos.y, &m_Core);
		m_Pos += m_Core;

		// Adopt the new position for all outgoing laser beams
		for(auto &DraggerBeam : m_apDraggerBeam)
		{
			if(DraggerBeam != nullptr)
			{
				DraggerBeam->SetPos(m_Pos);
			}
		}

		LookForPlayersToDrag();
	}
}

void CDragger::LookForPlayersToDrag()
{
	// Create a list of players who are in the range of the dragger
	CEntity *apPlayersInRange[MAX_CLIENTS];
	std::fill(std::begin(apPlayersInRange), std::end(apPlayersInRange), nullptr);

	int NumPlayersInRange = GameWorld()->FindEntities(m_Pos,
		g_Config.m_SvDraggerRange - CCharacterCore::PhysicalSize(),
		apPlayersInRange, MAX_CLIENTS, CGameWorld::ENTTYPE_CHARACTER);

	// The closest player (within range) in a team is selected as the target
	int aClosestTargetIdInTeam[MAX_CLIENTS];
	bool aCanStillBeTeamTarget[MAX_CLIENTS];
	bool aIsTarget[MAX_CLIENTS];
	int aMinDistInTeam[MAX_CLIENTS];
	std::fill(std::begin(aCanStillBeTeamTarget), std::end(aCanStillBeTeamTarget), false);
	std::fill(std::begin(aMinDistInTeam), std::end(aMinDistInTeam), 0);
	std::fill(std::begin(aIsTarget), std::end(aIsTarget), false);
	std::fill(std::begin(aClosestTargetIdInTeam), std::end(aClosestTargetIdInTeam), -1);

	for(int i = 0; i < NumPlayersInRange; i++)
	{
		CCharacter *pTarget = static_cast<CCharacter *>(apPlayersInRange[i]);
		const int &TargetTeam = pTarget->Team();

		// Do not create a dragger beam for super player
		if(TargetTeam == TEAM_SUPER)
		{
			continue;
		}
		// If the dragger is disabled for the target's team, no dragger beam will be generated
		if(m_Layer == LAYER_SWITCH && m_Number > 0 &&
			!Switchers()[m_Number].m_aStatus[TargetTeam])
		{
			continue;
		}

		// Dragger beams can be created only for reachable, alive players
		int IsReachable =
			m_IgnoreWalls ?
				!Collision()->IntersectNoLaserNoWalls(m_Pos, pTarget->m_Pos, nullptr, nullptr) :
				!Collision()->IntersectNoLaser(m_Pos, pTarget->m_Pos, nullptr, nullptr);
		if(IsReachable && pTarget->IsAlive())
		{
			const int &TargetClientId = pTarget->GetCid();
			// Solo players are dragged independently from the rest of the team
			if(pTarget->TeamsCore()->GetSolo(TargetClientId))
			{
				aIsTarget[TargetClientId] = true;
			}
			else
			{
				int Distance = distance(pTarget->m_Pos, m_Pos);
				if(aMinDistInTeam[TargetTeam] == 0 || aMinDistInTeam[TargetTeam] > Distance)
				{
					aMinDistInTeam[TargetTeam] = Distance;
					aClosestTargetIdInTeam[TargetTeam] = TargetClientId;
				}
				aCanStillBeTeamTarget[TargetClientId] = true;
			}
		}
	}

	// Set the closest player for each team as a target if the team does not have a target player yet
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if((m_aTargetIdInTeam[i] != -1 && !aCanStillBeTeamTarget[m_aTargetIdInTeam[i]]) || m_aTargetIdInTeam[i] == -1)
		{
			m_aTargetIdInTeam[i] = aClosestTargetIdInTeam[i];
		}
		if(m_aTargetIdInTeam[i] != -1)
		{
			aIsTarget[m_aTargetIdInTeam[i]] = true;
		}
	}

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		// Create Dragger Beams which have not been created yet
		if(aIsTarget[i] && m_apDraggerBeam[i] == nullptr)
		{
			m_apDraggerBeam[i] = new CDraggerBeam(GameWorld(), this, m_Pos, m_Strength, m_IgnoreWalls, i, m_Layer, m_Number);
			// The generated dragger beam is placed in the first position in the tick sequence and would therefore
			// no longer be executed automatically in this tick. To execute the dragger beam nevertheless already
			// this tick we call it manually (we do this to keep the old game logic)
			m_apDraggerBeam[i]->Tick();
		}
		// Remove dragger beams that have not yet been deleted
		else if(!aIsTarget[i] && m_apDraggerBeam[i] != nullptr)
		{
			m_apDraggerBeam[i]->Reset();
		}
	}
}

void CDragger::RemoveDraggerBeam(int ClientId)
{
	m_apDraggerBeam[ClientId] = nullptr;
}

void CDragger::Reset()
{
	m_MarkedForDestroy = true;
}

void CDragger::SwapClients(int Client1, int Client2)
{
	std::swap(m_apDraggerBeam[Client1], m_apDraggerBeam[Client2]);
	for(int &TargetId : m_aTargetIdInTeam)
	{
		TargetId = TargetId == Client1 ? Client2 : (TargetId == Client2 ? Client1 : TargetId);
	}
}
