/* copyright (c) 2007 magnus auvinen, see licence.txt for more info */
#include "gun.h"

#include "character.h"
#include "plasma.h"

#include <engine/server.h>
#include <engine/shared/config.h>

#include <game/mapitems.h>
#include <game/server/gamecontext.h>
#include <game/server/player.h>
#include <game/server/teams.h>

CGun::CGun(CGameWorld *pGameWorld, vec2 Pos, bool Freeze, bool Explosive, int Layer, int Number) :
	CEntity(pGameWorld, EEntityClass::GUN, true)
{
	m_Core = vec2(0.0f, 0.0f);
	m_Pos = Pos;
	m_Freeze = Freeze;
	m_Explosive = Explosive;
	m_Layer = Layer;
	m_Number = Number;
	m_EvalTick = GameWorld()->GameTick();

	std::fill(std::begin(m_aLastFireTeam), std::end(m_aLastFireTeam), 0);
	std::fill(std::begin(m_aLastFireSolo), std::end(m_aLastFireSolo), 0);
	GameWorld()->InsertEntity(this);
}

void CGun::Tick()
{
	if(GameWorld()->GameTick() % (int)(GameWorld()->GameTickSpeed() * 0.15f) == 0)
	{
		m_EvalTick = GameWorld()->GameTick();
		Collision()->MoverSpeed(m_Pos.x, m_Pos.y, &m_Core);
		m_Pos += m_Core;
	}
	if(g_Config.m_SvPlasmaPerSec > 0)
	{
		Fire();
	}
}

void CGun::Fire()
{
	// Create a list of players who are in the range of the turret
	CEntity *apPlayersInRange[MAX_CLIENTS];
	std::fill(std::begin(apPlayersInRange), std::end(apPlayersInRange), nullptr);

	int NumPlayersInRange = GameWorld()->FindEntities(m_Pos, g_Config.m_SvPlasmaRange,
		apPlayersInRange, MAX_CLIENTS, CGameWorld::ENTTYPE_CHARACTER);

	// The closest player (within range) in a team is selected as the target
	int aTargetIdInTeam[MAX_CLIENTS];
	bool aIsTarget[MAX_CLIENTS];
	int aMinDistInTeam[MAX_CLIENTS];
	std::fill(std::begin(aMinDistInTeam), std::end(aMinDistInTeam), 0);
	std::fill(std::begin(aIsTarget), std::end(aIsTarget), false);
	std::fill(std::begin(aTargetIdInTeam), std::end(aTargetIdInTeam), -1);

	for(int i = 0; i < NumPlayersInRange; i++)
	{
		CCharacter *pTarget = static_cast<CCharacter *>(apPlayersInRange[i]);
		const int &TargetTeam = pTarget->Team();
		// Do not fire at super players
		if(TargetTeam == TEAM_SUPER)
		{
			continue;
		}
		// If the turret is disabled for the target's team, the turret will not fire
		if(m_Layer == LAYER_SWITCH && m_Number > 0 &&
			!Switchers()[m_Number].m_aStatus[TargetTeam])
		{
			continue;
		}

		// Turrets can only shoot at a speed of sv_plasma_per_sec
		const int &TargetClientId = pTarget->GetCid();
		const bool &TargetIsSolo = pTarget->TeamsCore()->GetSolo(TargetClientId);
		if((TargetIsSolo &&
			   m_aLastFireSolo[TargetClientId] + GameWorld()->GameTickSpeed() / g_Config.m_SvPlasmaPerSec > GameWorld()->GameTick()) ||
			(!TargetIsSolo &&
				m_aLastFireTeam[TargetTeam] + GameWorld()->GameTickSpeed() / g_Config.m_SvPlasmaPerSec > GameWorld()->GameTick()))
		{
			continue;
		}

		// Turrets can shoot only at reachable, alive players
		int IsReachable = !Collision()->IntersectLine(m_Pos, pTarget->m_Pos, nullptr, nullptr);
		if(IsReachable && pTarget->IsAlive())
		{
			// Turrets fire on solo players regardless of the rest of the team
			if(TargetIsSolo)
			{
				aIsTarget[TargetClientId] = true;
				m_aLastFireSolo[TargetClientId] = GameWorld()->GameTick();
			}
			else
			{
				int Distance = distance(pTarget->m_Pos, m_Pos);
				if(aMinDistInTeam[TargetTeam] == 0 || aMinDistInTeam[TargetTeam] > Distance)
				{
					aMinDistInTeam[TargetTeam] = Distance;
					aTargetIdInTeam[TargetTeam] = TargetClientId;
				}
			}
		}
	}

	// Set the closest player for each team as a target
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(aTargetIdInTeam[i] != -1)
		{
			aIsTarget[aTargetIdInTeam[i]] = true;
			m_aLastFireTeam[i] = GameWorld()->GameTick();
		}
	}

	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		// Fire at each target
		if(aIsTarget[i])
		{
			CCharacter *pTarget = GameWorld()->GetCharacterById(i);
			new CPlasma(GameWorld(), m_Pos, normalize(pTarget->m_Pos - m_Pos), m_Freeze, m_Explosive, i);
		}
	}
}

void CGun::Reset()
{
	m_MarkedForDestroy = true;
}
