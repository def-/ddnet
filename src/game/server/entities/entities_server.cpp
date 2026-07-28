/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */

/*
	The parts of the entities that only the server has.

	The entities themselves are shared with the client's prediction; what is left
	here is the bookkeeping the prediction has no way to do, because it does not
	know who is connected or which clients can see what.
*/

#include <engine/shared/config.h>

#include <game/entities/character.h>
#include <game/entities/door.h>
#include <game/entities/laser.h>
#include <game/server/gamecontext.h>
#include <game/server/player.h>

void CLaser::SyncInteractState()
{
	const bool OwnerConnected = Env()->IsPlayerConnected(m_Owner);
	CCharacter *pOwnerChar = GameWorld()->GetCharacterById(m_Owner);

	// as long as the owner is connected
	// refill the state on tick
	// as soon as the owner disconnects keep that state
	if(OwnerConnected)
	{
		bool NoHitOthers = g_Config.m_SvHit;
		if(pOwnerChar)
			NoHitOthers = (m_Type == WEAPON_LASER && pOwnerChar->LaserHitDisabled()) || (m_Type == WEAPON_SHOTGUN && pOwnerChar->ShotgunHitDisabled());
		bool NoHitSelf = g_Config.m_SvOldLaser || (m_Bounces == 0 && !m_WasTele);
		m_InteractState.FillOwnerConnected(
			pOwnerChar && pOwnerChar->IsAlive(),
			OwnerConnected ? Env()->GetDDRaceTeam(m_Owner) : 0,
			pOwnerChar && pOwnerChar->Core()->m_Solo,
			NoHitOthers,
			NoHitSelf);
	}
	else
	{
		m_InteractState.FillOwnerDisconnected();
	}
}

CClientMask CLaser::BounceMask()
{
	return m_InteractState.CanSeeMask(GameWorld()->GameServer());
}

bool CLaser::CanHit(int ClientId)
{
	return m_InteractState.CanHit(GameWorld()->GameServer(), ClientId);
}

// Doors on the server live as long as the map does and set their collision once.
// Only a predicted door has to take its collision back when it goes away.
void CDoor::Destroy()
{
	delete this;
}
