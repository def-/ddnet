/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */

/*
	Snapshotting for every entity.

	This lives apart from the entities themselves because the entity classes are
	shared with the client's prediction, which has no snapshots to build and no
	CGameContext to build them from. Snap() is therefore not a virtual on
	CEntity: CGameWorld::Snap dispatches on CEntity::EntityClass() and calls the
	non-virtual Snap() of the concrete class, all of which are defined here.
*/

#include "entities/character.h"
#include "entities/door.h"
#include "entities/dragger.h"
#include "entities/dragger_beam.h"
#include "entities/gun.h"
#include "entities/laser.h"
#include "entities/light.h"
#include "entities/pickup.h"
#include "entities/plasma.h"
#include "entities/projectile.h"
#include "gamecontext.h"
#include "gamecontroller.h"
#include <game/entities/gameworld.h>
#include "player.h"
#include "teams.h"

#include <engine/shared/config.h>

#include <generated/protocol.h>
#include <generated/server_data.h>

#include <game/mapitems.h>
#include <game/teamscore.h>

#include <algorithm>

//////////////////////////////////////////////////
// CGameWorld
//////////////////////////////////////////////////
static void SnapEntity(CEntity *pEnt, int SnappingClient)
{
	switch(pEnt->EntityClass())
	{
	case EEntityClass::CHARACTER:
		static_cast<CCharacter *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::PROJECTILE:
		static_cast<CProjectile *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::LASER:
		static_cast<CLaser *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::DOOR:
		static_cast<CDoor *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::DRAGGER:
		static_cast<CDragger *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::DRAGGER_BEAM:
		static_cast<CDraggerBeam *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::PLASMA:
		static_cast<CPlasma *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::LIGHT:
		static_cast<CLight *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::GUN:
		static_cast<CGun *>(pEnt)->Snap(SnappingClient);
		return;
	case EEntityClass::PICKUP:
		static_cast<CPickup *>(pEnt)->Snap(SnappingClient);
		return;
	}
	dbg_assert(false, "unhandled EEntityClass");
}

void CGameWorld::Snap(int SnappingClient)
{
	for(CEntity *pEnt = m_apFirstEntityTypes[ENTTYPE_CHARACTER]; pEnt;)
	{
		m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
		SnapEntity(pEnt, SnappingClient);
		pEnt = m_pNextTraverseEntity;
	}

	for(int i = 0; i < NUM_ENTTYPES; i++)
	{
		if(i == ENTTYPE_CHARACTER)
			continue;

		for(CEntity *pEnt = m_apFirstEntityTypes[i]; pEnt;)
		{
			m_pNextTraverseEntity = pEnt->m_pNextTypeEntity;
			SnapEntity(pEnt, SnappingClient);
			pEnt = m_pNextTraverseEntity;
		}
	}
}

//////////////////////////////////////////////////
// CCharacter
//////////////////////////////////////////////////

void CCharacter::SnapCharacter(int SnappingClient, int Id)
{
	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);
	CCharacterCore *pCore;
	int Weapon = m_Core.m_ActiveWeapon, AmmoCount = 0,
	    Health = 0, Armor = 0;
	int Emote = DetermineEyeEmote();
	int Tick;
	if(!m_ReckoningTick || GameServer()->m_pController->IsGamePaused())
	{
		Tick = 0;
		pCore = &m_Core;
	}
	else
	{
		Tick = m_ReckoningTick;
		pCore = &m_SendCore;
	}

	// use ninja graphic for old clients if player is frozen
	if((m_Core.m_DeepFrozen || m_FreezeTime > 0) && SnappingClientVersion < VERSION_DDNET_NEW_HUD)
		Weapon = WEAPON_NINJA;

	// solo, collision, jetpack and ninjajetpack prediction
	if(m_pPlayer->GetCid() == SnappingClient)
	{
		int Faketuning = 0;
		if(m_pPlayer->GetClientVersion() < VERSION_DDNET_NEW_HUD)
		{
			if(m_Core.m_Jetpack && Weapon != WEAPON_NINJA)
				Faketuning |= FAKETUNE_JETPACK;
			if(m_Core.m_Solo)
				Faketuning |= FAKETUNE_SOLO;
			if(m_Core.m_HammerHitDisabled)
				Faketuning |= FAKETUNE_NOHAMMER;
			if(m_Core.m_CollisionDisabled)
				Faketuning |= FAKETUNE_NOCOLL;
			if(m_Core.m_HookHitDisabled)
				Faketuning |= FAKETUNE_NOHOOK;
			if(!m_Core.m_EndlessJump && m_Core.m_Jumps == 0)
				Faketuning |= FAKETUNE_NOJUMP;
		}
		if(Faketuning != m_NeededFaketuning)
		{
			m_NeededFaketuning = Faketuning;
			GameServer()->SendTuningParams(m_pPlayer->GetCid(), m_TuneZone); // update tunings
		}
	}

	// use ninja graphic and set ammo count if player has ninjajetpack
	if(m_pPlayer->m_NinjaJetpack && m_Core.m_Jetpack && m_Core.m_ActiveWeapon == WEAPON_GUN && !m_Core.m_DeepFrozen && m_FreezeTime == 0 && !m_Core.m_HasTelegunGun)
	{
		Weapon = WEAPON_NINJA;
		AmmoCount = 10;
	}

	if(m_pPlayer->GetCid() == SnappingClient || SnappingClient == SERVER_DEMO_CLIENT ||
		(!g_Config.m_SvStrictSpectateMode && m_pPlayer->GetCid() == GameServer()->m_apPlayers[SnappingClient]->SpectatorId()))
	{
		Health = m_Health;
		Armor = m_Armor;
		AmmoCount = (m_FreezeTime == 0 && m_Core.m_ActiveWeapon >= 0) ? m_Core.m_aWeapons[m_Core.m_ActiveWeapon].m_Ammo : 0;
	}

	if(!Server()->IsSixup(SnappingClient))
	{
		CNetObj_Character Character = {};

		pCore->Write(&Character);

		Character.m_Tick = Tick;
		Character.m_Emote = Emote;

		if(Character.m_HookedPlayer != -1)
		{
			if(!Server()->Translate(Character.m_HookedPlayer, SnappingClient))
				Character.m_HookedPlayer = -1;
		}

		Character.m_AttackTick = m_AttackTick;
		Character.m_Direction = m_Input.m_Direction;
		Character.m_Weapon = Weapon;
		Character.m_AmmoCount = AmmoCount;
		Character.m_Health = Health;
		Character.m_Armor = Armor;
		Character.m_PlayerFlags = GetPlayer()->m_PlayerFlags;

		Server()->SnapNewItem(Id, Character);
	}
	else
	{
		protocol7::CNetObj_Character Character = {};

		pCore->Write(reinterpret_cast<CNetObj_CharacterCore *>(static_cast<protocol7::CNetObj_CharacterCore *>(&Character)));
		if(Character.m_Angle > (int)(pi * 256.0f))
		{
			Character.m_Angle -= (int)(2.0f * pi * 256.0f);
		}

		// m_HookTick can be negative when using the hook_duration tune, which 0.7 clients
		// will consider invalid. https://github.com/ddnet/ddnet/issues/3915
		Character.m_HookTick = std::max(0, Character.m_HookTick);

		Character.m_Tick = Tick;
		Character.m_Emote = Emote;
		Character.m_AttackTick = m_AttackTick;
		Character.m_Direction = m_Input.m_Direction;
		Character.m_Weapon = Weapon;
		Character.m_AmmoCount = AmmoCount;

		if(m_FreezeTime > 0 || m_Core.m_DeepFrozen)
			Character.m_AmmoCount = m_Core.m_FreezeStart + g_Config.m_SvFreezeDelay * Server()->TickSpeed();
		else if(Weapon == WEAPON_NINJA)
			Character.m_AmmoCount = m_Core.m_Ninja.m_ActivationTick + g_pData->m_Weapons.m_Ninja.m_Duration * Server()->TickSpeed() / 1000;

		Character.m_Health = Health;
		Character.m_Armor = Armor;
		Character.m_TriggeredEvents = m_TriggeredEvents7;

		Server()->SnapNewItem(Id, Character);
	}
}

bool CCharacter::CanSnapCharacter(int SnappingClient)
{
	if(SnappingClient == SERVER_DEMO_CLIENT)
		return true;

	CCharacter *pSnapChar = GameServer()->GetPlayerChar(SnappingClient);
	CPlayer *pSnapPlayer = GameServer()->m_apPlayers[SnappingClient];

	if(pSnapPlayer->GetTeam() == TEAM_SPECTATORS || pSnapPlayer->IsPaused())
	{
		if(pSnapPlayer->SpectatorId() != SPEC_FREEVIEW && !CanCollide(pSnapPlayer->SpectatorId()) && (pSnapPlayer->m_ShowOthers == SHOW_OTHERS_OFF || (pSnapPlayer->m_ShowOthers == SHOW_OTHERS_ONLY_TEAM && !SameTeam(pSnapPlayer->SpectatorId()))))
			return false;
		else if(pSnapPlayer->SpectatorId() == SPEC_FREEVIEW && !CanCollide(SnappingClient) && pSnapPlayer->m_SpecTeam && !SameTeam(SnappingClient))
			return false;
	}
	else if(pSnapChar && !pSnapChar->m_Core.m_Super && !CanCollide(SnappingClient) && (pSnapPlayer->m_ShowOthers == SHOW_OTHERS_OFF || (pSnapPlayer->m_ShowOthers == SHOW_OTHERS_ONLY_TEAM && !SameTeam(SnappingClient))))
		return false;

	return true;
}

bool CCharacter::IsSnappingCharacterInView(int SnappingClientId)
{
	int Id = m_pPlayer->GetCid();

	// A player may not be clipped away if their hook or a hook attached to them is in the field of view
	bool PlayerAndHookNotInView = NetworkClippedLine(SnappingClientId, m_Pos, m_Core.m_HookPos);
	bool AttachedHookInView = false;
	if(PlayerAndHookNotInView)
	{
		for(const auto &AttachedPlayerId : m_Core.m_AttachedPlayers)
		{
			const CCharacter *pOtherPlayer = GameServer()->GetPlayerChar(AttachedPlayerId);
			if(pOtherPlayer && pOtherPlayer->m_Core.HookedPlayer() == Id)
			{
				if(!NetworkClippedLine(SnappingClientId, m_Pos, pOtherPlayer->m_Pos))
				{
					AttachedHookInView = true;
					break;
				}
			}
		}
	}
	if(PlayerAndHookNotInView && !AttachedHookInView)
	{
		return false;
	}
	return true;
}

void CCharacter::Snap(int SnappingClient)
{
	int Id = m_pPlayer->GetCid();

	if(!Server()->Translate(Id, SnappingClient))
		return;

	if(!CanSnapCharacter(SnappingClient))
	{
		return;
	}

	// always snap the snapping client, even if it is not in view
	if(!IsSnappingCharacterInView(SnappingClient) && Id != SnappingClient)
		return;

	SnapCharacter(SnappingClient, Id);

	CNetObj_DDNetCharacter DDNetCharacter = {};

	DDNetCharacter.m_Flags = 0;
	if(m_Core.m_Solo)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_SOLO;
	if(m_Core.m_Super)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_SUPER;
	if(m_Core.m_Invincible)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_INVINCIBLE;
	if(m_Core.m_EndlessHook)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_ENDLESS_HOOK;
	if(m_Core.m_CollisionDisabled || !GetTuning(m_TuneZone)->m_PlayerCollision)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_COLLISION_DISABLED;
	if(m_Core.m_HookHitDisabled || !GetTuning(m_TuneZone)->m_PlayerHooking)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_HOOK_HIT_DISABLED;
	if(m_Core.m_EndlessJump)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_ENDLESS_JUMP;
	if(m_Core.m_Jetpack)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_JETPACK;
	if(m_Core.m_HammerHitDisabled)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_HAMMER_HIT_DISABLED;
	if(m_Core.m_ShotgunHitDisabled)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_SHOTGUN_HIT_DISABLED;
	if(m_Core.m_GrenadeHitDisabled)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_GRENADE_HIT_DISABLED;
	if(m_Core.m_LaserHitDisabled)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_LASER_HIT_DISABLED;
	if(m_Core.m_HasTelegunGun)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_TELEGUN_GUN;
	if(m_Core.m_HasTelegunGrenade)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_TELEGUN_GRENADE;
	if(m_Core.m_HasTelegunLaser)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_TELEGUN_LASER;
	if(m_Core.m_aWeapons[WEAPON_HAMMER].m_Got)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_WEAPON_HAMMER;
	if(m_Core.m_aWeapons[WEAPON_GUN].m_Got)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_WEAPON_GUN;
	if(m_Core.m_aWeapons[WEAPON_SHOTGUN].m_Got)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_WEAPON_SHOTGUN;
	if(m_Core.m_aWeapons[WEAPON_GRENADE].m_Got)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_WEAPON_GRENADE;
	if(m_Core.m_aWeapons[WEAPON_LASER].m_Got)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_WEAPON_LASER;
	if(m_Core.m_ActiveWeapon == WEAPON_NINJA)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_WEAPON_NINJA;
	if(m_Core.m_LiveFrozen)
		DDNetCharacter.m_Flags |= CHARACTERFLAG_MOVEMENTS_DISABLED;

	DDNetCharacter.m_FreezeEnd = m_Core.m_DeepFrozen ? -1 : (m_FreezeTime == 0 ? 0 : Server()->Tick() + m_FreezeTime);
	DDNetCharacter.m_Jumps = m_Core.m_Jumps;
	DDNetCharacter.m_TeleCheckpoint = m_TeleCheckpoint;
	DDNetCharacter.m_StrongWeakId = m_StrongWeakId;

	// Display Information
	DDNetCharacter.m_JumpedTotal = m_Core.m_JumpedTotal;
	DDNetCharacter.m_NinjaActivationTick = m_Core.m_Ninja.m_ActivationTick;
	DDNetCharacter.m_FreezeStart = m_Core.m_FreezeStart;
	if(m_Core.m_IsInFreeze)
	{
		DDNetCharacter.m_Flags |= CHARACTERFLAG_IN_FREEZE;
	}
	if(Teams()->IsPractice(Team()))
	{
		DDNetCharacter.m_Flags |= CHARACTERFLAG_PRACTICE_MODE;
	}
	if(Teams()->TeamLocked(Team()))
	{
		DDNetCharacter.m_Flags |= CHARACTERFLAG_LOCK_MODE;
	}
	if(Teams()->TeamFlock(Team()))
	{
		DDNetCharacter.m_Flags |= CHARACTERFLAG_TEAM0_MODE;
	}
	DDNetCharacter.m_TargetX = m_Core.m_Input.m_TargetX;
	DDNetCharacter.m_TargetY = m_Core.m_Input.m_TargetY;

	// OVERRIDE_NONE is the default value, the object is zeroed, so it would incorrectly become 0
	DDNetCharacter.m_TuneZoneOverride = TuneZone::OVERRIDE_NONE;

	Server()->SnapNewItem(Id, DDNetCharacter);
}

void CCharacter::PostGlobalSnap()
{
	m_TriggeredEvents7 = 0;
}

//////////////////////////////////////////////////
// CProjectile
//////////////////////////////////////////////////

CNetObj_Projectile CProjectile::NetInfoVanilla() const
{
	CNetObj_Projectile Result = {};
	Result.m_X = (int)m_Pos.x;
	Result.m_Y = (int)m_Pos.y;
	Result.m_VelX = (int)(m_Direction.x * 100.0f);
	Result.m_VelY = (int)(m_Direction.y * 100.0f);
	Result.m_StartTick = m_StartTick;
	Result.m_Type = m_Type;
	return Result;
}

bool CProjectile::NetIsInfoLegacyCompatible() const
{
	const int MaxPos = 0x7fffffff / 100;
	if(absolute((int)m_Pos.y) + 1 >= MaxPos || absolute((int)m_Pos.x) + 1 >= MaxPos)
	{
		//If the modified data would be too large to fit in an integer, send normal data instead
		return false;
	}
	return true;
}

CNetObj_DDRaceProjectile CProjectile::NetInfoLegacy() const
{
	dbg_assert(NetIsInfoLegacyCompatible(), "can't send incompatible projectile");

	//Send additional/modified info, by modifying the fields of the netobj
	float Angle = -std::atan2(m_Direction.x, m_Direction.y);

	int Data = 0;
	Data |= (absolute(m_Owner) & 255) << 0;
	if(m_Owner < 0)
		Data |= LEGACYPROJECTILEFLAG_NO_OWNER;
	//This bit tells the client to use the extra info
	Data |= LEGACYPROJECTILEFLAG_IS_DDNET;
	// LEGACYPROJECTILEFLAG_BOUNCE_HORIZONTAL, LEGACYPROJECTILEFLAG_BOUNCE_VERTICAL
	Data |= (m_Bouncing & 3) << 10;
	if(m_Explosive)
		Data |= LEGACYPROJECTILEFLAG_EXPLOSIVE;
	if(m_Freeze)
		Data |= LEGACYPROJECTILEFLAG_FREEZE;

	CNetObj_DDRaceProjectile Result = {};
	Result.m_X = (int)(m_Pos.x * 100.0f);
	Result.m_Y = (int)(m_Pos.y * 100.0f);
	Result.m_Angle = (int)(Angle * 1000000.0f);
	Result.m_Data = Data;
	Result.m_StartTick = m_StartTick;
	Result.m_Type = m_Type;
	return Result;
}

CNetObj_DDNetProjectile CProjectile::NetInfo() const
{
	CNetObj_DDNetProjectile Result = {};

	int Flags = 0;
	if(m_Bouncing & 1)
	{
		Flags |= PROJECTILEFLAG_BOUNCE_HORIZONTAL;
	}
	if(m_Bouncing & 2)
	{
		Flags |= PROJECTILEFLAG_BOUNCE_VERTICAL;
	}
	if(m_Explosive)
	{
		Flags |= PROJECTILEFLAG_EXPLOSIVE;
	}
	if(m_Freeze)
	{
		Flags |= PROJECTILEFLAG_FREEZE;
	}

	if(m_Owner < 0)
	{
		Result.m_VelX = round_to_int(m_Direction.x * 1e6f);
		Result.m_VelY = round_to_int(m_Direction.y * 1e6f);
	}
	else
	{
		Result.m_VelX = round_to_int(m_InitDir.x);
		Result.m_VelY = round_to_int(m_InitDir.y);
		Flags |= PROJECTILEFLAG_NORMALIZE_VEL;
	}

	Result.m_X = round_to_int(m_Pos.x * 100.0f);
	Result.m_Y = round_to_int(m_Pos.y * 100.0f);
	Result.m_Type = m_Type;
	Result.m_StartTick = m_StartTick;
	Result.m_Owner = m_Owner;
	Result.m_SwitchNumber = m_Number;
	Result.m_TuneZone = m_TuneZone;
	Result.m_Flags = Flags;
	return Result;
}

void CProjectile::Snap(int SnappingClient)
{
	float Ct = (Server()->Tick() - m_StartTick) / (float)Server()->TickSpeed();

	if(NetworkClipped(SnappingClient, GetPos(Ct)) || !GetId().has_value())
		return;

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);
	if(SnappingClientVersion < VERSION_DDNET_ENTITY_NETOBJS)
	{
		CCharacter *pSnapChar = GameServer()->GetPlayerChar(SnappingClient);
		int Tick = (Server()->Tick() % Server()->TickSpeed()) % ((m_Explosive) ? 6 : 20);
		if(pSnapChar && pSnapChar->IsAlive() && (m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[pSnapChar->Team()] && (!Tick)))
			return;
	}

	CCharacter *pOwnerChar = nullptr;
	CClientMask TeamMask = CClientMask().set();

	if(m_Owner >= 0)
		pOwnerChar = GameServer()->GetPlayerChar(m_Owner);

	if(pOwnerChar && pOwnerChar->IsAlive())
		TeamMask = pOwnerChar->TeamMask();

	if(SnappingClient != SERVER_DEMO_CLIENT && m_Owner != -1 && !TeamMask.test(SnappingClient))
		return;

	if(SnappingClientVersion >= VERSION_DDNET_ENTITY_NETOBJS)
	{
		Server()->SnapNewItem(GetId().value(), NetInfo());
	}
	else if(SnappingClientVersion >= VERSION_DDNET_ANTIPING_PROJECTILE && NetIsInfoLegacyCompatible())
	{
		if(SnappingClientVersion >= VERSION_DDNET_MSG_LEGACY)
		{
			Server()->SnapNewItem(GetId().value(), NetInfoLegacy());
		}
		else
		{
			CNetObj_DDRaceProjectile DDRaceProjectile = NetInfoLegacy();
			CNetObj_Projectile Projectile = {};
			static_assert(sizeof(DDRaceProjectile) == sizeof(Projectile));
			mem_copy(&Projectile, &DDRaceProjectile, sizeof(Projectile));
			Server()->SnapNewItem(GetId().value(), Projectile);
		}
	}
	else
	{
		Server()->SnapNewItem(GetId().value(), NetInfoVanilla());
	}
}

//////////////////////////////////////////////////
// CLaser
//////////////////////////////////////////////////

void CLaser::Snap(int SnappingClient)
{
	if((NetworkClipped(SnappingClient) && NetworkClipped(SnappingClient, m_From)) || !GetId().has_value())
		return;

	if(SnappingClient != SERVER_DEMO_CLIENT && !m_InteractState.CanSee(GameServer(), SnappingClient))
		return;

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);
	int LaserType = m_Type == WEAPON_LASER ? LASERTYPE_RIFLE : (m_Type == WEAPON_SHOTGUN ? LASERTYPE_SHOTGUN : -1);

	GameServer()->SnapLaserObject(CSnapContext(SnappingClientVersion, Server()->IsSixup(SnappingClient), SnappingClient), GetId().value(),
		m_Pos, m_From, m_EvalTick, m_Owner, LaserType, 0, m_Number);
}

//////////////////////////////////////////////////
// CPickup
//////////////////////////////////////////////////

void CPickup::Snap(int SnappingClient)
{
	if(NetworkClipped(SnappingClient) || !GetId().has_value())
		return;

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);
	bool Sixup = Server()->IsSixup(SnappingClient);

	if(SnappingClientVersion < VERSION_DDNET_ENTITY_NETOBJS)
	{
		CCharacter *pChar = GameServer()->GetPlayerChar(SnappingClient);

		if(SnappingClient != SERVER_DEMO_CLIENT && (GameServer()->m_apPlayers[SnappingClient]->GetTeam() == TEAM_SPECTATORS || GameServer()->m_apPlayers[SnappingClient]->IsPaused()) && GameServer()->m_apPlayers[SnappingClient]->SpectatorId() != SPEC_FREEVIEW)
			pChar = GameServer()->GetPlayerChar(GameServer()->m_apPlayers[SnappingClient]->SpectatorId());

		int Tick = (Server()->Tick() % Server()->TickSpeed()) % 11;
		if(pChar && pChar->IsAlive() && m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[pChar->Team()] && !Tick)
			return;
	}

	GameServer()->SnapPickup(CSnapContext(SnappingClientVersion, Sixup, SnappingClient), GetId().value(), m_Pos, m_Type, m_Subtype, m_Number, m_Flags);
}

//////////////////////////////////////////////////
// CPlasma
//////////////////////////////////////////////////

void CPlasma::Snap(int SnappingClient)
{
	// Only players who can see the targeted player can see the plasma bullet
	CCharacter *pTarget = GameServer()->GetPlayerChar(m_ForClientId);
	if(!pTarget || !pTarget->CanSnapCharacter(SnappingClient))
	{
		return;
	}

	// Only players with the plasma bullet in their field of view or who want to see everything will receive the snap
	if(NetworkClipped(SnappingClient) || !GetId().has_value())
		return;

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);

	int Subtype = (m_Explosive ? 1 : 0) | (m_Freeze ? 2 : 0);
	GameServer()->SnapLaserObject(CSnapContext(SnappingClientVersion, Server()->IsSixup(SnappingClient), SnappingClient), GetId().value(),
		m_Pos, m_Pos, m_EvalTick, m_ForClientId, LASERTYPE_PLASMA, Subtype, m_Number);
}

//////////////////////////////////////////////////
// CDoor
//////////////////////////////////////////////////

void CDoor::Snap(int SnappingClient)
{
	if((NetworkClipped(SnappingClient, m_Pos) && NetworkClipped(SnappingClient, m_To)) || !GetId().has_value())
		return;

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);

	vec2 From;
	int StartTick;

	if(SnappingClientVersion >= VERSION_DDNET_ENTITY_NETOBJS)
	{
		From = m_To;
		StartTick = -1;
	}
	else
	{
		CCharacter *pChr = GameServer()->GetPlayerChar(SnappingClient);

		if(SnappingClient != SERVER_DEMO_CLIENT && (GameServer()->m_apPlayers[SnappingClient]->GetTeam() == TEAM_SPECTATORS || GameServer()->m_apPlayers[SnappingClient]->IsPaused()) && GameServer()->m_apPlayers[SnappingClient]->SpectatorId() != SPEC_FREEVIEW)
			pChr = GameServer()->GetPlayerChar(GameServer()->m_apPlayers[SnappingClient]->SpectatorId());

		if(pChr && pChr->Team() != TEAM_SUPER && pChr->IsAlive() && !Switchers().empty() && Switchers()[m_Number].m_aStatus[pChr->Team()])
		{
			From = m_To;
		}
		else
		{
			From = m_Pos;
		}
		StartTick = Server()->Tick();
	}

	GameServer()->SnapLaserObject(CSnapContext(SnappingClientVersion, Server()->IsSixup(SnappingClient), SnappingClient), GetId().value(),
		m_Pos, From, StartTick, -1, LASERTYPE_DOOR, 0, m_Number);
}

//////////////////////////////////////////////////
// CDragger
//////////////////////////////////////////////////

bool CDragger::WillDraggerBeamUseDraggerId(int TargetClientId, int SnappingClientId)
{
	// For each snapping client, this must return true for at most one target (i.e. only one of the dragger beams),
	// in which case the dragger itself must not be snapped
	CCharacter *pTargetChar = GameServer()->GetPlayerChar(TargetClientId);
	CCharacter *pSnapChar = GameServer()->GetPlayerChar(SnappingClientId);
	if(pTargetChar && pSnapChar && m_apDraggerBeam[TargetClientId] != nullptr)
	{
		const int SnapTeam = pSnapChar->Team();
		const int TargetTeam = pTargetChar->Team();
		if(SnapTeam == TargetTeam && SnapTeam < MAX_CLIENTS)
		{
			if(pSnapChar->Teams()->m_Core.GetSolo(SnappingClientId) || m_aTargetIdInTeam[SnapTeam] < 0)
			{
				return SnappingClientId == TargetClientId;
			}
			else
			{
				return m_aTargetIdInTeam[SnapTeam] == TargetClientId;
			}
		}
	}
	return false;
}

void CDragger::Snap(int SnappingClient)
{
	// Only players with the dragger in their field of view or who want to see everything will receive the snap
	if(NetworkClipped(SnappingClient) || !GetId().has_value())
		return;

	// Send the dragger in its resting position if the player would not otherwise see a dragger beam within its own team
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		if(WillDraggerBeamUseDraggerId(i, SnappingClient))
		{
			return;
		}
	}

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);

	int Subtype = (m_IgnoreWalls ? 1 : 0) | (std::clamp(round_to_int(m_Strength - 1.f), 0, 2) << 1);

	int StartTick;
	if(SnappingClientVersion >= VERSION_DDNET_ENTITY_NETOBJS)
	{
		StartTick = -1;
	}
	else
	{
		// Emulate turned off blinking dragger for old clients
		CCharacter *pChar = GameServer()->GetPlayerChar(SnappingClient);
		if(SnappingClient != SERVER_DEMO_CLIENT &&
			(GameServer()->m_apPlayers[SnappingClient]->GetTeam() == TEAM_SPECTATORS ||
				GameServer()->m_apPlayers[SnappingClient]->IsPaused()) &&
			GameServer()->m_apPlayers[SnappingClient]->SpectatorId() != SPEC_FREEVIEW)
			pChar = GameServer()->GetPlayerChar(GameServer()->m_apPlayers[SnappingClient]->SpectatorId());

		int Tick = (Server()->Tick() % Server()->TickSpeed()) % 11;
		if(pChar && m_Layer == LAYER_SWITCH && m_Number > 0 &&
			!Switchers()[m_Number].m_aStatus[pChar->Team()] && !Tick)
			return;

		StartTick = m_EvalTick;
		if(StartTick < Server()->Tick() - 4)
			StartTick = Server()->Tick() - 4;
		else if(StartTick > Server()->Tick())
			StartTick = Server()->Tick();
	}

	GameServer()->SnapLaserObject(CSnapContext(SnappingClientVersion, Server()->IsSixup(SnappingClient), SnappingClient), GetId().value(),
		m_Pos, m_Pos, StartTick, -1, LASERTYPE_DRAGGER, Subtype, m_Number);
}

//////////////////////////////////////////////////
// CDraggerBeam
//////////////////////////////////////////////////

void CDraggerBeam::Snap(int SnappingClient)
{
	if(!m_Active)
	{
		return;
	}

	// Only players who can see the player attached to the dragger can see the dragger beam
	CCharacter *pTarget = GameServer()->GetPlayerChar(m_ForClientId);
	if(!pTarget || !pTarget->CanSnapCharacter(SnappingClient))
	{
		return;
	}
	// Only players with the dragger beam in their field of view or who want to see everything will receive the snap
	vec2 TargetPos = vec2(pTarget->m_Pos.x, pTarget->m_Pos.y);
	if(distance(pTarget->m_Pos, m_Pos) >= g_Config.m_SvDraggerRange || NetworkClippedLine(SnappingClient, m_Pos, TargetPos))
	{
		return;
	}

	int Subtype = (m_IgnoreWalls ? 1 : 0) | (std::clamp(round_to_int(m_Strength - 1.f), 0, 2) << 1);

	int StartTick = m_EvalTick;
	if(StartTick < Server()->Tick() - 4)
	{
		StartTick = Server()->Tick() - 4;
	}
	else if(StartTick > Server()->Tick())
	{
		StartTick = Server()->Tick();
	}

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);
	if(SnappingClientVersion >= VERSION_DDNET_ENTITY_NETOBJS)
	{
		StartTick = -1;
	}

	std::optional<int> SnapObjId = GetId();
	if(m_pDragger->WillDraggerBeamUseDraggerId(m_ForClientId, SnappingClient) && m_pDragger->GetId().has_value())
	{
		SnapObjId = m_pDragger->GetId();
	}

	if(!SnapObjId.has_value())
		return;

	GameServer()->SnapLaserObject(CSnapContext(SnappingClientVersion, Server()->IsSixup(SnappingClient), SnappingClient), SnapObjId.value(),
		TargetPos, m_Pos, StartTick, m_ForClientId, LASERTYPE_DRAGGER, Subtype, m_Number);
}

//////////////////////////////////////////////////
// CLight
//////////////////////////////////////////////////

void CLight::Snap(int SnappingClient)
{
	if((NetworkClipped(SnappingClient, m_Pos) && NetworkClipped(SnappingClient, m_To)) || !GetId().has_value())
		return;

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);

	CCharacter *pChr = GameServer()->GetPlayerChar(SnappingClient);

	if(SnappingClient != SERVER_DEMO_CLIENT && (GameServer()->m_apPlayers[SnappingClient]->GetTeam() == TEAM_SPECTATORS || GameServer()->m_apPlayers[SnappingClient]->IsPaused()) && GameServer()->m_apPlayers[SnappingClient]->SpectatorId() != SPEC_FREEVIEW)
		pChr = GameServer()->GetPlayerChar(GameServer()->m_apPlayers[SnappingClient]->SpectatorId());

	vec2 From = m_Pos;
	int StartTick = -1;

	if(pChr && pChr->Team() == TEAM_SUPER)
	{
		From = m_Pos;
	}
	else if(pChr && m_Layer == LAYER_SWITCH && m_Number > 0 && Switchers()[m_Number].m_aStatus[pChr->Team()])
	{
		From = m_To;
	}
	// light on game and switch layer with a number 0 is always on
	else if(m_Layer != LAYER_SWITCH || (m_Layer == LAYER_SWITCH && m_Number == 0))
	{
		From = m_To;
	}

	if(SnappingClientVersion < VERSION_DDNET_ENTITY_NETOBJS)
	{
		int Tick = (Server()->Tick() % Server()->TickSpeed()) % 6;
		if(pChr && pChr->IsAlive() && m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[pChr->Team()] && Tick)
			return;

		StartTick = m_EvalTick;
		if(StartTick < Server()->Tick() - 4)
			StartTick = Server()->Tick() - 4;
		else if(StartTick > Server()->Tick())
			StartTick = Server()->Tick();
	}

	GameServer()->SnapLaserObject(CSnapContext(SnappingClientVersion, Server()->IsSixup(SnappingClient), SnappingClient), GetId().value(),
		m_Pos, From, StartTick, -1, LASERTYPE_FREEZE, 0, m_Number);
}

//////////////////////////////////////////////////
// CGun
//////////////////////////////////////////////////

void CGun::Snap(int SnappingClient)
{
	if(NetworkClipped(SnappingClient) || !GetId().has_value())
		return;

	int SnappingClientVersion = GameServer()->GetClientVersion(SnappingClient);

	int Subtype = (m_Explosive ? 1 : 0) | (m_Freeze ? 2 : 0);

	int StartTick;
	if(SnappingClientVersion >= VERSION_DDNET_ENTITY_NETOBJS)
	{
		StartTick = -1;
	}
	else
	{
		// Emulate turned off blinking turret for old clients
		CCharacter *pChar = GameServer()->GetPlayerChar(SnappingClient);

		if(SnappingClient != SERVER_DEMO_CLIENT &&
			(GameServer()->m_apPlayers[SnappingClient]->GetTeam() == TEAM_SPECTATORS ||
				GameServer()->m_apPlayers[SnappingClient]->IsPaused()) &&
			GameServer()->m_apPlayers[SnappingClient]->SpectatorId() != SPEC_FREEVIEW)
			pChar = GameServer()->GetPlayerChar(GameServer()->m_apPlayers[SnappingClient]->SpectatorId());

		int Tick = (Server()->Tick() % Server()->TickSpeed()) % 11;
		if(pChar && m_Layer == LAYER_SWITCH && m_Number > 0 &&
			!Switchers()[m_Number].m_aStatus[pChar->Team()] && (!Tick))
			return;

		StartTick = m_EvalTick;
	}

	GameServer()->SnapLaserObject(CSnapContext(SnappingClientVersion, Server()->IsSixup(SnappingClient), SnappingClient), GetId().value(),
		m_Pos, m_Pos, StartTick, -1, LASERTYPE_GUN, Subtype, m_Number);
}
