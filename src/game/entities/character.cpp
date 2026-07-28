/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#include "character.h"

#include <antibot/antibot_data.h>

#include <base/log.h>
#include <base/time.h>

#include <engine/antibot.h>
#include <engine/shared/config.h>

#include <generated/protocol.h>
#include <generated/server_data.h>

#include <game/mapitems.h>
// Still reached for by FireWeapon and the pickup tiles. These three classes are
// the rest of what has to move here before this file can be compiled once and
// shared with the prediction rather than only built into the server.
#include <game/server/entities/laser.h>
#include <game/server/entities/pickup.h>
#include <game/server/entities/projectile.h>
#include <game/server/gamecontext.h>
#include <game/server/gamecontroller.h>
#include <game/server/player.h>
#include <game/server/score.h>
#include <game/server/teams.h>
#include <game/team_state.h>

MACRO_ALLOC_POOL_ID_IMPL(CCharacter, MAX_CLIENTS)

// Character, "physical" player's part
CCharacter::CCharacter(CGameWorld *pWorld, CNetObj_PlayerInput LastInput) :
	CEntity(pWorld, EEntityClass::CHARACTER, false, vec2(0, 0), CCharacterCore::PhysicalSize())
{
	m_Health = 0;
	m_Armor = 0;
	m_TriggeredEvents7 = 0;
	m_StrongWeakId = 0;

	m_Input = LastInput;
	// never initialize both to zero
	m_Input.m_TargetX = 0;
	m_Input.m_TargetY = -1;

	m_LatestPrevPrevInput = m_LatestPrevInput = m_LatestInput = m_PrevInput = m_SavedInput = m_Input;

	m_LastTimeCp = -1;
	m_LastTimeCpBroadcasted = -1;
	for(float &CurrentTimeCp : m_aCurrentTimeCp)
	{
		CurrentTimeCp = 0.0f;
	}
}

void CCharacter::Reset()
{
	StopRecording();
	Destroy();
}

void CCharacter::Destroy()
{
	GameWorld()->m_Core.m_apCharacters[GetCid()] = nullptr;
	m_Alive = false;
	SetSolo(false);
}

void CCharacter::SetWeapon(int W)
{
	if(W == m_Core.m_ActiveWeapon)
		return;

	m_LastWeapon = m_Core.m_ActiveWeapon;
	m_QueuedWeapon = -1;
	m_Core.m_ActiveWeapon = W;
	Env()->CreateSound(m_Pos, SOUND_WEAPON_SWITCH, TeamMask(), GetCid());

	if(m_Core.m_ActiveWeapon < 0 || m_Core.m_ActiveWeapon >= NUM_WEAPONS)
		m_Core.m_ActiveWeapon = 0;
}

void CCharacter::SetJetpack(bool Active)
{
	m_Core.m_Jetpack = Active;
}

void CCharacter::SetEndlessJump(bool Active)
{
	m_Core.m_EndlessJump = Active;
}

void CCharacter::SetJumps(int Jumps)
{
	m_Core.m_Jumps = Jumps;
}

void CCharacter::SetSolo(bool Solo)
{
	m_Core.m_Solo = Solo;
	TeamsCore()->SetSolo(GetCid(), Solo);
}

void CCharacter::SetSuper(bool Super)
{
	// Disable invincible mode before activating super mode. Both modes active at the same time wouldn't necessarily break anything but it's not useful.
	if(Super)
		SetInvincible(false);

	bool WasSuper = m_Core.m_Super;
	m_Core.m_Super = Super;
	if(Super && !WasSuper)
	{
		m_TeamBeforeSuper = Team();
		char aError[512];
		if(!Env()->SetCharacterTeam(GetCid(), TEAM_SUPER, aError, sizeof(aError)))
			log_error("character", "failed to set super: %s", aError);
		m_DDRaceState = ERaceState::CHEATED;
	}
	else if(!Super && WasSuper)
	{
		Env()->SetForceCharacterTeam(GetCid(), m_TeamBeforeSuper);
	}
}

void CCharacter::SetInvincible(bool Invincible)
{
	// Disable super mode before activating invincible mode. Both modes active at the same time wouldn't necessarily break anything but it's not useful.
	if(Invincible)
		SetSuper(false);

	m_Core.m_Invincible = Invincible;
	if(Invincible)
		Unfreeze();

	SetEndlessJump(Invincible);
}

void CCharacter::SetCollisionDisabled(bool CollisionDisabled)
{
	m_Core.m_CollisionDisabled = CollisionDisabled;
}

void CCharacter::SetHookHitDisabled(bool HookHitDisabled)
{
	m_Core.m_HookHitDisabled = HookHitDisabled;
}

void CCharacter::SetLiveFrozen(bool Active)
{
	m_Core.m_LiveFrozen = Active;
}

void CCharacter::SetDeepFrozen(bool Active)
{
	m_Core.m_DeepFrozen = Active;
}

bool CCharacter::IsGrounded()
{
	if(Collision()->IsOnGround(m_Pos, GetProximityRadius()))
		return true;

	int MoveRestrictionsBelow = Collision()->GetMoveRestrictions(m_Pos + vec2(0, GetProximityRadius() / 2 + 4), 0.0f);
	return (MoveRestrictionsBelow & CANTMOVE_DOWN) != 0;
}

void CCharacter::HandleJetpack()
{
	if(m_Core.m_ActiveWeapon < 0)
		return;

	vec2 Direction = normalize(vec2(m_LatestInput.m_TargetX, m_LatestInput.m_TargetY));

	bool FullAuto = false;
	if(m_Core.m_ActiveWeapon == WEAPON_GRENADE || m_Core.m_ActiveWeapon == WEAPON_SHOTGUN || m_Core.m_ActiveWeapon == WEAPON_LASER)
		FullAuto = true;
	if(m_Core.m_Jetpack && m_Core.m_ActiveWeapon == WEAPON_GUN)
		FullAuto = true;

	// check if we gonna fire
	bool WillFire = false;
	if(CountInput(m_LatestPrevInput.m_Fire, m_LatestInput.m_Fire).m_Presses)
		WillFire = true;

	if(FullAuto && (m_LatestInput.m_Fire & 1) && m_Core.m_aWeapons[m_Core.m_ActiveWeapon].m_Ammo)
		WillFire = true;

	if(!WillFire)
		return;

	// check for ammo
	if(!m_Core.m_aWeapons[m_Core.m_ActiveWeapon].m_Ammo || m_FreezeTime)
	{
		return;
	}

	switch(m_Core.m_ActiveWeapon)
	{
	case WEAPON_GUN:
	{
		if(m_Core.m_Jetpack)
		{
			float Strength = GetTuning(m_TuneZone)->m_JetpackStrength;
			TakeDamage(Direction * -1.0f * (Strength / 100.0f / 6.11f), 0, GetCid(), m_Core.m_ActiveWeapon);
		}
	}
	}
}

void CCharacter::HandleNinja()
{
	if(m_Core.m_ActiveWeapon != WEAPON_NINJA)
		return;

	if((GameWorld()->GameTick() - m_Core.m_Ninja.m_ActivationTick) > (g_pData->m_Weapons.m_Ninja.m_Duration * GameWorld()->GameTickSpeed() / 1000))
	{
		// time's up, return
		RemoveNinja();
		return;
	}

	int NinjaTime = m_Core.m_Ninja.m_ActivationTick + (g_pData->m_Weapons.m_Ninja.m_Duration * GameWorld()->GameTickSpeed() / 1000) - GameWorld()->GameTick();

	if(NinjaTime % GameWorld()->GameTickSpeed() == 0 && NinjaTime / GameWorld()->GameTickSpeed() <= 5)
	{
		Env()->CreateDamageInd(m_Pos, 0, NinjaTime / GameWorld()->GameTickSpeed(), TeamMask() & Env()->ClientsMaskExcludeClientVersionAndHigher(VERSION_DDNET_NEW_HUD), GetCid());
	}

	Env()->SetArmorProgress(this, NinjaTime);

	// force ninja Weapon
	SetWeapon(WEAPON_NINJA);

	m_Core.m_Ninja.m_CurrentMoveTime--;

	if(m_Core.m_Ninja.m_CurrentMoveTime == 0)
	{
		// reset velocity
		m_Core.m_Vel = m_Core.m_Ninja.m_ActivationDir * m_Core.m_Ninja.m_OldVelAmount;
	}

	if(m_Core.m_Ninja.m_CurrentMoveTime > 0)
	{
		// Set velocity
		m_Core.m_Vel = m_Core.m_Ninja.m_ActivationDir * g_pData->m_Weapons.m_Ninja.m_Velocity;
		vec2 OldPos = m_Pos;
		vec2 GroundElasticity = vec2(
			GetTuning(m_TuneZone)->m_GroundElasticityX,
			GetTuning(m_TuneZone)->m_GroundElasticityY);

		Collision()->MoveBox(&m_Core.m_Pos, &m_Core.m_Vel, vec2(GetProximityRadius(), GetProximityRadius()), GroundElasticity);

		// reset velocity so the client doesn't predict stuff
		ResetVelocity();

		// check if we Hit anything along the way
		{
			CEntity *apEnts[MAX_CLIENTS];
			float Radius = GetProximityRadius() * 2.0f;
			int Num = GameWorld()->FindEntities(OldPos, Radius, apEnts, MAX_CLIENTS, CGameWorld::ENTTYPE_CHARACTER);

			// check that we're not in solo part
			if(TeamsCore()->GetSolo(GetCid()))
				return;

			for(int i = 0; i < Num; ++i)
			{
				auto *pChr = static_cast<CCharacter *>(apEnts[i]);
				if(pChr == this)
					continue;

				// Don't hit players in other teams
				if(Team() != pChr->Team())
					continue;

				const int ClientId = pChr->GetCid();

				// Don't hit players in solo parts
				if(TeamsCore()->GetSolo(ClientId))
					continue;

				// make sure we haven't Hit this object before
				bool AlreadyHit = false;
				for(int j = 0; j < m_NumObjectsHit; j++)
				{
					if(m_aHitObjects[j] == ClientId)
						AlreadyHit = true;
				}
				if(AlreadyHit)
					continue;

				// check so we are sufficiently close
				if(distance(pChr->m_Pos, m_Pos) > Radius)
					continue;

				// Hit a player, give them damage and stuffs...
				Env()->CreateSound(pChr->m_Pos, SOUND_NINJA_HIT, TeamMask(), GetCid());
				// set their velocity to fast upward (for now)
				dbg_assert(m_NumObjectsHit < MAX_CLIENTS, "m_aHitObjects overflow");
				m_aHitObjects[m_NumObjectsHit++] = ClientId;

				pChr->TakeDamage(vec2(0, -10.0f), g_pData->m_Weapons.m_Ninja.m_pBase->m_Damage, GetCid(), WEAPON_NINJA);
			}
		}

		return;
	}
}

void CCharacter::DoWeaponSwitch()
{
	// make sure we can switch
	if(m_ReloadTimer != 0 || m_QueuedWeapon == -1)
		return;
	if(m_Core.m_aWeapons[WEAPON_NINJA].m_Got || !m_Core.m_aWeapons[m_QueuedWeapon].m_Got)
		return;

	// switch Weapon
	SetWeapon(m_QueuedWeapon);
}

void CCharacter::HandleWeaponSwitch()
{
	int WantedWeapon = m_Core.m_ActiveWeapon;
	if(m_QueuedWeapon != -1)
		WantedWeapon = m_QueuedWeapon;

	bool Anything = false;
	for(int i = 0; i < NUM_WEAPONS - 1; ++i)
		if(m_Core.m_aWeapons[i].m_Got)
			Anything = true;
	if(!Anything)
		return;
	// select Weapon
	int Next = CountInput(m_LatestPrevInput.m_NextWeapon, m_LatestInput.m_NextWeapon).m_Presses;
	int Prev = CountInput(m_LatestPrevInput.m_PrevWeapon, m_LatestInput.m_PrevWeapon).m_Presses;

	if(Next < 128) // make sure we only try sane stuff
	{
		while(Next) // Next Weapon selection
		{
			WantedWeapon = (WantedWeapon + 1) % NUM_WEAPONS;
			if(m_Core.m_aWeapons[WantedWeapon].m_Got)
				Next--;
		}
	}

	if(Prev < 128) // make sure we only try sane stuff
	{
		while(Prev) // Prev Weapon selection
		{
			WantedWeapon = (WantedWeapon - 1) < 0 ? NUM_WEAPONS - 1 : WantedWeapon - 1;
			if(m_Core.m_aWeapons[WantedWeapon].m_Got)
				Prev--;
		}
	}

	// Direct Weapon selection
	if(m_LatestInput.m_WantedWeapon)
		WantedWeapon = m_Input.m_WantedWeapon - 1;

	// check for insane values
	if(WantedWeapon >= 0 && WantedWeapon < NUM_WEAPONS && WantedWeapon != m_Core.m_ActiveWeapon && m_Core.m_aWeapons[WantedWeapon].m_Got)
		m_QueuedWeapon = WantedWeapon;

	DoWeaponSwitch();
}

void CCharacter::FireWeapon()
{
	if(m_ReloadTimer != 0)
	{
		if(m_LatestInput.m_Fire & 1)
		{
			Env()->AntibotOnHammerFireReloading(GetCid());
		}
		return;
	}

	DoWeaponSwitch();
	vec2 MouseTarget = vec2(m_LatestInput.m_TargetX, m_LatestInput.m_TargetY);
	vec2 Direction = normalize(MouseTarget);

	bool FullAuto = false;
	if(m_Core.m_ActiveWeapon == WEAPON_GRENADE || m_Core.m_ActiveWeapon == WEAPON_SHOTGUN || m_Core.m_ActiveWeapon == WEAPON_LASER)
		FullAuto = true;
	if(m_Core.m_Jetpack && m_Core.m_ActiveWeapon == WEAPON_GUN)
		FullAuto = true;
	// allow firing directly after coming out of freeze or being unfrozen
	// by something
	if(m_FrozenLastTick)
		FullAuto = true;

	// don't fire hammer when player is deep and sv_deepfly is disabled
	if(!g_Config.m_SvDeepfly && m_Core.m_ActiveWeapon == WEAPON_HAMMER && m_Core.m_DeepFrozen)
		return;

	// check if we gonna fire
	bool WillFire = false;
	if(CountInput(m_LatestPrevInput.m_Fire, m_LatestInput.m_Fire).m_Presses)
		WillFire = true;

	if(FullAuto && (m_LatestInput.m_Fire & 1) && m_Core.m_ActiveWeapon >= 0 && m_Core.m_aWeapons[m_Core.m_ActiveWeapon].m_Ammo)
		WillFire = true;

	if(!WillFire)
		return;

	if(m_FreezeTime)
	{
		// Timer stuff to avoid shrieking orchestra caused by unfreeze-plasma
		if(m_PainSoundTimer <= 0 && !(m_LatestPrevInput.m_Fire & 1))
		{
			m_PainSoundTimer = 1 * GameWorld()->GameTickSpeed();
			Env()->CreateSound(m_Pos, SOUND_PLAYER_PAIN_LONG, TeamMask(), GetCid()); // NOLINT(clang-analyzer-unix.Malloc)
		}
		return;
	}

	// check for ammo
	if(m_Core.m_ActiveWeapon < 0 || !m_Core.m_aWeapons[m_Core.m_ActiveWeapon].m_Ammo)
		return;

	vec2 ProjStartPos = m_Pos + Direction * GetProximityRadius() * 0.75f;

	switch(m_Core.m_ActiveWeapon)
	{
	case WEAPON_HAMMER:
	{
		Env()->CreateSound(m_Pos, SOUND_HAMMER_FIRE, TeamMask(), GetCid()); // NOLINT(clang-analyzer-unix.Malloc)

		Env()->AntibotOnHammerFire(GetCid());

		if(m_Core.m_HammerHitDisabled)
			break;

		CEntity *apEnts[MAX_CLIENTS];
		int Hits = 0;
		int Num = GameWorld()->FindEntities(ProjStartPos, GetProximityRadius() * 0.5f, apEnts,
			MAX_CLIENTS, CGameWorld::ENTTYPE_CHARACTER);

		for(int i = 0; i < Num; ++i)
		{
			auto *pTarget = static_cast<CCharacter *>(apEnts[i]);

			if((pTarget == this || (pTarget->IsAlive() && !CanCollide(pTarget->GetCid()))))
				continue;

			// set their velocity to fast upward (for now)
			if(length(pTarget->m_Pos - ProjStartPos) > 0.0f)
				Env()->CreateHammerHit(pTarget->m_Pos - normalize(pTarget->m_Pos - ProjStartPos) * GetProximityRadius() * 0.5f, TeamMask(), GetCid());
			else
				Env()->CreateHammerHit(ProjStartPos, TeamMask(), GetCid());

			vec2 Dir;
			if(length(pTarget->m_Pos - m_Pos) > 0.0f)
				Dir = normalize(pTarget->m_Pos - m_Pos);
			else
				Dir = vec2(0.f, -1.f);

			float Strength = GetTuning(m_TuneZone)->m_HammerStrength;

			vec2 Temp = pTarget->m_Core.m_Vel + normalize(Dir + vec2(0.f, -1.1f)) * 10.0f;
			Temp = ClampVel(pTarget->m_MoveRestrictions, Temp);
			Temp -= pTarget->m_Core.m_Vel;
			pTarget->TakeDamage((vec2(0.f, -1.0f) + Temp) * Strength, g_pData->m_Weapons.m_Hammer.m_pBase->m_Damage,
				GetCid(), m_Core.m_ActiveWeapon);
			pTarget->Unfreeze();

			Env()->AntibotOnHammerHit(GetCid(), pTarget->GetCid());

			Hits++;
		}

		// if we Hit anything, we have to wait for the reload
		if(Hits)
		{
			float FireDelay = GetTuning(m_TuneZone)->m_HammerHitFireDelay;
			m_ReloadTimer = FireDelay * GameWorld()->GameTickSpeed() / 1000;
		}
	}
	break;

	case WEAPON_GUN:
	{
		if(!m_Core.m_Jetpack || !HasNinjaJetpack() || m_Core.m_HasTelegunGun)
		{
			int Lifetime = (int)(GameWorld()->GameTickSpeed() * GetTuning(m_TuneZone)->m_GunLifetime);

			new CProjectile(
				GameWorld(),
				WEAPON_GUN, //Type
				GetCid(), //Owner
				ProjStartPos, //Pos
				Direction, //Dir
				Lifetime, //Span
				false, //Freeze
				false, //Explosive
				-1, //SoundImpact
				MouseTarget //InitDir
			);

			Env()->CreateSound(m_Pos, SOUND_GUN_FIRE, TeamMask(), GetCid()); // NOLINT(clang-analyzer-unix.Malloc)
		}
	}
	break;

	case WEAPON_SHOTGUN:
	{
		float LaserReach = GetTuning(m_TuneZone)->m_LaserReach;

		new CLaser(GameWorld(), m_Pos, Direction, LaserReach, GetCid(), WEAPON_SHOTGUN);
		Env()->CreateSound(m_Pos, SOUND_SHOTGUN_FIRE, TeamMask(), GetCid()); // NOLINT(clang-analyzer-unix.Malloc)
	}
	break;

	case WEAPON_GRENADE:
	{
		int Lifetime = (int)(GameWorld()->GameTickSpeed() * GetTuning(m_TuneZone)->m_GrenadeLifetime);

		new CProjectile(
			GameWorld(),
			WEAPON_GRENADE, //Type
			GetCid(), //Owner
			ProjStartPos, //Pos
			Direction, //Dir
			Lifetime, //Span
			false, //Freeze
			true, //Explosive
			SOUND_GRENADE_EXPLODE, //SoundImpact
			MouseTarget // MouseTarget
		);

		Env()->CreateSound(m_Pos, SOUND_GRENADE_FIRE, TeamMask(), GetCid()); // NOLINT(clang-analyzer-unix.Malloc)
	}
	break;

	case WEAPON_LASER:
	{
		float LaserReach = GetTuning(m_TuneZone)->m_LaserReach;

		new CLaser(GameWorld(), m_Pos, Direction, LaserReach, GetCid(), WEAPON_LASER);
		Env()->CreateSound(m_Pos, SOUND_LASER_FIRE, TeamMask(), GetCid()); // NOLINT(clang-analyzer-unix.Malloc)
	}
	break;

	case WEAPON_NINJA:
	{
		// reset Hit objects
		m_NumObjectsHit = 0;

		m_Core.m_Ninja.m_ActivationDir = Direction;
		m_Core.m_Ninja.m_CurrentMoveTime = g_pData->m_Weapons.m_Ninja.m_Movetime * GameWorld()->GameTickSpeed() / 1000;

		// clamp to prevent massive MoveBox calculation lag with SG bug
		m_Core.m_Ninja.m_OldVelAmount = std::clamp(length(m_Core.m_Vel), 0.0f, 6000.0f);

		Env()->CreateSound(m_Pos, SOUND_NINJA_FIRE, TeamMask(), GetCid()); // NOLINT(clang-analyzer-unix.Malloc)
	}
	break;
	}

	m_AttackTick = GameWorld()->GameTick();

	// -1 is no weapon, handled here so pain sound still plays when firing in freeze
	if(!m_ReloadTimer && m_Core.m_ActiveWeapon != -1)
	{
		m_ReloadTimer = GetTuning(m_TuneZone)->GetWeaponFireDelay(m_Core.m_ActiveWeapon) * GameWorld()->GameTickSpeed();
	}
}

void CCharacter::HandleWeapons()
{
	//ninja
	HandleNinja();
	HandleJetpack();

	if(m_PainSoundTimer > 0)
		m_PainSoundTimer--;

	// check reload timer
	if(m_ReloadTimer)
	{
		m_ReloadTimer--;
		return;
	}

	// fire Weapon, if wanted
	FireWeapon();
}

void CCharacter::GiveNinja()
{
	m_Core.m_Ninja.m_ActivationTick = GameWorld()->GameTick();
	m_Core.m_aWeapons[WEAPON_NINJA].m_Got = true;
	m_Core.m_aWeapons[WEAPON_NINJA].m_Ammo = -1;
	if(m_Core.m_ActiveWeapon != WEAPON_NINJA)
		m_LastWeapon = m_Core.m_ActiveWeapon;
	m_Core.m_ActiveWeapon = WEAPON_NINJA;

	// not used on ddrace
	// Env()->CreateSound(m_Pos, SOUND_PICKUP_NINJA, TeamMask(), GetCid());
}

void CCharacter::RemoveNinja()
{
	m_Core.m_Ninja.m_ActivationDir = vec2(0, 0);
	m_Core.m_Ninja.m_ActivationTick = 0;
	m_Core.m_Ninja.m_CurrentMoveTime = 0;
	m_Core.m_Ninja.m_OldVelAmount = 0;
	m_Core.m_aWeapons[WEAPON_NINJA].m_Got = false;
	m_Core.m_aWeapons[WEAPON_NINJA].m_Ammo = 0;
	m_Core.m_ActiveWeapon = m_LastWeapon;

	SetWeapon(m_Core.m_ActiveWeapon);
}

void CCharacter::SetEmote(int Emote, int Tick)
{
	m_EmoteType = Emote;
	m_EmoteStop = Tick;
}

void CCharacter::OnPredictedInput(const CNetObj_PlayerInput *pNewInput)
{
	// check for changes
	if(mem_comp(&m_SavedInput, pNewInput, sizeof(CNetObj_PlayerInput)) != 0)
		m_LastAction = GameWorld()->GameTick();

	// copy new input
	mem_copy(&m_Input, pNewInput, sizeof(m_Input));

	// it is not allowed to aim in the center
	if(m_Input.m_TargetX == 0 && m_Input.m_TargetY == 0)
		m_Input.m_TargetY = -1;

	mem_copy(&m_SavedInput, &m_Input, sizeof(m_SavedInput));
}

void CCharacter::OnDirectInput(const CNetObj_PlayerInput *pNewInput)
{
	mem_copy(&m_LatestPrevInput, &m_LatestInput, sizeof(m_LatestInput));
	mem_copy(&m_LatestInput, pNewInput, sizeof(m_LatestInput));
	m_NumInputs++;

	// it is not allowed to aim in the center
	if(m_LatestInput.m_TargetX == 0 && m_LatestInput.m_TargetY == 0)
		m_LatestInput.m_TargetY = -1;

	Env()->AntibotOnDirectInput(GetCid());

	if(m_NumInputs > 1 && GetPlayerTeam() != TEAM_SPECTATORS)
	{
		HandleWeaponSwitch();
		FireWeapon();
	}

	mem_copy(&m_LatestPrevPrevInput, &m_LatestPrevInput, sizeof(m_LatestInput));
	mem_copy(&m_LatestPrevInput, &m_LatestInput, sizeof(m_LatestInput));
}

void CCharacter::ReleaseHook()
{
	m_Core.SetHookedPlayer(-1);
	m_Core.m_HookState = HOOK_RETRACTED;
	m_Core.m_TriggeredEvents |= COREEVENT_HOOK_RETRACT;
}

void CCharacter::ResetHook()
{
	ReleaseHook();
	m_Core.m_HookPos = m_Core.m_Pos;
}

void CCharacter::ResetInput()
{
	m_Input.m_Direction = 0;
	// simulate releasing the fire button
	if((m_Input.m_Fire & 1) != 0)
		m_Input.m_Fire++;
	m_Input.m_Fire &= INPUT_STATE_MASK;
	m_Input.m_Jump = 0;
	m_LatestPrevInput = m_LatestInput = m_Input;
}

void CCharacter::PreTick()
{
	if(m_StartTime > GameWorld()->GameTick())
	{
		// Prevent the player from getting a negative time
		// The main reason why this can happen is because of time penalty tiles
		// However, other reasons are hereby also excluded
		Env()->SendChatInfo(GetCid(), "You died of old age");
		Die(GetCid(), WEAPON_WORLD);
	}

	if(m_Paused)
		return;

	// set emote
	if(m_EmoteStop < GameWorld()->GameTick())
	{
		SetDefaultEmote();
	}

	DDRaceTick();

	Env()->AntibotOnCharacterTick(GetCid());

	m_Core.m_Input = m_Input;
	m_Core.Tick(true, !g_Config.m_SvNoWeakHook);
}

void CCharacter::Tick()
{
	if(g_Config.m_SvNoWeakHook)
	{
		if(m_Paused)
			return;

		m_Core.TickDeferred();
	}
	else
	{
		PreTick();
	}

	if(!m_PrevInput.m_Hook && m_Input.m_Hook && !(m_Core.m_TriggeredEvents & COREEVENT_HOOK_ATTACH_PLAYER))
	{
		Env()->AntibotOnHookAttach(GetCid(), false);
	}

	// handle Weapons
	HandleWeapons();

	DDRacePostCoreTick();

	if(m_Core.m_TriggeredEvents & COREEVENT_HOOK_ATTACH_PLAYER)
	{
		const int HookedPlayer = m_Core.HookedPlayer();
		if(HookedPlayer != -1 && Env()->IsPlayerInGame(HookedPlayer))
		{
			Env()->AntibotOnHookAttach(GetCid(), true);
		}
	}

	// Previnput
	m_PrevInput = m_Input;

	m_PrevPos = m_Core.m_Pos;
}

void CCharacter::TickDeferred()
{
	// advance the dummy
	{
		CWorldCore TempWorld;
		m_ReckoningCore.Init(&TempWorld, Collision(), TeamsCore());
		m_ReckoningCore.m_Id = GetCid();
		m_ReckoningCore.m_Tuning = CTuningParams();
		m_ReckoningCore.Tick(false);
		m_ReckoningCore.Move();
		m_ReckoningCore.Quantize();
	}

	//lastsentcore
	vec2 StartPos = m_Core.m_Pos;
	vec2 StartVel = m_Core.m_Vel;
	bool StuckBefore = Collision()->TestBox(m_Core.m_Pos, CCharacterCore::PhysicalSizeVec2());

	m_Core.m_Id = GetCid();
	m_Core.Move();
	bool StuckAfterMove = Collision()->TestBox(m_Core.m_Pos, CCharacterCore::PhysicalSizeVec2());
	m_Core.Quantize();
	bool StuckAfterQuant = Collision()->TestBox(m_Core.m_Pos, CCharacterCore::PhysicalSizeVec2());
	m_Pos = m_Core.m_Pos;

	if(!StuckBefore && (StuckAfterMove || StuckAfterQuant))
	{
		// Hackish solution to get rid of strict-aliasing warning
		union
		{
			float f;
			unsigned u;
		} StartPosX, StartPosY, StartVelX, StartVelY;

		StartPosX.f = StartPos.x;
		StartPosY.f = StartPos.y;
		StartVelX.f = StartVel.x;
		StartVelY.f = StartVel.y;

		char aBuf[256];
		str_format(aBuf, sizeof(aBuf), "STUCK!!! %d %d %d %f %f %f %f %x %x %x %x",
			StuckBefore,
			StuckAfterMove,
			StuckAfterQuant,
			StartPos.x, StartPos.y,
			StartVel.x, StartVel.y,
			StartPosX.u, StartPosY.u,
			StartVelX.u, StartVelY.u);
		Env()->PrintDebug(aBuf);
	}

	{
		int Events = m_Core.m_TriggeredEvents;

		// Some sounds are triggered client-side for the acting player (or for all players on Sixup)
		// so we need to avoid duplicating them
		CClientMask TeamMaskExceptSelfAndSixup = TeamMaskWithoutSelfAndSixup();
		// Some are triggered client-side but only on Sixup
		CClientMask TeamMaskExceptSixup = TeamMaskWithoutSixup();

		if(Events & COREEVENT_GROUND_JUMP)
			Env()->CreateSound(m_Pos, SOUND_PLAYER_JUMP, TeamMaskExceptSelfAndSixup, GetCid());

		if(Events & COREEVENT_HOOK_ATTACH_PLAYER)
			Env()->CreateSound(m_Pos, SOUND_HOOK_ATTACH_PLAYER, TeamMaskExceptSixup, GetCid());

		if(Events & COREEVENT_HOOK_ATTACH_GROUND)
			Env()->CreateSound(m_Pos, SOUND_HOOK_ATTACH_GROUND, TeamMaskExceptSelfAndSixup, GetCid());

		if(Events & COREEVENT_HOOK_HIT_NOHOOK)
			Env()->CreateSound(m_Pos, SOUND_HOOK_NOATTACH, TeamMaskExceptSelfAndSixup, GetCid());

		if(Events & COREEVENT_GROUND_JUMP)
			m_TriggeredEvents7 |= protocol7::COREEVENTFLAG_GROUND_JUMP;
		if(Events & COREEVENT_AIR_JUMP)
			m_TriggeredEvents7 |= protocol7::COREEVENTFLAG_AIR_JUMP;
		if(Events & COREEVENT_HOOK_ATTACH_PLAYER)
			m_TriggeredEvents7 |= protocol7::COREEVENTFLAG_HOOK_ATTACH_PLAYER;
		if(Events & COREEVENT_HOOK_ATTACH_GROUND)
			m_TriggeredEvents7 |= protocol7::COREEVENTFLAG_HOOK_ATTACH_GROUND;
		if(Events & COREEVENT_HOOK_HIT_NOHOOK)
			m_TriggeredEvents7 |= protocol7::COREEVENTFLAG_HOOK_HIT_NOHOOK;
	}

	if(GetPlayerTeam() == TEAM_SPECTATORS)
	{
		m_Pos.x = m_Input.m_TargetX;
		m_Pos.y = m_Input.m_TargetY;
	}

	// update the m_SendCore if needed
	{
		CNetObj_Character Predicted;
		CNetObj_Character Current;
		mem_zero(&Predicted, sizeof(Predicted));
		mem_zero(&Current, sizeof(Current));
		m_ReckoningCore.Write(&Predicted);
		m_Core.Write(&Current);

		// only allow dead reckoning for a top of 3 seconds
		if(m_Core.m_Reset || m_ReckoningTick + GameWorld()->GameTickSpeed() * 3 < GameWorld()->GameTick() || mem_comp(&Predicted, &Current, sizeof(CNetObj_Character)) != 0)
		{
			m_ReckoningTick = GameWorld()->GameTick();
			m_SendCore = m_Core;
			m_ReckoningCore = m_Core;
			m_Core.m_Reset = false;
		}
	}
}

void CCharacter::TickPaused()
{
	++m_AttackTick;
	++m_DamageTakenTick;
	++m_Core.m_Ninja.m_ActivationTick;
	++m_ReckoningTick;
	if(m_LastAction != -1)
		++m_LastAction;
	if(m_Core.m_ActiveWeapon >= 0 && m_Core.m_aWeapons[m_Core.m_ActiveWeapon].m_AmmoRegenStart > -1)
		++m_Core.m_aWeapons[m_Core.m_ActiveWeapon].m_AmmoRegenStart;
	if(m_EmoteStop > -1)
		++m_EmoteStop;
}

bool CCharacter::TakeDamage(vec2 Force, int Dmg, int From, int Weapon)
{
	if(Dmg)
	{
		SetEmote(EMOTE_PAIN, GameWorld()->GameTick() + 500 * GameWorld()->GameTickSpeed() / 1000);
	}

	vec2 Temp = m_Core.m_Vel + Force;
	m_Core.m_Vel = ClampVel(m_MoveRestrictions, Temp);

	return true;
}

// DDRace

bool CCharacter::CanCollide(int ClientId)
{
	return TeamsCore()->CanCollide(GetCid(), ClientId);
}
bool CCharacter::SameTeam(int ClientId)
{
	return TeamsCore()->SameTeam(GetCid(), ClientId);
}

CTeamsCore *CCharacter::TeamsCore()
{
	return GameWorld()->TeamsCore();
}

int CCharacter::Team()
{
	return TeamsCore()->Team(GetCid());
}

void CCharacter::HandleSkippableTiles(int Index)
{
	// handle death-tiles and leaving gamelayer
	if((Collision()->GetCollisionAt(m_Pos.x + GetProximityRadius() / 3.f, m_Pos.y - GetProximityRadius() / 3.f) == TILE_DEATH ||
		   Collision()->GetCollisionAt(m_Pos.x + GetProximityRadius() / 3.f, m_Pos.y + GetProximityRadius() / 3.f) == TILE_DEATH ||
		   Collision()->GetCollisionAt(m_Pos.x - GetProximityRadius() / 3.f, m_Pos.y - GetProximityRadius() / 3.f) == TILE_DEATH ||
		   Collision()->GetCollisionAt(m_Pos.x - GetProximityRadius() / 3.f, m_Pos.y + GetProximityRadius() / 3.f) == TILE_DEATH ||
		   Collision()->GetFrontCollisionAt(m_Pos.x + GetProximityRadius() / 3.f, m_Pos.y - GetProximityRadius() / 3.f) == TILE_DEATH ||
		   Collision()->GetFrontCollisionAt(m_Pos.x + GetProximityRadius() / 3.f, m_Pos.y + GetProximityRadius() / 3.f) == TILE_DEATH ||
		   Collision()->GetFrontCollisionAt(m_Pos.x - GetProximityRadius() / 3.f, m_Pos.y - GetProximityRadius() / 3.f) == TILE_DEATH ||
		   Collision()->GetFrontCollisionAt(m_Pos.x - GetProximityRadius() / 3.f, m_Pos.y + GetProximityRadius() / 3.f) == TILE_DEATH) &&
		!m_Core.m_Super && !m_Core.m_Invincible && !(Team() && Env()->TeeFinished(GetCid())))
	{
		if(Env()->TeamIsPractice(Team()))
		{
			Freeze();
			// Rate limit death effects to once per second
			if(GameWorld()->GameTick() - GetDieTick() >= GameWorld()->GameTickSpeed())
			{
				SetDieTick(GameWorld()->GameTick());
				Env()->CreateSound(m_Pos, SOUND_PLAYER_DIE, TeamMask(), GetCid());
				Env()->CreateDeath(m_Pos, GetCid(), TeamMask());
			}
		}
		else
		{
			Die(GetCid(), WEAPON_WORLD);
			return;
		}
	}

	if(GameLayerClipped(m_Pos))
	{
		Die(GetCid(), WEAPON_WORLD);
		return;
	}

	if(Index < 0)
		return;

	// handle speedup tiles
	if(Collision()->IsSpeedup(Index))
	{
		vec2 Direction, TempVel = m_Core.m_Vel;
		int Force, Type, MaxSpeed = 0;
		Collision()->GetSpeedup(Index, &Direction, &Force, &MaxSpeed, &Type);

		if(Type == TILE_SPEED_BOOST_OLD)
		{
			float TeeAngle, SpeederAngle, DiffAngle, SpeedLeft, TeeSpeed;
			if(Force == 255 && MaxSpeed)
			{
				m_Core.m_Vel = Direction * (MaxSpeed / 5);
			}
			else
			{
				if(MaxSpeed > 0 && MaxSpeed < 5)
					MaxSpeed = 5;
				if(MaxSpeed > 0)
				{
					if(Direction.x > 0.0000001f)
						SpeederAngle = -std::atan(Direction.y / Direction.x);
					else if(Direction.x < 0.0000001f)
						SpeederAngle = std::atan(Direction.y / Direction.x) + 2.0f * std::asin(1.0f);
					else if(Direction.y > 0.0000001f)
						SpeederAngle = std::asin(1.0f);
					else
						SpeederAngle = std::asin(-1.0f);

					if(SpeederAngle < 0)
						SpeederAngle = 4.0f * std::asin(1.0f) + SpeederAngle;

					if(TempVel.x > 0.0000001f)
						TeeAngle = -std::atan(TempVel.y / TempVel.x);
					else if(TempVel.x < 0.0000001f)
						TeeAngle = std::atan(TempVel.y / TempVel.x) + 2.0f * std::asin(1.0f);
					else if(TempVel.y > 0.0000001f)
						TeeAngle = std::asin(1.0f);
					else
						TeeAngle = std::asin(-1.0f);

					if(TeeAngle < 0)
						TeeAngle = 4.0f * std::asin(1.0f) + TeeAngle;

					TeeSpeed = std::sqrt(std::pow(TempVel.x, 2) + std::pow(TempVel.y, 2));

					DiffAngle = SpeederAngle - TeeAngle;
					SpeedLeft = MaxSpeed / 5.0f - std::cos(DiffAngle) * TeeSpeed;
					if(absolute((int)SpeedLeft) > Force && SpeedLeft > 0.0000001f)
						TempVel += Direction * Force;
					else if(absolute((int)SpeedLeft) > Force)
						TempVel += Direction * -Force;
					else
						TempVel += Direction * SpeedLeft;
				}
				else
					TempVel += Direction * Force;

				m_Core.m_Vel = ClampVel(m_MoveRestrictions, TempVel);
			}
		}
		else if(Type == TILE_SPEED_BOOST)
		{
			constexpr float MaxSpeedScale = 5.0f;
			if(MaxSpeed == 0)
			{
				float MaxRampSpeed = GetTuning(m_TuneZone)->m_VelrampRange / (50 * log(std::max((float)GetTuning(m_TuneZone)->m_VelrampCurvature, 1.01f)));
				MaxSpeed = std::max(MaxRampSpeed, GetTuning(m_TuneZone)->m_VelrampStart / 50) * MaxSpeedScale;
			}

			// (signed) length of projection
			float CurrentDirectionalSpeed = dot(Direction, m_Core.m_Vel);
			float TempMaxSpeed = MaxSpeed / MaxSpeedScale;
			if(CurrentDirectionalSpeed + Force > TempMaxSpeed)
				TempVel += Direction * (TempMaxSpeed - CurrentDirectionalSpeed);
			else
				TempVel += Direction * Force;

			m_Core.m_Vel = ClampVel(m_MoveRestrictions, TempVel);
		}
	}
}

bool CCharacter::IsSwitchActiveCb(unsigned char Number, void *pUser)
{
	CCharacter *pThis = (CCharacter *)pUser;
	auto &aSwitchers = pThis->Switchers();
	return !aSwitchers.empty() && pThis->Team() != TEAM_SUPER && aSwitchers[Number].m_aStatus[pThis->Team()];
}

void CCharacter::HandleTiles(int Index)
{
	int MapIndex = Index;
	m_TileIndex = Collision()->GetTileIndex(MapIndex);
	m_TileFIndex = Collision()->GetFrontTileIndex(MapIndex);
	m_MoveRestrictions = Collision()->GetMoveRestrictions(IsSwitchActiveCb, this, m_Pos, 18.0f, MapIndex);
	if(Index < 0)
	{
		m_LastRefillJumps = false;
		m_LastPenalty = false;
		m_LastBonus = false;
		return;
	}
	SetTimeCheckpoint(Collision()->IsTimeCheckpoint(MapIndex));
	SetTimeCheckpoint(Collision()->IsFrontTimeCheckpoint(MapIndex));
	int TeleCheckpoint = Collision()->IsTeleCheckpoint(MapIndex);
	if(TeleCheckpoint)
		m_TeleCheckpoint = TeleCheckpoint;

	Env()->OnCharacterTiles(this, Index);
	if(!m_Alive)
		return;

	// freeze
	if(((m_TileIndex == TILE_FREEZE) || (m_TileFIndex == TILE_FREEZE)) && !m_Core.m_Super && !m_Core.m_Invincible && !m_Core.m_DeepFrozen)
	{
		Freeze();
	}
	else if(((m_TileIndex == TILE_UNFREEZE) || (m_TileFIndex == TILE_UNFREEZE)) && !m_Core.m_DeepFrozen)
		Unfreeze();

	// deep freeze
	if(((m_TileIndex == TILE_DFREEZE) || (m_TileFIndex == TILE_DFREEZE)) && !m_Core.m_Super && !m_Core.m_Invincible && !m_Core.m_DeepFrozen)
		m_Core.m_DeepFrozen = true;
	else if(((m_TileIndex == TILE_DUNFREEZE) || (m_TileFIndex == TILE_DUNFREEZE)) && !m_Core.m_Super && !m_Core.m_Invincible && m_Core.m_DeepFrozen)
		m_Core.m_DeepFrozen = false;

	// live freeze
	if(((m_TileIndex == TILE_LFREEZE) || (m_TileFIndex == TILE_LFREEZE)) && !m_Core.m_Super && !m_Core.m_Invincible)
	{
		m_Core.m_LiveFrozen = true;
	}
	else if(((m_TileIndex == TILE_LUNFREEZE) || (m_TileFIndex == TILE_LUNFREEZE)) && !m_Core.m_Super && !m_Core.m_Invincible)
	{
		m_Core.m_LiveFrozen = false;
	}

	// endless hook
	if(((m_TileIndex == TILE_EHOOK_ENABLE) || (m_TileFIndex == TILE_EHOOK_ENABLE)))
	{
		SetEndlessHook(true);
	}
	else if(((m_TileIndex == TILE_EHOOK_DISABLE) || (m_TileFIndex == TILE_EHOOK_DISABLE)))
	{
		SetEndlessHook(false);
	}

	// hit others
	if(((m_TileIndex == TILE_HIT_DISABLE) || (m_TileFIndex == TILE_HIT_DISABLE)) && (!m_Core.m_HammerHitDisabled || !m_Core.m_ShotgunHitDisabled || !m_Core.m_GrenadeHitDisabled || !m_Core.m_LaserHitDisabled))
	{
		Env()->SendChatInfo(GetCid(), "You can't hit others");
		m_Core.m_HammerHitDisabled = true;
		m_Core.m_ShotgunHitDisabled = true;
		m_Core.m_GrenadeHitDisabled = true;
		m_Core.m_LaserHitDisabled = true;
	}
	else if(((m_TileIndex == TILE_HIT_ENABLE) || (m_TileFIndex == TILE_HIT_ENABLE)) && (m_Core.m_HammerHitDisabled || m_Core.m_ShotgunHitDisabled || m_Core.m_GrenadeHitDisabled || m_Core.m_LaserHitDisabled))
	{
		Env()->SendChatInfo(GetCid(), "You can hit others");
		m_Core.m_ShotgunHitDisabled = false;
		m_Core.m_GrenadeHitDisabled = false;
		m_Core.m_HammerHitDisabled = false;
		m_Core.m_LaserHitDisabled = false;
	}

	// collide with others
	if(((m_TileIndex == TILE_NPC_DISABLE) || (m_TileFIndex == TILE_NPC_DISABLE)) && !m_Core.m_CollisionDisabled)
	{
		Env()->SendChatInfo(GetCid(), "You can't collide with others");
		m_Core.m_CollisionDisabled = true;
	}
	else if(((m_TileIndex == TILE_NPC_ENABLE) || (m_TileFIndex == TILE_NPC_ENABLE)) && m_Core.m_CollisionDisabled)
	{
		Env()->SendChatInfo(GetCid(), "You can collide with others");
		m_Core.m_CollisionDisabled = false;
	}

	// hook others
	if(((m_TileIndex == TILE_NPH_DISABLE) || (m_TileFIndex == TILE_NPH_DISABLE)) && !m_Core.m_HookHitDisabled)
	{
		Env()->SendChatInfo(GetCid(), "You can't hook others");
		m_Core.m_HookHitDisabled = true;
	}
	else if(((m_TileIndex == TILE_NPH_ENABLE) || (m_TileFIndex == TILE_NPH_ENABLE)) && m_Core.m_HookHitDisabled)
	{
		Env()->SendChatInfo(GetCid(), "You can hook others");
		m_Core.m_HookHitDisabled = false;
	}

	// unlimited air jumps
	if(((m_TileIndex == TILE_UNLIMITED_JUMPS_ENABLE) || (m_TileFIndex == TILE_UNLIMITED_JUMPS_ENABLE)) && !m_Core.m_EndlessJump)
	{
		Env()->SendChatInfo(GetCid(), "You have unlimited air jumps");
		m_Core.m_EndlessJump = true;
	}
	else if(((m_TileIndex == TILE_UNLIMITED_JUMPS_DISABLE) || (m_TileFIndex == TILE_UNLIMITED_JUMPS_DISABLE)) && m_Core.m_EndlessJump)
	{
		Env()->SendChatInfo(GetCid(), "You don't have unlimited air jumps");
		m_Core.m_EndlessJump = false;
	}

	// walljump
	if((m_TileIndex == TILE_WALLJUMP) || (m_TileFIndex == TILE_WALLJUMP))
	{
		if(m_Core.m_Vel.y > 0 && m_Core.m_Colliding && m_Core.m_LeftWall)
		{
			m_Core.m_LeftWall = false;
			m_Core.m_JumpedTotal = m_Core.m_Jumps >= 2 ? m_Core.m_Jumps - 2 : 0;
			m_Core.m_Jumped = 1;
		}
	}

	// jetpack gun
	if(((m_TileIndex == TILE_JETPACK_ENABLE) || (m_TileFIndex == TILE_JETPACK_ENABLE)) && !m_Core.m_Jetpack)
	{
		Env()->SendChatInfo(GetCid(), "You have a jetpack gun");
		m_Core.m_Jetpack = true;
	}
	else if(((m_TileIndex == TILE_JETPACK_DISABLE) || (m_TileFIndex == TILE_JETPACK_DISABLE)) && m_Core.m_Jetpack)
	{
		Env()->SendChatInfo(GetCid(), "You lost your jetpack gun");
		m_Core.m_Jetpack = false;
	}

	// refill jumps
	if(((m_TileIndex == TILE_REFILL_JUMPS) || (m_TileFIndex == TILE_REFILL_JUMPS)) && !m_LastRefillJumps)
	{
		m_Core.m_JumpedTotal = 0;
		m_Core.m_Jumped = 0;
		m_LastRefillJumps = true;
	}
	if((m_TileIndex != TILE_REFILL_JUMPS) && (m_TileFIndex != TILE_REFILL_JUMPS))
	{
		m_LastRefillJumps = false;
	}

	// Teleport gun
	if(((m_TileIndex == TILE_TELE_GUN_ENABLE) || (m_TileFIndex == TILE_TELE_GUN_ENABLE)) && !m_Core.m_HasTelegunGun)
	{
		m_Core.m_HasTelegunGun = true;

		Env()->SendChatInfo(GetCid(), "Teleport gun enabled");
	}
	else if(((m_TileIndex == TILE_TELE_GUN_DISABLE) || (m_TileFIndex == TILE_TELE_GUN_DISABLE)) && m_Core.m_HasTelegunGun)
	{
		m_Core.m_HasTelegunGun = false;

		Env()->SendChatInfo(GetCid(), "Teleport gun disabled");
	}

	if(((m_TileIndex == TILE_TELE_GRENADE_ENABLE) || (m_TileFIndex == TILE_TELE_GRENADE_ENABLE)) && !m_Core.m_HasTelegunGrenade)
	{
		m_Core.m_HasTelegunGrenade = true;

		Env()->SendChatInfo(GetCid(), "Teleport grenade enabled");
	}
	else if(((m_TileIndex == TILE_TELE_GRENADE_DISABLE) || (m_TileFIndex == TILE_TELE_GRENADE_DISABLE)) && m_Core.m_HasTelegunGrenade)
	{
		m_Core.m_HasTelegunGrenade = false;

		Env()->SendChatInfo(GetCid(), "Teleport grenade disabled");
	}

	if(((m_TileIndex == TILE_TELE_LASER_ENABLE) || (m_TileFIndex == TILE_TELE_LASER_ENABLE)) && !m_Core.m_HasTelegunLaser)
	{
		m_Core.m_HasTelegunLaser = true;

		Env()->SendChatInfo(GetCid(), "Teleport laser enabled");
	}
	else if(((m_TileIndex == TILE_TELE_LASER_DISABLE) || (m_TileFIndex == TILE_TELE_LASER_DISABLE)) && m_Core.m_HasTelegunLaser)
	{
		m_Core.m_HasTelegunLaser = false;

		Env()->SendChatInfo(GetCid(), "Teleport laser disabled");
	}

	// stopper
	if(m_Core.m_Vel.y > 0 && (m_MoveRestrictions & CANTMOVE_DOWN))
	{
		m_Core.m_Jumped = 0;
		m_Core.m_JumpedTotal = 0;
	}
	ApplyMoveRestrictions();

	// handle switch tiles
	const int SwitchType = Collision()->GetSwitchType(MapIndex);
	const int SwitchNumber = Collision()->GetSwitchNumber(MapIndex);
	const int SwitchDelay = Collision()->GetSwitchDelay(MapIndex);
	if(SwitchType == TILE_SWITCHOPEN && Team() != TEAM_SUPER && SwitchNumber > 0)
	{
		Switchers()[SwitchNumber].m_aStatus[Team()] = true;
		Switchers()[SwitchNumber].m_aEndTick[Team()] = 0;
		Switchers()[SwitchNumber].m_aType[Team()] = TILE_SWITCHOPEN;
		Switchers()[SwitchNumber].m_aLastUpdateTick[Team()] = GameWorld()->GameTick();
	}
	else if(SwitchType == TILE_SWITCHTIMEDOPEN && Team() != TEAM_SUPER && SwitchNumber > 0)
	{
		Switchers()[SwitchNumber].m_aStatus[Team()] = true;
		Switchers()[SwitchNumber].m_aEndTick[Team()] = GameWorld()->GameTick() + 1 + SwitchDelay * GameWorld()->GameTickSpeed();
		Switchers()[SwitchNumber].m_aType[Team()] = TILE_SWITCHTIMEDOPEN;
		Switchers()[SwitchNumber].m_aLastUpdateTick[Team()] = GameWorld()->GameTick();
	}
	else if(SwitchType == TILE_SWITCHTIMEDCLOSE && Team() != TEAM_SUPER && SwitchNumber > 0)
	{
		Switchers()[SwitchNumber].m_aStatus[Team()] = false;
		Switchers()[SwitchNumber].m_aEndTick[Team()] = GameWorld()->GameTick() + 1 + SwitchDelay * GameWorld()->GameTickSpeed();
		Switchers()[SwitchNumber].m_aType[Team()] = TILE_SWITCHTIMEDCLOSE;
		Switchers()[SwitchNumber].m_aLastUpdateTick[Team()] = GameWorld()->GameTick();
	}
	else if(SwitchType == TILE_SWITCHCLOSE && Team() != TEAM_SUPER && SwitchNumber > 0)
	{
		Switchers()[SwitchNumber].m_aStatus[Team()] = false;
		Switchers()[SwitchNumber].m_aEndTick[Team()] = 0;
		Switchers()[SwitchNumber].m_aType[Team()] = TILE_SWITCHCLOSE;
		Switchers()[SwitchNumber].m_aLastUpdateTick[Team()] = GameWorld()->GameTick();
	}
	else if(SwitchType == TILE_FREEZE && Team() != TEAM_SUPER && !m_Core.m_Invincible)
	{
		if(SwitchNumber == 0 || Switchers()[SwitchNumber].m_aStatus[Team()])
		{
			Freeze(SwitchDelay);
		}
	}
	else if(SwitchType == TILE_DFREEZE && Team() != TEAM_SUPER && !m_Core.m_Invincible)
	{
		if(SwitchNumber == 0 || Switchers()[SwitchNumber].m_aStatus[Team()])
			m_Core.m_DeepFrozen = true;
	}
	else if(SwitchType == TILE_DUNFREEZE && Team() != TEAM_SUPER && !m_Core.m_Invincible)
	{
		if(SwitchNumber == 0 || Switchers()[SwitchNumber].m_aStatus[Team()])
			m_Core.m_DeepFrozen = false;
	}
	else if(SwitchType == TILE_LFREEZE && Team() != TEAM_SUPER && !m_Core.m_Invincible)
	{
		if(SwitchNumber == 0 || Switchers()[SwitchNumber].m_aStatus[Team()])
		{
			m_Core.m_LiveFrozen = true;
		}
	}
	else if(SwitchType == TILE_LUNFREEZE && Team() != TEAM_SUPER && !m_Core.m_Invincible)
	{
		if(SwitchNumber == 0 || Switchers()[SwitchNumber].m_aStatus[Team()])
		{
			m_Core.m_LiveFrozen = false;
		}
	}
	else if(SwitchType == TILE_HIT_ENABLE && m_Core.m_HammerHitDisabled && SwitchDelay == WEAPON_HAMMER)
	{
		Env()->SendChatInfo(GetCid(), "You can hammer hit others");
		m_Core.m_HammerHitDisabled = false;
	}
	else if(SwitchType == TILE_HIT_DISABLE && !(m_Core.m_HammerHitDisabled) && SwitchDelay == WEAPON_HAMMER)
	{
		Env()->SendChatInfo(GetCid(), "You can't hammer hit others");
		m_Core.m_HammerHitDisabled = true;
	}
	else if(SwitchType == TILE_HIT_ENABLE && m_Core.m_ShotgunHitDisabled && SwitchDelay == WEAPON_SHOTGUN)
	{
		Env()->SendChatInfo(GetCid(), "You can shoot others with shotgun");
		m_Core.m_ShotgunHitDisabled = false;
	}
	else if(SwitchType == TILE_HIT_DISABLE && !(m_Core.m_ShotgunHitDisabled) && SwitchDelay == WEAPON_SHOTGUN)
	{
		Env()->SendChatInfo(GetCid(), "You can't shoot others with shotgun");
		m_Core.m_ShotgunHitDisabled = true;
	}
	else if(SwitchType == TILE_HIT_ENABLE && m_Core.m_GrenadeHitDisabled && SwitchDelay == WEAPON_GRENADE)
	{
		Env()->SendChatInfo(GetCid(), "You can shoot others with grenade");
		m_Core.m_GrenadeHitDisabled = false;
	}
	else if(SwitchType == TILE_HIT_DISABLE && !(m_Core.m_GrenadeHitDisabled) && SwitchDelay == WEAPON_GRENADE)
	{
		Env()->SendChatInfo(GetCid(), "You can't shoot others with grenade");
		m_Core.m_GrenadeHitDisabled = true;
	}
	else if(SwitchType == TILE_HIT_ENABLE && m_Core.m_LaserHitDisabled && SwitchDelay == WEAPON_LASER)
	{
		Env()->SendChatInfo(GetCid(), "You can shoot others with laser");
		m_Core.m_LaserHitDisabled = false;
	}
	else if(SwitchType == TILE_HIT_DISABLE && !(m_Core.m_LaserHitDisabled) && SwitchDelay == WEAPON_LASER)
	{
		Env()->SendChatInfo(GetCid(), "You can't shoot others with laser");
		m_Core.m_LaserHitDisabled = true;
	}
	else if(SwitchType == TILE_JUMP)
	{
		int NewJumps = SwitchDelay;
		if(NewJumps == 255)
		{
			NewJumps = -1;
		}

		if(NewJumps != m_Core.m_Jumps)
		{
			char aBuf[256];
			if(NewJumps == -1)
				str_copy(aBuf, "You only have your ground jump now");
			else if(NewJumps == 1)
				str_format(aBuf, sizeof(aBuf), "You can jump %d time", NewJumps);
			else
				str_format(aBuf, sizeof(aBuf), "You can jump %d times", NewJumps);
			Env()->SendChatInfo(GetCid(), aBuf);
			m_Core.m_Jumps = NewJumps;
		}
	}
	else if(SwitchType == TILE_ADD_TIME && !m_LastPenalty)
	{
		const int Minutes = SwitchDelay;
		const int Seconds = SwitchNumber;
		int Team = TeamsCore()->Team(m_Core.m_Id);

		m_StartTime -= (Minutes * 60 + Seconds) * GameWorld()->GameTickSpeed();

		if((g_Config.m_SvTeam == SV_TEAM_FORCED_SOLO || (Team != TEAM_FLOCK && !Env()->TeamFlock(Team))) && Team != TEAM_SUPER)
		{
			for(int i = 0; i < MAX_CLIENTS; i++)
			{
				if(TeamsCore()->Team(i) == Team && i != m_Core.m_Id)
				{
					CCharacter *pChar = GameWorld()->GetCharacterById(i);

					if(pChar)
						pChar->m_StartTime = m_StartTime;
				}
			}
		}

		m_LastPenalty = true;
	}
	else if(SwitchType == TILE_SUBTRACT_TIME && !m_LastBonus)
	{
		const int Minutes = SwitchDelay;
		const int Seconds = SwitchNumber;
		int Team = TeamsCore()->Team(m_Core.m_Id);

		m_StartTime += (Minutes * 60 + Seconds) * GameWorld()->GameTickSpeed();
		if(m_StartTime > GameWorld()->GameTick())
			m_StartTime = GameWorld()->GameTick();

		if((g_Config.m_SvTeam == SV_TEAM_FORCED_SOLO || (Team != TEAM_FLOCK && !Env()->TeamFlock(Team))) && Team != TEAM_SUPER)
		{
			for(int i = 0; i < MAX_CLIENTS; i++)
			{
				if(TeamsCore()->Team(i) == Team && i != m_Core.m_Id)
				{
					CCharacter *pChar = GameWorld()->GetCharacterById(i);

					if(pChar)
						pChar->m_StartTime = m_StartTime;
				}
			}
		}

		m_LastBonus = true;
	}

	if(SwitchType != TILE_ADD_TIME)
	{
		m_LastPenalty = false;
	}

	if(SwitchType != TILE_SUBTRACT_TIME)
	{
		m_LastBonus = false;
	}

	int z = Collision()->IsTeleport(MapIndex);
	if(!g_Config.m_SvOldTeleportHook && !g_Config.m_SvOldTeleportWeapons && z && !Collision()->TeleOuts(z - 1).empty())
	{
		if(m_Core.m_Super || m_Core.m_Invincible)
			return;
		int TeleOut = GameWorld()->m_Core.RandomOr0(Collision()->TeleOuts(z - 1).size());
		m_Core.m_Pos = Collision()->TeleOuts(z - 1)[TeleOut];
		if(!g_Config.m_SvTeleportHoldHook)
		{
			ResetHook();
		}
		if(g_Config.m_SvTeleportLoseWeapons)
			ResetPickups();
		return;
	}
	const int EvilTeleport = Collision()->IsEvilTeleport(MapIndex);
	if(EvilTeleport && !Collision()->TeleOuts(EvilTeleport - 1).empty())
	{
		if(m_Core.m_Super || m_Core.m_Invincible)
			return;
		int TeleOut = GameWorld()->m_Core.RandomOr0(Collision()->TeleOuts(EvilTeleport - 1).size());
		m_Core.m_Pos = Collision()->TeleOuts(EvilTeleport - 1)[TeleOut];
		if(!g_Config.m_SvOldTeleportHook && !g_Config.m_SvOldTeleportWeapons)
		{
			m_Core.m_Vel = vec2(0, 0);

			if(!g_Config.m_SvTeleportHoldHook)
			{
				ResetHook();
				GameWorld()->ReleaseHooked(GetCid());
			}
			if(g_Config.m_SvTeleportLoseWeapons)
			{
				ResetPickups();
			}
		}
		return;
	}
	if(Collision()->IsCheckEvilTeleport(MapIndex))
	{
		if(m_Core.m_Super || m_Core.m_Invincible)
			return;
		// first check if there is a TeleCheckOut for the current recorded checkpoint, if not check previous checkpoints
		for(int k = m_TeleCheckpoint - 1; k >= 0; k--)
		{
			if(!Collision()->TeleCheckOuts(k).empty())
			{
				int TeleOut = GameWorld()->m_Core.RandomOr0(Collision()->TeleCheckOuts(k).size());
				m_Core.m_Pos = Collision()->TeleCheckOuts(k)[TeleOut];
				m_Core.m_Vel = vec2(0, 0);

				if(!g_Config.m_SvTeleportHoldHook)
				{
					ResetHook();
					GameWorld()->ReleaseHooked(GetCid());
				}

				return;
			}
		}
		// if no checkpointout have been found (or if there no recorded checkpoint), teleport to start
		vec2 SpawnPos;
		if(Env()->CanSpawn(GetPlayerTeam(), &SpawnPos, GetCid()))
		{
			m_Core.m_Pos = SpawnPos;
			m_Core.m_Vel = vec2(0, 0);

			if(!g_Config.m_SvTeleportHoldHook)
			{
				ResetHook();
				GameWorld()->ReleaseHooked(GetCid());
			}
		}
		return;
	}
	if(Collision()->IsCheckTeleport(MapIndex))
	{
		if(m_Core.m_Super || m_Core.m_Invincible)
			return;
		// first check if there is a TeleCheckOut for the current recorded checkpoint, if not check previous checkpoints
		for(int k = m_TeleCheckpoint - 1; k >= 0; k--)
		{
			if(!Collision()->TeleCheckOuts(k).empty())
			{
				int TeleOut = GameWorld()->m_Core.RandomOr0(Collision()->TeleCheckOuts(k).size());
				m_Core.m_Pos = Collision()->TeleCheckOuts(k)[TeleOut];

				if(!g_Config.m_SvTeleportHoldHook)
				{
					ResetHook();
				}

				return;
			}
		}
		// if no checkpointout have been found (or if there no recorded checkpoint), teleport to start
		vec2 SpawnPos;
		if(Env()->CanSpawn(GetPlayerTeam(), &SpawnPos, GetCid()))
		{
			m_Core.m_Pos = SpawnPos;

			if(!g_Config.m_SvTeleportHoldHook)
			{
				ResetHook();
			}
		}
		return;
	}
}

void CCharacter::HandleTuneLayer()
{
	m_TuneZoneOld = m_TuneZone;
	int CurrentIndex = Collision()->GetMapIndex(m_Pos);
	m_TuneZone = Collision()->IsTune(CurrentIndex);
	m_Core.m_Tuning = TuningList()[m_TuneZone]; // throw tunings from specific zone into gamecore

	if(m_TuneZone != m_TuneZoneOld) // don't send tunigs all the time
	{
		// send zone msgs
		SendZoneMsgs();
	}
}

void CCharacter::DDRaceTick()
{
	mem_copy(&m_Input, &m_SavedInput, sizeof(m_Input));
	Env()->SetArmorProgress(this, m_FreezeTime);
	if(m_Input.m_Direction != 0 || m_Input.m_Jump != 0)
		m_LastMove = GameWorld()->GameTick();

	if(m_Core.m_LiveFrozen && !m_Core.m_Super && !m_Core.m_Invincible)
	{
		m_Input.m_Direction = 0;
		m_Input.m_Jump = 0;
		// Hook is possible in live freeze
	}
	if(m_FreezeTime > 0)
	{
		if(m_FreezeTime % GameWorld()->GameTickSpeed() == GameWorld()->GameTickSpeed() - 1)
		{
			Env()->CreateDamageInd(m_Pos, 0, (m_FreezeTime + 1) / GameWorld()->GameTickSpeed(), TeamMask() & Env()->ClientsMaskExcludeClientVersionAndHigher(VERSION_DDNET_NEW_HUD), GetCid());
		}
		m_FreezeTime--;
		m_Input.m_Direction = 0;
		m_Input.m_Jump = 0;
		m_Input.m_Hook = 0;
		if(m_FreezeTime == 1)
			Unfreeze();
	}

	HandleTuneLayer(); // need this before coretick

	// check if the tee is in any type of freeze
	int Index = Collision()->GetPureMapIndex(m_Pos);
	const int aTiles[] = {
		Collision()->GetTileIndex(Index),
		Collision()->GetFrontTileIndex(Index),
		Collision()->GetSwitchType(Index)};
	m_Core.m_IsInFreeze = false;
	for(const int Tile : aTiles)
	{
		if(Tile == TILE_FREEZE || Tile == TILE_DFREEZE || Tile == TILE_LFREEZE || Tile == TILE_DEATH)
		{
			m_Core.m_IsInFreeze = true;
			break;
		}
	}
	m_Core.m_IsInFreeze |= (Collision()->GetCollisionAt(m_Pos.x + GetProximityRadius() / 3.f, m_Pos.y - GetProximityRadius() / 3.f) == TILE_DEATH ||
				Collision()->GetCollisionAt(m_Pos.x + GetProximityRadius() / 3.f, m_Pos.y + GetProximityRadius() / 3.f) == TILE_DEATH ||
				Collision()->GetCollisionAt(m_Pos.x - GetProximityRadius() / 3.f, m_Pos.y - GetProximityRadius() / 3.f) == TILE_DEATH ||
				Collision()->GetCollisionAt(m_Pos.x - GetProximityRadius() / 3.f, m_Pos.y + GetProximityRadius() / 3.f) == TILE_DEATH ||
				Collision()->GetFrontCollisionAt(m_Pos.x + GetProximityRadius() / 3.f, m_Pos.y - GetProximityRadius() / 3.f) == TILE_DEATH ||
				Collision()->GetFrontCollisionAt(m_Pos.x + GetProximityRadius() / 3.f, m_Pos.y + GetProximityRadius() / 3.f) == TILE_DEATH ||
				Collision()->GetFrontCollisionAt(m_Pos.x - GetProximityRadius() / 3.f, m_Pos.y - GetProximityRadius() / 3.f) == TILE_DEATH ||
				Collision()->GetFrontCollisionAt(m_Pos.x - GetProximityRadius() / 3.f, m_Pos.y + GetProximityRadius() / 3.f) == TILE_DEATH);

	// look for save position for rescue feature
	// always update auto rescue
	TrySetRescue(RESCUEMODE_AUTO);

	m_Core.m_Id = GetCid();
}

void CCharacter::DDRacePostCoreTick()
{
	m_Time = (float)(GameWorld()->GameTick() - m_StartTime) / ((float)GameWorld()->GameTickSpeed());

	if(m_Core.m_EndlessHook || (m_Core.m_Super && g_Config.m_SvEndlessSuperHook))
		m_Core.m_HookTick = 0;

	m_FrozenLastTick = false;

	if(m_Core.m_DeepFrozen && !m_Core.m_Super && !m_Core.m_Invincible)
		Freeze();

	// following jump rules can be overridden by tiles, like Refill Jumps, Stopper and Wall Jump
	if(m_Core.m_Jumps == -1)
	{
		// The player has only one ground jump, so their feet are always dark
		m_Core.m_Jumped |= 2;
	}
	else if(m_Core.m_Jumps == 0)
	{
		// The player has no jumps at all, so their feet are always dark
		m_Core.m_Jumped |= 2;
	}
	else if(m_Core.m_Jumps == 1 && m_Core.m_Jumped > 0)
	{
		// If the player has only one jump, each jump is the last one
		m_Core.m_Jumped |= 2;
	}
	else if(m_Core.m_JumpedTotal < m_Core.m_Jumps - 1 && m_Core.m_Jumped > 1)
	{
		// The player has not yet used up all their jumps, so their feet remain light
		m_Core.m_Jumped = 1;
	}

	if((m_Core.m_Super || m_Core.m_EndlessJump) && m_Core.m_Jumped > 1)
	{
		// Super players and players with infinite jumps always have light feet
		m_Core.m_Jumped = 1;
	}

	int CurrentIndex = Collision()->GetMapIndex(m_Pos);
	HandleSkippableTiles(CurrentIndex);
	if(!m_Alive)
		return;

	// handle Anti-Skip tiles
	std::vector<int> vIndices = Collision()->GetMapIndices(m_PrevPos, m_Pos);
	if(!vIndices.empty())
	{
		for(int &Index : vIndices)
		{
			HandleTiles(Index);
			if(!m_Alive)
				return;
		}
	}
	else
	{
		HandleTiles(CurrentIndex);
		if(!m_Alive)
			return;
	}

	// teleport gun
	if(m_TeleGunTeleport)
	{
		Env()->CreateDeath(m_Pos, GetCid(), TeamMask());
		m_Core.m_Pos = m_TeleGunPos;
		if(!m_IsBlueTeleGunTeleport)
			m_Core.m_Vel = vec2(0, 0);
		Env()->CreateDeath(m_TeleGunPos, GetCid(), TeamMask());
		Env()->CreateSound(m_TeleGunPos, SOUND_WEAPON_SPAWN, TeamMask(), GetCid());
		m_TeleGunTeleport = false;
		m_IsBlueTeleGunTeleport = false;
	}

	HandleBroadcast();
}

bool CCharacter::Freeze(int Seconds)
{
	if(Seconds <= 0 || m_Core.m_Super || m_Core.m_Invincible || m_FreezeTime > Seconds * GameWorld()->GameTickSpeed())
		return false;
	if(m_FreezeTime == 0 || m_Core.m_FreezeStart < GameWorld()->GameTick() - GameWorld()->GameTickSpeed())
	{
		m_Armor = 0;
		m_FreezeTime = Seconds * GameWorld()->GameTickSpeed();
		m_Core.m_FreezeStart = GameWorld()->GameTick();
		return true;
	}
	return false;
}

bool CCharacter::Freeze()
{
	return Freeze(g_Config.m_SvFreezeDelay);
}

bool CCharacter::Unfreeze()
{
	if(m_FreezeTime > 0)
	{
		m_Armor = 10;
		if(m_Core.m_ActiveWeapon >= 0 && !m_Core.m_aWeapons[m_Core.m_ActiveWeapon].m_Got)
			m_Core.m_ActiveWeapon = WEAPON_GUN;
		m_FreezeTime = 0;
		m_Core.m_FreezeStart = 0;
		m_FrozenLastTick = true;
		return true;
	}
	return false;
}

void CCharacter::GiveWeapon(int Weapon, bool Remove)
{
	if(Weapon == WEAPON_NINJA)
	{
		if(Remove)
			RemoveNinja();
		else
			GiveNinja();
		return;
	}

	if(Remove)
	{
		if(GetActiveWeapon() == Weapon)
			SetActiveWeapon(WEAPON_GUN);
	}
	else
	{
		m_Core.m_aWeapons[Weapon].m_Ammo = -1;
	}

	m_Core.m_aWeapons[Weapon].m_Got = !Remove;
}

void CCharacter::GiveAllWeapons()
{
	for(int i = WEAPON_GUN; i < NUM_WEAPONS - 1; i++)
	{
		GiveWeapon(i);
	}
}

void CCharacter::ResetPickups()
{
	for(int i = WEAPON_SHOTGUN; i < NUM_WEAPONS - 1; i++)
	{
		m_Core.m_aWeapons[i].m_Got = false;
		if(m_Core.m_ActiveWeapon == i)
			m_Core.m_ActiveWeapon = WEAPON_GUN;
	}
}

void CCharacter::SetEndlessHook(bool Enable)
{
	if(m_Core.m_EndlessHook == Enable)
	{
		return;
	}
	Env()->SendChatInfo(GetCid(), Enable ? "Endless hook has been activated" : "Endless hook has been deactivated");

	m_Core.m_EndlessHook = Enable;
}

void CCharacter::SetPosition(const vec2 &Position)
{
	m_Core.m_Pos = Position;
}

void CCharacter::Move(vec2 RelPos)
{
	m_Core.m_Pos += RelPos;
}

void CCharacter::ResetVelocity()
{
	m_Core.m_Vel = vec2(0, 0);
}

void CCharacter::SetVelocity(vec2 NewVelocity)
{
	m_Core.m_Vel = ClampVel(m_MoveRestrictions, NewVelocity);
}

// The method is needed only to reproduce 'shotgun bug' ddnet#5258
// Use SetVelocity() instead.
void CCharacter::SetRawVelocity(vec2 NewVelocity)
{
	m_Core.m_Vel = NewVelocity;
}

void CCharacter::AddVelocity(vec2 Addition)
{
	SetVelocity(m_Core.m_Vel + Addition);
}

void CCharacter::ApplyMoveRestrictions()
{
	m_Core.m_Vel = ClampVel(m_MoveRestrictions, m_Core.m_Vel);
}

void CCharacter::SwapClients(int Client1, int Client2)
{
	const int HookedPlayer = m_Core.HookedPlayer();
	m_Core.SetHookedPlayer(HookedPlayer == Client1 ? Client2 : (HookedPlayer == Client2 ? Client1 : HookedPlayer));
}
