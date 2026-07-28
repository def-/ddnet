/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#include "projectile.h"

#include <engine/shared/config.h>

#include <generated/protocol.h>

#include <game/collision.h>
#include <game/entities/character.h>
#include <game/gameenv.h>
#include <game/mapbugs.h>
#include <game/mapitems.h>

CProjectile::CProjectile(
	CGameWorld *pGameWorld,
	int Type,
	int Owner,
	vec2 Pos,
	vec2 Dir,
	int Span,
	bool Freeze,
	bool Explosive,
	int SoundImpact,
	vec2 InitDir,
	int Layer,
	int Number) :
	CEntity(pGameWorld, EEntityClass::PROJECTILE, true)
{
	m_Type = Type;
	m_Pos = Pos;
	m_Direction = Dir;
	m_LifeSpan = Span;
	m_Owner = Owner;
	m_SoundImpact = SoundImpact;
	m_StartTick = GameWorld()->GameTick();
	m_Explosive = Explosive;

	m_Layer = Layer;
	m_Number = Number;
	m_Bouncing = 0;
	m_Freeze = Freeze;

	m_InitDir = InitDir;
	m_TuneZone = Collision()->IsTune(Collision()->GetMapIndex(m_Pos));

	CCharacter *pOwnerChar = GameWorld()->GetCharacterById(m_Owner);
	m_BelongsToPracticeTeam = pOwnerChar && Env()->TeamIsPractice(pOwnerChar->Team());
	m_DDRaceTeam = m_Owner == -1 ? 0 : Env()->GetDDRaceTeam(m_Owner);
	m_IsSolo = pOwnerChar && pOwnerChar->GetCore().m_Solo;

	GameWorld()->InsertEntity(this);
}

void CProjectile::Reset()
{
	m_MarkedForDestroy = true;
}

vec2 CProjectile::GetPos(float Time)
{
	float Curvature = 0;
	float Speed = 0;
	CTuningParams *pTuning = GetTuning(m_TuneZone);

	switch(m_Type)
	{
	case WEAPON_GRENADE:
		Curvature = pTuning->m_GrenadeCurvature;
		Speed = pTuning->m_GrenadeSpeed;
		break;

	case WEAPON_SHOTGUN:
		Curvature = pTuning->m_ShotgunCurvature;
		Speed = pTuning->m_ShotgunSpeed;
		break;

	case WEAPON_GUN:
		Curvature = pTuning->m_GunCurvature;
		Speed = pTuning->m_GunSpeed;
		break;
	}

	return CalcPos(m_Pos, m_Direction, Curvature, Speed, Time);
}

void CProjectile::Tick()
{
	float Pt = (GameWorld()->GameTick() - m_StartTick - 1) / (float)GameWorld()->GameTickSpeed();
	float Ct = (GameWorld()->GameTick() - m_StartTick) / (float)GameWorld()->GameTickSpeed();
	vec2 PrevPos = GetPos(Pt);
	vec2 CurPos = GetPos(Ct);
	vec2 ColPos;
	vec2 NewPos;
	int Collide = Collision()->IntersectLine(PrevPos, CurPos, &ColPos, &NewPos);
	CCharacter *pOwnerChar = nullptr;

	if(m_Owner >= 0)
		pOwnerChar = GameWorld()->GetCharacterById(m_Owner);

	CCharacter *pTargetChr = nullptr;

	if(pOwnerChar ? !pOwnerChar->GrenadeHitDisabled() : g_Config.m_SvHit)
		pTargetChr = GameWorld()->IntersectCharacter(PrevPos, ColPos, m_Freeze ? 1.0f : 6.0f, ColPos, pOwnerChar, m_Owner);

	if(m_LifeSpan > -1)
		m_LifeSpan--;

	CClientMask TeamMask = CClientMask().set();
	bool IsWeaponCollide = false;
	if(
		pOwnerChar &&
		pTargetChr &&
		pOwnerChar->IsAlive() &&
		pTargetChr->IsAlive() &&
		!pTargetChr->CanCollide(m_Owner))
	{
		IsWeaponCollide = true;
	}
	if(pOwnerChar && pOwnerChar->IsAlive())
	{
		TeamMask = pOwnerChar->TeamMask();
	}
	else if(m_Owner >= 0 && (m_Type != WEAPON_GRENADE || g_Config.m_SvDestroyBulletsOnDeath || m_BelongsToPracticeTeam))
	{
		m_MarkedForDestroy = true;
		return;
	}

	if(((pTargetChr && (pOwnerChar ? !pOwnerChar->GrenadeHitDisabled() : g_Config.m_SvHit || m_Owner == -1 || pTargetChr == pOwnerChar)) || Collide || GameLayerClipped(CurPos)) && !IsWeaponCollide)
	{
		if(m_Explosive /*??*/ && (!pTargetChr || (pTargetChr && (!m_Freeze || (m_Type == WEAPON_SHOTGUN && Collide)))))
		{
			int Number = 1;
			if(GameWorld()->EmulateBug(BUG_GRENADE_DOUBLEEXPLOSION) && m_LifeSpan == -1)
			{
				Number = 2;
			}
			for(int i = 0; i < Number; i++)
			{
				GameWorld()->CreateExplosion(ColPos, m_Owner, m_Type, m_Owner == -1, (!pTargetChr ? -1 : pTargetChr->Team()),
					(m_Owner != -1) ? TeamMask : CClientMask().set());
				Env()->CreateSound(ColPos, m_SoundImpact,
					(m_Owner != -1) ? TeamMask : CClientMask().set(), m_StartTick);
			}
		}
		else if(m_Freeze)
		{
			CEntity *apEnts[MAX_CLIENTS];
			int Num = GameWorld()->FindEntities(CurPos, 1.0f, apEnts, MAX_CLIENTS, CGameWorld::ENTTYPE_CHARACTER);
			for(int i = 0; i < Num; ++i)
			{
				auto *pChr = static_cast<CCharacter *>(apEnts[i]);
				if(pChr && (m_Layer != LAYER_SWITCH || (m_Layer == LAYER_SWITCH && m_Number > 0 && Switchers()[m_Number].m_aStatus[pChr->Team()])))
					pChr->Freeze();
			}
		}
		else if(pTargetChr)
			pTargetChr->TakeDamage(vec2(0, 0), 0, m_Owner, m_Type);

		if(pOwnerChar && !GameLayerClipped(ColPos) &&
			((m_Type == WEAPON_GRENADE && pOwnerChar->HasTelegunGrenade()) || (m_Type == WEAPON_GUN && pOwnerChar->HasTelegunGun())))
		{
			int MapIndex = Collision()->GetPureMapIndex(pTargetChr ? pTargetChr->m_Pos : ColPos);
			int TileFIndex = Collision()->GetFrontTileIndex(MapIndex);
			bool IsSwitchTeleGun = Collision()->GetSwitchType(MapIndex) == TILE_ALLOW_TELE_GUN;
			bool IsBlueSwitchTeleGun = Collision()->GetSwitchType(MapIndex) == TILE_ALLOW_BLUE_TELE_GUN;

			if(IsSwitchTeleGun || IsBlueSwitchTeleGun)
			{
				// Delay specifies which weapon the tile should work for.
				// Delay = 0 means all.
				int Delay = Collision()->GetSwitchDelay(MapIndex);

				if(Delay == 1 && m_Type != WEAPON_GUN)
					IsSwitchTeleGun = IsBlueSwitchTeleGun = false;
				if(Delay == 2 && m_Type != WEAPON_GRENADE)
					IsSwitchTeleGun = IsBlueSwitchTeleGun = false;
				if(Delay == 3 && m_Type != WEAPON_LASER)
					IsSwitchTeleGun = IsBlueSwitchTeleGun = false;
			}

			if(TileFIndex == TILE_ALLOW_TELE_GUN || TileFIndex == TILE_ALLOW_BLUE_TELE_GUN || IsSwitchTeleGun || IsBlueSwitchTeleGun || pTargetChr)
			{
				bool Found;
				vec2 PossiblePos;

				if(!Collide)
					Found = GetNearestAirPosPlayer(pTargetChr ? pTargetChr->m_Pos : ColPos, &PossiblePos);
				else
					Found = GetNearestAirPos(NewPos, CurPos, &PossiblePos);

				if(Found)
				{
					pOwnerChar->m_TeleGunPos = PossiblePos;
					pOwnerChar->m_TeleGunTeleport = true;
					pOwnerChar->m_IsBlueTeleGunTeleport = TileFIndex == TILE_ALLOW_BLUE_TELE_GUN || IsBlueSwitchTeleGun;
				}
			}
		}

		if(Collide && m_Bouncing != 0)
		{
			m_StartTick = GameWorld()->GameTick();
			m_Pos = NewPos + (-(m_Direction * 4));
			if(m_Bouncing == 1)
				m_Direction.x = -m_Direction.x;
			else if(m_Bouncing == 2)
				m_Direction.y = -m_Direction.y;
			if(absolute(m_Direction.x) < 1e-6f)
				m_Direction.x = 0;
			if(absolute(m_Direction.y) < 1e-6f)
				m_Direction.y = 0;
			m_Pos += m_Direction;
		}
		else if(m_Type == WEAPON_GUN)
		{
			Env()->CreateDamageInd(CurPos, -std::atan2(m_Direction.x, m_Direction.y), 10, (m_Owner != -1) ? TeamMask : CClientMask().set(), m_StartTick);
			m_MarkedForDestroy = true;
			return;
		}
		else
		{
			if(!m_Freeze)
			{
				m_MarkedForDestroy = true;
				return;
			}
		}
	}
	if(m_LifeSpan == -1)
	{
		if(m_Explosive)
		{
			if(m_Owner >= 0)
				pOwnerChar = GameWorld()->GetCharacterById(m_Owner);

			TeamMask = CClientMask().set();
			if(pOwnerChar && pOwnerChar->IsAlive())
			{
				TeamMask = pOwnerChar->TeamMask();
			}

			GameWorld()->CreateExplosion(ColPos, m_Owner, m_Type, m_Owner == -1, (!pOwnerChar ? -1 : pOwnerChar->Team()),
				(m_Owner != -1) ? TeamMask : CClientMask().set());
			Env()->CreateSound(ColPos, m_SoundImpact,
				(m_Owner != -1) ? TeamMask : CClientMask().set(), m_StartTick);
		}
		m_MarkedForDestroy = true;
		return;
	}

	int x = Collision()->GetIndex(PrevPos, CurPos);
	int z;
	if(g_Config.m_SvOldTeleportWeapons)
		z = Collision()->IsTeleport(x);
	else
		z = Collision()->IsTeleportWeapon(x);
	if(z && !Collision()->TeleOuts(z - 1).empty())
	{
		int TeleOut = GameWorld()->m_Core.RandomOr0(Collision()->TeleOuts(z - 1).size());
		m_Pos = Collision()->TeleOuts(z - 1)[TeleOut];
		m_StartTick = GameWorld()->GameTick();
	}
}

void CProjectile::TickPaused()
{
	++m_StartTick;
}

void CProjectile::SwapClients(int Client1, int Client2)
{
	m_Owner = m_Owner == Client1 ? Client2 : (m_Owner == Client2 ? Client1 : m_Owner);
}

// DDRace

bool CProjectile::CanCollide(int ClientId)
{
	if(m_DDRaceTeam != Env()->GetDDRaceTeam(ClientId))
		return false;
	if(m_IsSolo)
		return m_Owner == ClientId;
	return true;
}

void CProjectile::SetBouncing(int Value)
{
	m_Bouncing = Value;
}
