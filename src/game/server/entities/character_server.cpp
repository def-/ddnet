/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */

/*
	The parts of CCharacter that only the server has.

	Everything a character does to itself is simulation and lives in
	src/game/entities/character.cpp, shared with the client's prediction. What is
	here either belongs to the player behind the character - dying, spawning,
	race timing, demos, the things a player is told - or is only ever reached
	from server code. The declarations sit in the shared header; the prediction
	answers the ones it needs in src/game/client/prediction/.
*/

#include <antibot/antibot_data.h>

#include <base/log.h>
#include <base/time.h>

#include <engine/antibot.h>
#include <engine/shared/config.h>

#include <generated/protocol.h>
#include <generated/server_data.h>

#include <game/entities/character.h>
#include <game/mapitems.h>
#include <game/server/entities/laser.h>
#include <game/server/entities/pickup.h>
#include <game/server/entities/projectile.h>
#include <game/server/gamecontext.h>
#include <game/server/gamecontroller.h>
#include <game/server/player.h>
#include <game/server/score.h>
#include <game/server/teams.h>
#include <game/team_state.h>
#include <game/teamscore.h>

bool CCharacter::Spawn(CPlayer *pPlayer, vec2 Pos)
{
	m_EmoteStop = -1;
	m_LastAction = -1;
	m_LastWeapon = WEAPON_HAMMER;
	m_QueuedWeapon = -1;
	m_LastRefillJumps = false;
	m_LastPenalty = false;
	m_LastBonus = false;

	m_TeleGunTeleport = false;
	m_IsBlueTeleGunTeleport = false;

	m_pPlayer = pPlayer;
	m_Pos = Pos;

	mem_zero(&m_LatestPrevPrevInput, sizeof(m_LatestPrevPrevInput));
	m_LatestPrevPrevInput.m_TargetY = -1;
	m_NumInputs = 0;
	m_SpawnTick = GameWorld()->GameTick();
	m_WeaponChangeTick = GameWorld()->GameTick();
	Antibot()->OnSpawn(GetCid());

	m_Core.Reset();
	m_Core.Init(&GameWorld()->m_Core, Collision());
	m_Core.m_ActiveWeapon = WEAPON_GUN;
	m_Core.m_Pos = m_Pos;
	m_Core.m_Id = GetCid();
	int TuneZone = Collision()->IsTune(Collision()->GetMapIndex(Pos));
	m_Core.m_Tuning = TuningList()[TuneZone];
	GameWorld()->m_Core.m_apCharacters[GetCid()] = &m_Core;

	m_ReckoningTick = 0;
	m_SendCore = CCharacterCore();
	m_ReckoningCore = CCharacterCore();

	GameWorld()->InsertEntity(this);
	m_Alive = true;

	GameServer()->m_pController->OnCharacterSpawn(this);

	DDRaceInit();

	m_TuneZone = TuneZone;
	m_TuneZoneOld = -1; // no zone leave msg on spawn
	m_NeededFaketuning = 0; // reset fake tunings on respawn and send the client
	SendZoneMsgs(); // we want a entermessage also on spawn
	GameServer()->SendTuningParams(GetCid(), m_TuneZone);

	TrySetRescue(RESCUEMODE_MANUAL);
	Server()->StartRecord(GetCid());

	int Team = GameServer()->m_aTeamMapping[GetCid()];

	if(Team != -1)
	{
		GameServer()->m_pController->Teams().SetForceCharacterTeam(GetCid(), Team);
		GameServer()->m_aTeamMapping[GetCid()] = -1;

		if(GameServer()->m_apSavedTeams[Team])
		{
			GameServer()->m_apSavedTeams[Team]->Load(GameServer(), Team, true, true);
			delete GameServer()->m_apSavedTeams[Team];
			GameServer()->m_apSavedTeams[Team] = nullptr;
		}

		if(GameServer()->m_apSavedTees[GetCid()])
		{
			GameServer()->m_apSavedTees[GetCid()]->Load(m_pPlayer->GetCharacter(), Team);
			delete GameServer()->m_apSavedTees[GetCid()];
			GameServer()->m_apSavedTees[GetCid()] = nullptr;
		}
	}

	return true;
}

int CCharacter::DetermineEyeEmote()
{
	const bool IsFrozen = m_Core.m_DeepFrozen || m_FreezeTime > 0 || m_Core.m_LiveFrozen;
	const bool HasNinjajetpack = m_pPlayer->m_NinjaJetpack && m_Core.m_Jetpack && m_Core.m_ActiveWeapon == WEAPON_GUN;

	if(GetPlayer()->IsAfk() || GetPlayer()->IsPaused())
		return (m_Core.m_DeepFrozen || m_FreezeTime > 0) ? EMOTE_NORMAL : EMOTE_BLINK;
	if(m_EmoteType != EMOTE_NORMAL) // user manually set an eye emote using /emote
		return m_EmoteType;
	if(IsFrozen)
		return (m_Core.m_DeepFrozen || m_Core.m_LiveFrozen) ? EMOTE_PAIN : EMOTE_BLINK;
	if(HasNinjajetpack && !m_Core.m_DeepFrozen && m_FreezeTime == 0 && !m_Core.m_HasTelegunGun)
		return EMOTE_HAPPY;
	if(5 * GameWorld()->GameTickSpeed() - ((GameWorld()->GameTick() - m_LastAction) % (5 * GameWorld()->GameTickSpeed())) < 5)
		return EMOTE_BLINK;
	return EMOTE_NORMAL;
}

bool CCharacter::IncreaseHealth(int Amount)
{
	if(m_Health >= 10)
		return false;
	m_Health = std::clamp(m_Health + Amount, 0, 10);
	return true;
}

bool CCharacter::IncreaseArmor(int Amount)
{
	if(m_Armor >= 10)
		return false;
	m_Armor = std::clamp(m_Armor + Amount, 0, 10);
	return true;
}

void CCharacter::StopRecording()
{
	if(Server()->IsRecording(GetCid()))
	{
		CPlayerData *pData = GameServer()->Score()->PlayerData(GetCid());

		if(pData->m_RecordStopTick - GameWorld()->GameTick() <= GameWorld()->GameTickSpeed() && pData->m_RecordStopTick != -1)
			Server()->SaveDemo(GetCid(), pData->m_RecordFinishTime);
		else
			Server()->StopRecord(GetCid());

		pData->m_RecordStopTick = -1;
	}
}

void CCharacter::Die(int Killer, int Weapon, bool SendKillMsg)
{
	if(Killer != WEAPON_GAME && m_SetSavePos[RESCUEMODE_AUTO])
		GetPlayer()->m_LastDeath = m_RescueTee[RESCUEMODE_AUTO];
	StopRecording();
	int ModeSpecial = GameServer()->m_pController->OnCharacterDeath(this, GameServer()->m_apPlayers[Killer], Weapon);

	log_info("game", "kill killer='%d:%s' victim='%d:%s' weapon=%d special=%d",
		Killer, Server()->ClientName(Killer),
		GetCid(), Server()->ClientName(GetCid()), Weapon, ModeSpecial);

	if(SendKillMsg)
	{
		SendDeathMessageIfNotInLockedTeam(Killer, Weapon, ModeSpecial);
	}

	// a nice sound, and bursting tee death effect
	Env()->CreateSound(m_Pos, SOUND_PLAYER_DIE, TeamMask(), GetCid());
	Env()->CreateDeath(m_Pos, GetCid(), TeamMask());

	// this is to rate limit respawning to 3 secs
	m_pPlayer->m_PreviousDieTick = m_pPlayer->m_DieTick;
	m_pPlayer->m_DieTick = GameWorld()->GameTick();

	m_Alive = false;
	SetSolo(false);

	GameWorld()->RemoveEntity(this);
	GameWorld()->m_Core.m_apCharacters[GetCid()] = nullptr;
	Teams()->OnCharacterDeath(GetCid(), Weapon);
	CancelSwapRequests();
}

void CCharacter::SendDeathMessageIfNotInLockedTeam(int Killer, int Weapon, int ModeSpecial)
{
	if((Team() == TEAM_FLOCK || Teams()->TeamFlock(Team()) || Teams()->TeamSize(Team()) == 1 || Teams()->GetTeamState(Team()) == ETeamState::OPEN || !Teams()->TeamLocked(Team())))
	{
		CNetMsg_Sv_KillMsg Msg;
		Msg.m_Killer = Killer;
		Msg.m_Victim = GetCid();
		Msg.m_Weapon = Weapon;
		Msg.m_ModeSpecial = ModeSpecial;
		Server()->SendPackMsg(&Msg, MSGFLAG_VITAL, -1);
	}
}

void CCharacter::CancelSwapRequests()
{
	for(auto &pPlayer : GameServer()->m_apPlayers)
	{
		if(pPlayer && pPlayer->m_SwapTargetsClientId == GetCid())
			pPlayer->m_SwapTargetsClientId = -1;
	}
	GetPlayer()->m_SwapTargetsClientId = -1;
}

CClientMask CCharacter::TeamMaskWithoutSelfAndSixup()
{
	// Some sounds are triggered client-side for the acting player (or for all
	// players on Sixup) so we need to avoid duplicating them.
	return Teams()->TeamMask(Team(), GetCid(), GetCid(), CGameContext::FLAG_SIX);
}

CClientMask CCharacter::TeamMaskWithoutSixup()
{
	return Teams()->TeamMask(Team(), -1, GetCid(), CGameContext::FLAG_SIX);
}

int CCharacter::GetPlayerTeam() const
{
	return m_pPlayer->GetTeam();
}

bool CCharacter::HasNinjaJetpack() const
{
	return m_pPlayer->m_NinjaJetpack;
}

void CCharacter::SetDefaultEmote()
{
	SetEmote(m_pPlayer->GetDefaultEmote(), -1);
}

int CCharacter::GetDieTick() const
{
	return m_pPlayer->m_DieTick;
}

void CCharacter::SetDieTick(int Tick)
{
	m_pPlayer->m_DieTick = Tick;
}

int CCharacter::GetCid() const
{
	return m_pPlayer->GetCid();
}

void CCharacter::FillAntibot(CAntibotCharacterData *pData)
{
	pData->m_Pos = m_Pos;
	pData->m_Vel = m_Core.m_Vel;
	pData->m_Angle = m_Core.m_Angle;
	pData->m_HookedPlayer = m_Core.HookedPlayer();
	pData->m_SpawnTick = m_SpawnTick;
	pData->m_WeaponChangeTick = m_WeaponChangeTick;

	// 0
	pData->m_aLatestInputs[0].m_Direction = m_LatestInput.m_Direction;
	pData->m_aLatestInputs[0].m_TargetX = m_LatestInput.m_TargetX;
	pData->m_aLatestInputs[0].m_TargetY = m_LatestInput.m_TargetY;
	pData->m_aLatestInputs[0].m_Jump = m_LatestInput.m_Jump;
	pData->m_aLatestInputs[0].m_Fire = m_LatestInput.m_Fire;
	pData->m_aLatestInputs[0].m_Hook = m_LatestInput.m_Hook;
	pData->m_aLatestInputs[0].m_PlayerFlags = m_LatestInput.m_PlayerFlags;
	pData->m_aLatestInputs[0].m_WantedWeapon = m_LatestInput.m_WantedWeapon;
	pData->m_aLatestInputs[0].m_NextWeapon = m_LatestInput.m_NextWeapon;
	pData->m_aLatestInputs[0].m_PrevWeapon = m_LatestInput.m_PrevWeapon;

	// 1
	pData->m_aLatestInputs[1].m_Direction = m_LatestPrevInput.m_Direction;
	pData->m_aLatestInputs[1].m_TargetX = m_LatestPrevInput.m_TargetX;
	pData->m_aLatestInputs[1].m_TargetY = m_LatestPrevInput.m_TargetY;
	pData->m_aLatestInputs[1].m_Jump = m_LatestPrevInput.m_Jump;
	pData->m_aLatestInputs[1].m_Fire = m_LatestPrevInput.m_Fire;
	pData->m_aLatestInputs[1].m_Hook = m_LatestPrevInput.m_Hook;
	pData->m_aLatestInputs[1].m_PlayerFlags = m_LatestPrevInput.m_PlayerFlags;
	pData->m_aLatestInputs[1].m_WantedWeapon = m_LatestPrevInput.m_WantedWeapon;
	pData->m_aLatestInputs[1].m_NextWeapon = m_LatestPrevInput.m_NextWeapon;
	pData->m_aLatestInputs[1].m_PrevWeapon = m_LatestPrevInput.m_PrevWeapon;

	// 2
	pData->m_aLatestInputs[2].m_Direction = m_LatestPrevPrevInput.m_Direction;
	pData->m_aLatestInputs[2].m_TargetX = m_LatestPrevPrevInput.m_TargetX;
	pData->m_aLatestInputs[2].m_TargetY = m_LatestPrevPrevInput.m_TargetY;
	pData->m_aLatestInputs[2].m_Jump = m_LatestPrevPrevInput.m_Jump;
	pData->m_aLatestInputs[2].m_Fire = m_LatestPrevPrevInput.m_Fire;
	pData->m_aLatestInputs[2].m_Hook = m_LatestPrevPrevInput.m_Hook;
	pData->m_aLatestInputs[2].m_PlayerFlags = m_LatestPrevPrevInput.m_PlayerFlags;
	pData->m_aLatestInputs[2].m_WantedWeapon = m_LatestPrevPrevInput.m_WantedWeapon;
	pData->m_aLatestInputs[2].m_NextWeapon = m_LatestPrevPrevInput.m_NextWeapon;
	pData->m_aLatestInputs[2].m_PrevWeapon = m_LatestPrevPrevInput.m_PrevWeapon;
}

void CCharacter::HandleBroadcast()
{
	CPlayerData *pData = GameServer()->Score()->PlayerData(GetCid());

	if(m_DDRaceState == ERaceState::STARTED && m_pPlayer->GetClientVersion() == VERSION_VANILLA && !Server()->IsSixup(GetCid()) &&
		m_LastTimeCpBroadcasted != m_LastTimeCp && m_LastTimeCp > -1 &&
		m_TimeCpBroadcastEndTick > GameWorld()->GameTick() && pData->m_BestTime && pData->m_aBestTimeCp[m_LastTimeCp] != 0)
	{
		char aBroadcast[128];
		float Diff = m_aCurrentTimeCp[m_LastTimeCp] - pData->m_aBestTimeCp[m_LastTimeCp];
		str_format(aBroadcast, sizeof(aBroadcast), "Checkpoint | Diff : %+5.2f", Diff);
		GameServer()->SendBroadcast(aBroadcast, GetCid());
		m_LastTimeCpBroadcasted = m_LastTimeCp;
		m_LastBroadcast = GameWorld()->GameTick();
	}
	else if((m_pPlayer->m_TimerType == CPlayer::TIMERTYPE_BROADCAST || m_pPlayer->m_TimerType == CPlayer::TIMERTYPE_GAMETIMER_AND_BROADCAST) && m_DDRaceState == ERaceState::STARTED && m_LastBroadcast + GameWorld()->GameTickSpeed() * g_Config.m_SvTimeInBroadcastInterval <= GameWorld()->GameTick())
	{
		char aBuf[32];
		int Time = (int64_t)100 * ((float)(GameWorld()->GameTick() - m_StartTime) / ((float)GameWorld()->GameTickSpeed()));
		str_time(Time, ETimeFormat::HOURS, aBuf, sizeof(aBuf));
		GameServer()->SendBroadcast(aBuf, GetCid(), false);
		m_LastTimeCpBroadcasted = m_LastTimeCp;
		m_LastBroadcast = GameWorld()->GameTick();
	}
}

void CCharacter::SetTimeCheckpoint(int TimeCheckpoint)
{
	if(TimeCheckpoint > -1 && m_DDRaceState == ERaceState::STARTED && m_aCurrentTimeCp[TimeCheckpoint] == 0.0f && m_Time != 0.0f)
	{
		m_LastTimeCp = TimeCheckpoint;
		m_aCurrentTimeCp[m_LastTimeCp] = m_Time;
		m_TimeCpBroadcastEndTick = GameWorld()->GameTick() + GameWorld()->GameTickSpeed() * 2;
		if(m_pPlayer->GetClientVersion() >= VERSION_DDRACE || Server()->IsSixup(GetCid()))
		{
			CPlayerData *pData = GameServer()->Score()->PlayerData(GetCid());
			if(pData->m_aBestTimeCp[m_LastTimeCp] != 0.0f)
			{
				if(Server()->IsSixup(GetCid()))
				{
					protocol7::CNetMsg_Sv_Checkpoint Msg;
					float Diff = (m_aCurrentTimeCp[m_LastTimeCp] - pData->m_aBestTimeCp[m_LastTimeCp]) * 1000;
					Msg.m_Diff = (int)Diff;
					Server()->SendPackMsg(&Msg, MSGFLAG_VITAL, GetCid());
				}
				else
				{
					CNetMsg_Sv_DDRaceTime Msg;
					Msg.m_Time = (int)(m_Time * 100.0f);
					Msg.m_Finish = 0;
					float Diff = (m_aCurrentTimeCp[m_LastTimeCp] - pData->m_aBestTimeCp[m_LastTimeCp]) * 100;
					Msg.m_Check = (int)Diff;
					Server()->SendPackMsg(&Msg, MSGFLAG_VITAL, GetCid());
				}
			}
		}
	}
}

void CCharacter::SendZoneMsgs()
{
	// send zone leave msg
	// (m_TuneZoneOld >= 0: avoid zone leave msgs on spawn)
	if(m_TuneZoneOld >= 0 && GameServer()->m_aaZoneLeaveMsg[m_TuneZoneOld][0])
	{
		const char *pCur = GameServer()->m_aaZoneLeaveMsg[m_TuneZoneOld];
		const char *pPos;
		while((pPos = str_find(pCur, "\\n")))
		{
			char aBuf[256];
			str_copy(aBuf, pCur, pPos - pCur + 1);
			aBuf[pPos - pCur + 1] = '\0';
			pCur = pPos + 2;
			Env()->SendChatInfo(GetCid(), aBuf);
		}
		Env()->SendChatInfo(GetCid(), pCur);
	}
	// send zone enter msg
	if(GameServer()->m_aaZoneEnterMsg[m_TuneZone][0])
	{
		const char *pCur = GameServer()->m_aaZoneEnterMsg[m_TuneZone];
		const char *pPos;
		while((pPos = str_find(pCur, "\\n")))
		{
			char aBuf[256];
			str_copy(aBuf, pCur, pPos - pCur + 1);
			aBuf[pPos - pCur + 1] = '\0';
			pCur = pPos + 2;
			Env()->SendChatInfo(GetCid(), aBuf);
		}
		Env()->SendChatInfo(GetCid(), pCur);
	}
}

IAntibot *CCharacter::Antibot()
{
	return GameServer()->Antibot();
}

void CCharacter::SetTeams(CGameTeams *pTeams)
{
	m_pTeams = pTeams;
	m_Core.SetTeamsCore(&m_pTeams->m_Core);
}

bool CCharacter::TrySetRescue(int RescueMode)
{
	bool Set = false;
	if(g_Config.m_SvRescue || ((g_Config.m_SvTeam == SV_TEAM_FORCED_SOLO || Team() > TEAM_FLOCK) && Teams()->IsValidTeamNumber(Team())))
	{
		// check for nearby health pickups (also freeze)
		bool InHealthPickup = false;
		if(!m_Core.m_IsInFreeze)
		{
			CEntity *apEnts[9];
			int Num = GameWorld()->FindEntities(m_Pos, GetProximityRadius() + CPickup::ms_CollisionExtraSize, apEnts, std::size(apEnts), CGameWorld::ENTTYPE_PICKUP);
			for(int i = 0; i < Num; ++i)
			{
				CPickup *pPickup = static_cast<CPickup *>(apEnts[i]);
				if(pPickup->Type() == POWERUP_HEALTH)
				{
					// This uses a separate variable InHealthPickup instead of setting m_Core.m_IsInFreeze
					// as the latter causes freezebars to flicker when standing in the freeze range of a
					// health pickup. When the same code for client prediction is added, the freezebars
					// still flicker, but only when standing at the edge of the health pickup's freeze range.
					InHealthPickup = true;
					break;
				}
			}
		}

		if(!m_Core.m_IsInFreeze && IsGrounded() && !m_Core.m_DeepFrozen && !InHealthPickup)
		{
			ForceSetRescue(RescueMode);
			Set = true;
		}
	}

	return Set;
}

void CCharacter::ForceSetRescue(int RescueMode)
{
	m_RescueTee[RescueMode].Save(this);
	m_SetSavePos[RescueMode] = true;
}

void CCharacter::ResetJumps()
{
	m_Core.m_JumpedTotal = 0;
	m_Core.m_Jumped = 0;
}

void CCharacter::Pause(bool Pause)
{
	m_Paused = Pause;
	if(Pause)
	{
		GameWorld()->m_Core.m_apCharacters[GetCid()] = nullptr;
		GameWorld()->RemoveEntity(this);

		if(m_Core.HookedPlayer() != -1) // Keeping hook would allow cheats
		{
			ResetHook();
			GameWorld()->ReleaseHooked(GetCid());
		}
		m_PausedTick = GameWorld()->GameTick();
	}
	else
	{
		m_Core.m_Vel = vec2(0, 0);
		GameWorld()->m_Core.m_apCharacters[GetCid()] = &m_Core;
		GameWorld()->InsertEntity(this);
		if(m_Core.m_FreezeStart > 0 && m_PausedTick >= 0)
		{
			m_Core.m_FreezeStart += GameWorld()->GameTick() - m_PausedTick;
		}
	}
}

void CCharacter::DDRaceInit()
{
	m_Paused = false;
	m_DDRaceState = ERaceState::NONE;
	m_PrevPos = m_Pos;
	for(bool &Set : m_SetSavePos)
		Set = false;
	m_LastBroadcast = 0;
	m_TeamBeforeSuper = 0;
	m_Core.m_Id = GetCid();
	m_TeleCheckpoint = 0;
	m_Core.m_EndlessHook = g_Config.m_SvEndlessDrag;
	if(g_Config.m_SvHit)
	{
		m_Core.m_HammerHitDisabled = false;
		m_Core.m_ShotgunHitDisabled = false;
		m_Core.m_GrenadeHitDisabled = false;
		m_Core.m_LaserHitDisabled = false;
	}
	else
	{
		m_Core.m_HammerHitDisabled = true;
		m_Core.m_ShotgunHitDisabled = true;
		m_Core.m_GrenadeHitDisabled = true;
		m_Core.m_LaserHitDisabled = true;
	}
	m_Core.m_Jumps = 2;

	int Team = TeamsCore()->Team(m_Core.m_Id);

	if(Teams()->TeamLocked(Team) && !Env()->TeamFlock(Team))
	{
		for(int i = 0; i < MAX_CLIENTS; i++)
		{
			if(TeamsCore()->Team(i) == Team && i != m_Core.m_Id)
			{
				CCharacter *pChar = GameWorld()->GetCharacterById(i);

				if(pChar)
				{
					m_DDRaceState = pChar->m_DDRaceState;
					m_StartTime = pChar->m_StartTime;
				}
			}
		}
	}

	if(g_Config.m_SvTeam == SV_TEAM_MANDATORY && Team == TEAM_FLOCK)
	{
		GameServer()->SendStartWarning(GetCid(), "Please join a team before you start");
	}
}

void CCharacter::Rescue()
{
	if(m_SetSavePos[GetPlayer()->m_RescueMode] && !m_Core.m_Super && !m_Core.m_Invincible)
	{
		if(m_LastRescue + (int64_t)g_Config.m_SvRescueDelay * GameWorld()->GameTickSpeed() > GameWorld()->GameTick() && !Teams()->IsPractice(Team()))
		{
			char aBuf[256];
			str_format(aBuf, sizeof(aBuf), "You have to wait %d seconds until you can rescue yourself", (int)((m_LastRescue + (int64_t)g_Config.m_SvRescueDelay * GameWorld()->GameTickSpeed() - GameWorld()->GameTick()) / GameWorld()->GameTickSpeed()));
			Env()->SendChatInfo(GetCid(), aBuf);
			return;
		}

		m_LastRescue = GameWorld()->GameTick();
		int StartTime = m_StartTime;
		ERaceState DDRaceState = m_DDRaceState;
		m_RescueTee[GetPlayer()->m_RescueMode].Load(this);
		// Don't load these from saved tee:
		m_Core.m_Vel = vec2(0, 0);
		m_Core.m_HookState = HOOK_IDLE;
		m_StartTime = StartTime;
		m_DDRaceState = DDRaceState;
		m_SavedInput.m_Direction = 0;
		m_SavedInput.m_Jump = 0;
		// simulate releasing the fire button
		if((m_SavedInput.m_Fire & 1) != 0)
			m_SavedInput.m_Fire++;
		m_SavedInput.m_Fire &= INPUT_STATE_MASK;
		m_SavedInput.m_Hook = 0;
		m_pPlayer->Pause(CPlayer::PAUSE_NONE, true);
	}
}

CClientMask CCharacter::TeamMask()
{
	return Teams()->TeamMask(Team(), -1, GetCid());
}
