/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef GAME_GAMEENV_H
#define GAME_GAMEENV_H

#include <base/vmath.h>

#include <engine/shared/protocol.h>

#include <optional>

/*
	Class: IGameEnvironment
		The part of the game logic that the server and the client's prediction
		cannot share.

		Everything else lives on CGameWorld and CEntity, which are compiled once
		and used by both sides. What is left here are the effects: the server
		broadcasts them to clients as net events, while the prediction records
		them locally so that it can suppress the duplicate arriving from the
		server a moment later.

		Server settings are deliberately not part of this interface. Every
		setting the shared logic reads is flagged CFGFLAG_GAME and therefore
		stored in the map, which the client executes in
		CGameClient::LoadMapSettings(), so g_Config holds the same value on both
		sides. The one exception is sv_endless_super_hook, which is server-only;
		the client keeps it at its default, which is what it would have to assume
		in any case.
*/
class IGameEnvironment
{
public:
	virtual ~IGameEnvironment() = default;

	/*
		The Id argument identifies whatever caused the effect, usually an entity
		id or a start tick. Prediction runs several times per tick, so it uses
		Id to recognise an effect it has already recorded. The server ignores it.
	*/
	virtual void CreateSound(vec2 Pos, int Sound, CClientMask Mask, int Id) = 0;
	virtual void CreateExplosionEvent(vec2 Pos, CClientMask Mask, int Id) = 0;
	virtual void CreateDamageInd(vec2 Pos, float Angle, int Amount, CClientMask Mask, int Id) = 0;
	virtual void CreateHammerHit(vec2 Pos, CClientMask Mask, int Id) = 0;

	/*
		Death effects are not predicted: the client plays NETEVENTTYPE_DEATH
		straight from the snapshot without checking it against the predicted
		events, so predicting it too would show the particles twice.
	*/
	virtual void CreateDeath(vec2 Pos, int ClientId, CClientMask Mask) = 0;

	/*
		Returns the clients whose version is below Version. Prediction only ever
		simulates for the local client, so it answers with a full mask.
	*/
	virtual CClientMask ClientsMaskExcludeClientVersionAndHigher(int Version) const = 0;

	/*
		Ids that entities occupy in the snapshot. There are only so many, so
		allocation can fail and entities have to cope with having none. The
		prediction builds no snapshot and always answers with none; its entities
		take their id from the snapshot they were read out of instead.
	*/
	virtual std::optional<int> AllocateSnapId() { return std::nullopt; }
	virtual void FreeSnapId(int Id) {}

	/*
		Things the game logic tells a player about, or asks the rest of the
		server for. None of it changes the simulation, so the prediction ignores
		it or answers with what it can see.
	*/
	virtual void SendChatInfo(int ClientId, const char *pText) {}
	virtual void SendStartWarningInfo(int ClientId, const char *pMessage) {}
	virtual void SendZoneMessage(int ClientId, int TuneZone, bool Entering) {}
	virtual void SendBroadcastInfo(int ClientId, const char *pText, bool Important) {}
	virtual void PrintDebug(const char *pMessage) {}

	/*
		The game mode's say in what a tile does and where a player may spawn.
		The prediction has no game controller, so tiles the controller owns -
		start, finish, checkpoints - simply do nothing, and nothing may spawn.
	*/
	virtual void OnCharacterTiles(class CCharacter *pChr, int MapIndex) {}
	virtual void SetArmorProgress(class CCharacter *pChr, int Progress) {}
	virtual bool CanSpawn(int Team, vec2 *pOutPos, int ClientId) { return false; }

	// Whether the client is connected and playing rather than spectating.
	virtual bool IsPlayerInGame(int ClientId) { return true; }

	/*
		Teams. The prediction has the teams core, which is what the simulation
		reads, but none of the bookkeeping around it.
	*/
	virtual bool TeamFlock(int Team) { return false; }
	virtual bool TeamIsPractice(int Team) { return false; }
	virtual bool TeeFinished(int ClientId) { return false; }
	virtual bool SetCharacterTeam(int ClientId, int Team, char *pError, int ErrorSize) { return true; }
	virtual void SetForceCharacterTeam(int ClientId, int Team) {}

	/*
		Antibot only ever observes.
	*/
	virtual void AntibotOnCharacterTick(int ClientId) {}
	virtual void AntibotOnHookAttach(int ClientId, bool Player) {}
	virtual void AntibotOnDirectInput(int ClientId) {}
	virtual void AntibotOnHammerFire(int ClientId) {}
	virtual void AntibotOnHammerFireReloading(int ClientId) {}
	virtual void AntibotOnHammerHit(int ClientId, int TargetClientId) {}
};

#endif
