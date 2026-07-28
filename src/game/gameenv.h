/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef GAME_GAMEENV_H
#define GAME_GAMEENV_H

#include <base/vmath.h>

#include <engine/shared/protocol.h>

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
};

#endif
