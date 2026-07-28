/* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
/* If you are missing that file, acquire a complete release at teeworlds.com.                */
#ifndef GAME_ENTITIES_GAMEWORLD_H
#define GAME_ENTITIES_GAMEWORLD_H

#include <game/gamecore.h>
#include <game/teamscore.h>

#include <list>
#include <vector>

class CCollision;
class CEntity;
class CCharacter;
class CMapBugs;
class IGameEnvironment;

/*
	Class: Game World
		Tracks all entities in the game. Propagates tick and
		snap calls to all entities.
*/
class CGameWorld
{
public:
	enum
	{
		ENTTYPE_PROJECTILE = 0,
		ENTTYPE_LASER,
		ENTTYPE_PICKUP,
		ENTTYPE_FLAG,
		ENTTYPE_CHARACTER,
		NUM_ENTTYPES
	};

private:
	void Reset();
	void RemoveEntities();

	CEntity *m_pNextTraverseEntity = nullptr;
	CEntity *m_apFirstEntityTypes[NUM_ENTTYPES];
	CCharacter *m_apCharacters[MAX_CLIENTS];

	class CGameContext *m_pGameServer;
	class CConfig *m_pConfig;
	class IServer *m_pServer;
	CTuningParams *m_pTuningList;
	CCollision *m_pCollision;
	const CMapBugs *m_pMapBugs;
	IGameEnvironment *m_pEnv = nullptr;

public:
	// Server only.
	class CGameContext *GameServer() { return m_pGameServer; }
	class CConfig *Config() { return m_pConfig; }
	class IServer *Server() { return m_pServer; }

	IGameEnvironment *Env() { return m_pEnv; }
	void SetEnv(IGameEnvironment *pEnv) { m_pEnv = pEnv; }
	CCollision *Collision() { return m_pCollision; }
	const CCollision *Collision() const { return m_pCollision; }
	bool EmulateBug(int Bug) const;
	std::vector<SSwitchers> &Switchers() { return m_Core.m_vSwitchers; }

	// Spelled the same way as in the client's prediction, so that the game logic
	// reading them can be shared. Defined in gameworld.cpp because they need the
	// complete CGameContext and IServer.
	int GameTick() const;
	int GameTickSpeed() const;
	CCharacter *GetCharacterById(int ClientId);
	class CTeamsCore *TeamsCore();
	// Which tuning an explosion takes its strength from. The server asks the
	// owning player, whose tune zone follows their view; the prediction has only
	// the character.
	int ExplosionTuneZone(int Owner);
	void CreateExplosion(vec2 Pos, int Owner, int Weapon, bool NoDamage, int ActivatedTeam, CClientMask Mask = CClientMask().set(), int Id = -1);

	bool m_ResetRequested;
	bool m_Paused;
	CWorldCore m_Core;

	CGameWorld();
	~CGameWorld();

	void SetGameServer(CGameContext *pGameServer);
	void Init(CCollision *pCollision, CTuningParams *pTuningList, const CMapBugs *pMapBugs);

	CEntity *FindFirst(int Type);
	CEntity *FindLast(int Type);

	/*
		Function: FindEntities
			Finds entities close to a position and returns them in a list.

		Arguments:
			Pos - Position.
			Radius - How close the entities have to be.
			ppEnts - Pointer to a list that should be filled with the pointers
				to the entities.
			Max - Number of entities that fits into the ents array.
			Type - Type of the entities to find.

		Returns:
			Number of entities found and added to the ents array.
	*/
	int FindEntities(vec2 Pos, float Radius, CEntity **ppEnts, int Max, int Type);

	/**
	 * Finds the CCharacter that intersects the line.
	 *
	 * @see IntersectEntity
	 *
	 * @param Pos0 Start position
	 * @param Pos1 End position
	 * @param Radius How far from the line the @link CCharacter @endlink is allowed to be
	 * @param NewPos Intersection position
	 * @param pNotThis Character to ignore intersecting with
	 * @param CollideWith Only find entities that can collide with that Client Id (pass -1 to ignore this check)
	 * @param pThisOnly Only search this specific character and ignore all others
	 *
	 * @return Pointer to the closest hit or `nullptr` if there is no intersection.
	 */
	CCharacter *IntersectCharacter(vec2 Pos0, vec2 Pos1, float Radius, vec2 &NewPos, const CCharacter *pNotThis = nullptr, int CollideWith = -1, const CCharacter *pThisOnly = nullptr);

	/**
	 * Finds the CEntity that intersects the line.
	 *
	 * @see IntersectCharacter
	 *
	 * @param Pos0 Start position
	 * @param Pos1 End position
	 * @param Radius How far from the line the @link CEntity @endlink is allowed to be
	 * @param Type Type of the entity to intersect
	 * @param NewPos Intersection position
	 * @param pNotThis Entity to ignore intersecting with
	 * @param CollideWith Only find entities that can collide with that Client Id (pass -1 to ignore this check)
	 * @param pThisOnly Only search this specific entity and ignore all others
	 *
	 * @return Pointer to the closest hit or `nullptr` if there is no intersection.
	 */
	CEntity *IntersectEntity(vec2 Pos0, vec2 Pos1, float Radius, int Type, vec2 &NewPos, const CEntity *pNotThis = nullptr, int CollideWith = -1, const CEntity *pThisOnly = nullptr);

	/*
		Function: ClosestCharacter
			Finds the closest CCharacter to a specific point.

		Arguments:
			Pos - The center position.
			Radius - How far off the CCharacter is allowed to be
			pNotThis - Entity to ignore

		Returns:
			Returns a pointer to the closest CCharacter or nullptr if no CCharacter is close enough.
	*/
	CCharacter *ClosestCharacter(vec2 Pos, float Radius, const CEntity *pNotThis);

	/*
		Function: InsertEntity
			Adds an entity to the world.

		Arguments:
			pEntity - Entity to add
	*/
	void InsertEntity(CEntity *pEntity, bool Last = false);

	/*
		Function: RemoveEntity
			Removes an entity from the world.

		Arguments:
			pEntity - Entity to remove
	*/
	void RemoveEntity(CEntity *pEntity);

	void RemoveEntitiesFromPlayer(int PlayerId);
	void RemoveEntitiesFromPlayers(int PlayerIds[], int NumPlayers);

	/*
		Function: Snap
			Calls Snap on all the entities in the world to create
			the snapshot.

		Arguments:
			SnappingClient - ID of the client which snapshot
			is being created.
	*/
	void Snap(int SnappingClient);

	/*
		Function: Tick
			Calls Tick on all the entities in the world to progress
			the world to the next tick.
	*/
	void Tick();

	/*
		Function: SwapClients
			Calls SwapClients on all the entities in the world to ensure that /swap
			command is handled safely.
	*/
	void SwapClients(int Client1, int Client2);

	// DDRace
	void ReleaseHooked(int ClientId);

	/*
		Function: IntersectedCharacters
			Finds all CCharacters that intersect the line.

		Arguments:
			Pos0 - Start position
			Pos1 - End position
			Radius - How for from the line the CCharacter is allowed to be.
			pNotThis - Entity to ignore intersecting with

		Returns:
			Returns list with all Characters on line.
	*/
	std::vector<CCharacter *> IntersectedCharacters(vec2 Pos0, vec2 Pos1, float Radius, const CEntity *pNotThis = nullptr);

	/*
		Everything from here down belongs to the client's prediction. The server
		carries the members - a few pointers per world and per entity - and never
		looks at them; the functions are defined in
		src/game/client/prediction/gameworld_predict.cpp.

		The prediction keeps three worlds: the one built from the last snapshot,
		the one it simulates forward, and the previous simulated one to
		interpolate from. CopyWorld clones entities between them, and each copy
		remembers the entity it came from so render state survives a
		repredict. FindMatch pairs an entity in the snapshot with the one the
		prediction already has, so it can be kept rather than rebuilt.
	*/
	CTeamsCore m_Teams;
	int m_GameTick;
	int m_LocalClientId;
	bool m_IsValidCopy;
	CGameWorld *m_pParent;
	CGameWorld *m_pChild;

	// The prediction expires timed switches as part of its own tick; the server
	// does it in CGameContext::OnTick and must not do it twice.
	bool m_ExpireSwitchersInTick = false;

	struct
	{
		bool m_IsDDRace;
		bool m_IsVanilla;
		bool m_IsFNG;
		bool m_InfiniteAmmo;
		bool m_PredictTiles;
		int m_PredictFreeze;
		bool m_PredictWeapons;
		bool m_PredictDDRace;
		bool m_IsSolo;
		bool m_UseTuneZones;
		bool m_BugDDRaceInput;
		bool m_NoWeakHookAndBounce;
		bool m_PredictEvents;
	} m_WorldConfig;

	void RemoveCharacter(CCharacter *pChar);
	bool IsLocalTeam(int OwnerId) const;
	void OnModified() const;
	void NetObjBegin(CTeamsCore Teams, int LocalClientId);
	void NetCharAdd(int ObjId, CNetObj_Character *pChar, CNetObj_DDNetCharacter *pExtended, int GameTeam, bool IsLocal);
	void NetObjAdd(int ObjId, int ObjType, const void *pObjData, const CNetObj_EntityEx *pDataEx);
	void NetObjEnd();
	void CopyWorld(CGameWorld *pFrom);
	CEntity *FindMatch(int ObjId, int ObjType, const void *pObjData);
	CEntity *GetEntity(int Id, int EntityType);
	void Clear();
	void ExpireSwitchers();

	class CPredictedEvent
	{
	public:
		int m_EventId;
		vec2 m_Pos; // NetEvent's Pos are integers
		int m_Id; // identifier to prevent adding the same event multiple times
		int m_Tick;

		int m_ExtraInfo;
		bool m_Handled = false;

		CPredictedEvent(int EventId, vec2 Pos, int Id, int Tick, int ExtraInfo = -1) :
			m_EventId(EventId), m_Pos(vec2((int)Pos.x, (int)Pos.y)), m_Id(Id), m_Tick(Tick), m_ExtraInfo(ExtraInfo)
		{
		}
	};

	std::vector<CPredictedEvent> m_PredictedEvents;

	void CreatePredictedEvent(const CPredictedEvent &NewEvent);
	bool CheckPredictedEventHandled(const CPredictedEvent &CheckEvent);
	void PlayPredictedEvents(int Tick);

	void CreatePredictedSound(vec2 Pos, int SoundId, int Id = -1);
	void CreatePredictedExplosionEvent(vec2 Pos, int Id = -1);
	void CreatePredictedHammerHitEvent(vec2 Pos, int Id = -1);
	void CreatePredictedDamageIndEvent(vec2 Pos, float Angle, int Amount, int Id = -1);

	const CTuningParams *TuningList() const { return m_pTuningList; }
	CTuningParams *TuningList() { return m_pTuningList; }
	const CTuningParams *GlobalTuning() const { return &TuningList()[0]; }
	CTuningParams *GlobalTuning() { return &TuningList()[0]; }
	const CTuningParams *GetTuning(int i) const { return &TuningList()[i]; }
	CTuningParams *GetTuning(int i) { return &TuningList()[i]; }
};

// Prediction only: the order in which the client believes characters were
// inserted, which decides who has the strong hook against whom.
class CCharOrder
{
public:
	std::list<int> m_Ids; // reverse of the order in the gameworld, since entities will be inserted in reverse
	CCharOrder()
	{
		Reset();
	}
	void Reset()
	{
		m_Ids.clear();
		for(int i = 0; i < MAX_CLIENTS; i++)
			m_Ids.push_back(i);
	}
	void GiveStrong(int c)
	{
		if(0 <= c && c < MAX_CLIENTS)
		{
			m_Ids.remove(c);
			m_Ids.push_front(c);
		}
	}
	void GiveWeak(int c)
	{
		if(0 <= c && c < MAX_CLIENTS)
		{
			m_Ids.remove(c);
			m_Ids.push_back(c);
		}
	}
	bool HasStrongAgainst(int From, int To)
	{
		for(int i : m_Ids)
		{
			if(i == To)
				return false;
			else if(i == From)
				return true;
		}
		return false;
	}
};

#endif
