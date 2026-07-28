/* (c) Shereef Marzouk. See "licence DDRace.txt" and the readme.txt in the root of the distribution for more information. */
#include "light.h"

#include <engine/server.h>

#include <game/collision.h>
#include <game/entities/character.h>
#include <game/gameenv.h>
#include <game/mapitems.h>

CLight::CLight(CGameWorld *pGameWorld, vec2 Pos, float Rotation, int Length,
	int Layer, int Number) :
	CEntity(pGameWorld, EEntityClass::LIGHT, true)
{
	m_To = vec2(0.0f, 0.0f);
	m_Core = vec2(0.0f, 0.0f);
	m_Layer = Layer;
	m_Number = Number;
	m_Tick = (GameWorld()->GameTickSpeed() * 0.15f);
	m_Pos = Pos;
	m_Rotation = Rotation;
	m_Length = Length;
	m_EvalTick = GameWorld()->GameTick();
	GameWorld()->InsertEntity(this);
	Step();
}

bool CLight::HitCharacter()
{
	std::vector<CCharacter *> vpHitCharacters = GameWorld()->IntersectedCharacters(m_Pos, m_To, 0.0f, nullptr);
	if(vpHitCharacters.empty())
		return false;
	for(auto *pChar : vpHitCharacters)
	{
		if(m_Layer == LAYER_SWITCH && m_Number > 0 && !Switchers()[m_Number].m_aStatus[pChar->Team()])
			continue;
		pChar->Freeze();
	}
	return true;
}

void CLight::Move()
{
	if(m_Speed != 0)
	{
		if((m_CurveLength >= m_Length && m_Speed > 0) || (m_CurveLength <= 0 && m_Speed < 0))
			m_Speed = -m_Speed;
		m_CurveLength += m_Speed * m_Tick + m_LengthL;
		m_LengthL = 0;
		if(m_CurveLength > m_Length)
		{
			m_LengthL = m_CurveLength - m_Length;
			m_CurveLength = m_Length;
		}
		else if(m_CurveLength < 0)
		{
			m_LengthL = 0 + m_CurveLength;
			m_CurveLength = 0;
		}
	}

	m_Rotation += m_AngularSpeed * m_Tick;
	if(m_Rotation > pi * 2)
		m_Rotation -= pi * 2;
	else if(m_Rotation < 0)
		m_Rotation += pi * 2;
}

void CLight::Step()
{
	Move();
	const vec2 Direction = vec2(std::sin(m_Rotation), std::cos(m_Rotation));
	const vec2 NextPosition = m_Pos + normalize(Direction) * m_CurveLength;
	Collision()->IntersectNoLaser(m_Pos, NextPosition, &m_To, nullptr);
}

void CLight::Reset()
{
	m_MarkedForDestroy = true;
}

void CLight::Tick()
{
	if(GameWorld()->GameTick() % (int)(GameWorld()->GameTickSpeed() * 0.15f) == 0)
	{
		m_EvalTick = GameWorld()->GameTick();
		Collision()->MoverSpeed(m_Pos.x, m_Pos.y, &m_Core);
		m_Pos += m_Core;
		Step();
	}

	HitCharacter();
}
