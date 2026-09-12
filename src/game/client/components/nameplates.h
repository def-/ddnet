#ifndef GAME_CLIENT_COMPONENTS_NAMEPLATES_H
#define GAME_CLIENT_COMPONENTS_NAMEPLATES_H

#include <base/color.h>
#include <base/vmath.h>

#include <game/client/component.h>
#include <game/client/components/multi_server.h>

struct CNetObj_PlayerInfo;

class CNamePlates : public CComponent
{
private:
	class CNamePlatesData;
	CNamePlatesData *m_pData;

public:
	void RenderNamePlateGame(vec2 Position, const CNetObj_PlayerInfo *pPlayerInfo, float Alpha);
	/**
	 * Renders the name plate of a player that is on one of the observed servers.
	 *
	 * Their direction and hook strength are not known here, so those parts are left out.
	 */
	void RenderNamePlateRemote(vec2 Position, int Server, int ClientId, const CMultiServer::CRemotePlayer &Player);
	void RenderNamePlatePreview(vec2 Position, int Dummy);
	void ResetNamePlates();
	int Sizeof() const override { return sizeof(*this); }
	void OnWindowResize() override;
	void OnRender() override;
	CNamePlates();
	~CNamePlates() override;
};

#endif
