#include "graphics.h"

// helper functions
void IGraphics::CalcScreenParams(float Aspect, float Zoom, float *pWidth, float *pHeight) const
{
	const float Amount = 1150 * 1000;
	const float WMax = 1500;
	const float MinAspect = 5.0f / 4.0f;

	const float f = std::sqrt(Amount) / std::sqrt(Aspect);
	*pWidth = f * Aspect;
	*pHeight = f;

	// A screen narrower than 5:4 shows the width a 5:4 screen would and uses
	// the extra room for more of the world vertically, rather than zooming in.
	// Only reachable with gfx_limit_aspect_ratio off, the viewport is clamped
	// to 5:4 otherwise, which is also why the height needs no upper limit.
	if(Aspect < MinAspect)
	{
		*pWidth = std::sqrt(Amount * MinAspect);
		*pHeight = *pWidth / Aspect;
	}

	// limit the view
	if(*pWidth > WMax)
	{
		*pWidth = WMax;
		*pHeight = *pWidth / Aspect;
	}

	*pWidth *= Zoom;
	*pHeight *= Zoom;
}

CScreenRect IGraphics::MapScreenToWorld(float CenterX, float CenterY, float ParallaxX, float ParallaxY,
	float ParallaxZoom, float OffsetX, float OffsetY, float Aspect, float Zoom) const
{
	float Width, Height;
	CalcScreenParams(Aspect, Zoom, &Width, &Height);

	float Scale = (ParallaxZoom * (Zoom - 1.0f) + 100.0f) / 100.0f / Zoom;
	Width *= Scale;
	Height *= Scale;

	CenterX *= ParallaxX / 100.0f;
	CenterY *= ParallaxY / 100.0f;

	return CScreenRect(
		OffsetX + CenterX - Width / 2,
		OffsetY + CenterY - Height / 2,
		Width,
		Height);
}

void IGraphics::MapScreenToInterface(float CenterX, float CenterY, float Zoom)
{
	CScreenRect ScreenRect = MapScreenToWorld(CenterX, CenterY, 100.0f, 100.0f, 100.0f,
		0, 0, ScreenAspect(), Zoom);
	MapScreen(ScreenRect);
}

void IGraphics::MapScreenToSize(float Width, float Height)
{
	MapScreen(CScreenRect(0, 0, Width, Height));
}
