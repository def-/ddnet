// Throwaway: what the game layer holds at a position.
#include <base/logger.h>
#include <base/os.h>
#include <base/str.h>

#include <engine/shared/datafile.h>
#include <engine/shared/map.h>
#include <engine/storage.h>

#include <game/collision.h>
#include <game/layers.h>
#include <game/mapitems.h>

#include <memory>

int main(int argc, const char *argv[])
{
	std::unique_ptr<IStorage> pStorage = CreateLocalStorage();
	CCmdlineFix CmdlineFix(&argc, &argv);
	log_set_global_logger_default();
	if(argc < 4)
	{
		printf("Usage: %s <map> <x> <y> [radius]\n", argv[0]);
		return -1;
	}
	CMap Map;
	if(!Map.Load(pStorage.get(), argv[1], IStorage::TYPE_ALL_OR_ABSOLUTE))
	{
		printf("failed to load map\n");
		return -1;
	}
	CLayers Layers;
	Layers.Init(&Map, false, false);
	CCollision Collision;
	Collision.Init(&Layers);
	const int X = str_toint(argv[2]);
	const int Y = str_toint(argv[3]);
	const int Radius = argc > 4 ? str_toint(argv[4]) : 1;
	if(str_comp(argv[2], "find") == 0)
	{
		// Every tile of an index, game layer and front layer
		const int Wanted = str_toint(argv[3]);
		const int Width = Layers.GameLayer()->m_Width;
		const int Height = Layers.GameLayer()->m_Height;
		int Found = 0;
		for(int Ty = 0; Ty < Height; Ty++)
		{
			for(int Tx = 0; Tx < Width; Tx++)
			{
				const int Index = Collision.GetPureMapIndex(vec2(Tx * 32 + 16, Ty * 32 + 16));
				if(Collision.GetTileIndex(Index) == Wanted || Collision.GetFrontTileIndex(Index) == Wanted)
				{
					if(Found < 40)
						printf("tile %d,%d at %d,%d\n", Tx, Ty, Tx * 32 + 16, Ty * 32 + 16);
					Found++;
				}
			}
		}
		printf("%d tiles with index %d\n", Found, Wanted);
		return 0;
	}
	if(argc > 5 && str_comp(argv[5], "grid") == 0)
	{
		// One character per tile: . empty, # solid, F freeze, D deep freeze,
		// U unfreeze, S speedup, T tele, other indexes as their number mod 10
		for(int Ty = Y / 32 - Radius; Ty <= Y / 32 + Radius; Ty++)
		{
			printf("%5d ", Ty);
			for(int Tx = X / 32 - Radius; Tx <= X / 32 + Radius; Tx++)
			{
				const int Index = Collision.GetPureMapIndex(vec2(Tx * 32 + 16, Ty * 32 + 16));
				const int Game = Collision.GetTileIndex(Index);
				const int Front = Collision.GetFrontTileIndex(Index);
				const int Tile = Game != 0 ? Game : Front;
				char Char = Tile == 0 ? '.' : Tile == TILE_SOLID ? '#' :
						      Tile == TILE_NOHOOK        ? 'N' :
						      Tile == TILE_FREEZE        ? 'F' :
						      Tile == TILE_DFREEZE       ? 'D' :
						      Tile == TILE_UNFREEZE      ? 'U' :
						      Tile == TILE_DUNFREEZE     ? 'u' :
										   (char)('0' + Tile % 10);
				printf("%c", Char);
			}
			printf("\n");
		}
		return 0;
	}
	for(int Ty = Y / 32 - Radius; Ty <= Y / 32 + Radius; Ty++)
	{
		for(int Tx = X / 32 - Radius; Tx <= X / 32 + Radius; Tx++)
		{
			const int Index = Collision.GetPureMapIndex(vec2(Tx * 32 + 16, Ty * 32 + 16));
			printf("tile %4d,%4d game %3d front %3d %s%s\n", Tx, Ty,
				Collision.GetTileIndex(Index), Collision.GetFrontTileIndex(Index),
				Collision.GetTileIndex(Index) == TILE_FREEZE || Collision.GetFrontTileIndex(Index) == TILE_FREEZE ? "FREEZE " : "",
				Collision.GetTileIndex(Index) == TILE_UNFREEZE || Collision.GetFrontTileIndex(Index) == TILE_UNFREEZE ? "UNFREEZE" : "");
		}
	}
	return 0;
}
