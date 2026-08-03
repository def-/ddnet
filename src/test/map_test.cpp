#include "test.h"

#include <base/io.h>
#include <base/mem.h>

#include <engine/shared/datafile.h>
#include <engine/shared/map.h>
#include <engine/storage.h>

#include <game/mapitems.h>

#include <gtest/gtest.h>

#include <memory>

// Writes a map containing a single tile layer using the tile skip encoding,
// then corrupts the compressed tile data so that it cannot be uncompressed.
static void WriteMapWithBrokenTileData(IStorage *pStorage, const char *pFilename, int Width, int Height)
{
	{
		CDataFileWriter Writer;
		ASSERT_TRUE(Writer.Open(pStorage, pFilename));

		CMapItemVersion Version;
		Version.m_Version = 1;
		Writer.AddItem(MAPITEMTYPE_VERSION, 0, sizeof(Version), &Version);

		std::vector<CTile> vTiles(Width * Height);
		const int Data = Writer.AddData(vTiles.size() * sizeof(CTile), vTiles.data());

		CMapItemLayerTilemap Tilemap = {};
		Tilemap.m_Layer.m_Version = 0;
		Tilemap.m_Layer.m_Type = LAYERTYPE_TILES;
		Tilemap.m_Layer.m_Flags = 0;
		Tilemap.m_Version = CMapItemLayerTilemap::VERSION_TEEWORLDS_TILESKIP;
		Tilemap.m_Width = Width;
		Tilemap.m_Height = Height;
		Tilemap.m_Data = Data;
		Tilemap.m_Image = -1;
		Tilemap.m_ColorEnv = -1;
		Writer.AddItem(MAPITEMTYPE_LAYER, 0, sizeof(Tilemap), &Tilemap);

		CMapItemGroup Group = {};
		Group.m_Version = 3;
		Group.m_ParallaxX = 100;
		Group.m_ParallaxY = 100;
		Group.m_StartLayer = 0;
		Group.m_NumLayers = 1;
		Writer.AddItem(MAPITEMTYPE_GROUP, 0, sizeof(Group), &Group);

		Writer.Finish();
	}

	// The tile data is the only compressed data of this map, so corrupting the
	// last byte of the file makes uncompressing it fail.
	void *pFileData;
	unsigned FileSize;
	IOHANDLE File = pStorage->OpenFile(pFilename, IOFLAG_READ, IStorage::TYPE_ALL);
	ASSERT_NE(File, nullptr);
	ASSERT_TRUE(io_read_all(File, &pFileData, &FileSize));
	io_close(File);
	ASSERT_GT(FileSize, 0u);
	static_cast<char *>(pFileData)[FileSize - 1] ^= 0xff;

	File = pStorage->OpenFile(pFilename, IOFLAG_WRITE, IStorage::TYPE_SAVE);
	ASSERT_NE(File, nullptr);
	ASSERT_EQ(io_write(File, pFileData, FileSize), FileSize);
	io_close(File);
	free(pFileData);
}

// Loading a map with tile data that cannot be uncompressed must not crash and
// must not leave the tiles of the affected layer uninitialized.
TEST(Map, BrokenTileData)
{
	std::unique_ptr<IStorage> pStorage = CreateLocalStorage();
	ASSERT_NE(pStorage, nullptr) << "Error creating local storage";

	CTestInfo Info;
	const int Width = 50;
	const int Height = 70;
	WriteMapWithBrokenTileData(pStorage.get(), Info.m_aFilename, Width, Height);

	{
		CMap Map;
		if(Map.Load(pStorage.get(), Info.m_aFilename, IStorage::TYPE_ALL))
		{
			const CMapItemLayerTilemap *pTilemap = static_cast<CMapItemLayerTilemap *>(Map.FindItem(MAPITEMTYPE_LAYER, 0));
			ASSERT_NE(pTilemap, nullptr);
			const CTile *pTiles = static_cast<CTile *>(Map.GetData(pTilemap->m_Data));
			if(pTiles != nullptr)
			{
				EXPECT_EQ(Map.GetDataSize(pTilemap->m_Data), (int)(Width * Height * sizeof(CTile)));
				for(int i = 0; i < Width * Height; i++)
				{
					EXPECT_EQ(pTiles[i].m_Index, 0);
				}
			}
			Map.Unload();
		}
	}

	if(!HasFailure())
	{
		pStorage->RemoveFile(Info.m_aFilename, IStorage::TYPE_SAVE);
	}
}

TEST(Map, ExtractTiles)
{
	CTile aPacked[2] = {};
	aPacked[0].m_Index = 1;
	aPacked[0].m_Skip = 2;
	aPacked[1].m_Index = 3;
	aPacked[1].m_Skip = 0;

	CTile aTiles[5];
	mem_zero(aTiles, sizeof(aTiles));
	CMap::ExtractTiles(aTiles, std::size(aTiles), aPacked, std::size(aPacked));

	EXPECT_EQ(aTiles[0].m_Index, 1);
	EXPECT_EQ(aTiles[1].m_Index, 1);
	EXPECT_EQ(aTiles[2].m_Index, 1);
	EXPECT_EQ(aTiles[3].m_Index, 3);
	EXPECT_EQ(aTiles[4].m_Index, 0);
	for(const CTile &Tile : aTiles)
	{
		EXPECT_EQ(Tile.m_Skip, 0);
		EXPECT_EQ(Tile.m_MustBe0, 0);
	}
}
