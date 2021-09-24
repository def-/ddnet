#include "engine/client/graphics_threaded.h"
#include "engine/graphics.h"
#include "engine/shared/image_manipulation.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <base/math.h>
#include <base/system.h>

#include <cstddef>
#include <limits>
#include <string>

#include <set>
#include <vector>
#include <vulkan/vulkan.h>

#include <engine/client/backend/backend_base.h>
#include <engine/client/backend_sdl.h>

#include <engine/shared/config.h>

#include <engine/storage.h>
#include <vulkan/vulkan_core.h>

#include <array>

class CCommandProcessorFragment_Vulkan : public CCommandProcessorFragment_GLBase
{
	void NotImplemented()
	{
		//	dbg_msg("not", "implemented!");
	}

	struct SDeviceMemoryBlock
	{
		VkDeviceMemory m_Mem = VK_NULL_HANDLE;
		VkDeviceSize m_Size = 0;
	};

	template<size_t ID>
	struct SMemoryBlock
	{
		VkDeviceSize m_Offset;
		VkDeviceSize m_UsedSize;

		// optional
		VkBuffer m_Buffer;

		SDeviceMemoryBlock m_BufferMem;
		void *m_pMappedBuffer;

		bool m_IsCached;
		size_t m_CacheIndex;
	};

	template<size_t ID>
	struct SMemoryBlockCache
	{
		std::vector<SMemoryBlock<ID>> m_aMemoryBlocks[4];
		std::vector<std::vector<SMemoryBlock<ID>>> m_FrameDelayedCachedBufferCleanup;

		void Init(size_t SwapChainImageCount)
		{
			m_FrameDelayedCachedBufferCleanup.resize(SwapChainImageCount);
		}

		void Cleanup(size_t ImgIndex)
		{
			for(auto &MemBlock : m_FrameDelayedCachedBufferCleanup[ImgIndex])
			{
				MemBlock.m_UsedSize = 0;
				m_aMemoryBlocks[MemBlock.m_CacheIndex].push_back(MemBlock);
			}
			m_FrameDelayedCachedBufferCleanup[ImgIndex].clear();
		}

		void FreeMemBlock(SMemoryBlock<ID> &Block, size_t ImgIndex)
		{
			m_FrameDelayedCachedBufferCleanup[ImgIndex].push_back(Block);
		}
	};

	static constexpr size_t s_StagingBufferCacheID = 0;
	static constexpr size_t s_VertexBufferCacheID = 1;
	static constexpr size_t s_ImageBufferCacheID = 2;

	SMemoryBlockCache<s_StagingBufferCacheID> m_StagingBufferCache;
	SMemoryBlockCache<s_VertexBufferCacheID> m_VertexBufferCache;
	SMemoryBlockCache<s_ImageBufferCacheID> m_ImageBufferCache;

	struct CTexture
	{
		VkImage m_Img = VK_NULL_HANDLE;
		SMemoryBlock<s_ImageBufferCacheID> m_ImgMem;
		VkImageView m_ImgView = VK_NULL_HANDLE;
		VkSampler m_aSamplers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

		VkImage m_Img3D = VK_NULL_HANDLE;
		SMemoryBlock<s_ImageBufferCacheID> m_Img3DMem;
		VkImageView m_Img3DView = VK_NULL_HANDLE;
		VkSampler m_Sampler3D = VK_NULL_HANDLE;

		int m_MemSize = 0;

		int m_Width = 0;
		int m_Height = 0;
		int m_Depth = 0;
		int m_RescaleCount = 0;
		float m_ResizeWidth = 0;
		float m_ResizeHeight = 0;

		std::array<VkDescriptorSet, 2> m_aVKStandardTexturedDescrSets;
		VkDescriptorSet m_VKStandard3DTexturedDescrSet;
		VkDescriptorSet m_VKTextDescrSet;
	};
	std::vector<CTexture> m_Textures;
	std::atomic<uint64_t> *m_pTextureMemoryUsage;

	TWGLint m_MaxTexSize;

	int m_LastBlendMode;
	bool m_LastClipEnable;

	int m_GlobalTextureLodBIAS;

	bool m_RecreateSwapChain = false;

	VkMemoryRequirements m_DummyImageMemRequirements;
	bool m_AllowsLinearBlitting = false;

	VkBuffer m_IndexBuffer;
	SDeviceMemoryBlock m_IndexBufferMemory;

	VkBuffer m_RenderIndexBuffer;
	SDeviceMemoryBlock m_RenderIndexBufferMemory;
	size_t m_CurRenderIndexPrimitiveCount;

	VkDeviceSize m_NonCoherentMemAlignment;

	class IStorage *m_pStorage;

	std::vector<std::vector<std::pair<VkBuffer, SDeviceMemoryBlock>>> m_FrameDelayedBufferCleanup;

private:
	std::vector<VkImageView> m_VKSwapChainImageViewList;
	std::vector<VkFramebuffer> m_VKFramebufferList;
	std::vector<VkCommandBuffer> m_CommandBuffers;
	std::vector<VkCommandBuffer> m_MemoryCommandBuffers;

	std::vector<bool> m_UsedMemoryCommandBuffer;

	// swapped by use case
	std::vector<VkSemaphore> m_WaitSemaphores;
	std::vector<VkSemaphore> m_SigSemaphores;

	std::vector<VkSemaphore> m_MemorySemaphores;

	std::vector<VkFence> m_FrameFences;
	std::vector<VkFence> m_ImagesFences;

	struct SBufferObject
	{
		SMemoryBlock<s_VertexBufferCacheID> m_Mem;
	};

	struct SBufferObjectFrame
	{
		SBufferObject m_BufferObject;

		// since stream buffers can be used the cur buffer should always be used for rendering
		bool m_IsStreamedBuffer = false;
		VkBuffer m_CurBuffer = VK_NULL_HANDLE;
		size_t m_CurBufferOffset = 0;
	};

	std::vector<SBufferObjectFrame> m_BufferObjects;

	enum EBufferContainerLastUsedPipelineType
	{
		BUFFER_CONTAINER_LAST_USED_PIPELINE_TYPE_NONE = 0,
		BUFFER_CONTAINER_LAST_USED_PIPELINE_TYPE_Text,
	};

	struct SBufferContainer
	{
		int m_BufferObjectIndex;
		EBufferContainerLastUsedPipelineType m_PipeType = BUFFER_CONTAINER_LAST_USED_PIPELINE_TYPE_NONE;
	};
	std::vector<SBufferContainer> m_BufferContainers;

	VkInstance m_VKInstance;
	VkPhysicalDevice m_VKGPU;
	uint32_t m_VKGraphicsQueueIndex = std::numeric_limits<uint32_t>::max();
	VkDevice m_VKDevice;
	VkQueue m_VKGraphicsQueue, m_VKPresentQueue;
	VkSurfaceKHR m_VKPresentSurface;
	VkExtent2D m_VKSwapImgExtent;

	VkDescriptorSetLayout m_StandardTexturedDescriptorSetLayout;
	VkDescriptorSetLayout m_Standard3DTexturedDescriptorSetLayout;

	VkDescriptorSetLayout m_TextDescriptorSetLayout;

	VkDescriptorSetLayout m_UniformDescriptorSetLayout;

	VkPipelineLayout m_StandardPipeLineLayout;
	VkPipelineLayout m_StandardLinePipeLineLayout;
	VkPipelineLayout m_StandardTexturedPipeLineLayout;

	VkPipelineLayout m_TextPipeLineLayout;

	VkPipelineLayout m_TilePipeLineLayout;
	VkPipelineLayout m_TileTexturedPipeLineLayout;
	VkPipelineLayout m_TileBorderPipeLineLayout;
	VkPipelineLayout m_TileBorderTexturedPipeLineLayout;
	VkPipelineLayout m_TileBorderLinePipeLineLayout;
	VkPipelineLayout m_TileBorderLineTexturedPipeLineLayout;

	VkPipelineLayout m_PrimExPipeLineLayout;
	VkPipelineLayout m_PrimExTexPipeLineLayout;
	VkPipelineLayout m_PrimExRotationlessPipeLineLayout;
	VkPipelineLayout m_PrimExRotationlessTexPipeLineLayout;

	VkPipelineLayout m_SpriteMultiPipeLineLayout;

	VkPipeline m_StandardPipeline;
	VkPipeline m_StandardLinePipeline;
	VkPipeline m_StandardTexturedPipeline;

	VkPipeline m_TextPipeline;

	VkPipeline m_TilePipeline;
	VkPipeline m_TileTexturedPipeline;
	VkPipeline m_TileBorderPipeline;
	VkPipeline m_TileBorderTexturedPipeline;
	VkPipeline m_TileBorderLinePipeline;
	VkPipeline m_TileBorderLineTexturedPipeline;

	VkPipeline m_PrimExPipeline;
	VkPipeline m_PrimExTexPipeline;
	VkPipeline m_PrimExRotationlessPipeline;
	VkPipeline m_PrimExRotationlessTexPipeline;

	VkPipeline m_SpriteMultiPipeline;

	VkPipeline m_LastPipeline = VK_NULL_HANDLE;

	VkCommandPool m_CommandPool;

	VkRenderPass m_VKRenderPass;

	VkSurfaceFormatKHR m_VKSurfFormat;

	VkDescriptorPool m_VKDescrPool;

	VkSwapchainKHR m_VKSwapChain = VK_NULL_HANDLE;
	std::vector<VkImage> m_VKChainImages;

	struct SFrameBuffers
	{
		VkBuffer m_Buffer;
		SDeviceMemoryBlock m_BufferMem;
		size_t m_Size;
		size_t m_Offset;
		void *m_MappedBufferData;
	};

	struct SFrameUniformBuffers
	{
		VkBuffer m_Buffer;
		SDeviceMemoryBlock m_BufferMem;
		size_t m_OffsetInBuffer;
		void *m_pMappedBufferData;
		VkDescriptorSet m_UniformSet;

		size_t m_UsedSize;
	};

	std::vector<std::vector<SFrameBuffers>> m_VKBuffersOfFrame;
	std::vector<std::vector<VkMappedMemoryRange>> m_VKBuffersOfFrameRangeData;

	std::vector<std::vector<SFrameUniformBuffers>> m_VKUniformBufferObjectsOfFrame;
	std::vector<std::vector<VkMappedMemoryRange>> m_VKUniformBufferObjectsOfFrameRangeData;
	std::vector<size_t> m_CurrentUniformUsedCount;

	uint32_t m_MaxFramesChain = 0;

	uint32_t m_CurFrames = 0;
	uint32_t m_CurImageIndex = 0;

	int64_t m_FrameCounter = 0;

	uint32_t m_CanvasWidth;
	uint32_t m_CanvasHeight;

	std::array<float, 4> m_aClearColor;

protected:
	void SetError(const char *pErr)
	{
		dbg_assert(false, pErr);
	}

	/************************
	* ENGINE IMPLEMENTATION
	************************/

	void GetBufferImpl(VkDeviceSize RequiredSize, VkBuffer &Buffer, SDeviceMemoryBlock &BufferMemory, VkBufferUsageFlags BufferUsage, VkMemoryPropertyFlags BufferProperties)
	{
		CreateBuffer(RequiredSize, BufferUsage, BufferProperties, Buffer, BufferMemory);
	}

	template<size_t ID,
		ssize_t MemoryBlockSize1, size_t BlockCount1,
		ssize_t MemoryBlockSize2, size_t BlockCount2,
		ssize_t MemoryBlockSize3, size_t BlockCount3,
		ssize_t MemoryBlockSize4, size_t BlockCount4,
		bool RequiresMapping>
	SMemoryBlock<ID> GetBufferBlockImpl(SMemoryBlockCache<ID> &MemoryCache, VkBufferUsageFlags BufferUsage, VkMemoryPropertyFlags BufferProperties, const void *pBufferData, VkDeviceSize RequiredSize)
	{
		SMemoryBlock<ID> RetBlock;

		auto &&CreateCacheBlock = [&](size_t CacheIndex, ssize_t MemoryBlockSize, size_t BlockCount) {
			if(!MemoryCache.m_aMemoryBlocks[CacheIndex].empty())
			{
				RetBlock = MemoryCache.m_aMemoryBlocks[CacheIndex].back();
				MemoryCache.m_aMemoryBlocks[CacheIndex].pop_back();
			}
			else
			{
				VkBuffer TmpBuffer;
				SDeviceMemoryBlock TmpBufferMemory;
				GetBufferImpl(MemoryBlockSize * BlockCount, TmpBuffer, TmpBufferMemory, BufferUsage, BufferProperties);

				void *pMapData = nullptr;

				if(RequiresMapping)
					vkMapMemory(m_VKDevice, TmpBufferMemory.m_Mem, 0, RequiredSize, 0, &pMapData);

				for(VkDeviceSize i = 0; i < BlockCount - 1; ++i)
				{
					SMemoryBlock<ID> NewBlock;
					NewBlock.m_Buffer = TmpBuffer;
					NewBlock.m_BufferMem = TmpBufferMemory;
					if(RequiresMapping)
						NewBlock.m_pMappedBuffer = ((char *)pMapData) + (MemoryBlockSize * i);
					else
						NewBlock.m_pMappedBuffer = nullptr;
					NewBlock.m_IsCached = true;
					NewBlock.m_CacheIndex = CacheIndex;
					NewBlock.m_Offset = MemoryBlockSize * i;
					NewBlock.m_UsedSize = 0;
					MemoryCache.m_aMemoryBlocks[CacheIndex].push_back(NewBlock);
				}

				RetBlock.m_Buffer = TmpBuffer;
				RetBlock.m_BufferMem = TmpBufferMemory;
				if(RequiresMapping)
					RetBlock.m_pMappedBuffer = ((char *)pMapData) + (MemoryBlockSize * (BlockCount - 1));
				else
					RetBlock.m_pMappedBuffer = nullptr;
				RetBlock.m_IsCached = true;
				RetBlock.m_CacheIndex = CacheIndex;
				RetBlock.m_Offset = MemoryBlockSize * (BlockCount - 1);
				RetBlock.m_UsedSize = RequiredSize;
			}

			if(RequiresMapping)
				mem_copy(RetBlock.m_pMappedBuffer, pBufferData, RequiredSize);
		};

		if(RequiredSize < (VkDeviceSize)MemoryBlockSize1)
		{
			CreateCacheBlock(0, MemoryBlockSize1, BlockCount1);
		}
		else if(MemoryBlockSize2 != -1 && RequiredSize < (VkDeviceSize)MemoryBlockSize2)
		{
			CreateCacheBlock(1, MemoryBlockSize2, BlockCount2);
		}
		else if(MemoryBlockSize3 != -1 && RequiredSize < (VkDeviceSize)MemoryBlockSize3)
		{
			CreateCacheBlock(2, MemoryBlockSize3, BlockCount3);
		}
		else if(MemoryBlockSize4 != -1 && RequiredSize < (VkDeviceSize)MemoryBlockSize4)
		{
			CreateCacheBlock(3, MemoryBlockSize4, BlockCount4);
		}
		else
		{
			VkBuffer TmpBuffer;
			SDeviceMemoryBlock TmpBufferMemory;
			GetBufferImpl(RequiredSize, TmpBuffer, TmpBufferMemory, BufferUsage, BufferProperties);

			void *pMapData = nullptr;
			if(RequiresMapping)
			{
				vkMapMemory(m_VKDevice, TmpBufferMemory.m_Mem, 0, RequiredSize, 0, &pMapData);
				mem_copy(pMapData, pBufferData, static_cast<size_t>(RequiredSize));
				vkUnmapMemory(m_VKDevice, TmpBufferMemory.m_Mem);
			}

			RetBlock.m_Buffer = TmpBuffer;
			RetBlock.m_BufferMem = TmpBufferMemory;
			RetBlock.m_pMappedBuffer = nullptr;
			RetBlock.m_IsCached = false;
			RetBlock.m_CacheIndex = 0;
			RetBlock.m_Offset = 0;
			RetBlock.m_UsedSize = RequiredSize;
		}

		return RetBlock;
	}

	SMemoryBlock<s_StagingBufferCacheID> GetStagingBuffer(const void *pBufferData, VkDeviceSize RequiredSize)
	{
		return GetBufferBlockImpl<s_StagingBufferCacheID, 16 * 1024, 200, 64 * 1024, 40, 1024 * 1024, 20, 8 * 1024 * 1024, 10, true>(m_StagingBufferCache, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pBufferData, RequiredSize);
	}

	void FreeStagingMemBlock(SMemoryBlock<s_StagingBufferCacheID> &Block)
	{
		if(!Block.m_IsCached)
		{
			m_FrameDelayedBufferCleanup[m_CurImageIndex].push_back({Block.m_Buffer, Block.m_BufferMem});
		}
		else
		{
			m_StagingBufferCache.FreeMemBlock(Block, m_CurImageIndex);
		}
	}

	SMemoryBlock<s_VertexBufferCacheID> GetVertexBuffer(VkDeviceSize RequiredSize)
	{
		return GetBufferBlockImpl<s_VertexBufferCacheID, 16 * 1024, 150, 1024 * 1024, 10, -1, 0, -1, 0, false>(m_VertexBufferCache, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, nullptr, RequiredSize);
	}

	void FreeVertexMemBlock(SMemoryBlock<s_VertexBufferCacheID> &Block)
	{
		if(!Block.m_IsCached)
		{
			m_FrameDelayedBufferCleanup[m_CurImageIndex].push_back({Block.m_Buffer, Block.m_BufferMem});
		}
		else
		{
			m_VertexBufferCache.FreeMemBlock(Block, m_CurImageIndex);
		}
	}

	static size_t ImageMemorySizeWithMipMaps(size_t Width, size_t Height, size_t BPP)
	{
		size_t ImgSize = Width * Height * BPP;
		size_t Ret = ImgSize;
		for(size_t i = 0; i < 15; ++i)
		{
			ImgSize /= 2;
			Ret += ImgSize;
		}

		return Ret;
	}

	static size_t ImageMipLevelCount(size_t Width, size_t Height, size_t Depth)
	{
		return floor(log2(maximum(Width, maximum(Height, Depth)))) + 1;
	}

	static size_t ImageMipLevelCount(VkExtent3D &ImgExtent)
	{
		return ImageMipLevelCount(ImgExtent.width, ImgExtent.height, ImgExtent.depth);
	}

	// no c++20 for consteval, so just precalculate them, aligned
	static constexpr ssize_t s_128x128ImgSize = 131070 + (65536 - (131070 % 65536)); // ImageMemorySizeWithMipMaps(128, 128, 4);
	static constexpr ssize_t s_256x256ImgSize = 524280 + (65536 - (524280 % 65536)); // ImageMemorySizeWithMipMaps(256, 256, 4);
	static constexpr ssize_t s_512x512ImgSize = 2097120 + (65536 - (2097120 % 65536)); // ImageMemorySizeWithMipMaps(512, 512, 4);
	static constexpr ssize_t s_1024x1024ImgSize = 8388480 + (65536 - (8388480 % 65536)); // ImageMemorySizeWithMipMaps(1024, 1024, 4);

	bool GetImageMemoryImpl(VkDeviceSize RequiredSize, SDeviceMemoryBlock &BufferMemory, VkMemoryPropertyFlags BufferProperties)
	{
		VkMemoryAllocateInfo MemAllocInfo{};
		MemAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		MemAllocInfo.allocationSize = RequiredSize;
		MemAllocInfo.memoryTypeIndex = FindMemoryType(m_VKGPU, m_DummyImageMemRequirements.memoryTypeBits, BufferProperties);

		BufferMemory.m_Size = RequiredSize;
		m_pTextureMemoryUsage->store(m_pTextureMemoryUsage->load(std::memory_order_relaxed) + RequiredSize, std::memory_order_relaxed);

		if(g_Config.m_DbgGfx >= 3)
		{
			dbg_msg("vulkan", "allocated chunk of memory with size: %zu for frame %zu (image)", RequiredSize, (size_t)m_CurImageIndex);
		}

		if(vkAllocateMemory(m_VKDevice, &MemAllocInfo, nullptr, &BufferMemory.m_Mem) != VK_SUCCESS)
		{
			SetError("Allocation from buffer object failed.");
			return false;
		}

		return true;
	}

	template<size_t ID,
		ssize_t MemoryBlockSize1, size_t BlockCount1,
		ssize_t MemoryBlockSize2, size_t BlockCount2,
		ssize_t MemoryBlockSize3, size_t BlockCount3,
		ssize_t MemoryBlockSize4, size_t BlockCount4>
	SMemoryBlock<ID> GetImageMemoryBlockImpl(SMemoryBlockCache<ID> &MemoryCache, VkMemoryPropertyFlags BufferProperties, VkDeviceSize RequiredSize, bool IsRGBAAnd2D)
	{
		SMemoryBlock<ID> RetBlock;

		auto &&CreateCacheBlock = [&](size_t CacheIndex, ssize_t MemoryBlockSize, size_t BlockCount) {
			if(!MemoryCache.m_aMemoryBlocks[CacheIndex].empty())
			{
				RetBlock = MemoryCache.m_aMemoryBlocks[CacheIndex].back();
				MemoryCache.m_aMemoryBlocks[CacheIndex].pop_back();
			}
			else
			{
				SDeviceMemoryBlock TmpBufferMemory;
				GetImageMemoryImpl(MemoryBlockSize * BlockCount, TmpBufferMemory, BufferProperties);

				for(VkDeviceSize i = 0; i < BlockCount - 1; ++i)
				{
					SMemoryBlock<ID> NewBlock;
					NewBlock.m_Buffer = VK_NULL_HANDLE;
					NewBlock.m_BufferMem = TmpBufferMemory;
					NewBlock.m_pMappedBuffer = nullptr;
					NewBlock.m_IsCached = true;
					NewBlock.m_CacheIndex = CacheIndex;
					NewBlock.m_Offset = MemoryBlockSize * i;
					NewBlock.m_UsedSize = 0;
					MemoryCache.m_aMemoryBlocks[CacheIndex].push_back(NewBlock);
				}

				RetBlock.m_Buffer = VK_NULL_HANDLE;
				RetBlock.m_BufferMem = TmpBufferMemory;
				RetBlock.m_pMappedBuffer = nullptr;
				RetBlock.m_IsCached = true;
				RetBlock.m_CacheIndex = CacheIndex;
				RetBlock.m_Offset = MemoryBlockSize * (BlockCount - 1);
				RetBlock.m_UsedSize = RequiredSize;
			}
		};

		if(IsRGBAAnd2D && RequiredSize < (VkDeviceSize)MemoryBlockSize1)
		{
			CreateCacheBlock(0, MemoryBlockSize1, BlockCount1);
		}
		else if(IsRGBAAnd2D && MemoryBlockSize2 != -1 && RequiredSize < (VkDeviceSize)MemoryBlockSize2)
		{
			CreateCacheBlock(1, MemoryBlockSize2, BlockCount2);
		}
		else if(IsRGBAAnd2D && MemoryBlockSize3 != -1 && RequiredSize < (VkDeviceSize)MemoryBlockSize3)
		{
			CreateCacheBlock(2, MemoryBlockSize3, BlockCount3);
		}
		else if(IsRGBAAnd2D && MemoryBlockSize4 != -1 && RequiredSize < (VkDeviceSize)MemoryBlockSize4)
		{
			CreateCacheBlock(3, MemoryBlockSize4, BlockCount4);
		}
		else
		{
			SDeviceMemoryBlock TmpBufferMemory;
			GetImageMemoryImpl(RequiredSize, TmpBufferMemory, BufferProperties);

			RetBlock.m_Buffer = VK_NULL_HANDLE;
			RetBlock.m_BufferMem = TmpBufferMemory;
			RetBlock.m_pMappedBuffer = nullptr;
			RetBlock.m_IsCached = false;
			RetBlock.m_CacheIndex = 0;
			RetBlock.m_Offset = 0;
			RetBlock.m_UsedSize = RequiredSize;
		}

		return RetBlock;
	}

	SMemoryBlock<s_ImageBufferCacheID> GetImageMemory(VkDeviceSize RequiredSize, bool Is2DAndRGBA)
	{
		return GetImageMemoryBlockImpl<s_ImageBufferCacheID, s_128x128ImgSize, 64, s_256x256ImgSize, 16, s_512x512ImgSize, 8, s_1024x1024ImgSize, 4>(m_ImageBufferCache, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, RequiredSize, Is2DAndRGBA);
	}

	void FreeImageMemBlock(SMemoryBlock<s_ImageBufferCacheID> &Block)
	{
		if(!Block.m_IsCached)
		{
			m_FrameDelayedBufferCleanup[m_CurImageIndex].push_back({Block.m_Buffer, Block.m_BufferMem});
		}
		else
		{
			m_ImageBufferCache.FreeMemBlock(Block, m_CurImageIndex);
		}
	}

	void WaitFrame()
	{
		size_t RangeUpdateCount = 0;
		for(auto &BufferOfFrame : m_VKBuffersOfFrame[m_CurImageIndex])
		{
			if(BufferOfFrame.m_Offset > 0)
			{
				auto &MemRange = m_VKBuffersOfFrameRangeData[m_CurImageIndex][RangeUpdateCount++];
				MemRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
				MemRange.memory = BufferOfFrame.m_BufferMem.m_Mem;
				MemRange.offset = 0;
				auto AlignmentMod = ((VkDeviceSize)BufferOfFrame.m_Offset % m_NonCoherentMemAlignment);
				auto AlignmentReq = (m_NonCoherentMemAlignment - AlignmentMod);
				if(AlignmentMod == 0)
					AlignmentReq = 0;
				MemRange.size = BufferOfFrame.m_Offset + AlignmentReq;
			}
		}
		if(RangeUpdateCount > 0)
		{
			vkFlushMappedMemoryRanges(m_VKDevice, RangeUpdateCount, m_VKBuffersOfFrameRangeData[m_CurImageIndex].data());
		}

		// now the buffer objects
		if(m_CurrentUniformUsedCount[m_CurImageIndex] > 0)
		{
			RangeUpdateCount = 0;
			for(size_t i = 0; i < m_CurrentUniformUsedCount[m_CurImageIndex]; ++i)
			{
				auto &BufferOfFrame = m_VKUniformBufferObjectsOfFrame[m_CurImageIndex][i];
				auto &MemRange = m_VKUniformBufferObjectsOfFrameRangeData[m_CurImageIndex][RangeUpdateCount++];
				MemRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
				MemRange.memory = BufferOfFrame.m_BufferMem.m_Mem;
				MemRange.offset = BufferOfFrame.m_OffsetInBuffer;
				auto AlignmentMod = ((VkDeviceSize)BufferOfFrame.m_UsedSize % m_NonCoherentMemAlignment);
				auto AlignmentReq = (m_NonCoherentMemAlignment - AlignmentMod);
				if(AlignmentMod == 0)
					AlignmentReq = 0;
				MemRange.size = BufferOfFrame.m_UsedSize + AlignmentReq;
			}
			if(RangeUpdateCount > 0)
			{
				vkFlushMappedMemoryRanges(m_VKDevice, RangeUpdateCount, m_VKUniformBufferObjectsOfFrameRangeData[m_CurImageIndex].data());
			}
			m_CurrentUniformUsedCount[m_CurImageIndex] = 0;
		}

		auto &CommandBuffer = m_CommandBuffers[m_CurImageIndex];
		vkCmdEndRenderPass(CommandBuffer);

		if(vkEndCommandBuffer(CommandBuffer) != VK_SUCCESS)
		{
			SetError("Command buffer cannot be ended anymore.");
		}

		VkSemaphore WaitSemaphore = m_WaitSemaphores[m_CurFrames];

		VkSubmitInfo SubmitInfo{};
		SubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		SubmitInfo.commandBufferCount = 1;
		SubmitInfo.pCommandBuffers = &CommandBuffer;

		VkCommandBuffer aCommandBuffers[2] = {};

		if(m_UsedMemoryCommandBuffer[m_CurImageIndex])
		{
			auto &MemoryCommandBuffer = m_MemoryCommandBuffers[m_CurImageIndex];
			vkEndCommandBuffer(MemoryCommandBuffer);

			aCommandBuffers[0] = MemoryCommandBuffer;
			aCommandBuffers[1] = CommandBuffer;
			SubmitInfo.commandBufferCount = 2;
			SubmitInfo.pCommandBuffers = aCommandBuffers;

			m_UsedMemoryCommandBuffer[m_CurImageIndex] = false;
		}

		VkSemaphore waitSemaphores[] = {WaitSemaphore};
		VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		SubmitInfo.waitSemaphoreCount = 1;
		SubmitInfo.pWaitSemaphores = waitSemaphores;
		SubmitInfo.pWaitDstStageMask = waitStages;

		VkSemaphore signalSemaphores[] = {m_SigSemaphores[m_CurFrames]};
		SubmitInfo.signalSemaphoreCount = 1;
		SubmitInfo.pSignalSemaphores = signalSemaphores;

		vkResetFences(m_VKDevice, 1, &m_FrameFences[m_CurFrames]);

		if(vkQueueSubmit(m_VKGraphicsQueue, 1, &SubmitInfo, m_FrameFences[m_CurFrames]) != VK_SUCCESS)
		{
			SetError("Submitting to graphics queue failed.");
		}

		std::swap(m_WaitSemaphores[m_CurFrames], m_SigSemaphores[m_CurFrames]);

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = {m_VKSwapChain};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &m_CurImageIndex;

		vkQueuePresentKHR(m_VKPresentQueue, &presentInfo);
		m_CurFrames = (m_CurFrames + 1) % m_MaxFramesChain;

		++m_FrameCounter;
	}

	void PrepareFrame()
	{
		vkWaitForFences(m_VKDevice, 1, &m_FrameFences[m_CurFrames], VK_TRUE, std::numeric_limits<uint64_t>::max());

		if(m_RecreateSwapChain)
		{
			m_RecreateSwapChain = false;
			RecreateSwapChain();
		}

		auto AcqResult = vkAcquireNextImageKHR(m_VKDevice, m_VKSwapChain, std::numeric_limits<uint64_t>::max(), m_SigSemaphores[m_CurFrames], VK_NULL_HANDLE, &m_CurImageIndex);
		if(AcqResult != VK_SUCCESS)
		{
			if(AcqResult == VK_ERROR_OUT_OF_DATE_KHR || m_RecreateSwapChain)
			{
				RecreateSwapChain();
				PrepareFrame();
				return;
			}
			else if(AcqResult != VK_SUBOPTIMAL_KHR)
				dbg_msg("vulkan", "acquire next image failed %d", (int)AcqResult);
		}
		std::swap(m_WaitSemaphores[m_CurFrames], m_SigSemaphores[m_CurFrames]);

		if(m_ImagesFences[m_CurImageIndex] != VK_NULL_HANDLE)
		{
			vkWaitForFences(m_VKDevice, 1, &m_ImagesFences[m_CurImageIndex], VK_TRUE, std::numeric_limits<uint64_t>::max());
		}
		m_ImagesFences[m_CurImageIndex] = m_FrameFences[m_CurFrames];

		// clear stream buffer offsets
		for(auto &BufferOfFrame : m_VKBuffersOfFrame[m_CurImageIndex])
		{
			BufferOfFrame.m_Offset = 0;
		}

		// clear pending buffers, that require deletion
		for(auto &BufferPair : m_FrameDelayedBufferCleanup[m_CurImageIndex])
		{
			vkDestroyBuffer(m_VKDevice, BufferPair.first, nullptr);
			vkFreeMemory(m_VKDevice, BufferPair.second.m_Mem, nullptr);
			m_pTextureMemoryUsage->store(m_pTextureMemoryUsage->load(std::memory_order_relaxed) - BufferPair.second.m_Size, std::memory_order_relaxed);

			if(g_Config.m_DbgGfx >= 3)
			{
				dbg_msg("vulkan", "deallocated chunk of memory with size: %zu from frame %zu", (size_t)BufferPair.second.m_Size, (size_t)m_CurImageIndex);
			}
		}
		m_FrameDelayedBufferCleanup[m_CurImageIndex].clear();

		m_StagingBufferCache.Cleanup(m_CurImageIndex);
		m_VertexBufferCache.Cleanup(m_CurImageIndex);
		m_ImageBufferCache.Cleanup(m_CurImageIndex);

		// clear frame
		vkResetCommandBuffer(m_CommandBuffers[m_CurImageIndex], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

		auto &CommandBuffer = m_CommandBuffers[m_CurImageIndex];
		VkCommandBufferBeginInfo BeginInfo{};
		BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if(vkBeginCommandBuffer(CommandBuffer, &BeginInfo) != VK_SUCCESS)
		{
			SetError("Command buffer cannot be filled anymore.");
		}

		VkRenderPassBeginInfo RenderPassInfo{};
		RenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		RenderPassInfo.renderPass = m_VKRenderPass;
		RenderPassInfo.framebuffer = m_VKFramebufferList[m_CurImageIndex];
		RenderPassInfo.renderArea.offset = {0, 0};
		RenderPassInfo.renderArea.extent = m_VKSwapImgExtent;

		VkClearValue ClearColorVal = {{{m_aClearColor[0], m_aClearColor[1], m_aClearColor[2], m_aClearColor[3]}}};
		RenderPassInfo.clearValueCount = 1;
		RenderPassInfo.pClearValues = &ClearColorVal;

		vkCmdBeginRenderPass(CommandBuffer, &RenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		m_LastPipeline = VK_NULL_HANDLE;
	}

	void NextFrame()
	{
		WaitFrame();
		PrepareFrame();
	}

	void SetState(const CCommandBuffer::SState &State)
	{
	}

	void Cmd_Shutdown(const SCommand_Shutdown *pCommand) {}

	void Cmd_Init(const SCommand_Init *pCommand)
	{
		pCommand->m_pCapabilities->m_TileBuffering = true;
		pCommand->m_pCapabilities->m_QuadBuffering = false;
		pCommand->m_pCapabilities->m_TextBuffering = true;
		pCommand->m_pCapabilities->m_QuadContainerBuffering = true;
		pCommand->m_pCapabilities->m_ShaderSupport = true;

		pCommand->m_pCapabilities->m_MipMapping = true;
		pCommand->m_pCapabilities->m_3DTextures = false;
		pCommand->m_pCapabilities->m_2DArrayTextures = true;
		pCommand->m_pCapabilities->m_NPOTTextures = true;

		pCommand->m_pCapabilities->m_ContextMajor = 1;
		pCommand->m_pCapabilities->m_ContextMinor = 1;
		pCommand->m_pCapabilities->m_ContextPatch = 0;

		pCommand->m_pCapabilities->m_TrianglesAsQuads = true;

		m_GlobalTextureLodBIAS = g_Config.m_GfxGLTextureLODBIAS;
		m_pTextureMemoryUsage = pCommand->m_pTextureMemoryUsage;

		*pCommand->m_pInitError = InitVulkanSDL(pCommand->m_pWindow, pCommand->m_Width, pCommand->m_Height, pCommand->m_pRendererString, pCommand->m_pVendorString, pCommand->m_pVersionString);
		m_pStorage = pCommand->m_pStorage;
		InitVulkan(pCommand->m_pStorage);

		std::array<uint32_t, (size_t)CCommandBuffer::MAX_VERTICES / 4 * 6> aIndices;
		int Primq = 0;
		for(int i = 0; i < CCommandBuffer::MAX_VERTICES / 4 * 6; i += 6)
		{
			aIndices[i] = Primq;
			aIndices[i + 1] = Primq + 1;
			aIndices[i + 2] = Primq + 2;
			aIndices[i + 3] = Primq;
			aIndices[i + 4] = Primq + 2;
			aIndices[i + 5] = Primq + 3;
			Primq += 4;
		}
		PrepareFrame();

		CreateIndexBuffer(aIndices.data(), sizeof(uint32_t) * aIndices.size(), m_IndexBuffer, m_IndexBufferMemory);
		CreateIndexBuffer(aIndices.data(), sizeof(uint32_t) * aIndices.size(), m_RenderIndexBuffer, m_RenderIndexBufferMemory);
		m_CurRenderIndexPrimitiveCount = CCommandBuffer::MAX_VERTICES / 4;
	}

	void Cmd_Texture_Update(const CCommandBuffer::SCommand_Texture_Update *pCommand)
	{
		size_t IndexTex = pCommand->m_Slot;
		size_t ImageSize = (size_t)pCommand->m_Width * pCommand->m_Height * TexFormatToPixelChannelCount(pCommand->m_Format);
		auto StagingBuffer = GetStagingBuffer(pCommand->m_pData, ImageSize);

		auto &Tex = m_Textures[IndexTex];

		VkFormat ImgFormat = TexFormatToVulkanFormat(pCommand->m_Format);

		if(ImgFormat == VK_FORMAT_R8G8B8_UNORM)
		{
			SetError("RGB images cannot be updated currently.");
		}

		ImageBarrier(Tex.m_Img, 0, 1, 0, 1, ImgFormat, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		CopyBufferToImage(StagingBuffer.m_Buffer, StagingBuffer.m_Offset, Tex.m_Img, pCommand->m_X, pCommand->m_Y, static_cast<uint32_t>(pCommand->m_Width), static_cast<uint32_t>(pCommand->m_Height), 1);
		ImageBarrier(Tex.m_Img, 0, 1, 0, 1, ImgFormat, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		FreeStagingMemBlock(StagingBuffer);

		free(pCommand->m_pData);
	}

	void DestroyTexture(int Slot)
	{
	}

	void Cmd_Texture_Destroy(const CCommandBuffer::SCommand_Texture_Destroy *pCommand)
	{
		size_t ImageIndex = (size_t)pCommand->m_Slot;
		auto &Texture = m_Textures[ImageIndex];
		if(Texture.m_Img != VK_NULL_HANDLE)
		{
			FreeImageMemBlock(Texture.m_ImgMem);
		}

		if(Texture.m_Img3D != VK_NULL_HANDLE)
		{
			FreeImageMemBlock(Texture.m_Img3DMem);
		}

		Texture = CTexture{};
	}

	void CreateTextureCMD(
		int Slot,
		int Width,
		int Height,
		int PixelSize,
		int Format,
		int StoreFormat,
		int Flags,
		void *&pData)
	{
		size_t ImageIndex = (size_t)Slot;
		int ImageColorChannels = TexFormatToImageColorChannelCount(Format);

		while(ImageIndex >= m_Textures.size())
		{
			m_Textures.resize((m_Textures.size() * 2) + 1);
		}

		// convert to RGBA (32 bit), better alignment
		if(Format == CCommandBuffer::TEXFORMAT_RGB)
		{
			uint8_t *pNewData = (uint8_t *)malloc(sizeof(uint8_t) * Width * Height * 4);
			const uint8_t *pCurData = (const uint8_t *)pData;
			for(size_t y = 0; y < (size_t)Height; ++y)
			{
				for(size_t x = 0; x < (size_t)Width; ++x)
				{
					size_t BuffOffCur = y * Width * 3 + x * 3;
					size_t BuffOffNew = y * Width * 4 + x * 4;
					mem_copy(&pNewData[BuffOffNew], &pCurData[BuffOffCur], sizeof(uint8_t) * 3);
					pNewData[BuffOffNew + 3] = 255;
				}
			}

			Format = CCommandBuffer::TEXFORMAT_RGBA;
			if(StoreFormat == CCommandBuffer::TEXFORMAT_RGB)
				StoreFormat = CCommandBuffer::TEXFORMAT_RGBA;
			PixelSize = 4;

			free(pData);
			pData = pNewData;
		}

		bool Requires2DTexture = (Flags & CCommandBuffer::TEXFLAG_NO_2D_TEXTURE) == 0;
		bool Requires2DTextureArray = (Flags & (CCommandBuffer::TEXFLAG_TO_2D_ARRAY_TEXTURE | CCommandBuffer::TEXFLAG_TO_2D_ARRAY_TEXTURE_SINGLE_LAYER)) != 0;
		bool Is2DTextureSingleLayer = (Flags & CCommandBuffer::TEXFLAG_TO_2D_ARRAY_TEXTURE_SINGLE_LAYER) != 0;
		bool RequiresMipMaps = (Flags & CCommandBuffer::TEXFLAG_NOMIPMAPS) == 0;
		size_t MipMapLevelCount = 1;
		if(RequiresMipMaps)
		{
			VkExtent3D ImgSize{(uint32_t)Width, (uint32_t)Height, 1};
			MipMapLevelCount = ImageMipLevelCount(ImgSize);
		}

		CTexture &Texture = m_Textures[ImageIndex];

		if(Requires2DTexture)
		{
			CreateTextureImage(ImageIndex, Texture.m_Img, Texture.m_ImgMem, pData, Format, Width, Height, 1, PixelSize, MipMapLevelCount);
			VkFormat ImgFormat = TexFormatToVulkanFormat(Format);
			VkImageView ImgView = CreateTextureImageView(Texture.m_Img, ImgFormat, VK_IMAGE_VIEW_TYPE_2D, 1, MipMapLevelCount);
			Texture.m_ImgView = ImgView;
			VkSampler ImgSampler = CreateTextureSampler(VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT, VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT, VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT);
			Texture.m_aSamplers[0] = ImgSampler;
			ImgSampler = CreateTextureSampler(VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
			Texture.m_aSamplers[1] = ImgSampler;

			CreateNewTexturedStandardDescriptorSets(ImageIndex, 0);
			CreateNewTexturedStandardDescriptorSets(ImageIndex, 1);
		}

		if(Requires2DTextureArray)
		{
			int Image3DWidth = Width;
			int Image3DHeight = Height;

			int ConvertWidth = Width;
			int ConvertHeight = Height;

			if(!Is2DTextureSingleLayer)
			{
				if(ConvertWidth == 0 || (ConvertWidth % 16) != 0 || ConvertHeight == 0 || (ConvertHeight % 16) != 0)
				{
					dbg_msg("gfx", "3D/2D array texture was resized");
					int NewWidth = maximum<int>(HighestBit(ConvertWidth), 16);
					int NewHeight = maximum<int>(HighestBit(ConvertHeight), 16);
					uint8_t *pNewTexData = (uint8_t *)Resize(ConvertWidth, ConvertHeight, NewWidth, NewHeight, Format, (const uint8_t *)pData);

					ConvertWidth = NewWidth;
					ConvertHeight = NewHeight;

					free(pData);
					pData = pNewTexData;
				}
			}

			void *p3DTexData = pData;
			if(!Is2DTextureSingleLayer)
			{
				p3DTexData = malloc((size_t)ImageColorChannels * Width * Height);
				if(!Texture2DTo3D(pData, ConvertWidth, ConvertHeight, ImageColorChannels, 16, 16, p3DTexData, Image3DWidth, Image3DHeight))
				{
					free(p3DTexData);
					p3DTexData = nullptr;
				}
			}

			if(p3DTexData != nullptr)
			{
				const size_t ImageDepth2DArray = (size_t)16 * 16;
				VkExtent3D ImgSize{(uint32_t)Image3DWidth, (uint32_t)Image3DHeight, 1};
				if(RequiresMipMaps)
					MipMapLevelCount = ImageMipLevelCount(ImgSize);

				CreateTextureImage(ImageIndex, Texture.m_Img3D, Texture.m_Img3DMem, p3DTexData, Format, Image3DWidth, Image3DHeight, ImageDepth2DArray, PixelSize, MipMapLevelCount);
				VkFormat ImgFormat = TexFormatToVulkanFormat(Format);
				VkImageView ImgView = CreateTextureImageView(Texture.m_Img3D, ImgFormat, VK_IMAGE_VIEW_TYPE_2D_ARRAY, ImageDepth2DArray, MipMapLevelCount);
				Texture.m_Img3DView = ImgView;
				VkSampler ImgSampler = CreateTextureSampler(VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
				Texture.m_Sampler3D = ImgSampler;

				CreateNew3DTexturedStandardDescriptorSets(ImageIndex);
			}
		}
	}

	void Cmd_Texture_Create(const CCommandBuffer::SCommand_Texture_Create *pCommand)
	{
		int Slot = pCommand->m_Slot;
		int Width = pCommand->m_Width;
		int Height = pCommand->m_Height;
		int PixelSize = pCommand->m_PixelSize;
		int Format = pCommand->m_Format;
		int StoreFormat = pCommand->m_StoreFormat;
		int Flags = pCommand->m_Flags;
		void *pData = pCommand->m_pData;

		CreateTextureCMD(Slot, Width, Height, PixelSize, Format, StoreFormat, Flags, pData);

		free(pData);
	}

	void Cmd_TextTextures_Create(const CCommandBuffer::SCommand_TextTextures_Create *pCommand)
	{
		int Slot = pCommand->m_Slot;
		int SlotOutline = pCommand->m_SlotOutline;
		int Width = pCommand->m_Width;
		int Height = pCommand->m_Height;

		int MemSize = Width * Height;
		void *pTmpData = malloc(MemSize);
		mem_zero(pTmpData, MemSize);

		CreateTextureCMD(Slot, Width, Height, 1, CCommandBuffer::TEXFORMAT_ALPHA, CCommandBuffer::TEXFORMAT_ALPHA, CCommandBuffer::TEXFLAG_NOMIPMAPS, pTmpData);
		CreateTextureCMD(SlotOutline, Width, Height, 1, CCommandBuffer::TEXFORMAT_ALPHA, CCommandBuffer::TEXFORMAT_ALPHA, CCommandBuffer::TEXFLAG_NOMIPMAPS, pTmpData);

		CreateNewTextDescriptorSets(Slot, SlotOutline);

		free(pTmpData);
	}

	void Cmd_TextTextures_Destroy(const CCommandBuffer::SCommand_TextTextures_Destroy *pCommand) { NotImplemented(); }

	void Cmd_Clear(const CCommandBuffer::SCommand_Clear *pCommand)
	{
		m_aClearColor[0] = pCommand->m_Color.r;
		m_aClearColor[1] = pCommand->m_Color.g;
		m_aClearColor[2] = pCommand->m_Color.b;
		m_aClearColor[3] = pCommand->m_Color.a;
	}

	void GetStateMatrix(const CCommandBuffer::SState &State, std::array<float, (size_t)4 * 2> &Matrix)
	{
		Matrix = {
			// column 1
			2.f / (State.m_ScreenBR.x - State.m_ScreenTL.x),
			0,
			// column 2
			0,
			2.f / (State.m_ScreenBR.y - State.m_ScreenTL.y),
			// column 3
			0,
			0,
			// column 4
			-((State.m_ScreenTL.x + State.m_ScreenBR.x) / (State.m_ScreenBR.x - State.m_ScreenTL.x)),
			-((State.m_ScreenTL.y + State.m_ScreenBR.y) / (State.m_ScreenBR.y - State.m_ScreenTL.y)),
		};
	}

	size_t GetAddressModeIndex(const CCommandBuffer::SState &State)
	{
		return State.m_WrapMode == CCommandBuffer::WRAP_REPEAT ? 0 : 1;
	}

	VkPipelineLayout &GetStandardPipeLayout(bool IsLineGeometry, bool IsTextured)
	{
		if(IsLineGeometry)
			return !IsTextured ? m_StandardLinePipeLineLayout : m_StandardTexturedPipeLineLayout;
		else
			return !IsTextured ? m_StandardPipeLineLayout : m_StandardTexturedPipeLineLayout;
	}

	VkPipeline &GetStandardPipe(bool IsLineGeometry, bool IsTextured)
	{
		if(IsLineGeometry)
			return !IsTextured ? m_StandardLinePipeline : m_StandardTexturedPipeline;
		else
			return !IsTextured ? m_StandardPipeline : m_StandardTexturedPipeline;
	}

	void Cmd_Render(const CCommandBuffer::SCommand_Render *pCommand)
	{
		std::array<float, (size_t)4 * 2> m;
		GetStateMatrix(pCommand->m_State, m);
		size_t AddressModeIndex = GetAddressModeIndex(pCommand->m_State);

		bool IsTextured = pCommand->m_State.m_Texture != -1;
		bool IsLineGeometry = pCommand->m_PrimType == CCommandBuffer::PRIMTYPE_LINES;

		auto &PipeLayout = GetStandardPipeLayout(IsLineGeometry, IsTextured);
		auto &PipeLine = GetStandardPipe(IsLineGeometry, IsTextured);

		auto &CommandBuffer = m_CommandBuffers[m_CurImageIndex];

		if(m_LastPipeline != PipeLine)
		{
			vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLine);
			m_LastPipeline = PipeLine;
		}

		size_t VertPerPrim = 2;
		bool IsIndexed = false;
		if(pCommand->m_PrimType == CCommandBuffer::PRIMTYPE_QUADS)
		{
			VertPerPrim = 4;
			IsIndexed = true;
		}
		else if(pCommand->m_PrimType == CCommandBuffer::PRIMTYPE_TRIANGLES)
		{
			VertPerPrim = 3;
		}

		VkBuffer VKBuffer;
		SDeviceMemoryBlock VKBufferMem;
		size_t BufferOff = 0;
		CreateStreamVertexBuffer(VKBuffer, VKBufferMem, BufferOff, pCommand->m_pVertices, VertPerPrim * sizeof(CCommandBuffer::SVertex) * pCommand->m_PrimCount);

		VkBuffer aVertexBuffers[] = {VKBuffer};
		VkDeviceSize aOffsets[] = {BufferOff};
		vkCmdBindVertexBuffers(CommandBuffer, 0, 1, aVertexBuffers, aOffsets);

		if(IsIndexed)
			vkCmdBindIndexBuffer(CommandBuffer, m_IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

		if(IsTextured)
		{
			auto &DescrSet = m_Textures[pCommand->m_State.m_Texture].m_aVKStandardTexturedDescrSets[AddressModeIndex];
			vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLayout, 0, 1, &DescrSet, 0, nullptr);
		}

		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SUniformGPos), m.data());

		if(IsIndexed)
			vkCmdDrawIndexed(CommandBuffer, static_cast<uint32_t>(pCommand->m_PrimCount * 6), 1, 0, 0, 0);
		else
			vkCmdDraw(CommandBuffer, static_cast<uint32_t>(pCommand->m_PrimCount * VertPerPrim), 1, 0, 0);
	}

	void Cmd_Screenshot(const CCommandBuffer::SCommand_Screenshot *pCommand)
	{
		NotImplemented();
	}

	void Cmd_RenderTex3D(const CCommandBuffer::SCommand_RenderTex3D *pCommand)
	{
		NotImplemented();
	}

	void Cmd_Update_Viewport(const CCommandBuffer::SCommand_Update_Viewport *pCommand)
	{
		if(pCommand->m_ByResize)
			m_RecreateSwapChain = true;
		else
		{
			NotImplemented();
		}
	}

	void Cmd_VSync(const CCommandBuffer::SCommand_VSync *pCommand)
	{
		m_RecreateSwapChain = true;
		*pCommand->m_pRetOk = true;
	}

	void Cmd_Finish(const CCommandBuffer::SCommand_Finish *pCommand)
	{ // just ignore it with vulkan
	}

	void Cmd_Swap(const CCommandBuffer::SCommand_Swap *pCommand)
	{
		NextFrame();
	}

	void CreateBufferObject(size_t BufferIndex, const void *pUploadData, VkDeviceSize BufferDataSize, bool IsOneFrameBuffer)
	{
		void *pUploadDataTmp = nullptr;
		if(pUploadData == nullptr)
		{
			pUploadDataTmp = malloc(BufferDataSize);
			pUploadData = pUploadDataTmp;
		}

		while(BufferIndex >= m_BufferObjects.size())
		{
			m_BufferObjects.resize((m_BufferObjects.size() * 2) + 1);
		}
		auto &BufferObject = m_BufferObjects[BufferIndex];

		VkBuffer VertexBuffer;
		size_t BufferOffset = 0;
		if(!IsOneFrameBuffer)
		{
			auto StagingBuffer = GetStagingBuffer(pUploadData, BufferDataSize);

			auto Mem = GetVertexBuffer(BufferDataSize);

			BufferObject.m_BufferObject.m_Mem = Mem;
			VertexBuffer = Mem.m_Buffer;
			BufferOffset = Mem.m_Offset;

			MemoryBarrier(VertexBuffer, Mem.m_Offset, BufferDataSize, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, true);
			CopyBuffer(StagingBuffer.m_Buffer, VertexBuffer, StagingBuffer.m_Offset, Mem.m_Offset, BufferDataSize);
			MemoryBarrier(VertexBuffer, Mem.m_Offset, BufferDataSize, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, false);
			FreeStagingMemBlock(StagingBuffer);
		}
		else
		{
			SDeviceMemoryBlock VertexBufferMemory;
			CreateStreamVertexBuffer(VertexBuffer, VertexBufferMemory, BufferOffset, pUploadData, BufferDataSize);
		}
		BufferObject.m_IsStreamedBuffer = IsOneFrameBuffer;
		BufferObject.m_CurBuffer = VertexBuffer;
		BufferObject.m_CurBufferOffset = BufferOffset;

		if(pUploadDataTmp != nullptr)
			free(pUploadDataTmp);
	}

	void DeleteBufferObject(size_t BufferIndex)
	{
		auto &BufferObject = m_BufferObjects[BufferIndex];
		if(!BufferObject.m_IsStreamedBuffer)
		{
			FreeVertexMemBlock(BufferObject.m_BufferObject.m_Mem);
		}
	}

	void Cmd_CreateBufferObject(const CCommandBuffer::SCommand_CreateBufferObject *pCommand)
	{
		bool IsOneFrameBuffer = (pCommand->m_Flags & IGraphics::EBufferObjectCreateFlags::BUFFER_OBJECT_CREATE_FLAGS_ONE_TIME_USE_BIT) != 0;
		CreateBufferObject((size_t)pCommand->m_BufferIndex, pCommand->m_pUploadData, (VkDeviceSize)pCommand->m_DataSize, IsOneFrameBuffer);
		if(pCommand->m_DeletePointer)
			free(pCommand->m_pUploadData);
	}

	void Cmd_UpdateBufferObject(const CCommandBuffer::SCommand_UpdateBufferObject *pCommand)
	{
		size_t BufferIndex = (size_t)pCommand->m_BufferIndex;
		bool DeletePointer = pCommand->m_DeletePointer;
		VkDeviceSize Offset = (VkDeviceSize)((intptr_t)pCommand->m_pOffset);
		void *pUploadData = pCommand->m_pUploadData;
		VkDeviceSize DataSize = (VkDeviceSize)pCommand->m_DataSize;

		auto StagingBuffer = GetStagingBuffer(pUploadData, DataSize);

		auto &MemBlock = m_BufferObjects[BufferIndex].m_BufferObject.m_Mem;
		VkBuffer VertexBuffer = MemBlock.m_Buffer;
		MemoryBarrier(VertexBuffer, Offset + MemBlock.m_Offset, DataSize, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, true);
		CopyBuffer(StagingBuffer.m_Buffer, VertexBuffer, StagingBuffer.m_Offset, Offset + MemBlock.m_Offset, DataSize);
		MemoryBarrier(VertexBuffer, Offset + MemBlock.m_Offset, DataSize, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, false);

		FreeStagingMemBlock(StagingBuffer);

		if(DeletePointer)
			free(pUploadData);
	}

	void Cmd_RecreateBufferObject(const CCommandBuffer::SCommand_RecreateBufferObject *pCommand)
	{
		DeleteBufferObject((size_t)pCommand->m_BufferIndex);
		bool IsOneFrameBuffer = (pCommand->m_Flags & IGraphics::EBufferObjectCreateFlags::BUFFER_OBJECT_CREATE_FLAGS_ONE_TIME_USE_BIT) != 0;
		CreateBufferObject((size_t)pCommand->m_BufferIndex, pCommand->m_pUploadData, (VkDeviceSize)pCommand->m_DataSize, IsOneFrameBuffer);
	}

	void Cmd_CopyBufferObject(const CCommandBuffer::SCommand_CopyBufferObject *pCommand)
	{
		size_t ReadBufferIndex = (size_t)pCommand->m_ReadBufferIndex;
		size_t WriteBufferIndex = (size_t)pCommand->m_WriteBufferIndex;
		auto &ReadMemBlock = m_BufferObjects[ReadBufferIndex].m_BufferObject.m_Mem;
		auto &WriteMemBlock = m_BufferObjects[WriteBufferIndex].m_BufferObject.m_Mem;
		VkBuffer ReadBuffer = ReadMemBlock.m_Buffer;
		VkBuffer WriteBuffer = WriteMemBlock.m_Buffer;

		VkDeviceSize DataSize = (VkDeviceSize)pCommand->m_CopySize;
		VkDeviceSize ReadOffset = (VkDeviceSize)pCommand->m_pReadOffset + ReadMemBlock.m_Offset;
		VkDeviceSize WriteOffset = (VkDeviceSize)pCommand->m_pWriteOffset + WriteMemBlock.m_Offset;

		MemoryBarrier(ReadBuffer, ReadOffset, DataSize, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, true);
		MemoryBarrier(WriteBuffer, WriteOffset, DataSize, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, true);
		CopyBuffer(ReadBuffer, WriteBuffer, ReadOffset, WriteOffset, DataSize);
		MemoryBarrier(WriteBuffer, WriteOffset, DataSize, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, false);
		MemoryBarrier(ReadBuffer, ReadOffset, DataSize, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, false);
	}

	void Cmd_DeleteBufferObject(const CCommandBuffer::SCommand_DeleteBufferObject *pCommand)
	{
		DeleteBufferObject((size_t)pCommand->m_BufferIndex);
	}

	void Cmd_CreateBufferContainer(const CCommandBuffer::SCommand_CreateBufferContainer *pCommand)
	{
		size_t ContainerIndex = (size_t)pCommand->m_BufferContainerIndex;
		while(ContainerIndex >= m_BufferContainers.size())
			m_BufferContainers.resize((m_BufferContainers.size() * 2) + 1);

		m_BufferContainers[ContainerIndex].m_BufferObjectIndex = pCommand->m_Attributes[0].m_VertBufferBindingIndex;

		m_BufferContainers[ContainerIndex].m_PipeType = BUFFER_CONTAINER_LAST_USED_PIPELINE_TYPE_NONE;

		NotImplemented();
	}

	void Cmd_UpdateBufferContainer(const CCommandBuffer::SCommand_UpdateBufferContainer *pCommand)
	{
		size_t ContainerIndex = (size_t)pCommand->m_BufferContainerIndex;
		m_BufferContainers[ContainerIndex].m_PipeType = BUFFER_CONTAINER_LAST_USED_PIPELINE_TYPE_NONE;
		NotImplemented();
	}

	void Cmd_DeleteBufferContainer(const CCommandBuffer::SCommand_DeleteBufferContainer *pCommand)
	{
		size_t ContainerIndex = (size_t)pCommand->m_BufferContainerIndex;
		bool DeleteAllBO = pCommand->m_DestroyAllBO;
		m_BufferContainers[ContainerIndex].m_PipeType = BUFFER_CONTAINER_LAST_USED_PIPELINE_TYPE_NONE;
		if(DeleteAllBO)
		{
			DeleteBufferObject(m_BufferContainers[ContainerIndex].m_BufferObjectIndex);
		}
		NotImplemented();
	}

	void Cmd_IndicesRequiredNumNotify(const CCommandBuffer::SCommand_IndicesRequiredNumNotify *pCommand)
	{
		size_t IndicesCount = pCommand->m_RequiredIndicesNum;
		if(m_CurRenderIndexPrimitiveCount < IndicesCount / 6)
		{
			m_FrameDelayedBufferCleanup[m_CurImageIndex].push_back({m_RenderIndexBuffer, m_RenderIndexBufferMemory});
			std::vector<uint32_t> Indices(IndicesCount);
			uint32_t Primq = 0;
			for(size_t i = 0; i < IndicesCount; i += 6)
			{
				Indices[i] = Primq;
				Indices[i + 1] = Primq + 1;
				Indices[i + 2] = Primq + 2;
				Indices[i + 3] = Primq;
				Indices[i + 4] = Primq + 2;
				Indices[i + 5] = Primq + 3;
				Primq += 4;
			}
			CreateIndexBuffer(Indices.data(), Indices.size() * sizeof(uint32_t), m_RenderIndexBuffer, m_RenderIndexBufferMemory);
			m_CurRenderIndexPrimitiveCount = IndicesCount / 6;
		}
	}

	VkPipelineLayout &GetTileLayerPipeLayout(int Type, bool IsTextured)
	{
		if(Type == 0)
			return !IsTextured ? m_TilePipeLineLayout : m_TileTexturedPipeLineLayout;
		else if(Type == 1)
			return !IsTextured ? m_TileBorderPipeLineLayout : m_TileBorderTexturedPipeLineLayout;
		else
			return !IsTextured ? m_TileBorderLinePipeLineLayout : m_TileBorderLineTexturedPipeLineLayout;
	}

	VkPipeline &GetTileLayerPipe(int Type, bool IsTextured)
	{
		if(Type == 0)
			return !IsTextured ? m_TilePipeline : m_TileTexturedPipeline;
		else if(Type == 1)
			return !IsTextured ? m_TileBorderPipeline : m_TileBorderTexturedPipeline;
		else
			return !IsTextured ? m_TileBorderLinePipeline : m_TileBorderLineTexturedPipeline;
	}

	void RenderTileLayer(const CCommandBuffer::SState &State, size_t BufferContainerIndex, int Type, const GL_SColorf &Color, const vec2 &Dir, const vec2 &Off, int32_t JumpIndex, size_t IndicesDrawNum, char *const *pIndicesOffsets, const unsigned int *pDrawCount, size_t InstanceCount)
	{
		std::array<float, (size_t)4 * 2> m;
		GetStateMatrix(State, m);

		bool IsTextured = State.m_Texture != -1;
		auto &PipeLayout = GetTileLayerPipeLayout(Type, IsTextured);
		auto &PipeLine = GetTileLayerPipe(Type, IsTextured);

		auto &CommandBuffer = m_CommandBuffers[m_CurImageIndex];

		if(m_LastPipeline != PipeLine)
		{
			vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLine);
			m_LastPipeline = PipeLine;
		}

		size_t BufferObjectIndex = (size_t)m_BufferContainers[BufferContainerIndex].m_BufferObjectIndex;
		auto &BufferObject = m_BufferObjects[BufferObjectIndex];

		VkBuffer VKBuffer = BufferObject.m_CurBuffer;
		size_t BufferOff = BufferObject.m_CurBufferOffset;

		VkBuffer aVertexBuffers[] = {VKBuffer};
		VkDeviceSize aOffsets[] = {BufferOff};
		vkCmdBindVertexBuffers(CommandBuffer, 0, 1, aVertexBuffers, aOffsets);

		if(IsTextured)
		{
			auto &DescrSet = m_Textures[State.m_Texture].m_VKStandard3DTexturedDescrSet;
			vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLayout, 0, 1, &DescrSet, 0, nullptr);
		}

		SUniformTileGPosBorder VertexPushConstants;
		size_t VertexPushConstantSize = sizeof(SUniformTileGPos);
		SUniformTileGVertColor FragPushConstants;
		size_t FragPushConstantSize = sizeof(SUniformTileGVertColor);

		mem_copy(VertexPushConstants.m_aPos, m.data(), m.size() * sizeof(float));
		mem_copy(FragPushConstants.m_aColor, &Color, sizeof(FragPushConstants.m_aColor));

		if(Type == 1)
		{
			mem_copy(&VertexPushConstants.m_Dir, &Dir, sizeof(Dir));
			mem_copy(&VertexPushConstants.m_Offset, &Off, sizeof(Off));
			VertexPushConstants.m_JumpIndex = JumpIndex;
			VertexPushConstantSize = sizeof(SUniformTileGPosBorder);
		}
		else if(Type == 2)
		{
			mem_copy(&VertexPushConstants.m_Dir, &Dir, sizeof(Dir));
			mem_copy(&VertexPushConstants.m_Offset, &Off, sizeof(Off));
			VertexPushConstantSize = sizeof(SUniformTileGPosBorderLine);
		}

		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, VertexPushConstantSize, &VertexPushConstants);
		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(SUniformTileGPosBorder) + sizeof(SUniformTileGVertColorAlign), FragPushConstantSize, &FragPushConstants);

		size_t DrawCount = (size_t)IndicesDrawNum;
		for(size_t i = 0; i < DrawCount; ++i)
		{
			VkDeviceSize IndexOffset = (VkDeviceSize)((ptrdiff_t)pIndicesOffsets[i]);
			vkCmdBindIndexBuffer(CommandBuffer, m_RenderIndexBuffer, IndexOffset, VK_INDEX_TYPE_UINT32);

			vkCmdDrawIndexed(CommandBuffer, static_cast<uint32_t>(pDrawCount[i]), InstanceCount, 0, 0, 0);
		}
	}

	void Cmd_RenderTileLayer(const CCommandBuffer::SCommand_RenderTileLayer *pCommand)
	{
		int Type = 0;
		vec2 Dir{};
		vec2 Off{};
		int32_t JumpIndex = 0;
		RenderTileLayer(pCommand->m_State, (size_t)pCommand->m_BufferContainerIndex, Type, pCommand->m_Color, Dir, Off, JumpIndex, (size_t)pCommand->m_IndicesDrawNum, pCommand->m_pIndicesOffsets, pCommand->m_pDrawCount, 1);
	}

	void Cmd_RenderBorderTile(const CCommandBuffer::SCommand_RenderBorderTile *pCommand)
	{
		int Type = 1;
		vec2 Dir = {pCommand->m_Dir[0], pCommand->m_Dir[1]};
		vec2 Off = {pCommand->m_Offset[0], pCommand->m_Offset[1]};
		unsigned int DrawNum = 6;
		RenderTileLayer(pCommand->m_State, (size_t)pCommand->m_BufferContainerIndex, Type, pCommand->m_Color, Dir, Off, pCommand->m_JumpIndex, (size_t)1, &pCommand->m_pIndicesOffset, &DrawNum, pCommand->m_DrawNum);
	}

	void Cmd_RenderBorderTileLine(const CCommandBuffer::SCommand_RenderBorderTileLine *pCommand)
	{
		int Type = 2;
		vec2 Dir = {pCommand->m_Dir[0], pCommand->m_Dir[1]};
		vec2 Off = {pCommand->m_Offset[0], pCommand->m_Offset[1]};
		RenderTileLayer(pCommand->m_State, (size_t)pCommand->m_BufferContainerIndex, Type, pCommand->m_Color, Dir, Off, 0, (size_t)1, &pCommand->m_pIndicesOffset, &pCommand->m_IndexDrawNum, pCommand->m_DrawNum);
	}

	void Cmd_RenderQuadLayer(const CCommandBuffer::SCommand_RenderQuadLayer *pCommand)
	{
		NotImplemented();
	}

	void Cmd_RenderText(const CCommandBuffer::SCommand_RenderText *pCommand)
	{
		std::array<float, (size_t)4 * 2> m = {
			// column 1
			2.f / (pCommand->m_State.m_ScreenBR.x - pCommand->m_State.m_ScreenTL.x),
			0,
			// column 2
			0,
			2.f / (pCommand->m_State.m_ScreenBR.y - pCommand->m_State.m_ScreenTL.y),
			// column 3
			0,
			0,
			// column 4
			-((pCommand->m_State.m_ScreenTL.x + pCommand->m_State.m_ScreenBR.x) / (pCommand->m_State.m_ScreenBR.x - pCommand->m_State.m_ScreenTL.x)),
			-((pCommand->m_State.m_ScreenTL.y + pCommand->m_State.m_ScreenBR.y) / (pCommand->m_State.m_ScreenBR.y - pCommand->m_State.m_ScreenTL.y)),
		};

		auto &PipeLayout = m_TextPipeLineLayout;
		auto &PipeLine = m_TextPipeline;

		auto &CommandBuffer = m_CommandBuffers[m_CurImageIndex];

		if(m_LastPipeline != PipeLine)
		{
			vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLine);
			m_LastPipeline = PipeLine;
		}

		size_t ContainerIndex = (size_t)pCommand->m_BufferContainerIndex;
		auto &BufferContainer = m_BufferContainers[ContainerIndex];

		size_t BufferIndex = (size_t)BufferContainer.m_BufferObjectIndex;
		auto &Buffer = m_BufferObjects[BufferIndex];

		VkBuffer VKBuffer = Buffer.m_CurBuffer;
		size_t BufferOff = Buffer.m_CurBufferOffset;

		VkBuffer aVertexBuffers[] = {VKBuffer};
		VkDeviceSize aOffsets[] = {BufferOff};
		vkCmdBindVertexBuffers(CommandBuffer, 0, 1, aVertexBuffers, aOffsets);

		vkCmdBindIndexBuffer(CommandBuffer, m_RenderIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

		auto &TextTexture = m_Textures[pCommand->m_TextTextureIndex];
		auto &DescrSet1 = TextTexture.m_VKTextDescrSet;
		vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLayout, 0, 1, &DescrSet1, 0, nullptr);

		SUniformGTextPos PosTexSizeConstant;
		mem_copy(PosTexSizeConstant.m_aPos, m.data(), m.size() * sizeof(float));
		PosTexSizeConstant.m_TextureSize = pCommand->m_TextureSize;

		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SUniformGTextPos), &PosTexSizeConstant);

		SUniformTextFragment FragmentConstants;

		mem_copy(FragmentConstants.m_Constants.m_aTextColor, pCommand->m_aTextColor, sizeof(FragmentConstants.m_Constants.m_aTextColor));
		mem_copy(FragmentConstants.m_Constants.m_aTextOutlineColor, pCommand->m_aTextOutlineColor, sizeof(FragmentConstants.m_Constants.m_aTextOutlineColor));
		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(SUniformGTextPos) + sizeof(SUniformTextGFragmentOffset), sizeof(SUniformTextFragment), &FragmentConstants);

		vkCmdDrawIndexed(CommandBuffer, static_cast<uint32_t>(pCommand->m_DrawNum), 1, 0, 0, 0);
	}

	void Cmd_RenderQuadContainer(const CCommandBuffer::SCommand_RenderQuadContainer *pCommand)
	{
		std::array<float, (size_t)4 * 2> m;
		GetStateMatrix(pCommand->m_State, m);
		size_t AddressModeIndex = GetAddressModeIndex(pCommand->m_State);

		bool IsTextured = pCommand->m_State.m_Texture != -1;
		auto &PipeLayout = !IsTextured ? m_StandardPipeLineLayout : m_StandardTexturedPipeLineLayout;
		auto &PipeLine = !IsTextured ? m_StandardPipeline : m_StandardTexturedPipeline;

		auto &CommandBuffer = m_CommandBuffers[m_CurImageIndex];

		if(m_LastPipeline != PipeLine)
		{
			vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLine);
			m_LastPipeline = PipeLine;
		}

		size_t BufferContainerIndex = (size_t)pCommand->m_BufferContainerIndex;
		size_t BufferObjectIndex = (size_t)m_BufferContainers[BufferContainerIndex].m_BufferObjectIndex;
		auto &BufferObject = m_BufferObjects[BufferObjectIndex];

		VkBuffer VKBuffer = BufferObject.m_CurBuffer;
		size_t BufferOff = BufferObject.m_CurBufferOffset;

		VkBuffer aVertexBuffers[] = {VKBuffer};
		VkDeviceSize aOffsets[] = {BufferOff};
		vkCmdBindVertexBuffers(CommandBuffer, 0, 1, aVertexBuffers, aOffsets);

		VkDeviceSize IndexOffset = (VkDeviceSize)((ptrdiff_t)pCommand->m_pOffset);

		vkCmdBindIndexBuffer(CommandBuffer, m_RenderIndexBuffer, IndexOffset, VK_INDEX_TYPE_UINT32);

		if(IsTextured)
		{
			auto &DescrSet = m_Textures[pCommand->m_State.m_Texture].m_aVKStandardTexturedDescrSets[AddressModeIndex];
			vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLayout, 0, 1, &DescrSet, 0, nullptr);
		}

		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SUniformGPos), m.data());

		vkCmdDrawIndexed(CommandBuffer, static_cast<uint32_t>(pCommand->m_DrawNum), 1, 0, 0, 0);
	}

	void Cmd_RenderQuadContainerEx(const CCommandBuffer::SCommand_RenderQuadContainerEx *pCommand)
	{
		std::array<float, (size_t)4 * 2> m;
		GetStateMatrix(pCommand->m_State, m);
		size_t AddressModeIndex = GetAddressModeIndex(pCommand->m_State);

		bool IsTextured = pCommand->m_State.m_Texture != -1;
		bool IsRotationless = !(pCommand->m_Rotation != 0);
		auto &PipeLayout = !IsTextured ? (IsRotationless ? m_PrimExRotationlessPipeLineLayout : m_PrimExPipeLineLayout) : (IsRotationless ? m_PrimExRotationlessTexPipeLineLayout : m_PrimExTexPipeLineLayout);
		auto &PipeLine = !IsTextured ? (IsRotationless ? m_PrimExRotationlessPipeline : m_PrimExPipeline) : (IsRotationless ? m_PrimExRotationlessTexPipeline : m_PrimExTexPipeline);

		auto &CommandBuffer = m_CommandBuffers[m_CurImageIndex];

		if(m_LastPipeline != PipeLine)
		{
			vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLine);
			m_LastPipeline = PipeLine;
		}

		size_t BufferContainerIndex = (size_t)pCommand->m_BufferContainerIndex;
		size_t BufferObjectIndex = (size_t)m_BufferContainers[BufferContainerIndex].m_BufferObjectIndex;
		auto &BufferObject = m_BufferObjects[BufferObjectIndex];

		VkBuffer VKBuffer = BufferObject.m_CurBuffer;
		size_t BufferOff = BufferObject.m_CurBufferOffset;

		VkBuffer aVertexBuffers[] = {VKBuffer};
		VkDeviceSize aOffsets[] = {BufferOff};
		vkCmdBindVertexBuffers(CommandBuffer, 0, 1, aVertexBuffers, aOffsets);

		VkDeviceSize IndexOffset = (VkDeviceSize)((ptrdiff_t)pCommand->m_pOffset);

		vkCmdBindIndexBuffer(CommandBuffer, m_RenderIndexBuffer, IndexOffset, VK_INDEX_TYPE_UINT32);

		if(IsTextured)
		{
			auto &DescrSet = m_Textures[pCommand->m_State.m_Texture].m_aVKStandardTexturedDescrSets[AddressModeIndex];
			vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLayout, 0, 1, &DescrSet, 0, nullptr);
		}

		SUniformPrimExGVertColor PushConstantColor;
		SUniformPrimExGPos PushConstantVertex;
		size_t VertexPushConstantSize = sizeof(PushConstantVertex);

		mem_copy(PushConstantColor.m_aColor, &pCommand->m_VertexColor, sizeof(PushConstantColor.m_aColor));

		mem_copy(PushConstantVertex.m_aPos, m.data(), sizeof(PushConstantVertex.m_aPos));

		if(!IsRotationless)
		{
			PushConstantVertex.m_Rotation = pCommand->m_Rotation;
			PushConstantVertex.m_Center = {pCommand->m_Center.x, pCommand->m_Center.y};
		}
		else
		{
			VertexPushConstantSize = sizeof(SUniformPrimExGPosRotationless);
		}

		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, VertexPushConstantSize, &PushConstantVertex);
		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(SUniformPrimExGPos) + sizeof(SUniformPrimExGVertColorAlign), sizeof(PushConstantColor), &PushConstantColor);

		vkCmdDrawIndexed(CommandBuffer, static_cast<uint32_t>(pCommand->m_DrawNum), 1, 0, 0, 0);
	}

	void Cmd_RenderQuadContainerAsSpriteMultiple(const CCommandBuffer::SCommand_RenderQuadContainerAsSpriteMultiple *pCommand)
	{
		std::array<float, (size_t)4 * 2> m;
		GetStateMatrix(pCommand->m_State, m);
		size_t AddressModeIndex = GetAddressModeIndex(pCommand->m_State);

		dbg_assert(pCommand->m_State.m_Texture != -1, "Texture must be valid, when rendering sprites.");
		auto &PipeLayout = m_SpriteMultiPipeLineLayout;
		auto &PipeLine = m_SpriteMultiPipeline;

		auto &CommandBuffer = m_CommandBuffers[m_CurImageIndex];

		if(m_LastPipeline != PipeLine)
		{
			vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLine);
			m_LastPipeline = PipeLine;
		}

		size_t BufferContainerIndex = (size_t)pCommand->m_BufferContainerIndex;
		size_t BufferObjectIndex = (size_t)m_BufferContainers[BufferContainerIndex].m_BufferObjectIndex;
		auto &BufferObject = m_BufferObjects[BufferObjectIndex];

		VkBuffer VKBuffer = BufferObject.m_CurBuffer;
		size_t BufferOff = BufferObject.m_CurBufferOffset;

		VkBuffer aVertexBuffers[] = {VKBuffer};
		VkDeviceSize aOffsets[] = {BufferOff};
		vkCmdBindVertexBuffers(CommandBuffer, 0, 1, aVertexBuffers, aOffsets);

		VkDeviceSize IndexOffset = (VkDeviceSize)((ptrdiff_t)pCommand->m_pOffset);

		vkCmdBindIndexBuffer(CommandBuffer, m_RenderIndexBuffer, IndexOffset, VK_INDEX_TYPE_UINT32);

		auto &DescrSet = m_Textures[pCommand->m_State.m_Texture].m_aVKStandardTexturedDescrSets[AddressModeIndex];
		vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLayout, 0, 1, &DescrSet, 0, nullptr);

		SUniformSpriteMultiGVertColor PushConstantColor;
		SUniformSpriteMultiGPos PushConstantVertex;

		mem_copy(PushConstantColor.m_aColor, &pCommand->m_VertexColor, sizeof(PushConstantColor.m_aColor));

		mem_copy(PushConstantVertex.m_aPos, m.data(), sizeof(PushConstantVertex.m_aPos));
		mem_copy(&PushConstantVertex.m_Center, &pCommand->m_Center, sizeof(PushConstantVertex.m_Center));

		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantVertex), &PushConstantVertex);
		vkCmdPushConstants(CommandBuffer, PipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(SUniformSpriteMultiGPos) + sizeof(SUniformSpriteMultiGVertColorAlign), sizeof(PushConstantColor), &PushConstantColor);

		const int RSPCount = 512;
		int DrawCount = pCommand->m_DrawCount;
		size_t RenderOffset = 0;

		while(DrawCount > 0)
		{
			int UniformCount = (DrawCount > RSPCount ? RSPCount : DrawCount);

			// create uniform buffer
			VkDescriptorSet UniDescrSet;
			GetUniformBufferObject(UniDescrSet, (const float *)(pCommand->m_pRenderInfo + RenderOffset), UniformCount * sizeof(IGraphics::SRenderSpriteInfo));

			vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLayout, 1, 1, &UniDescrSet, 0, nullptr);

			vkCmdDrawIndexed(CommandBuffer, static_cast<uint32_t>(pCommand->m_DrawNum), UniformCount, 0, 0, 0);

			RenderOffset += RSPCount;
			DrawCount -= RSPCount;
		}
	}

public:
	CCommandProcessorFragment_Vulkan()
	{
		m_Textures.reserve(CCommandBuffer::MAX_TEXTURES);
	}

	~CCommandProcessorFragment_Vulkan() override = default;

	bool RunCommand(const CCommandBuffer::SCommand *pBaseCommand) override
	{
		switch(pBaseCommand->m_Cmd)
		{
		case CCommandProcessorFragment_GLBase::CMD_INIT:
			Cmd_Init(static_cast<const SCommand_Init *>(pBaseCommand));
			break;
		case CCommandProcessorFragment_GLBase::CMD_SHUTDOWN:
			Cmd_Shutdown(static_cast<const SCommand_Shutdown *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_TEXTURE_CREATE:
			Cmd_Texture_Create(static_cast<const CCommandBuffer::SCommand_Texture_Create *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_TEXTURE_DESTROY:
			Cmd_Texture_Destroy(static_cast<const CCommandBuffer::SCommand_Texture_Destroy *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_TEXT_TEXTURES_CREATE:
			Cmd_TextTextures_Create(static_cast<const CCommandBuffer::SCommand_TextTextures_Create *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_TEXT_TEXTURES_DESTROY:
			Cmd_TextTextures_Destroy(static_cast<const CCommandBuffer::SCommand_TextTextures_Destroy *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_TEXTURE_UPDATE:
			Cmd_Texture_Update(static_cast<const CCommandBuffer::SCommand_Texture_Update *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_CLEAR:
			Cmd_Clear(static_cast<const CCommandBuffer::SCommand_Clear *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RENDER:
			Cmd_Render(static_cast<const CCommandBuffer::SCommand_Render *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RENDER_TEX3D:
			Cmd_RenderTex3D(static_cast<const CCommandBuffer::SCommand_RenderTex3D *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_SCREENSHOT:
			Cmd_Screenshot(static_cast<const CCommandBuffer::SCommand_Screenshot *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_UPDATE_VIEWPORT:
			Cmd_Update_Viewport(static_cast<const CCommandBuffer::SCommand_Update_Viewport *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_VSYNC:
			Cmd_VSync(static_cast<const CCommandBuffer::SCommand_VSync *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_FINISH:
			Cmd_Finish(static_cast<const CCommandBuffer::SCommand_Finish *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_SWAP:
			Cmd_Swap(static_cast<const CCommandBuffer::SCommand_Swap *>(pBaseCommand));
			break;

		case CCommandBuffer::CMD_CREATE_BUFFER_OBJECT:
			Cmd_CreateBufferObject(static_cast<const CCommandBuffer::SCommand_CreateBufferObject *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_UPDATE_BUFFER_OBJECT:
			Cmd_UpdateBufferObject(static_cast<const CCommandBuffer::SCommand_UpdateBufferObject *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RECREATE_BUFFER_OBJECT:
			Cmd_RecreateBufferObject(static_cast<const CCommandBuffer::SCommand_RecreateBufferObject *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_COPY_BUFFER_OBJECT:
			Cmd_CopyBufferObject(static_cast<const CCommandBuffer::SCommand_CopyBufferObject *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_DELETE_BUFFER_OBJECT:
			Cmd_DeleteBufferObject(static_cast<const CCommandBuffer::SCommand_DeleteBufferObject *>(pBaseCommand));
			break;

		case CCommandBuffer::CMD_CREATE_BUFFER_CONTAINER:
			Cmd_CreateBufferContainer(static_cast<const CCommandBuffer::SCommand_CreateBufferContainer *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_UPDATE_BUFFER_CONTAINER:
			Cmd_UpdateBufferContainer(static_cast<const CCommandBuffer::SCommand_UpdateBufferContainer *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_DELETE_BUFFER_CONTAINER:
			Cmd_DeleteBufferContainer(static_cast<const CCommandBuffer::SCommand_DeleteBufferContainer *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_INDICES_REQUIRED_NUM_NOTIFY:
			Cmd_IndicesRequiredNumNotify(static_cast<const CCommandBuffer::SCommand_IndicesRequiredNumNotify *>(pBaseCommand));
			break;

		case CCommandBuffer::CMD_RENDER_TILE_LAYER:
			Cmd_RenderTileLayer(static_cast<const CCommandBuffer::SCommand_RenderTileLayer *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RENDER_BORDER_TILE:
			Cmd_RenderBorderTile(static_cast<const CCommandBuffer::SCommand_RenderBorderTile *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RENDER_BORDER_TILE_LINE:
			Cmd_RenderBorderTileLine(static_cast<const CCommandBuffer::SCommand_RenderBorderTileLine *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RENDER_QUAD_LAYER:
			Cmd_RenderQuadLayer(static_cast<const CCommandBuffer::SCommand_RenderQuadLayer *>(pBaseCommand));
			break;

		case CCommandBuffer::CMD_RENDER_TEXT:
			Cmd_RenderText(static_cast<const CCommandBuffer::SCommand_RenderText *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RENDER_QUAD_CONTAINER:
			Cmd_RenderQuadContainer(static_cast<const CCommandBuffer::SCommand_RenderQuadContainer *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RENDER_QUAD_CONTAINER_EX:
			Cmd_RenderQuadContainerEx(static_cast<const CCommandBuffer::SCommand_RenderQuadContainerEx *>(pBaseCommand));
			break;
		case CCommandBuffer::CMD_RENDER_QUAD_CONTAINER_SPRITE_MULTIPLE:
			Cmd_RenderQuadContainerAsSpriteMultiple(static_cast<const CCommandBuffer::SCommand_RenderQuadContainerAsSpriteMultiple *>(pBaseCommand));
			break;

		default:
			return false;
		}

		return true;
	}

	/************************
	* VULKAN SETUP CODE
	************************/

	bool GetVulkanExtensions(SDL_Window *pWindow, std::vector<std::string> &VKExtensions)
	{
		unsigned int ExtCount = 0;
		if(!SDL_Vulkan_GetInstanceExtensions(pWindow, &ExtCount, nullptr))
		{
			SetError("Could not get instance extensions from SDL.");
			return false;
		}

		std::vector<const char *> ExtensionList(ExtCount);
		if(!SDL_Vulkan_GetInstanceExtensions(pWindow, &ExtCount, ExtensionList.data()))
		{
			SetError("Could not get instance extensions from SDL.");
			return false;
		}

		for(uint32_t i = 0; i < ExtCount; i++)
		{
			VKExtensions.emplace_back(ExtensionList[i]);
		}

		if(g_Config.m_DbgGfx >= 1)
		{
			// debug message support
			VKExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return true;
	}

	std::set<std::string> OurVKLayers()
	{
		std::set<std::string> OurLayers;

		if(g_Config.m_DbgGfx >= 1)
		{
			OurLayers.emplace("VK_LAYER_KHRONOS_validation");
			// deprecated, but VK_LAYER_KHRONOS_validation was released after vulkan 1.1
			OurLayers.emplace("VK_LAYER_LUNARG_standard_validation");
		}

		return OurLayers;
	}

	std::set<std::string> OurDeviceExtensions()
	{
		std::set<std::string> OurExt;
		OurExt.emplace(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		return OurExt;
	}

	std::vector<VkImageUsageFlags> OurImageUsages()
	{
		std::vector<VkImageUsageFlags> ImgUsages;

		ImgUsages.emplace_back(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

		return ImgUsages;
	}

	bool GetVulkanLayers(std::vector<std::string> &VKLayers)
	{
		uint32_t LayerCount = 0;
		VkResult res = vkEnumerateInstanceLayerProperties(&LayerCount, NULL);
		if(res != VK_SUCCESS)
		{
			SetError("Could not get vulkan layers.");
			return false;
		}

		std::vector<VkLayerProperties> VKInstanceLayers(LayerCount);
		res = vkEnumerateInstanceLayerProperties(&LayerCount, VKInstanceLayers.data());
		if(res != VK_SUCCESS)
		{
			SetError("Could not get vulkan layers.");
			return false;
		}

		std::set<std::string> ReqLayerNames = OurVKLayers();
		VKLayers.clear();
		for(const auto &LayerName : VKInstanceLayers)
		{
			auto it = ReqLayerNames.find(std::string(LayerName.layerName));
			if(it != ReqLayerNames.end())
			{
				VKLayers.emplace_back(LayerName.layerName);
			}
		}

		return true;
	}

	bool CreateVulkanInstance(const std::vector<std::string> &VKLayers, const std::vector<std::string> &VKExtensions)
	{
		std::vector<const char *> LayersCStr;
		LayersCStr.reserve(VKLayers.size());
		for(const auto &Layer : VKLayers)
			LayersCStr.emplace_back(Layer.c_str());

		std::vector<const char *> ExtCStr;
		ExtCStr.reserve(VKExtensions.size());
		for(const auto &Ext : VKExtensions)
			ExtCStr.emplace_back(Ext.c_str());

		uint32_t VKAPIVersion;
		if(vkEnumerateInstanceVersion(&VKAPIVersion) != VK_SUCCESS)
		{
			SetError("Could not get vulkan version. This is usually an indicator of a failed vulkan extension initialization.");
			return false;
		}

		VkApplicationInfo VKAppInfo = {};
		VKAppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		VKAppInfo.pNext = NULL;
		VKAppInfo.pApplicationName = "DDNet";
		VKAppInfo.applicationVersion = 1;
		VKAppInfo.pEngineName = "DDNet-Vulkan";
		VKAppInfo.engineVersion = 1;
		VKAppInfo.apiVersion = VK_API_VERSION_1_1;

		VkValidationFeaturesEXT *pExt = nullptr;
		VkValidationFeaturesEXT Features = {};
		if(g_Config.m_DbgGfx >= 2)
		{
			VkValidationFeatureEnableEXT aEnables[] = {VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
			Features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
			Features.enabledValidationFeatureCount = 1;
			Features.pEnabledValidationFeatures = aEnables;

			pExt = &Features;
		}

		VkInstanceCreateInfo VKInstanceInfo = {};
		VKInstanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		VKInstanceInfo.pNext = pExt;
		VKInstanceInfo.flags = 0;
		VKInstanceInfo.pApplicationInfo = &VKAppInfo;
		VKInstanceInfo.enabledExtensionCount = static_cast<uint32_t>(ExtCStr.size());
		VKInstanceInfo.ppEnabledExtensionNames = ExtCStr.data();
		VKInstanceInfo.enabledLayerCount = static_cast<uint32_t>(LayersCStr.size());
		VKInstanceInfo.ppEnabledLayerNames = LayersCStr.data();

		VkResult res = vkCreateInstance(&VKInstanceInfo, NULL, &m_VKInstance);
		switch(res)
		{
		case VK_SUCCESS:
			break;
		case VK_ERROR_INCOMPATIBLE_DRIVER:
			SetError("No compatible driver found. Vulkan 1.1 is required.");
			return false;
		default:
			SetError("Non handled unknown error.");
			return false;
		}
		return true;
	}

	bool SelectGPU(char *pRendererName, char *pVendorName, char *pVersionName)
	{
		uint32_t DevicesCount = 0;
		vkEnumeratePhysicalDevices(m_VKInstance, &DevicesCount, nullptr);
		if(DevicesCount == 0)
		{
			SetError("No vulkan compatible devices found.");
			return false;
		}

		std::vector<VkPhysicalDevice> DeviceList(DevicesCount);
		vkEnumeratePhysicalDevices(m_VKInstance, &DevicesCount, DeviceList.data());

		size_t Index = 0;
		std::vector<VkPhysicalDeviceProperties> DevicePropList(DeviceList.size());
		for(auto &CurDevice : DeviceList)
		{
			vkGetPhysicalDeviceProperties(CurDevice, &(DevicePropList[Index]));

			auto &DeviceProp = DevicePropList[Index];

			Index++;
			if(DeviceProp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			{
				str_copy(pRendererName, DeviceProp.deviceName, gs_GPUInfoStringSize);
				const char *pVendorNameStr = NULL;
				switch(DeviceProp.vendorID)
				{
				case 0x1002:
					pVendorNameStr = "AMD";
					break;
				case 0x1010:
					pVendorNameStr = "ImgTec";
					break;
				case 0x10DE:
					pVendorNameStr = "NVIDIA";
					break;
				case 0x13B5:
					pVendorNameStr = "ARM";
					break;
				case 0x5143:
					pVendorNameStr = "Qualcomm";
					break;
				case 0x8086:
					pVendorNameStr = "INTEL";
					break;
				default:
					pVendorNameStr = "unknown";
					break;
				}
				str_copy(pVendorName, pVendorNameStr, gs_GPUInfoStringSize);
				str_format(pVersionName, gs_GPUInfoStringSize, "Vulkan %d.%d.%d", (int)VK_VERSION_MAJOR(DeviceProp.apiVersion), (int)VK_VERSION_MINOR(DeviceProp.apiVersion), (int)VK_VERSION_PATCH(DeviceProp.apiVersion));

				// get important device limits
				m_NonCoherentMemAlignment = DeviceProp.limits.nonCoherentAtomSize;

				break;
			}
		}

		VkPhysicalDevice CurDevice = DeviceList[Index - 1];

		uint32_t FamQueueCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(CurDevice, &FamQueueCount, nullptr);
		if(FamQueueCount == 0)
		{
			SetError("No vulkan queue family properties found.");
			return false;
		}

		std::vector<VkQueueFamilyProperties> QueuePropList(FamQueueCount);
		vkGetPhysicalDeviceQueueFamilyProperties(CurDevice, &FamQueueCount, QueuePropList.data());

		uint32_t QueueNodeIndex = std::numeric_limits<uint32_t>::max();
		for(uint32_t i = 0; i < FamQueueCount; i++)
		{
			if(QueuePropList[i].queueCount > 0 && (QueuePropList[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				QueueNodeIndex = i;
			}
			/*if(QueuePropList[i].queueCount > 0 && (QueuePropList[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
			{
				QueueNodeIndex = i;
			}*/
		}

		if(QueueNodeIndex == std::numeric_limits<uint32_t>::max())
		{
			SetError("No vulkan queue found that matches the requirements: graphics queue");
			return false;
		}

		m_VKGPU = CurDevice;
		m_VKGraphicsQueueIndex = QueueNodeIndex;
		return true;
	}

	bool CreateLogicalDevice(const std::vector<std::string> &VKLayers)
	{
		std::vector<const char *> LayerCNames;
		LayerCNames.reserve(VKLayers.size());
		for(const auto &Layer : VKLayers)
			LayerCNames.emplace_back(Layer.c_str());

		uint32_t DevPropCount = 0;
		if(vkEnumerateDeviceExtensionProperties(m_VKGPU, NULL, &DevPropCount, NULL) != VK_SUCCESS)
		{
			SetError("Querying logical device extension propterties failed.");
			return false;
		}

		std::vector<VkExtensionProperties> DevPropList(DevPropCount);
		if(vkEnumerateDeviceExtensionProperties(m_VKGPU, NULL, &DevPropCount, DevPropList.data()) != VK_SUCCESS)
		{
			SetError("Querying logical device extension propterties failed.");
			return false;
		}

		std::vector<const char *> DevPropCNames;
		std::set<std::string> OurDevExt = OurDeviceExtensions();

		for(const auto &CurExtProp : DevPropList)
		{
			auto it = OurDevExt.find(std::string(CurExtProp.extensionName));
			if(it != OurDevExt.end())
			{
				DevPropCNames.emplace_back(CurExtProp.extensionName);
			}
		}

		VkDeviceQueueCreateInfo VKQueueCreateInfo;
		VKQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		VKQueueCreateInfo.queueFamilyIndex = m_VKGraphicsQueueIndex;
		VKQueueCreateInfo.queueCount = 1;
		std::vector<float> queue_prio = {1.0f};
		VKQueueCreateInfo.pQueuePriorities = queue_prio.data();
		VKQueueCreateInfo.pNext = NULL;
		VKQueueCreateInfo.flags = 0;

		VkDeviceCreateInfo VKCreateInfo;
		VKCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		VKCreateInfo.queueCreateInfoCount = 1;
		VKCreateInfo.pQueueCreateInfos = &VKQueueCreateInfo;
		VKCreateInfo.ppEnabledLayerNames = LayerCNames.data();
		VKCreateInfo.enabledLayerCount = static_cast<uint32_t>(LayerCNames.size());
		VKCreateInfo.ppEnabledExtensionNames = DevPropCNames.data();
		VKCreateInfo.enabledExtensionCount = static_cast<uint32_t>(DevPropCNames.size());
		VKCreateInfo.pNext = NULL;
		VKCreateInfo.pEnabledFeatures = NULL;
		VKCreateInfo.flags = 0;

		VkResult res = vkCreateDevice(m_VKGPU, &VKCreateInfo, nullptr, &m_VKDevice);
		if(res != VK_SUCCESS)
		{
			SetError("Logical device could not be created.");
			return false;
		}

		return true;
	}

	bool CreateSurface(SDL_Window *pWindow)
	{
		if(!SDL_Vulkan_CreateSurface(pWindow, m_VKInstance, &m_VKPresentSurface))
		{
			SetError("Creating a vulkan surface for the SDL window failed.");
			return false;
		}

		VkBool32 IsSupported = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(m_VKGPU, m_VKGraphicsQueueIndex, m_VKPresentSurface, &IsSupported);
		if(!IsSupported)
		{
			SetError("The device surface does not support presenting the framebuffer to a screen. (maybe the wrong GPU was selected?)");
			return false;
		}

		return true;
	}

	bool GetPresentationMode(VkPresentModeKHR &VKIOMode)
	{
		uint32_t PresentModeCount = 0;
		if(vkGetPhysicalDeviceSurfacePresentModesKHR(m_VKGPU, m_VKPresentSurface, &PresentModeCount, NULL) != VK_SUCCESS)
		{
			SetError("The device surface presentation modes could not be fetched.");
			return false;
		}

		std::vector<VkPresentModeKHR> PresentModeList(PresentModeCount);
		if(vkGetPhysicalDeviceSurfacePresentModesKHR(m_VKGPU, m_VKPresentSurface, &PresentModeCount, PresentModeList.data()) != VK_SUCCESS)
		{
			SetError("The device surface presentation modes could not be fetched.");
			return false;
		}

		for(auto &Mode : PresentModeList)
		{
			if(Mode == VKIOMode)
				return true;
		}

		VKIOMode = g_Config.m_GfxVsync ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_IMMEDIATE_KHR;
		return true;
	}

	bool GetSurfaceProperties(VkSurfaceCapabilitiesKHR &VKSurfCapabilities)
	{
		if(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_VKGPU, m_VKPresentSurface, &VKSurfCapabilities) != VK_SUCCESS)
		{
			SetError("The device surface capabilities could not be fetched.");
			return false;
		}
		return true;
	}

	uint32_t GetNumberOfSwapImages(const VkSurfaceCapabilitiesKHR &VKCapabilities)
	{
		uint32_t ImgNumber = VKCapabilities.minImageCount + 1;
		return (VKCapabilities.maxImageCount > 0 && ImgNumber > VKCapabilities.maxImageCount) ? VKCapabilities.maxImageCount : ImgNumber;
	}

	VkExtent2D GetSwapImageSize(const VkSurfaceCapabilitiesKHR &VKCapabilities)
	{
		VkExtent2D RetSize = {m_CanvasWidth, m_CanvasHeight};

		if(VKCapabilities.currentExtent.width == std::numeric_limits<uint32_t>::max())
		{
			RetSize.width = clamp<uint32_t>(RetSize.width, VKCapabilities.minImageExtent.width, VKCapabilities.maxImageExtent.width);
			RetSize.height = clamp<uint32_t>(RetSize.height, VKCapabilities.maxImageExtent.height, VKCapabilities.maxImageExtent.height);
		}
		else
		{
			RetSize = VKCapabilities.currentExtent;
		}
		return RetSize;
	}

	bool GetImageUsage(const VkSurfaceCapabilitiesKHR &VKCapabilities, VkImageUsageFlags &VKOutUsage)
	{
		std::vector<VkImageUsageFlags> OurImgUsages = OurImageUsages();
		if(OurImgUsages.empty())
		{
			SetError("Framebuffer image attachment types not supported.");
			return false;
		}

		VKOutUsage = OurImgUsages[0];

		for(const auto &ImgUsage : OurImgUsages)
		{
			VkImageUsageFlags ImgUsageFlags = ImgUsage & VKCapabilities.supportedUsageFlags;
			if(ImgUsageFlags != ImgUsage)
				return false;

			VKOutUsage = (VKOutUsage | ImgUsage);
		}

		return true;
	}

	VkSurfaceTransformFlagBitsKHR GetTransform(const VkSurfaceCapabilitiesKHR &VKCapabilities)
	{
		if(VKCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
			return VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
		return VKCapabilities.currentTransform;
	}

	bool GetFormat()
	{
		uint32_t SurfFormats = 0;
		if(vkGetPhysicalDeviceSurfaceFormatsKHR(m_VKGPU, m_VKPresentSurface, &SurfFormats, nullptr) != VK_SUCCESS)
		{
			SetError("The device surface format fetching failed.");
			return false;
		}

		std::vector<VkSurfaceFormatKHR> SurfFormatList(SurfFormats);
		if(vkGetPhysicalDeviceSurfaceFormatsKHR(m_VKGPU, m_VKPresentSurface, &SurfFormats, SurfFormatList.data()) != VK_SUCCESS)
		{
			SetError("The device surface format fetching failed.");
			return false;
		}

		if(SurfFormatList.size() == 1 && SurfFormatList[0].format == VK_FORMAT_UNDEFINED)
		{
			m_VKSurfFormat.format = VK_FORMAT_B8G8R8A8_UNORM;
			m_VKSurfFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
			dbg_msg("vulkan", "warning: surface format was undefined. This can potentially cause bugs.");
			return true;
		}

		for(const auto &FindFormat : SurfFormatList)
		{
			if(FindFormat.format == VK_FORMAT_B8G8R8A8_UNORM && FindFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				m_VKSurfFormat = FindFormat;
				return true;
			}
			else if(FindFormat.format == VK_FORMAT_R8G8B8A8_UNORM && FindFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				m_VKSurfFormat = FindFormat;
				return true;
			}
		}

		dbg_msg("vulkan", "warning: surface format was not RGBA(or variants of it). This can potentially cause weird looking images(too bright etc.).");
		m_VKSurfFormat = SurfFormatList[0];
		return true;
	}

	bool CreateSwapChain()
	{
		VkSurfaceCapabilitiesKHR VKSurfCap;
		if(!GetSurfaceProperties(VKSurfCap))
			return false;

		VkPresentModeKHR PresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
		if(!GetPresentationMode(PresentMode))
			return false;

		uint32_t SwapImgCount = GetNumberOfSwapImages(VKSurfCap);
		m_MaxFramesChain = SwapImgCount;

		m_VKSwapImgExtent = GetSwapImageSize(VKSurfCap);

		VkImageUsageFlags UsageFlags;
		if(!GetImageUsage(VKSurfCap, UsageFlags))
			return false;

		VkSurfaceTransformFlagBitsKHR TransformFlagBits = GetTransform(VKSurfCap);

		if(!GetFormat())
			return false;

		VkSwapchainKHR OldSwapChain = m_VKSwapChain;

		VkSwapchainCreateInfoKHR SwapInfo;
		SwapInfo.pNext = nullptr;
		SwapInfo.flags = 0;
		SwapInfo.surface = m_VKPresentSurface;
		SwapInfo.minImageCount = SwapImgCount;
		SwapInfo.imageFormat = m_VKSurfFormat.format;
		SwapInfo.imageColorSpace = m_VKSurfFormat.colorSpace;
		SwapInfo.imageExtent = m_VKSwapImgExtent;
		SwapInfo.imageArrayLayers = 1;
		SwapInfo.imageUsage = UsageFlags;
		SwapInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		SwapInfo.queueFamilyIndexCount = 0;
		SwapInfo.pQueueFamilyIndices = nullptr;
		SwapInfo.preTransform = TransformFlagBits;
		SwapInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		SwapInfo.presentMode = PresentMode;
		SwapInfo.clipped = true;
		SwapInfo.oldSwapchain = VK_NULL_HANDLE;
		SwapInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;

		if(OldSwapChain != VK_NULL_HANDLE)
		{
			vkDestroySwapchainKHR(m_VKDevice, OldSwapChain, nullptr);
			OldSwapChain = VK_NULL_HANDLE;
		}

		if(vkCreateSwapchainKHR(m_VKDevice, &SwapInfo, nullptr, &OldSwapChain) != VK_SUCCESS)
		{
			SetError("Creating the swap chain failed.");
			return false;
		}

		m_VKSwapChain = OldSwapChain;
		return true;
	}

	bool GetSwapChainImageHandles()
	{
		uint32_t ImgCount = 0;
		VkResult res = vkGetSwapchainImagesKHR(m_VKDevice, m_VKSwapChain, &ImgCount, nullptr);
		if(res != VK_SUCCESS)
		{
			SetError("Could not get swap chain images.");
			return false;
		}

		m_VKChainImages.clear();
		m_VKChainImages.resize(ImgCount);
		if(vkGetSwapchainImagesKHR(m_VKDevice, m_VKSwapChain, &ImgCount, m_VKChainImages.data()) != VK_SUCCESS)
		{
			SetError("Could not get swap chain images.");
			return false;
		}

		return true;
	}

	void GetDeviceQueue()
	{
		vkGetDeviceQueue(m_VKDevice, m_VKGraphicsQueueIndex, 0, &m_VKGraphicsQueue);
		vkGetDeviceQueue(m_VKDevice, m_VKGraphicsQueueIndex, 0, &m_VKPresentQueue);
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL VKDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
	{
		dbg_msg("vulkan_debug", "%s", pCallbackData->pMessage);

		return VK_FALSE;
	}

	VkResult CreateDebugUtilsMessengerEXT(const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pDebugMessenger)
	{
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_VKInstance, "vkCreateDebugUtilsMessengerEXT");
		if(func != nullptr)
		{
			return func(m_VKInstance, pCreateInfo, pAllocator, pDebugMessenger);
		}
		else
		{
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void SetupDebugCallback()
	{
		bool FoundDebugCB = false;

		if(!FoundDebugCB)
		{
			VkDebugUtilsMessengerEXT DebugMessenger;

			VkDebugUtilsMessengerCreateInfoEXT CreateInfo = {};
			CreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			CreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			CreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT; // | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT <- too annoying
			CreateInfo.pfnUserCallback = VKDebugCallback;

			if(CreateDebugUtilsMessengerEXT(&CreateInfo, nullptr, &DebugMessenger) != VK_SUCCESS)
			{
				dbg_msg("vulkan", "no debug layers present.");
			}
			else
			{
				dbg_msg("vulkan", "Enabled vulkan debug context.");
			}
		}
	}

	bool CreateImageViews()
	{
		m_VKSwapChainImageViewList.resize(m_VKChainImages.size());

		for(size_t i = 0; i < m_VKChainImages.size(); i++)
		{
			VkImageViewCreateInfo CreateInfo{};
			CreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			CreateInfo.image = m_VKChainImages[i];
			CreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			CreateInfo.format = m_VKSurfFormat.format;
			CreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			CreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			CreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			CreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			CreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			CreateInfo.subresourceRange.baseMipLevel = 0;
			CreateInfo.subresourceRange.levelCount = 1;
			CreateInfo.subresourceRange.baseArrayLayer = 0;
			CreateInfo.subresourceRange.layerCount = 1;

			if(vkCreateImageView(m_VKDevice, &CreateInfo, nullptr, &m_VKSwapChainImageViewList[i]) != VK_SUCCESS)
			{
				SetError("Could not create image views for the swap chain framebuffers.");
				return false;
			}
		}

		return true;
	}

	bool CreateRenderPass(bool ClearAttachs)
	{
		VkAttachmentDescription ColorAttachment{};
		ColorAttachment.format = m_VKSurfFormat.format;
		ColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		ColorAttachment.loadOp = ClearAttachs ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		ColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		ColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		ColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		ColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		ColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference ColorAttachmentRef{};
		ColorAttachmentRef.attachment = 0;
		ColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription Subpass{};
		Subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		Subpass.colorAttachmentCount = 1;
		Subpass.pColorAttachments = &ColorAttachmentRef;

		VkSubpassDependency Dependency{};
		Dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		Dependency.dstSubpass = 0;
		Dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		Dependency.srcAccessMask = 0;
		Dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		Dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo CreateRenderPassInfo{};
		CreateRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		CreateRenderPassInfo.attachmentCount = 1;
		CreateRenderPassInfo.pAttachments = &ColorAttachment;
		CreateRenderPassInfo.subpassCount = 1;
		CreateRenderPassInfo.pSubpasses = &Subpass;
		CreateRenderPassInfo.dependencyCount = 1;
		CreateRenderPassInfo.pDependencies = &Dependency;

		if(vkCreateRenderPass(m_VKDevice, &CreateRenderPassInfo, nullptr, &m_VKRenderPass) != VK_SUCCESS)
		{
			SetError("Creating the render pass failed.");
			return false;
		}

		return true;
	}

	bool CreateFramebuffers()
	{
		m_VKFramebufferList.resize(m_VKSwapChainImageViewList.size());

		for(size_t i = 0; i < m_VKSwapChainImageViewList.size(); i++)
		{
			VkImageView aAttachments[] = {m_VKSwapChainImageViewList[i]};

			VkFramebufferCreateInfo FramebufferInfo{};
			FramebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			FramebufferInfo.renderPass = m_VKRenderPass;
			FramebufferInfo.attachmentCount = 1;
			FramebufferInfo.pAttachments = aAttachments;
			FramebufferInfo.width = m_VKSwapImgExtent.width;
			FramebufferInfo.height = m_VKSwapImgExtent.height;
			FramebufferInfo.layers = 1;

			if(vkCreateFramebuffer(m_VKDevice, &FramebufferInfo, nullptr, &m_VKFramebufferList[i]) != VK_SUCCESS)
			{
				SetError("Creating the framebuffers failed.");
				return false;
			}
		}

		return true;
	}

	bool CreateShaderModule(const std::vector<uint8_t> &Code, VkShaderModule &ShaderModule)
	{
		VkShaderModuleCreateInfo CreateInfo{};
		CreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		CreateInfo.codeSize = Code.size();
		CreateInfo.pCode = (const uint32_t *)(Code.data());

		if(vkCreateShaderModule(m_VKDevice, &CreateInfo, nullptr, &ShaderModule) != VK_SUCCESS)
		{
			SetError("Shader module was not created.");
			return false;
		}

		return true;
	}

	bool CreateDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding SamplerLayoutBinding{};
		SamplerLayoutBinding.binding = 0;
		SamplerLayoutBinding.descriptorCount = 1;
		SamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		SamplerLayoutBinding.pImmutableSamplers = nullptr;
		SamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding aBindings[] = {SamplerLayoutBinding};
		VkDescriptorSetLayoutCreateInfo LayoutInfo{};
		LayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		LayoutInfo.bindingCount = (sizeof(aBindings) / sizeof(aBindings[0]));
		LayoutInfo.pBindings = aBindings;

		if(vkCreateDescriptorSetLayout(m_VKDevice, &LayoutInfo, nullptr, &m_StandardTexturedDescriptorSetLayout) != VK_SUCCESS)
		{
			SetError("Creating descriptor layout failed.");
			return false;
		}

		if(vkCreateDescriptorSetLayout(m_VKDevice, &LayoutInfo, nullptr, &m_Standard3DTexturedDescriptorSetLayout) != VK_SUCCESS)
		{
			SetError("Creating descriptor layout failed.");
			return false;
		}
		return true;
	}

	struct SShaderModule
	{
		VkShaderModule m_VertShaderModule;
		VkShaderModule m_FragShaderModule;

		VkDevice m_VKDevice;

		~SShaderModule()
		{
			vkDestroyShaderModule(m_VKDevice, m_VertShaderModule, nullptr);
			vkDestroyShaderModule(m_VKDevice, m_FragShaderModule, nullptr);
		}
	};

	bool CreateShaders(const char *pVertName, const char *pFragName, VkPipelineShaderStageCreateInfo (&aShaderStages)[2], SShaderModule &ShaderModule)
	{
		auto *pVertShaderCodeFile = m_pStorage->OpenFile(pVertName, IOFLAG_READ, IStorage::TYPE_ALL);
		auto *pFragShaderCodeFile = m_pStorage->OpenFile(pFragName, IOFLAG_READ, IStorage::TYPE_ALL);

		bool ShaderLoaded = true;

		std::vector<uint8_t> VertBuff;
		std::vector<uint8_t> FragBuff;
		if(pVertShaderCodeFile)
		{
			long FileSize = io_length(pVertShaderCodeFile);
			VertBuff.resize(FileSize);
			io_read(pVertShaderCodeFile, VertBuff.data(), FileSize);
			io_close(pVertShaderCodeFile);
		}
		else
			ShaderLoaded = false;

		if(pFragShaderCodeFile)
		{
			long FileSize = io_length(pFragShaderCodeFile);
			FragBuff.resize(FileSize);
			io_read(pFragShaderCodeFile, FragBuff.data(), FileSize);
			io_close(pFragShaderCodeFile);
		}
		else
			ShaderLoaded = false;

		if(!ShaderLoaded)
		{
			SetError("A shader file could not load correctly");
			return false;
		}

		if(!CreateShaderModule(VertBuff, ShaderModule.m_VertShaderModule))
			return false;

		if(!CreateShaderModule(FragBuff, ShaderModule.m_FragShaderModule))
			return false;

		VkPipelineShaderStageCreateInfo &VertShaderStageInfo = aShaderStages[0];
		VertShaderStageInfo = {};
		VertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		VertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		VertShaderStageInfo.module = ShaderModule.m_VertShaderModule;
		VertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo &FragShaderStageInfo = aShaderStages[1];
		FragShaderStageInfo = {};
		FragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		FragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		FragShaderStageInfo.module = ShaderModule.m_FragShaderModule;
		FragShaderStageInfo.pName = "main";

		ShaderModule.m_VKDevice = m_VKDevice;
		return true;
	}

	bool GetStandardPipelineInfo(VkPipelineInputAssemblyStateCreateInfo &InputAssembly,
		VkViewport &Viewport,
		VkRect2D &Scissor,
		VkPipelineViewportStateCreateInfo &ViewportState,
		VkPipelineRasterizationStateCreateInfo &Rasterizer,
		VkPipelineMultisampleStateCreateInfo &Multisampling,
		VkPipelineColorBlendAttachmentState &ColorBlendAttachment,
		VkPipelineColorBlendStateCreateInfo &ColorBlending)
	{
		InputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		InputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		InputAssembly.primitiveRestartEnable = VK_FALSE;

		Viewport.x = 0.0f;
		Viewport.y = 0.0f;
		Viewport.width = (float)m_VKSwapImgExtent.width;
		Viewport.height = (float)m_VKSwapImgExtent.height;
		Viewport.minDepth = 0.0f;
		Viewport.maxDepth = 1.0f;

		Scissor.offset = {0, 0};
		Scissor.extent = m_VKSwapImgExtent;

		ViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		ViewportState.viewportCount = 1;
		ViewportState.pViewports = &Viewport;
		ViewportState.scissorCount = 1;
		ViewportState.pScissors = &Scissor;

		Rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		Rasterizer.depthClampEnable = VK_FALSE;
		Rasterizer.rasterizerDiscardEnable = VK_FALSE;
		Rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		Rasterizer.lineWidth = 1.0f;
		Rasterizer.cullMode = VK_CULL_MODE_NONE;
		Rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		Rasterizer.depthBiasEnable = VK_FALSE;

		Multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		Multisampling.sampleShadingEnable = VK_FALSE;
		Multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		ColorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		ColorBlendAttachment.blendEnable = VK_TRUE;

		ColorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		ColorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		ColorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		ColorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		ColorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		ColorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		ColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		ColorBlending.logicOpEnable = VK_FALSE;
		ColorBlending.logicOp = VK_LOGIC_OP_COPY;
		ColorBlending.attachmentCount = 1;
		ColorBlending.pAttachments = &ColorBlendAttachment;
		ColorBlending.blendConstants[0] = 0.0f;
		ColorBlending.blendConstants[1] = 0.0f;
		ColorBlending.blendConstants[2] = 0.0f;
		ColorBlending.blendConstants[3] = 0.0f;

		return true;
	}

	bool CreateStandardGraphicsPipeline(const char *pVertName, const char *pFragName, bool HasSampler, bool IsLinePipe)
	{
		VkPipelineShaderStageCreateInfo aShaderStages[2];
		SShaderModule Module;
		if(!CreateShaders(pVertName, pFragName, aShaderStages, Module))
			return false;

		VkPipelineVertexInputStateCreateInfo VertexInputInfo{};
		VertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		VkVertexInputBindingDescription BindingDescription{};
		BindingDescription.binding = 0;
		BindingDescription.stride = sizeof(float) * 2 * 2 + sizeof(uint8_t) * 4;
		BindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 3> aAttributeDescriptions = {};

		aAttributeDescriptions[0].binding = 0;
		aAttributeDescriptions[0].location = 0;
		aAttributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[0].offset = 0;

		aAttributeDescriptions[1].binding = 0;
		aAttributeDescriptions[1].location = 1;
		aAttributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[1].offset = sizeof(float) * 2;

		aAttributeDescriptions[2].binding = 0;
		aAttributeDescriptions[2].location = 2;
		aAttributeDescriptions[2].format = VK_FORMAT_R8G8B8A8_UNORM;
		aAttributeDescriptions[2].offset = sizeof(float) * 2 * 2;

		VertexInputInfo.vertexBindingDescriptionCount = 1;
		VertexInputInfo.vertexAttributeDescriptionCount = aAttributeDescriptions.size();
		VertexInputInfo.pVertexBindingDescriptions = &BindingDescription;
		VertexInputInfo.pVertexAttributeDescriptions = aAttributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo InputAssembly{};
		VkViewport Viewport{};
		VkRect2D Scissor{};
		VkPipelineViewportStateCreateInfo ViewportState{};
		VkPipelineRasterizationStateCreateInfo Rasterizer{};
		VkPipelineMultisampleStateCreateInfo Multisampling{};
		VkPipelineColorBlendAttachmentState ColorBlendAttachment{};
		VkPipelineColorBlendStateCreateInfo ColorBlending{};

		GetStandardPipelineInfo(InputAssembly, Viewport, Scissor, ViewportState, Rasterizer, Multisampling, ColorBlendAttachment, ColorBlending);
		InputAssembly.topology = IsLinePipe ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkPipelineLayoutCreateInfo PipelineLayoutInfo{};
		PipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		PipelineLayoutInfo.setLayoutCount = HasSampler ? 1 : 0;
		PipelineLayoutInfo.pSetLayouts = HasSampler ? &m_StandardTexturedDescriptorSetLayout : nullptr;

		std::array<VkPushConstantRange, 1> aPushConstants{};
		aPushConstants[0].offset = 0;
		aPushConstants[0].size = sizeof(SUniformGPos);
		aPushConstants[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		PipelineLayoutInfo.pushConstantRangeCount = 1;
		PipelineLayoutInfo.pPushConstantRanges = aPushConstants.data();

		VkPipelineLayout *pPipeLayout = &m_StandardPipeLineLayout;
		VkPipelineLayout *pPipeLayoutTex = &m_StandardTexturedPipeLineLayout;
		if(IsLinePipe)
		{
			pPipeLayout = &m_StandardLinePipeLineLayout;
		}

		VkPipeline *pPipe = &m_StandardPipeline;
		VkPipeline *pPipeTex = &m_StandardTexturedPipeline;
		if(IsLinePipe)
		{
			pPipe = &m_StandardLinePipeline;
		}

		if(vkCreatePipelineLayout(m_VKDevice, &PipelineLayoutInfo, nullptr, (HasSampler ? pPipeLayoutTex : pPipeLayout)) != VK_SUCCESS)
		{
			SetError("Creating pipeline layout failed.");
			return false;
		}

		VkGraphicsPipelineCreateInfo PipelineInfo{};
		PipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		PipelineInfo.stageCount = 2;
		PipelineInfo.pStages = aShaderStages;
		PipelineInfo.pVertexInputState = &VertexInputInfo;
		PipelineInfo.pInputAssemblyState = &InputAssembly;
		PipelineInfo.pViewportState = &ViewportState;
		PipelineInfo.pRasterizationState = &Rasterizer;
		PipelineInfo.pMultisampleState = &Multisampling;
		PipelineInfo.pColorBlendState = &ColorBlending;
		PipelineInfo.layout = *(HasSampler ? pPipeLayoutTex : pPipeLayout);
		PipelineInfo.renderPass = m_VKRenderPass;
		PipelineInfo.subpass = 0;
		PipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if(vkCreateGraphicsPipelines(m_VKDevice, VK_NULL_HANDLE, 1, &PipelineInfo, nullptr, (HasSampler ? pPipeTex : pPipe)) != VK_SUCCESS)
		{
			SetError("Creating the graphic pipeline failed.");
			return false;
		}

		return true;
	}

	bool CreateTextDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding SamplerLayoutBinding{};
		SamplerLayoutBinding.binding = 0;
		SamplerLayoutBinding.descriptorCount = 1;
		SamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		SamplerLayoutBinding.pImmutableSamplers = nullptr;
		SamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		auto SamplerLayoutBinding2 = SamplerLayoutBinding;
		SamplerLayoutBinding2.binding = 1;

		VkDescriptorSetLayoutBinding aBindings[] = {SamplerLayoutBinding, SamplerLayoutBinding2};
		VkDescriptorSetLayoutCreateInfo LayoutInfo{};
		LayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		LayoutInfo.bindingCount = sizeof(aBindings) / sizeof(aBindings[0]);
		LayoutInfo.pBindings = aBindings;

		if(vkCreateDescriptorSetLayout(m_VKDevice, &LayoutInfo, nullptr, &m_TextDescriptorSetLayout) != VK_SUCCESS)
		{
			SetError("Creating descriptor layout failed.");
			return false;
		}

		return true;
	}

	bool CreateTextGraphicsPipeline(const char *pVertName, const char *pFragName)
	{
		VkPipelineShaderStageCreateInfo aShaderStages[2];
		SShaderModule Module;
		if(!CreateShaders(pVertName, pFragName, aShaderStages, Module))
			return false;

		VkPipelineVertexInputStateCreateInfo VertexInputInfo{};
		VertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		VkVertexInputBindingDescription BindingDescription{};
		BindingDescription.binding = 0;
		BindingDescription.stride = sizeof(float) * 2 * 2 + sizeof(uint8_t) * 4;
		BindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 3> aAttributeDescriptions = {};

		aAttributeDescriptions[0].binding = 0;
		aAttributeDescriptions[0].location = 0;
		aAttributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[0].offset = 0;

		aAttributeDescriptions[1].binding = 0;
		aAttributeDescriptions[1].location = 1;
		aAttributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[1].offset = sizeof(float) * 2;

		aAttributeDescriptions[2].binding = 0;
		aAttributeDescriptions[2].location = 2;
		aAttributeDescriptions[2].format = VK_FORMAT_R8G8B8A8_UNORM;
		aAttributeDescriptions[2].offset = sizeof(float) * 2 * 2;

		VertexInputInfo.vertexBindingDescriptionCount = 1;
		VertexInputInfo.vertexAttributeDescriptionCount = aAttributeDescriptions.size();
		VertexInputInfo.pVertexBindingDescriptions = &BindingDescription;
		VertexInputInfo.pVertexAttributeDescriptions = aAttributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo InputAssembly{};
		VkViewport Viewport{};
		VkRect2D Scissor{};
		VkPipelineViewportStateCreateInfo ViewportState{};
		VkPipelineRasterizationStateCreateInfo Rasterizer{};
		VkPipelineMultisampleStateCreateInfo Multisampling{};
		VkPipelineColorBlendAttachmentState ColorBlendAttachment{};
		VkPipelineColorBlendStateCreateInfo ColorBlending{};
		GetStandardPipelineInfo(InputAssembly, Viewport, Scissor, ViewportState, Rasterizer, Multisampling, ColorBlendAttachment, ColorBlending);

		VkPipelineLayoutCreateInfo PipelineLayoutInfo{};
		PipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		PipelineLayoutInfo.setLayoutCount = 1;
		PipelineLayoutInfo.pSetLayouts = &m_TextDescriptorSetLayout;

		std::array<VkPushConstantRange, 2> aPushConstants{};
		aPushConstants[0].offset = 0;
		aPushConstants[0].size = sizeof(SUniformGTextPos);
		aPushConstants[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		aPushConstants[1].offset = sizeof(SUniformGTextPos) + sizeof(SUniformTextGFragmentOffset);
		aPushConstants[1].size = sizeof(SUniformTextGFragmentConstants);
		aPushConstants[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		PipelineLayoutInfo.pushConstantRangeCount = aPushConstants.size();
		PipelineLayoutInfo.pPushConstantRanges = aPushConstants.data();

		if(vkCreatePipelineLayout(m_VKDevice, &PipelineLayoutInfo, nullptr, &m_TextPipeLineLayout) != VK_SUCCESS)
		{
			SetError("Creating pipeline layout failed.");
			return false;
		}

		VkGraphicsPipelineCreateInfo PipelineInfo{};
		PipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		PipelineInfo.stageCount = 2;
		PipelineInfo.pStages = aShaderStages;
		PipelineInfo.pVertexInputState = &VertexInputInfo;
		PipelineInfo.pInputAssemblyState = &InputAssembly;
		PipelineInfo.pViewportState = &ViewportState;
		PipelineInfo.pRasterizationState = &Rasterizer;
		PipelineInfo.pMultisampleState = &Multisampling;
		PipelineInfo.pColorBlendState = &ColorBlending;
		PipelineInfo.layout = m_TextPipeLineLayout;
		PipelineInfo.renderPass = m_VKRenderPass;
		PipelineInfo.subpass = 0;
		PipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if(vkCreateGraphicsPipelines(m_VKDevice, VK_NULL_HANDLE, 1, &PipelineInfo, nullptr, &m_TextPipeline) != VK_SUCCESS)
		{
			SetError("Creating the graphic pipeline failed.");
			return false;
		}

		return true;
	}

	bool CreateTileGraphicsPipeline(const char *pVertName, const char *pFragName, bool HasSampler, int Type)
	{
		VkPipelineShaderStageCreateInfo aShaderStages[2];
		SShaderModule Module;
		if(!CreateShaders(pVertName, pFragName, aShaderStages, Module))
			return false;

		VkPipelineVertexInputStateCreateInfo VertexInputInfo{};
		VertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		VkVertexInputBindingDescription BindingDescription{};
		BindingDescription.binding = 0;
		BindingDescription.stride = HasSampler ? (sizeof(float) * (2 + 3)) : (sizeof(float) * 2);
		BindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 2> aAttributeDescriptions = {};

		aAttributeDescriptions[0].binding = 0;
		aAttributeDescriptions[0].location = 0;
		aAttributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[0].offset = 0;

		aAttributeDescriptions[1].binding = 0;
		aAttributeDescriptions[1].location = 1;
		aAttributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		aAttributeDescriptions[1].offset = sizeof(float) * 2;

		VertexInputInfo.vertexBindingDescriptionCount = 1;
		VertexInputInfo.vertexAttributeDescriptionCount = HasSampler ? 2 : 1;
		VertexInputInfo.pVertexBindingDescriptions = &BindingDescription;
		VertexInputInfo.pVertexAttributeDescriptions = aAttributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo InputAssembly{};
		VkViewport Viewport{};
		VkRect2D Scissor{};
		VkPipelineViewportStateCreateInfo ViewportState{};
		VkPipelineRasterizationStateCreateInfo Rasterizer{};
		VkPipelineMultisampleStateCreateInfo Multisampling{};
		VkPipelineColorBlendAttachmentState ColorBlendAttachment{};
		VkPipelineColorBlendStateCreateInfo ColorBlending{};
		GetStandardPipelineInfo(InputAssembly, Viewport, Scissor, ViewportState, Rasterizer, Multisampling, ColorBlendAttachment, ColorBlending);

		VkPipelineLayoutCreateInfo PipelineLayoutInfo{};
		PipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		PipelineLayoutInfo.setLayoutCount = HasSampler ? 1 : 0;
		PipelineLayoutInfo.pSetLayouts = HasSampler ? &m_Standard3DTexturedDescriptorSetLayout : nullptr;

		uint32_t VertPushConstantSize = sizeof(SUniformTileGPos);
		if(Type == 1)
			VertPushConstantSize = sizeof(SUniformTileGPosBorder);
		else if(Type == 2)
			VertPushConstantSize = sizeof(SUniformTileGPosBorderLine);

		uint32_t FragPushConstantSize = sizeof(SUniformTileGVertColor);

		std::array<VkPushConstantRange, 2> aPushConstants{};
		aPushConstants[0].offset = 0;
		aPushConstants[0].size = VertPushConstantSize;
		aPushConstants[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		aPushConstants[1].offset = sizeof(SUniformTileGPosBorder) + sizeof(SUniformTileGVertColorAlign);
		aPushConstants[1].size = FragPushConstantSize;
		aPushConstants[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		PipelineLayoutInfo.pushConstantRangeCount = aPushConstants.size();
		PipelineLayoutInfo.pPushConstantRanges = aPushConstants.data();

		VkPipelineLayout *pPipeLayout = &m_TilePipeLineLayout;
		VkPipelineLayout *pPipeLayoutTex = &m_TileTexturedPipeLineLayout;
		if(Type == 1)
		{
			pPipeLayout = &m_TileBorderPipeLineLayout;
			pPipeLayoutTex = &m_TileBorderTexturedPipeLineLayout;
		}
		else if(Type == 2)
		{
			pPipeLayout = &m_TileBorderLinePipeLineLayout;
			pPipeLayoutTex = &m_TileBorderLineTexturedPipeLineLayout;
		}

		if(vkCreatePipelineLayout(m_VKDevice, &PipelineLayoutInfo, nullptr, (HasSampler ? pPipeLayoutTex : pPipeLayout)) != VK_SUCCESS)
		{
			SetError("Creating pipeline layout failed.");
			return false;
		}

		VkGraphicsPipelineCreateInfo PipelineInfo{};
		PipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		PipelineInfo.stageCount = 2;
		PipelineInfo.pStages = aShaderStages;
		PipelineInfo.pVertexInputState = &VertexInputInfo;
		PipelineInfo.pInputAssemblyState = &InputAssembly;
		PipelineInfo.pViewportState = &ViewportState;
		PipelineInfo.pRasterizationState = &Rasterizer;
		PipelineInfo.pMultisampleState = &Multisampling;
		PipelineInfo.pColorBlendState = &ColorBlending;
		PipelineInfo.layout = (HasSampler ? *pPipeLayoutTex : *pPipeLayout);
		PipelineInfo.renderPass = m_VKRenderPass;
		PipelineInfo.subpass = 0;
		PipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		VkPipeline *pPipe = &m_TilePipeline;
		VkPipeline *pPipeTex = &m_TileTexturedPipeline;
		if(Type == 1)
		{
			pPipe = &m_TileBorderPipeline;
			pPipeTex = &m_TileBorderTexturedPipeline;
		}
		else if(Type == 2)
		{
			pPipe = &m_TileBorderLinePipeline;
			pPipeTex = &m_TileBorderLineTexturedPipeline;
		}

		if(vkCreateGraphicsPipelines(m_VKDevice, VK_NULL_HANDLE, 1, &PipelineInfo, nullptr, (HasSampler ? pPipeTex : pPipe)) != VK_SUCCESS)
		{
			SetError("Creating the graphic pipeline failed.");
			return false;
		}

		return true;
	}

	bool CreatePrimExGraphicsPipeline(const char *pVertName, const char *pFragName, bool HasSampler, bool Rotationless)
	{
		VkPipelineShaderStageCreateInfo aShaderStages[2];
		SShaderModule Module;
		if(!CreateShaders(pVertName, pFragName, aShaderStages, Module))
			return false;

		VkPipelineVertexInputStateCreateInfo VertexInputInfo{};
		VertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		VkVertexInputBindingDescription BindingDescription{};
		BindingDescription.binding = 0;
		BindingDescription.stride = sizeof(float) * 2 * 2 + sizeof(uint8_t) * 4;
		BindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 3> aAttributeDescriptions = {};

		aAttributeDescriptions[0].binding = 0;
		aAttributeDescriptions[0].location = 0;
		aAttributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[0].offset = 0;

		aAttributeDescriptions[1].binding = 0;
		aAttributeDescriptions[1].location = 1;
		aAttributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[1].offset = sizeof(float) * 2;

		aAttributeDescriptions[2].binding = 0;
		aAttributeDescriptions[2].location = 2;
		aAttributeDescriptions[2].format = VK_FORMAT_R8G8B8A8_UNORM;
		aAttributeDescriptions[2].offset = sizeof(float) * 2 * 2;

		VertexInputInfo.vertexBindingDescriptionCount = 1;
		VertexInputInfo.vertexAttributeDescriptionCount = aAttributeDescriptions.size();
		VertexInputInfo.pVertexBindingDescriptions = &BindingDescription;
		VertexInputInfo.pVertexAttributeDescriptions = aAttributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo InputAssembly{};
		VkViewport Viewport{};
		VkRect2D Scissor{};
		VkPipelineViewportStateCreateInfo ViewportState{};
		VkPipelineRasterizationStateCreateInfo Rasterizer{};
		VkPipelineMultisampleStateCreateInfo Multisampling{};
		VkPipelineColorBlendAttachmentState ColorBlendAttachment{};
		VkPipelineColorBlendStateCreateInfo ColorBlending{};
		GetStandardPipelineInfo(InputAssembly, Viewport, Scissor, ViewportState, Rasterizer, Multisampling, ColorBlendAttachment, ColorBlending);

		VkPipelineLayoutCreateInfo PipelineLayoutInfo{};
		PipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		PipelineLayoutInfo.setLayoutCount = HasSampler ? 1 : 0;
		PipelineLayoutInfo.pSetLayouts = HasSampler ? &m_StandardTexturedDescriptorSetLayout : nullptr;

		uint32_t VertPushConstantSize = sizeof(SUniformPrimExGPos);
		if(Rotationless)
			VertPushConstantSize = sizeof(SUniformPrimExGPosRotationless);

		uint32_t FragPushConstantSize = sizeof(SUniformPrimExGVertColor);

		std::array<VkPushConstantRange, 2> aPushConstants{};
		aPushConstants[0].offset = 0;
		aPushConstants[0].size = VertPushConstantSize;
		aPushConstants[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		aPushConstants[1].offset = sizeof(SUniformPrimExGPos) + sizeof(SUniformPrimExGVertColorAlign);
		aPushConstants[1].size = FragPushConstantSize;
		aPushConstants[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		PipelineLayoutInfo.pushConstantRangeCount = aPushConstants.size();
		PipelineLayoutInfo.pPushConstantRanges = aPushConstants.data();

		VkPipelineLayout *pPipeLayout = &m_PrimExPipeLineLayout;
		VkPipelineLayout *pPipeLayoutTex = &m_PrimExTexPipeLineLayout;
		if(Rotationless)
		{
			pPipeLayout = &m_PrimExRotationlessPipeLineLayout;
			pPipeLayoutTex = &m_PrimExRotationlessTexPipeLineLayout;
		}

		if(vkCreatePipelineLayout(m_VKDevice, &PipelineLayoutInfo, nullptr, (HasSampler ? pPipeLayoutTex : pPipeLayout)) != VK_SUCCESS)
		{
			SetError("Creating pipeline layout failed.");
			return false;
		}

		VkGraphicsPipelineCreateInfo PipelineInfo{};
		PipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		PipelineInfo.stageCount = 2;
		PipelineInfo.pStages = aShaderStages;
		PipelineInfo.pVertexInputState = &VertexInputInfo;
		PipelineInfo.pInputAssemblyState = &InputAssembly;
		PipelineInfo.pViewportState = &ViewportState;
		PipelineInfo.pRasterizationState = &Rasterizer;
		PipelineInfo.pMultisampleState = &Multisampling;
		PipelineInfo.pColorBlendState = &ColorBlending;
		PipelineInfo.layout = (HasSampler ? *pPipeLayoutTex : *pPipeLayout);
		PipelineInfo.renderPass = m_VKRenderPass;
		PipelineInfo.subpass = 0;
		PipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		VkPipeline *pPipe = &m_PrimExPipeline;
		VkPipeline *pPipeTex = &m_PrimExTexPipeline;
		if(Rotationless)
		{
			pPipe = &m_PrimExRotationlessPipeline;
			pPipeTex = &m_PrimExRotationlessTexPipeline;
		}

		if(vkCreateGraphicsPipelines(m_VKDevice, VK_NULL_HANDLE, 1, &PipelineInfo, nullptr, (HasSampler ? pPipeTex : pPipe)) != VK_SUCCESS)
		{
			SetError("Creating the graphic pipeline failed.");
			return false;
		}

		return true;
	}

	bool CreateUniformDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding SamplerLayoutBinding{};
		SamplerLayoutBinding.binding = 1;
		SamplerLayoutBinding.descriptorCount = 1;
		SamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		SamplerLayoutBinding.pImmutableSamplers = nullptr;
		SamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding aBindings[] = {SamplerLayoutBinding};
		VkDescriptorSetLayoutCreateInfo LayoutInfo{};
		LayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		LayoutInfo.bindingCount = (sizeof(aBindings) / sizeof(aBindings[0]));
		LayoutInfo.pBindings = aBindings;

		if(vkCreateDescriptorSetLayout(m_VKDevice, &LayoutInfo, nullptr, &m_UniformDescriptorSetLayout) != VK_SUCCESS)
		{
			SetError("Creating descriptor layout failed.");
			return false;
		}
		return true;
	}

	bool CreateUniformDescriptorSets(VkDescriptorSet *pSets, size_t SetCount, VkBuffer BindBuffer, size_t SingleBufferInstanceSize)
	{
		for(size_t i = 0; i < SetCount; ++i)
		{
			VkDescriptorSet DescrSet;
			VkDescriptorSetAllocateInfo DesAllocInfo{};
			DesAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			DesAllocInfo.descriptorPool = m_VKDescrPool;
			DesAllocInfo.descriptorSetCount = 1;
			DesAllocInfo.pSetLayouts = &m_UniformDescriptorSetLayout;

			if(vkAllocateDescriptorSets(m_VKDevice, &DesAllocInfo, &DescrSet) != VK_SUCCESS)
			{
				return false;
			}

			VkDescriptorBufferInfo BufferInfo{};
			BufferInfo.buffer = BindBuffer;
			BufferInfo.offset = SingleBufferInstanceSize * i;
			BufferInfo.range = SingleBufferInstanceSize;

			std::array<VkWriteDescriptorSet, 1> aDescriptorWrites{};

			aDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			aDescriptorWrites[0].dstSet = DescrSet;
			aDescriptorWrites[0].dstBinding = 1;
			aDescriptorWrites[0].dstArrayElement = 0;
			aDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			aDescriptorWrites[0].descriptorCount = 1;
			aDescriptorWrites[0].pBufferInfo = &BufferInfo;

			vkUpdateDescriptorSets(m_VKDevice, static_cast<uint32_t>(aDescriptorWrites.size()), aDescriptorWrites.data(), 0, nullptr);

			pSets[i] = DescrSet;
		}

		return true;
	}

	bool CreateSpriteMultiGraphicsPipeline(const char *pVertName, const char *pFragName)
	{
		VkPipelineShaderStageCreateInfo aShaderStages[2];
		SShaderModule Module;
		if(!CreateShaders(pVertName, pFragName, aShaderStages, Module))
			return false;

		VkPipelineVertexInputStateCreateInfo VertexInputInfo{};
		VertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		VkVertexInputBindingDescription BindingDescription{};
		BindingDescription.binding = 0;
		BindingDescription.stride = sizeof(float) * 2 * 2 + sizeof(uint8_t) * 4;
		BindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 3> aAttributeDescriptions = {};

		aAttributeDescriptions[0].binding = 0;
		aAttributeDescriptions[0].location = 0;
		aAttributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[0].offset = 0;

		aAttributeDescriptions[1].binding = 0;
		aAttributeDescriptions[1].location = 1;
		aAttributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		aAttributeDescriptions[1].offset = sizeof(float) * 2;

		aAttributeDescriptions[2].binding = 0;
		aAttributeDescriptions[2].location = 2;
		aAttributeDescriptions[2].format = VK_FORMAT_R8G8B8A8_UNORM;
		aAttributeDescriptions[2].offset = sizeof(float) * 2 * 2;

		VertexInputInfo.vertexBindingDescriptionCount = 1;
		VertexInputInfo.vertexAttributeDescriptionCount = aAttributeDescriptions.size();
		VertexInputInfo.pVertexBindingDescriptions = &BindingDescription;
		VertexInputInfo.pVertexAttributeDescriptions = aAttributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo InputAssembly{};
		VkViewport Viewport{};
		VkRect2D Scissor{};
		VkPipelineViewportStateCreateInfo ViewportState{};
		VkPipelineRasterizationStateCreateInfo Rasterizer{};
		VkPipelineMultisampleStateCreateInfo Multisampling{};
		VkPipelineColorBlendAttachmentState ColorBlendAttachment{};
		VkPipelineColorBlendStateCreateInfo ColorBlending{};
		GetStandardPipelineInfo(InputAssembly, Viewport, Scissor, ViewportState, Rasterizer, Multisampling, ColorBlendAttachment, ColorBlending);

		VkPipelineLayoutCreateInfo PipelineLayoutInfo{};
		PipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		PipelineLayoutInfo.setLayoutCount = 2;
		VkDescriptorSetLayout aSetLayouts[] = {m_StandardTexturedDescriptorSetLayout, m_UniformDescriptorSetLayout};
		PipelineLayoutInfo.pSetLayouts = aSetLayouts;

		uint32_t VertPushConstantSize = sizeof(SUniformSpriteMultiGPos);
		uint32_t FragPushConstantSize = sizeof(SUniformSpriteMultiGVertColor);

		std::array<VkPushConstantRange, 2> aPushConstants{};
		aPushConstants[0].offset = 0;
		aPushConstants[0].size = VertPushConstantSize;
		aPushConstants[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		aPushConstants[1].offset = sizeof(SUniformSpriteMultiGPos) + sizeof(SUniformSpriteMultiGVertColorAlign);
		aPushConstants[1].size = FragPushConstantSize;
		aPushConstants[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		PipelineLayoutInfo.pushConstantRangeCount = aPushConstants.size();
		PipelineLayoutInfo.pPushConstantRanges = aPushConstants.data();

		VkPipelineLayout *pPipeLayout = &m_SpriteMultiPipeLineLayout;

		if(vkCreatePipelineLayout(m_VKDevice, &PipelineLayoutInfo, nullptr, pPipeLayout) != VK_SUCCESS)
		{
			SetError("Creating pipeline layout failed.");
			return false;
		}

		VkGraphicsPipelineCreateInfo PipelineInfo{};
		PipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		PipelineInfo.stageCount = 2;
		PipelineInfo.pStages = aShaderStages;
		PipelineInfo.pVertexInputState = &VertexInputInfo;
		PipelineInfo.pInputAssemblyState = &InputAssembly;
		PipelineInfo.pViewportState = &ViewportState;
		PipelineInfo.pRasterizationState = &Rasterizer;
		PipelineInfo.pMultisampleState = &Multisampling;
		PipelineInfo.pColorBlendState = &ColorBlending;
		PipelineInfo.layout = *pPipeLayout;
		PipelineInfo.renderPass = m_VKRenderPass;
		PipelineInfo.subpass = 0;
		PipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		VkPipeline *pPipe = &m_SpriteMultiPipeline;

		if(vkCreateGraphicsPipelines(m_VKDevice, VK_NULL_HANDLE, 1, &PipelineInfo, nullptr, pPipe) != VK_SUCCESS)
		{
			SetError("Creating the graphic pipeline failed.");
			return false;
		}

		return true;
	}

	bool CreateCommandPool()
	{
		VkCommandPoolCreateInfo CreatePoolInfo{};
		CreatePoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		CreatePoolInfo.queueFamilyIndex = m_VKGraphicsQueueIndex;
		CreatePoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		if(vkCreateCommandPool(m_VKDevice, &CreatePoolInfo, nullptr, &m_CommandPool) != VK_SUCCESS)
		{
			SetError("Creating the command pool failed.");
			return false;
		}
		return true;
	}

	bool CreateCommandBuffers()
	{
		m_CommandBuffers.resize(m_VKFramebufferList.size());
		m_MemoryCommandBuffers.resize(m_VKFramebufferList.size());
		m_UsedMemoryCommandBuffer.resize(m_VKFramebufferList.size(), false);

		VkCommandBufferAllocateInfo AllocInfo{};
		AllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		AllocInfo.commandPool = m_CommandPool;
		AllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		AllocInfo.commandBufferCount = (uint32_t)m_CommandBuffers.size();

		if(vkAllocateCommandBuffers(m_VKDevice, &AllocInfo, m_CommandBuffers.data()) != VK_SUCCESS)
		{
			SetError("Allocating command buffers failed.");
			return false;
		}

		AllocInfo.commandBufferCount = (uint32_t)m_MemoryCommandBuffers.size();

		if(vkAllocateCommandBuffers(m_VKDevice, &AllocInfo, m_MemoryCommandBuffers.data()) != VK_SUCCESS)
		{
			SetError("Allocating command buffers failed.");
			return false;
		}

		return true;
	}

	bool CreateSyncObjects()
	{
		m_WaitSemaphores.resize(m_MaxFramesChain);
		m_SigSemaphores.resize(m_MaxFramesChain);

		m_MemorySemaphores.resize(m_MaxFramesChain);

		m_FrameFences.resize(m_MaxFramesChain);
		m_ImagesFences.resize(m_VKChainImages.size(), VK_NULL_HANDLE);

		VkSemaphoreCreateInfo CreateSemaphoreInfo{};
		CreateSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for(size_t i = 0; i < m_MaxFramesChain; i++)
		{
			if(vkCreateSemaphore(m_VKDevice, &CreateSemaphoreInfo, nullptr, &m_WaitSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(m_VKDevice, &CreateSemaphoreInfo, nullptr, &m_SigSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(m_VKDevice, &CreateSemaphoreInfo, nullptr, &m_MemorySemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(m_VKDevice, &fenceInfo, nullptr, &m_FrameFences[i]) != VK_SUCCESS)
			{
				SetError("Creating swap chain sync objects(fences, semaphores) failed.");
				return false;
			}
		}

		return true;
	}

	void CleanupSwapChain()
	{
		for(auto &FrameBuffer : m_VKFramebufferList)
		{
			vkDestroyFramebuffer(m_VKDevice, FrameBuffer, nullptr);
		}
		m_VKFramebufferList.clear();

		vkFreeCommandBuffers(m_VKDevice, m_CommandPool, static_cast<uint32_t>(m_MemoryCommandBuffers.size()), m_MemoryCommandBuffers.data());
		vkFreeCommandBuffers(m_VKDevice, m_CommandPool, static_cast<uint32_t>(m_CommandBuffers.size()), m_CommandBuffers.data());
		vkDestroyCommandPool(m_VKDevice, m_CommandPool, nullptr);
		m_CommandPool = VK_NULL_HANDLE;
		m_CommandBuffers.clear();
		m_MemoryCommandBuffers.clear();

		vkDestroyPipeline(m_VKDevice, m_StandardPipeline, nullptr);
		vkDestroyPipeline(m_VKDevice, m_StandardTexturedPipeline, nullptr);
		vkDestroyPipelineLayout(m_VKDevice, m_StandardPipeLineLayout, nullptr);
		vkDestroyPipelineLayout(m_VKDevice, m_StandardTexturedPipeLineLayout, nullptr);
		vkDestroyRenderPass(m_VKDevice, m_VKRenderPass, nullptr);
		m_StandardPipeline = VK_NULL_HANDLE;
		m_StandardTexturedPipeline = VK_NULL_HANDLE;
		m_StandardPipeLineLayout = VK_NULL_HANDLE;
		m_StandardTexturedPipeLineLayout = VK_NULL_HANDLE;
		m_VKRenderPass = VK_NULL_HANDLE;

		for(auto &ImageView : m_VKSwapChainImageViewList)
		{
			vkDestroyImageView(m_VKDevice, ImageView, nullptr);
		}
		m_VKSwapChainImageViewList.clear();

		vkDestroySwapchainKHR(m_VKDevice, m_VKSwapChain, nullptr);
		m_VKSwapChain = VK_NULL_HANDLE;
	}

	void RecreateSwapChain()
	{
		vkDeviceWaitIdle(m_VKDevice);

		CleanupSwapChain();

		CreateSwapChain();
		GetSwapChainImageHandles();
		CreateImageViews();
		CreateRenderPass(true);
		CreateStandardGraphicsPipeline("shader/vulkan/prim.vert.spv", "shader/vulkan/prim.frag.spv", false, false);
		CreateStandardGraphicsPipeline("shader/vulkan/prim_textured.vert.spv", "shader/vulkan/prim_textured.frag.spv", true, false);
		CreateStandardGraphicsPipeline("shader/vulkan/prim.vert.spv", "shader/vulkan/prim.frag.spv", false, true);

		CreateTextGraphicsPipeline("shader/vulkan/text.vert.spv", "shader/vulkan/text.frag.spv");

		CreateTileGraphicsPipeline("shader/vulkan/tile.vert.spv", "shader/vulkan/tile.frag.spv", false, 0);
		CreateTileGraphicsPipeline("shader/vulkan/tile_textured.vert.spv", "shader/vulkan/tile_textured.frag.spv", true, 0);
		CreateTileGraphicsPipeline("shader/vulkan/tile_border.vert.spv", "shader/vulkan/tile_border.frag.spv", false, 1);
		CreateTileGraphicsPipeline("shader/vulkan/tile_border_textured.vert.spv", "shader/vulkan/tile_border_textured.frag.spv", true, 1);
		CreateTileGraphicsPipeline("shader/vulkan/tile_border_line.vert.spv", "shader/vulkan/tile_border_line.frag.spv", false, 2);
		CreateTileGraphicsPipeline("shader/vulkan/tile_border_line_textured.vert.spv", "shader/vulkan/tile_border_line_textured.frag.spv", true, 2);

		CreatePrimExGraphicsPipeline("shader/vulkan/primex_rotationless.vert.spv", "shader/vulkan/primex_rotationless.frag.spv", false, true);
		CreatePrimExGraphicsPipeline("shader/vulkan/primex_tex_rotationless.vert.spv", "shader/vulkan/primex_tex_rotationless.frag.spv", true, true);
		CreatePrimExGraphicsPipeline("shader/vulkan/primex.vert.spv", "shader/vulkan/primex.frag.spv", false, false);
		CreatePrimExGraphicsPipeline("shader/vulkan/primex_tex.vert.spv", "shader/vulkan/primex_tex.frag.spv", true, false);

		CreateSpriteMultiGraphicsPipeline("shader/vulkan/spritemulti.vert.spv", "shader/vulkan/spritemulti.frag.spv");

		CreateFramebuffers();
		CreateCommandPool();
		CreateCommandBuffers();

		// m_ImagesFences.clear();
		m_ImagesFences.resize(m_VKChainImages.size(), VK_NULL_HANDLE);
	}

	int InitVulkanSDL(SDL_Window *pWindow, uint32_t CanvasWidth, uint32_t CanvasHeight, char *pRendererString, char *pVendorString, char *pVersionString)
	{
		std::vector<std::string> VKExtensions;
		std::vector<std::string> VKLayers;

		m_CanvasWidth = CanvasWidth;
		m_CanvasHeight = CanvasHeight;

		if(!GetVulkanExtensions(pWindow, VKExtensions))
			return -1;

		if(!GetVulkanLayers(VKLayers))
			return -1;

		if(!CreateVulkanInstance(VKLayers, VKExtensions))
			return -1;

		if(g_Config.m_DbgGfx >= 1)
		{
			SetupDebugCallback();

			for(auto &VKLayer : VKLayers)
			{
				dbg_msg("vulkan", "Validation layer: %s", VKLayer.c_str());
			}
		}

		if(!SelectGPU(pRendererString, pVendorString, pVersionString))
			return -1;

		if(!CreateLogicalDevice(VKLayers))
			return -1;

		GetDeviceQueue();

		if(!CreateSurface(pWindow))
			return -1;

		if(!CreateSwapChain())
			return -1;

		return 0;
	}

	uint32_t FindMemoryType(VkPhysicalDevice PhyDevice, uint32_t TypeFilter, VkMemoryPropertyFlags Properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(PhyDevice, &memProperties);

		for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if((TypeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & Properties) == Properties)
			{
				return i;
			}
		}

		return 0;
	}

	bool CreateBuffer(VkDeviceSize BufferSize, VkBufferUsageFlags BufferUsage, VkMemoryPropertyFlags BufferProperties, VkBuffer &VKBuffer, SDeviceMemoryBlock &VKBufferMemory)
	{
		VkBufferCreateInfo BufferInfo{};
		BufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		BufferInfo.size = BufferSize;
		BufferInfo.usage = BufferUsage;
		BufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if(vkCreateBuffer(m_VKDevice, &BufferInfo, nullptr, &VKBuffer) != VK_SUCCESS)
		{
			SetError("Buffer creation failed.");
			return false;
		}

		VkMemoryRequirements MemRequirements;
		vkGetBufferMemoryRequirements(m_VKDevice, VKBuffer, &MemRequirements);

		VkMemoryAllocateInfo MemAllocInfo{};
		MemAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		MemAllocInfo.allocationSize = MemRequirements.size;
		MemAllocInfo.memoryTypeIndex = FindMemoryType(m_VKGPU, MemRequirements.memoryTypeBits, BufferProperties);

		VKBufferMemory.m_Size = MemRequirements.size;
		m_pTextureMemoryUsage->store(m_pTextureMemoryUsage->load(std::memory_order_relaxed) + MemRequirements.size, std::memory_order_relaxed);

		if(g_Config.m_DbgGfx >= 3)
		{
			dbg_msg("vulkan", "allocated chunk of memory with size: %zu for frame %zu (buffer)", (size_t)MemRequirements.size, (size_t)m_CurImageIndex);
		}

		if(vkAllocateMemory(m_VKDevice, &MemAllocInfo, nullptr, &VKBufferMemory.m_Mem) != VK_SUCCESS)
		{
			SetError("Allocation from buffer object failed.");
			return false;
		}

		if(vkBindBufferMemory(m_VKDevice, VKBuffer, VKBufferMemory.m_Mem, 0) != VK_SUCCESS)
		{
			SetError("Binding memory to buffer failed.");
			return false;
		}

		return true;
	}

	struct SUniformGPos
	{
		float m_aPos[4 * 2];
	};

	struct SUniformGTextPos
	{
		float m_aPos[4 * 2];
		float m_TextureSize;
	};

	struct SUniformTextGFragmentOffset
	{
		float m_Padding[3];
	};

	struct SUniformTextGFragmentConstants
	{
		float m_aTextColor[4];
		float m_aTextOutlineColor[4];
	};

	struct SUniformTextFragment
	{
		SUniformTextGFragmentConstants m_Constants;
	};

	struct SUniformTileGPos
	{
		float m_aPos[4 * 2];
	};

	struct SUniformTileGPosBorderLine : public SUniformTileGPos
	{
		vec2 m_Dir;
		vec2 m_Offset;
	};

	struct SUniformTileGPosBorder : public SUniformTileGPosBorderLine
	{
		int32_t m_JumpIndex;
	};

	struct SUniformTileGVertColor
	{
		float m_aColor[4];
	};

	struct SUniformTileGVertColorAlign
	{
		float m_aPad[(64 - 52) / 4];
	};

	struct SUniformPrimExGPosRotationless
	{
		float m_aPos[4 * 2];
	};

	struct SUniformPrimExGPos : public SUniformPrimExGPosRotationless
	{
		vec2 m_Center;
		float m_Rotation;
	};

	struct SUniformPrimExGVertColor
	{
		float m_aColor[4];
	};

	struct SUniformPrimExGVertColorAlign
	{
		float m_aPad[(48 - 44) / 4];
	};

	struct SUniformSpriteMultiGPos
	{
		float m_aPos[4 * 2];
		vec2 m_Center;
	};

	struct SUniformSpriteMultiGVertColor
	{
		float m_aColor[4];
	};

	struct SUniformSpriteMultiGVertColorAlign
	{
		float m_aPad[(48 - 40) / 4];
	};

	bool CreateDescriptorPool()
	{
		// TODO do multiple descriptor pools on fly
		std::array<VkDescriptorPoolSize, 2> aPoolSizes{};
		aPoolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		aPoolSizes[0].descriptorCount = static_cast<uint32_t>(m_VKChainImages.size()) * CCommandBuffer::MAX_TEXTURES * 2;
		aPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		aPoolSizes[1].descriptorCount = static_cast<uint32_t>(m_VKChainImages.size()) * CCommandBuffer::MAX_TEXTURES * 2;

		VkDescriptorPoolCreateInfo PoolInfo{};
		PoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		PoolInfo.poolSizeCount = static_cast<uint32_t>(aPoolSizes.size());
		PoolInfo.pPoolSizes = aPoolSizes.data();
		PoolInfo.maxSets = static_cast<uint32_t>(m_VKChainImages.size()) * CCommandBuffer::MAX_TEXTURES * 2;
		PoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

		if(vkCreateDescriptorPool(m_VKDevice, &PoolInfo, nullptr, &m_VKDescrPool) != VK_SUCCESS)
		{
			SetError("Creating the descriptor pool failed.");
			return false;
		}

		return true;
	}

	bool CreateNewTexturedStandardDescriptorSets(size_t TextureSlot, size_t DescrIndex)
	{
		auto &Texture = m_Textures[TextureSlot];

		auto &DescrSet = Texture.m_aVKStandardTexturedDescrSets[DescrIndex];

		VkDescriptorSetAllocateInfo DesAllocInfo{};
		DesAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		DesAllocInfo.descriptorPool = m_VKDescrPool;
		DesAllocInfo.descriptorSetCount = 1;
		DesAllocInfo.pSetLayouts = &m_StandardTexturedDescriptorSetLayout;

		if(vkAllocateDescriptorSets(m_VKDevice, &DesAllocInfo, &DescrSet) != VK_SUCCESS)
		{
			return false;
		}

		VkDescriptorImageInfo ImageInfo{};
		ImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		ImageInfo.imageView = Texture.m_ImgView;
		ImageInfo.sampler = Texture.m_aSamplers[DescrIndex];

		std::array<VkWriteDescriptorSet, 1> aDescriptorWrites{};

		aDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		aDescriptorWrites[0].dstSet = DescrSet;
		aDescriptorWrites[0].dstBinding = 0;
		aDescriptorWrites[0].dstArrayElement = 0;
		aDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		aDescriptorWrites[0].descriptorCount = 1;
		aDescriptorWrites[0].pImageInfo = &ImageInfo;

		vkUpdateDescriptorSets(m_VKDevice, static_cast<uint32_t>(aDescriptorWrites.size()), aDescriptorWrites.data(), 0, nullptr);

		return true;
	}

	bool CreateNew3DTexturedStandardDescriptorSets(size_t TextureSlot)
	{
		auto &Texture = m_Textures[TextureSlot];

		auto &DescrSet = Texture.m_VKStandard3DTexturedDescrSet;

		VkDescriptorSetAllocateInfo DesAllocInfo{};
		DesAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		DesAllocInfo.descriptorPool = m_VKDescrPool;
		DesAllocInfo.descriptorSetCount = 1;
		DesAllocInfo.pSetLayouts = &m_Standard3DTexturedDescriptorSetLayout;

		if(vkAllocateDescriptorSets(m_VKDevice, &DesAllocInfo, &DescrSet) != VK_SUCCESS)
		{
			return false;
		}

		VkDescriptorImageInfo ImageInfo{};
		ImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		ImageInfo.imageView = Texture.m_Img3DView;
		ImageInfo.sampler = Texture.m_Sampler3D;

		std::array<VkWriteDescriptorSet, 1> aDescriptorWrites{};

		aDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		aDescriptorWrites[0].dstSet = DescrSet;
		aDescriptorWrites[0].dstBinding = 0;
		aDescriptorWrites[0].dstArrayElement = 0;
		aDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		aDescriptorWrites[0].descriptorCount = 1;
		aDescriptorWrites[0].pImageInfo = &ImageInfo;

		vkUpdateDescriptorSets(m_VKDevice, static_cast<uint32_t>(aDescriptorWrites.size()), aDescriptorWrites.data(), 0, nullptr);

		return true;
	}

	bool CreateNewTextDescriptorSets(size_t Texture, size_t TextureOutline)
	{
		auto &TextureText = m_Textures[Texture];
		auto &TextureTextOutline = m_Textures[TextureOutline];
		auto &DescrSetText = TextureText.m_VKTextDescrSet;
		auto &DescrSetTextOutline = TextureText.m_VKTextDescrSet;

		VkDescriptorSetAllocateInfo DesAllocInfo{};
		DesAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		DesAllocInfo.descriptorPool = m_VKDescrPool;
		DesAllocInfo.descriptorSetCount = 1;
		DesAllocInfo.pSetLayouts = &m_TextDescriptorSetLayout;

		if(vkAllocateDescriptorSets(m_VKDevice, &DesAllocInfo, &DescrSetText) != VK_SUCCESS)
		{
			return false;
		}

		std::array<VkDescriptorImageInfo, 2> aImageInfo{};
		aImageInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		aImageInfo[0].imageView = TextureText.m_ImgView;
		aImageInfo[0].sampler = TextureText.m_aSamplers[0];
		aImageInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		aImageInfo[1].imageView = TextureTextOutline.m_ImgView;
		aImageInfo[1].sampler = TextureTextOutline.m_aSamplers[0];

		std::array<VkWriteDescriptorSet, 2> aDescriptorWrites{};

		aDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		aDescriptorWrites[0].dstSet = DescrSetText;
		aDescriptorWrites[0].dstBinding = 0;
		aDescriptorWrites[0].dstArrayElement = 0;
		aDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		aDescriptorWrites[0].descriptorCount = 1;
		aDescriptorWrites[0].pImageInfo = aImageInfo.data();
		aDescriptorWrites[1] = aDescriptorWrites[0];
		aDescriptorWrites[1].dstBinding = 1;
		aDescriptorWrites[1].pImageInfo = &aImageInfo[1];

		vkUpdateDescriptorSets(m_VKDevice, static_cast<uint32_t>(aDescriptorWrites.size()), aDescriptorWrites.data(), 0, nullptr);

		DescrSetTextOutline = DescrSetText;

		return true;
	}

	int InitVulkan(IStorage *m_pStorage)
	{
		if(!GetSwapChainImageHandles())
			return -1;

		if(!CreateImageViews())
			return -1;

		if(!CreateRenderPass(false))
			return -1;

		if(!CreateFramebuffers())
			return -1;

		if(!CreateDescriptorSetLayout())
			return -1;

		if(!CreateStandardGraphicsPipeline("shader/vulkan/prim.vert.spv", "shader/vulkan/prim.frag.spv", false, false))
			return -1;

		if(!CreateStandardGraphicsPipeline("shader/vulkan/prim_textured.vert.spv", "shader/vulkan/prim_textured.frag.spv", true, false))
			return -1;

		if(!CreateStandardGraphicsPipeline("shader/vulkan/prim.vert.spv", "shader/vulkan/prim.frag.spv", false, true))
			return -1;

		if(!CreateTextDescriptorSetLayout())
			return -1;

		if(!CreateTextGraphicsPipeline("shader/vulkan/text.vert.spv", "shader/vulkan/text.frag.spv"))
			return -1;

		if(!CreateTileGraphicsPipeline("shader/vulkan/tile.vert.spv", "shader/vulkan/tile.frag.spv", false, 0))
			return -1;

		if(!CreateTileGraphicsPipeline("shader/vulkan/tile_textured.vert.spv", "shader/vulkan/tile_textured.frag.spv", true, 0))
			return -1;

		if(!CreateTileGraphicsPipeline("shader/vulkan/tile_border.vert.spv", "shader/vulkan/tile_border.frag.spv", false, 1))
			return -1;

		if(!CreateTileGraphicsPipeline("shader/vulkan/tile_border_textured.vert.spv", "shader/vulkan/tile_border_textured.frag.spv", true, 1))
			return -1;

		if(!CreateTileGraphicsPipeline("shader/vulkan/tile_border_line.vert.spv", "shader/vulkan/tile_border_line.frag.spv", false, 2))
			return -1;

		if(!CreateTileGraphicsPipeline("shader/vulkan/tile_border_line_textured.vert.spv", "shader/vulkan/tile_border_line_textured.frag.spv", true, 2))
			return -1;

		if(!CreatePrimExGraphicsPipeline("shader/vulkan/primex_rotationless.vert.spv", "shader/vulkan/primex_rotationless.frag.spv", false, true))
			return -1;

		if(!CreatePrimExGraphicsPipeline("shader/vulkan/primex_tex_rotationless.vert.spv", "shader/vulkan/primex_tex_rotationless.frag.spv", true, true))
			return -1;

		if(!CreatePrimExGraphicsPipeline("shader/vulkan/primex.vert.spv", "shader/vulkan/primex.frag.spv", false, false))
			return -1;

		if(!CreatePrimExGraphicsPipeline("shader/vulkan/primex_tex.vert.spv", "shader/vulkan/primex_tex.frag.spv", true, false))
			return -1;

		if(!CreateUniformDescriptorSetLayout())
			return -1;

		if(!CreateSpriteMultiGraphicsPipeline("shader/vulkan/spritemulti.vert.spv", "shader/vulkan/spritemulti.frag.spv"))
			return -1;

		if(!CreateCommandPool())
			return -1;

		if(!CreateCommandBuffers())
			return -1;

		if(!CreateSyncObjects())
			return -1;

		if(!CreateDescriptorPool())
			return -1;

		m_VKBuffersOfFrame.resize(m_VKChainImages.size());
		m_VKBuffersOfFrameRangeData.resize(m_VKChainImages.size());

		m_VKUniformBufferObjectsOfFrame.resize(m_VKChainImages.size());
		m_VKUniformBufferObjectsOfFrameRangeData.resize(m_VKChainImages.size());
		m_CurrentUniformUsedCount.resize(m_VKChainImages.size());

		m_FrameDelayedBufferCleanup.resize(m_VKChainImages.size());
		m_StagingBufferCache.Init(m_VKChainImages.size());
		m_VertexBufferCache.Init(m_VKChainImages.size());
		m_ImageBufferCache.Init(m_VKChainImages.size());

		// create dummy texture
		VkImage DummyImage;
		VkImageCreateInfo ImageInfo{};
		ImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ImageInfo.imageType = VK_IMAGE_TYPE_2D;
		ImageInfo.extent.width = 1024;
		ImageInfo.extent.height = 1024;
		ImageInfo.extent.depth = 1;
		ImageInfo.mipLevels = ImageMipLevelCount(ImageInfo.extent);
		ImageInfo.arrayLayers = 1;
		ImageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		ImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		ImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		ImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		ImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if(vkCreateImage(m_VKDevice, &ImageInfo, nullptr, &DummyImage) != VK_SUCCESS)
		{
			dbg_msg("gfx", "failed to create dummy image!");
		}

		vkGetImageMemoryRequirements(m_VKDevice, DummyImage, &m_DummyImageMemRequirements);
		vkDestroyImage(m_VKDevice, DummyImage, nullptr);

		// check if image format supports linear blitting
		VkFormatProperties FormatProperties;
		vkGetPhysicalDeviceFormatProperties(m_VKGPU, VK_FORMAT_R8G8B8A8_UNORM, &FormatProperties);
		if((FormatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0)
		{
			m_AllowsLinearBlitting = true;
		}

		return 0;
	}

	VkCommandBuffer GetMemoryCommandBuffer()
	{
		VkCommandBuffer MemCommandBuffer = m_MemoryCommandBuffers[m_CurImageIndex];
		if(!m_UsedMemoryCommandBuffer[m_CurImageIndex])
		{
			m_UsedMemoryCommandBuffer[m_CurImageIndex] = true;

			VkCommandBufferBeginInfo BeginInfo{};
			BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			if(vkBeginCommandBuffer(MemCommandBuffer, &BeginInfo) != VK_SUCCESS)
			{
				SetError("Command buffer cannot be filled anymore.");
			}
		}
		return MemCommandBuffer;
	}

	VkCommandBuffer GetCurrentCommandBuffer()
	{
		return m_CommandBuffers[m_CurImageIndex];
	}

	void CopyBuffer(VkBuffer SrcBuffer, VkBuffer DstBuffer, VkDeviceSize SrcOffset, VkDeviceSize DstOffset, VkDeviceSize CopySize)
	{
		VkCommandBuffer CommandBuffer = GetMemoryCommandBuffer();
		VkBufferCopy CopyRegion{};
		CopyRegion.srcOffset = SrcOffset;
		CopyRegion.dstOffset = DstOffset;
		CopyRegion.size = CopySize;
		vkCmdCopyBuffer(CommandBuffer, SrcBuffer, DstBuffer, 1, &CopyRegion);
	}

	void CreateStreamVertexBuffer(VkBuffer &NewBuffer, SDeviceMemoryBlock &NewBufferMem, size_t &BufferOffset, const void *pData, size_t DataSize)
	{
		VkBuffer Buffer = VK_NULL_HANDLE;
		SDeviceMemoryBlock BufferMem;
		size_t Offset = 0;
		uint8_t *pMem = nullptr;
		for(auto &BufferOfFrame : m_VKBuffersOfFrame[m_CurImageIndex])
		{
			if(BufferOfFrame.m_Size >= DataSize + BufferOfFrame.m_Offset)
			{
				Buffer = BufferOfFrame.m_Buffer;
				BufferMem = BufferOfFrame.m_BufferMem;
				Offset = BufferOfFrame.m_Offset;
				BufferOfFrame.m_Offset += DataSize;
				pMem = (uint8_t *)BufferOfFrame.m_MappedBufferData;
				break;
			}
		}

		if(BufferMem.m_Mem == VK_NULL_HANDLE)
		{
			// create memory
			VkBuffer StreamBuffer;
			SDeviceMemoryBlock StreamBufferMemory;
			VkDeviceSize NewBufferSize = sizeof(GL_SVertex) * CCommandBuffer::MAX_VERTICES;
			CreateBuffer(NewBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, StreamBuffer, StreamBufferMemory);

			void *pMappedData = nullptr;
			vkMapMemory(m_VKDevice, StreamBufferMemory.m_Mem, 0, VK_WHOLE_SIZE, 0, &pMappedData);

			m_VKBuffersOfFrame[m_CurImageIndex].push_back(SFrameBuffers{StreamBuffer, StreamBufferMemory, NewBufferSize, DataSize, pMappedData});
			m_VKBuffersOfFrameRangeData[m_CurImageIndex].push_back({});
			Buffer = StreamBuffer;
			BufferMem = StreamBufferMemory;

			pMem = (uint8_t *)pMappedData;
		}

		{
			mem_copy(pMem + Offset, pData, (size_t)DataSize);
		}

		NewBuffer = Buffer;
		NewBufferMem = BufferMem;
		BufferOffset = Offset;
	}

	void GetUniformBufferObject(VkDescriptorSet &DescrSet, const void *pData, size_t DataSize)
	{
		size_t BufferObjectAlignment = 16;
		size_t DataSizeAlignment = (DataSize % BufferObjectAlignment);
		if(DataSizeAlignment != 0)
			DataSizeAlignment = BufferObjectAlignment - (DataSize % BufferObjectAlignment);
		size_t RealDataSize = DataSize + DataSizeAlignment;

		const size_t NewBufferInstances = 64;
		const VkDeviceSize NewBufferInstanceSize = sizeof(IGraphics::SRenderSpriteInfo) * 512;
		const VkDeviceSize NewBufferSize = NewBufferInstanceSize * NewBufferInstances;

		uint8_t *pMem = nullptr;
		for(size_t i = m_CurrentUniformUsedCount[m_CurImageIndex]; i < m_VKUniformBufferObjectsOfFrame[m_CurImageIndex].size(); ++i)
		{
			auto &BufferOfFrame = m_VKUniformBufferObjectsOfFrame[m_CurImageIndex][i];
			DescrSet = BufferOfFrame.m_UniformSet;
			pMem = (uint8_t *)BufferOfFrame.m_pMappedBufferData;
			BufferOfFrame.m_UsedSize = RealDataSize;
			m_CurrentUniformUsedCount[m_CurImageIndex]++;
			break;
		}

		if(pMem == nullptr)
		{
			// create memory
			VkBuffer UniformBuffer;
			SDeviceMemoryBlock UniformBufferMemory;
			CreateBuffer(NewBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, UniformBuffer, UniformBufferMemory);

			void *pMappedData = nullptr;
			vkMapMemory(m_VKDevice, UniformBufferMemory.m_Mem, 0, VK_WHOLE_SIZE, 0, &pMappedData);

			VkDescriptorSet aDescrSets[NewBufferInstances];
			CreateUniformDescriptorSets(aDescrSets, NewBufferInstances, UniformBuffer, NewBufferInstanceSize);

			for(size_t i = 0; i < NewBufferInstances; ++i)
			{
				m_VKUniformBufferObjectsOfFrame[m_CurImageIndex].push_back(SFrameUniformBuffers{UniformBuffer, UniformBufferMemory, i * NewBufferInstanceSize, ((char *)pMappedData) + (i * NewBufferInstanceSize), aDescrSets[i]});
				m_VKUniformBufferObjectsOfFrameRangeData[m_CurImageIndex].push_back({});
			}

			auto &BufferEl = m_VKUniformBufferObjectsOfFrame[m_CurImageIndex][m_CurrentUniformUsedCount[m_CurImageIndex]];
			pMem = (uint8_t *)BufferEl.m_pMappedBufferData;
			BufferEl.m_UsedSize = RealDataSize;
			DescrSet = BufferEl.m_UniformSet;
			m_CurrentUniformUsedCount[m_CurImageIndex]++;
		}

		{
			mem_copy(pMem, pData, (size_t)DataSize);
		}
	}

	bool CreateIndexBuffer(void *pData, size_t DataSize, VkBuffer &Buffer, SDeviceMemoryBlock &Memory)
	{
		VkDeviceSize BufferDataSize = DataSize;

		auto StagingBuffer = GetStagingBuffer(pData, DataSize);

		SDeviceMemoryBlock VertexBufferMemory;
		VkBuffer VertexBuffer;
		CreateBuffer(BufferDataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VertexBuffer, VertexBufferMemory);

		MemoryBarrier(VertexBuffer, 0, BufferDataSize, VK_ACCESS_INDEX_READ_BIT, true);
		CopyBuffer(StagingBuffer.m_Buffer, VertexBuffer, StagingBuffer.m_Offset, 0, BufferDataSize);
		MemoryBarrier(VertexBuffer, 0, BufferDataSize, VK_ACCESS_INDEX_READ_BIT, false);

		FreeStagingMemBlock(StagingBuffer);

		Buffer = VertexBuffer;
		Memory = VertexBufferMemory;
		return true;
	}

	VkFormat TexFormatToVulkanFormat(int TexFormat)
	{
		if(TexFormat == CCommandBuffer::TEXFORMAT_RGB)
			return VK_FORMAT_R8G8B8_UNORM;
		if(TexFormat == CCommandBuffer::TEXFORMAT_ALPHA)
			return VK_FORMAT_R8_UNORM;
		if(TexFormat == CCommandBuffer::TEXFORMAT_RGBA)
			return VK_FORMAT_R8G8B8A8_UNORM;
		return VK_FORMAT_R8G8B8A8_UNORM;
	}

	size_t TexFormatToPixelChannelCount(int TexFormat)
	{
		if(TexFormat == CCommandBuffer::TEXFORMAT_RGB)
			return 3;
		if(TexFormat == CCommandBuffer::TEXFORMAT_ALPHA)
			return 1;
		if(TexFormat == CCommandBuffer::TEXFORMAT_RGBA)
			return 4;
		return 4;
	}

	void BuildMipmaps(VkImage Image, VkFormat ImageFormat, size_t Width, size_t Height, size_t Depth, size_t MipMapLevelCount)
	{
		VkCommandBuffer MemCommandBuffer = GetMemoryCommandBuffer();

		VkImageMemoryBarrier Barrier{};
		Barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		Barrier.image = Image;
		Barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		Barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		Barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		Barrier.subresourceRange.levelCount = 1;
		Barrier.subresourceRange.baseArrayLayer = 0;
		Barrier.subresourceRange.layerCount = Depth;

		int32_t TmpMipWidth = (int32_t)Width;
		int32_t TmpMipHeight = (int32_t)Height;

		for(size_t i = 1; i < MipMapLevelCount; ++i)
		{
			Barrier.subresourceRange.baseMipLevel = i - 1;
			Barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			Barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			Barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			Barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(MemCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &Barrier);

			VkImageBlit Blit{};
			Blit.srcOffsets[0] = {0, 0, 0};
			Blit.srcOffsets[1] = {TmpMipWidth, TmpMipHeight, 1};
			Blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			Blit.srcSubresource.mipLevel = i - 1;
			Blit.srcSubresource.baseArrayLayer = 0;
			Blit.srcSubresource.layerCount = Depth;
			Blit.dstOffsets[0] = {0, 0, 0};
			Blit.dstOffsets[1] = {TmpMipWidth > 1 ? TmpMipWidth / 2 : 1, TmpMipHeight > 1 ? TmpMipHeight / 2 : 1, 1};
			Blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			Blit.dstSubresource.mipLevel = i;
			Blit.dstSubresource.baseArrayLayer = 0;
			Blit.dstSubresource.layerCount = Depth;

			vkCmdBlitImage(MemCommandBuffer,
				Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &Blit,
				m_AllowsLinearBlitting ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);

			Barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			Barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			Barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			Barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(MemCommandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &Barrier);

			if(TmpMipWidth > 1)
				TmpMipWidth /= 2;
			if(TmpMipHeight > 1)
				TmpMipHeight /= 2;
		}

		Barrier.subresourceRange.baseMipLevel = MipMapLevelCount - 1;
		Barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		Barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		Barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		Barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(MemCommandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &Barrier);
	}

	bool CreateTextureImage(size_t ImageIndex, VkImage &NewImage, SMemoryBlock<s_ImageBufferCacheID> &NewImgMem, const void *pData, int Format, size_t Width, size_t Height, size_t Depth, size_t PixelSize, size_t MipMapLevelCount)
	{
		int ImageSize = Width * Height * Depth * PixelSize;

		auto StagingBuffer = GetStagingBuffer(pData, ImageSize);

		VkFormat ImgFormat = TexFormatToVulkanFormat(Format);

		CreateImage(Width, Height, Depth, MipMapLevelCount, ImgFormat, VK_IMAGE_TILING_OPTIMAL, NewImage, NewImgMem);

		ImageBarrier(NewImage, 0, MipMapLevelCount, 0, Depth, ImgFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		CopyBufferToImage(StagingBuffer.m_Buffer, StagingBuffer.m_Offset, NewImage, 0, 0, static_cast<uint32_t>(Width), static_cast<uint32_t>(Height), Depth);
		//ImageBarrier(NewImage, 0, 1, 0, Depth, ImgFormat, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		FreeStagingMemBlock(StagingBuffer);

		if(MipMapLevelCount > 1)
			BuildMipmaps(NewImage, ImgFormat, Width, Height, Depth, MipMapLevelCount);
		else
			ImageBarrier(NewImage, 0, 1, 0, Depth, ImgFormat, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		return true;
	}

	VkImageView CreateTextureImageView(VkImage TexImage, VkFormat ImgFormat, VkImageViewType ViewType, size_t Depth, size_t MipMapLevelCount)
	{
		return CreateImageView(TexImage, ImgFormat, ViewType, Depth, MipMapLevelCount);
	}

	VkSampler CreateTextureSampler(VkSamplerAddressMode AddrModeU, VkSamplerAddressMode AddrModeV, VkSamplerAddressMode AddrModeW)
	{
		VkPhysicalDeviceProperties Properties{};
		vkGetPhysicalDeviceProperties(m_VKGPU, &Properties);

		VkSamplerCreateInfo SamplerInfo{};
		SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		SamplerInfo.magFilter = VK_FILTER_LINEAR;
		SamplerInfo.minFilter = VK_FILTER_LINEAR;
		SamplerInfo.addressModeU = AddrModeU;
		SamplerInfo.addressModeV = AddrModeV;
		SamplerInfo.addressModeW = AddrModeW;
		SamplerInfo.anisotropyEnable = VK_FALSE;
		SamplerInfo.maxAnisotropy = Properties.limits.maxSamplerAnisotropy;
		SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		SamplerInfo.unnormalizedCoordinates = VK_FALSE;
		SamplerInfo.compareEnable = VK_FALSE;
		SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		SamplerInfo.mipLodBias = (m_GlobalTextureLodBIAS / 1000.0f);
		SamplerInfo.minLod = -1000;
		SamplerInfo.maxLod = 1000;

		VkSampler TexSampler;
		if(vkCreateSampler(m_VKDevice, &SamplerInfo, nullptr, &TexSampler) != VK_SUCCESS)
		{
			dbg_msg("gfx", "failed to create texture sampler!");
		}

		return TexSampler;
	}

	VkImageView CreateImageView(VkImage Image, VkFormat Format, VkImageViewType ViewType, size_t Depth, size_t MipMapLevelCount)
	{
		VkImageViewCreateInfo ViewCreateInfo{};
		ViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		ViewCreateInfo.image = Image;
		ViewCreateInfo.viewType = ViewType;
		ViewCreateInfo.format = Format;
		ViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		ViewCreateInfo.subresourceRange.baseMipLevel = 0;
		ViewCreateInfo.subresourceRange.levelCount = MipMapLevelCount;
		ViewCreateInfo.subresourceRange.baseArrayLayer = 0;
		ViewCreateInfo.subresourceRange.layerCount = Depth;

		VkImageView ImageView;
		if(vkCreateImageView(m_VKDevice, &ViewCreateInfo, nullptr, &ImageView) != VK_SUCCESS)
		{
			return VK_NULL_HANDLE;
		}

		return ImageView;
	}

	void CreateImage(uint32_t Width, uint32_t Height, uint32_t Depth, size_t MipMapLevelCount, VkFormat Format, VkImageTiling Tiling, VkImage &Image, SMemoryBlock<s_ImageBufferCacheID> &ImageMemory)
	{
		VkImageCreateInfo ImageInfo{};
		ImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		ImageInfo.imageType = VK_IMAGE_TYPE_2D;
		ImageInfo.extent.width = Width;
		ImageInfo.extent.height = Height;
		ImageInfo.extent.depth = 1;
		ImageInfo.mipLevels = MipMapLevelCount;
		ImageInfo.arrayLayers = Depth;
		ImageInfo.format = Format;
		ImageInfo.tiling = Tiling;
		ImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		ImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		ImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		ImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if(vkCreateImage(m_VKDevice, &ImageInfo, nullptr, &Image) != VK_SUCCESS)
		{
			dbg_msg("gfx", "failed to create image!");
		}

		VkMemoryRequirements MemRequirements;
		vkGetImageMemoryRequirements(m_VKDevice, Image, &MemRequirements);

		bool Is2DAndRGBA = (Format == VK_FORMAT_R8G8B8A8_UNORM);
		auto ImageMem = GetImageMemory(MemRequirements.size, Is2DAndRGBA);

		ImageMemory = ImageMem;
		vkBindImageMemory(m_VKDevice, Image, ImageMem.m_BufferMem.m_Mem, ImageMem.m_Offset);
	}

	void ImageBarrier(VkImage &Image, size_t MipMapBase, size_t MipMapCount, size_t LayerBase, size_t LayerCount, VkFormat Format, VkImageLayout OldLayout, VkImageLayout NewLayout)
	{
		VkCommandBuffer MemCommandBuffer = GetMemoryCommandBuffer();

		VkImageMemoryBarrier Barrier{};
		Barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		Barrier.oldLayout = OldLayout;
		Barrier.newLayout = NewLayout;
		Barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		Barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		Barrier.image = Image;
		Barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		Barrier.subresourceRange.baseMipLevel = MipMapBase;
		Barrier.subresourceRange.levelCount = MipMapCount;
		Barrier.subresourceRange.baseArrayLayer = LayerBase;
		Barrier.subresourceRange.layerCount = LayerCount;

		VkPipelineStageFlags SourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkPipelineStageFlags DestinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

		if(OldLayout == VK_IMAGE_LAYOUT_UNDEFINED && NewLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			Barrier.srcAccessMask = 0;
			Barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			SourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			DestinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if(OldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && NewLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			Barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			Barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			SourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			DestinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if(OldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && NewLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			Barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
			Barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			SourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			DestinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else
		{
			dbg_msg("gfx", "unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			MemCommandBuffer,
			SourceStage, DestinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &Barrier);
	}

	void MemoryBarrier(VkBuffer Buffer, VkDeviceSize Offset, VkDeviceSize Size, VkAccessFlags BufferAccessType, bool BeforeCommand)
	{
		VkCommandBuffer MemCommandBuffer = GetMemoryCommandBuffer();

		VkBufferMemoryBarrier Barrier{};
		Barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		Barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		Barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		Barrier.buffer = Buffer;
		Barrier.offset = Offset;
		Barrier.size = Size;

		VkPipelineStageFlags SourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkPipelineStageFlags DestinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

		if(BeforeCommand)
		{
			Barrier.srcAccessMask = BufferAccessType;
			Barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			SourceStage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
			DestinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else
		{
			Barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			Barrier.dstAccessMask = BufferAccessType;

			SourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			DestinationStage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		}

		vkCmdPipelineBarrier(
			MemCommandBuffer,
			SourceStage, DestinationStage,
			0,
			0, nullptr,
			1, &Barrier,
			0, nullptr);
	}

	void CopyBufferToImage(VkBuffer Buffer, VkDeviceSize BufferOffset, VkImage Image, int32_t X, int32_t Y, uint32_t Width, uint32_t Height, size_t Depth)
	{
		VkCommandBuffer CommandBuffer = GetMemoryCommandBuffer();

		VkBufferImageCopy Region{};
		Region.bufferOffset = BufferOffset;
		Region.bufferRowLength = 0;
		Region.bufferImageHeight = 0;
		Region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		Region.imageSubresource.mipLevel = 0;
		Region.imageSubresource.baseArrayLayer = 0;
		Region.imageSubresource.layerCount = Depth;
		Region.imageOffset = {X, Y, 0};
		Region.imageExtent = {
			Width,
			Height,
			1};

		vkCmdCopyBufferToImage(CommandBuffer, Buffer, Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &Region);
	}
};

CCommandProcessorFragment_GLBase *CreateVulkanCommandProcessorFragment()
{
	return new CCommandProcessorFragment_Vulkan();
}
