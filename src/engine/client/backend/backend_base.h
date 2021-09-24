#ifndef ENGINE_CLIENT_BACKEND_BACKEND_BASE_H
#define ENGINE_CLIENT_BACKEND_BACKEND_BASE_H

#include "../backend_sdl.h"

class CCommandProcessorFragment_GLBase
{
protected:
	static int TexFormatToImageColorChannelCount(int TexFormat);
	static void *Resize(int Width, int Height, int NewWidth, int NewHeight, int Format, const unsigned char *pData);

	static bool Texture2DTo3D(void *pImageBuffer, int ImageWidth, int ImageHeight, int ImageColorChannelCount, int SplitCountWidth, int SplitCountHeight, void *pTarget3DImageData, int &Target3DImageWidth, int &Target3DImageHeight);

public:
	virtual ~CCommandProcessorFragment_GLBase() = default;
	virtual bool RunCommand(const CCommandBuffer::SCommand *pBaseCommand) = 0;

	enum
	{
		CMD_INIT = CCommandBuffer::CMDGROUP_PLATFORM_GL,
		CMD_SHUTDOWN = CMD_INIT + 1,
	};

	struct SCommand_Init : public CCommandBuffer::SCommand
	{
		SCommand_Init() :
			SCommand(CMD_INIT) {}

		SDL_Window *m_pWindow;
		uint32_t m_Width;
		uint32_t m_Height;

		class IStorage *m_pStorage;
		std::atomic<uint64_t> *m_pTextureMemoryUsage;
		SBackendCapabilites *m_pCapabilities;
		int *m_pInitError;

		const char **m_pErrStringPtr;

		char *m_pVendorString;
		char *m_pVersionString;
		char *m_pRendererString;

		int m_RequestedMajor;
		int m_RequestedMinor;
		int m_RequestedPatch;

		EBackendType m_RequestedBackend;

		int m_GlewMajor;
		int m_GlewMinor;
		int m_GlewPatch;
	};

	struct SCommand_Shutdown : public CCommandBuffer::SCommand
	{
		SCommand_Shutdown() :
			SCommand(CMD_SHUTDOWN) {}
	};
};

#endif
