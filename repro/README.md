# Reproducer: `VK_ERROR_DEVICE_LOST` / renderer freeze from OOM-recovery re-entrancy

This branch (`vk-oom-repro`) contains a small fault injection and a runner that
deterministically reproduces a real bug in the Vulkan backend that leads to a
frozen renderer and, on many drivers, `VK_ERROR_DEVICE_LOST`.

> This branch is a **reproducer, not a fix, and is not meant to be merged.**
> The `DBG_VK_FORCE_OOM_WORKER` injection exists only to trigger the buggy path
> on demand instead of waiting for real GPU-memory pressure.

## The bug

`CCommandProcessorFragment_Vulkan::AllocateVulkanMemory()` recovers from a failed
`vkAllocateMemory` by driving the **entire frame loop**:

```cpp
// backend_vulkan.cpp
if(Res == VK_ERROR_OUT_OF_HOST_MEMORY || Res == VK_ERROR_OUT_OF_DEVICE_MEMORY)
{
    vkDeviceWaitIdle(m_VKDevice);
    for(size_t i = 0; i < m_SwapChainImageCount + 1; ++i)
        if(!NextFrame())                // WaitFrame() + PrepareFrame()
            return false;
    Res = vkAllocateMemory(...);
}
```

`NextFrame()` → `WaitFrame()` calls `FinishRenderThreads()` and then
`vkQueueSubmit` / (via `PrepareFrame`) `vkAcquireNextImageKHR` /
`vkQueuePresentKHR`. All of that is only valid on the **main render thread**.

But `AllocateVulkanMemory()` also runs on render **worker** threads. With the
default `gfx_render_thread_count` (`3`, clamped to `>= 3`, i.e. two workers) the
first vertex/uniform render on a worker allocates its stream buffer:

```
RunThread(worker)
  -> Cmd_Render -> RenderStandard
    -> CreateStreamVertexBuffer -> CreateStreamBuffer -> CreateBuffer
      -> AllocateVulkanMemory        <- allocation fails here
```

When the allocation fails on a worker, the recovery runs `FinishRenderThreads()`
**from that worker**. `FinishRenderThreads()` locks each render thread's mutex
and waits for `!m_IsRendering`. The worker is *currently rendering* and already
holds its own mutex (taken at the top of `RunThread`), so:

* it re-locks a mutex it already holds → **deadlock**, and
* it would wait on its own `m_IsRendering` flag, which can never clear.

The renderer freezes. On drivers where the lock ordering doesn't deadlock the
same way, the worker instead runs `vkDeviceWaitIdle` / `vkQueueSubmit` /
`vkQueuePresentKHR` concurrently with the main thread on the same
un-externally-synchronized `VkQueue` → **`VK_ERROR_DEVICE_LOST`**.

This matches the field reports: intermittent freezes ("crash after ~1 minute")
and `VK_ERROR_DEVICE_LOST` ("Failed to swap framebuffers") on modern GPUs, which
have plenty of VRAM but still hit allocation limits via fragmentation, memory
budgets, integrated-GPU shared memory, or `maxMemoryAllocationCount`.

## Evidence

`sample-deadlock.txt` is a `sample(1)` capture of the frozen process. The
key stacks (line numbers are from this branch):

```
# Render WORKER thread — OOM recovery re-entered the frame loop and deadlocked:
RunThread(unsigned long)                         backend_vulkan.cpp:7783
  RenderStandard<GL_SVertex, false>(...)         backend_vulkan.cpp:3531
    CreateStreamBuffer<...>(...)                 backend_vulkan.cpp:6462
      CreateBuffer(...)                          backend_vulkan.cpp:5825
        AllocateVulkanMemory(...)                backend_vulkan.cpp:1659   <- recovery
          WaitFrame()                            backend_vulkan.cpp:2303
            FinishRenderThreads()                backend_vulkan.cpp:2258
              std::mutex::lock()
                __psynch_mutexwait               <- blocked forever

# Main command-processor thread — also blocked in FinishRenderThreads:
CCommandProcessor_SDL_GL::RunBuffer(...)         backend_sdl.cpp:382
  ... WaitFrame() -> FinishRenderThreads() -> std::mutex::lock -> __psynch_mutexwait
```

Two threads parked in `__psynch_mutexwait`, the second worker idle in
`__psynch_cvwait` — a fully deadlocked renderer.

## How to build (macOS / MoltenVK)

The Vulkan backend isn't in the default macOS build. This is how `build-vk/`
was produced on an Apple-silicon Mac with Homebrew `molten-vk`:

```sh
# Link MoltenVK directly (not the Vulkan loader) so the portability driver is
# used without the loader hiding it. Homebrew's pkg-config would otherwise make
# CMake pick the loader, so hide vulkan.pc just for the configure step:
PC=/opt/homebrew/lib/pkgconfig/vulkan.pc
mv "$PC" "$PC.bak"
cmake -S . -B build-vk -G Ninja -DVULKAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDEV=ON
mv "$PC.bak" "$PC"

ninja -C build-vk DDNet
```

Verify it links MoltenVK directly:

```sh
otool -L build-vk/DDNet | grep -i moltenvk
```

## How to run

```sh
./repro/repro_vk_oom_device_lost.sh
```

The script copies the compiled shaders next to the assets, launches the client
with `DBG_VK_FORCE_OOM_WORKER=1`, waits for the menu to render on worker
threads, samples the process and prints a verdict. Expected output:

```
injection fired on a worker thread : YES
renderer deadlocked in recovery    : YES
REPRODUCED: OOM recovery on a render worker thread deadlocked the renderer.
```

Negative control: run without the env var and the client renders normally.

## Suggested fix (not applied here)

Allocation-failure recovery must never drive the frame loop off the main render
thread. Options:

* From a worker thread, just fail the allocation (`SetError(... OUT_OF_MEMORY
  ...)` / return `false`) and let the main thread recover on the next frame, or
* Drop the aggressive `NextFrame()` recovery entirely and fail cleanly.

Pair this with the `VK_EXT_device_fault` diagnostics commit so genuine driver
faults can be told apart from this app-side bug in the wild.
