#!/usr/bin/env bash
#
# Reproducer for the VK_ERROR_DEVICE_LOST / renderer-freeze bug caused by the
# OOM-recovery path in CCommandProcessorFragment_Vulkan::AllocateVulkanMemory().
#
# Mechanism (see repro/README.md for the full write-up):
#   AllocateVulkanMemory()'s allocation-failure recovery drives the whole frame
#   loop (vkDeviceWaitIdle + NextFrame -> WaitFrame -> FinishRenderThreads ->
#   vkQueueSubmit / vkQueuePresentKHR). That is only valid on the MAIN render
#   thread. With the default gfx_render_thread_count (>= 3) the failing
#   allocation can happen on a render WORKER thread, where FinishRenderThreads()
#   waits on the worker executing it and re-locks that worker's own mutex ->
#   deadlock. On drivers that slip past the deadlock, the worker issues queue
#   submit/present concurrently with the main thread -> VK_ERROR_DEVICE_LOST.
#
# This script relies on the DBG_VK_FORCE_OOM_WORKER fault injection that is
# compiled into the backend on the `vk-oom-repro` branch. It forces the first
# allocation made on a worker thread to report VK_ERROR_OUT_OF_DEVICE_MEMORY.
#
# macOS + MoltenVK. See README.md for how the build under build-vk/ is produced.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/build-vk/DDNet"
RUN_SECONDS="${RUN_SECONDS:-8}"
LOG="${REPO_ROOT}/repro/repro-run.log"
SAMPLE="${REPO_ROOT}/repro/repro-run.sample.txt"

# MoltenVK ICD (only needed if DDNet is linked against the Vulkan loader; a
# MoltenVK-direct build ignores it). Adjust the path if your molten-vk differs.
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
	ICD="$(ls /opt/homebrew/Cellar/molten-vk/*/etc/vulkan/icd.d/MoltenVK_icd.json 2>/dev/null | head -1)"
	[ -n "${ICD}" ] && export VK_ICD_FILENAMES="${ICD}"
fi

if [ ! -x "${BIN}" ]; then
	echo "error: ${BIN} not found. Build it first (see repro/README.md)." >&2
	exit 2
fi

# The client loads precompiled SPIR-V from data/shader/vulkan/*.spv. The build
# emits them under build-vk/data/shader/vulkan; make sure the client can find
# them next to the rest of the assets in the repo-root data/ tree.
mkdir -p "${REPO_ROOT}/data/shader/vulkan"
cp -n "${REPO_ROOT}"/build-vk/data/shader/vulkan/*.spv "${REPO_ROOT}/data/shader/vulkan/" 2>/dev/null || true

rm -f "${LOG}" "${SAMPLE}"

echo "== launching DDNet with DBG_VK_FORCE_OOM_WORKER=1 (Vulkan backend) =="
cd "${REPO_ROOT}"
DBG_VK_FORCE_OOM_WORKER=1 "${BIN}" \
	"gfx_backend Vulkan; gfx_fullscreen 0; gfx_screen_width 800; gfx_screen_height 600; snd_enable 0" \
	> "${LOG}" 2>&1 &
PID=$!
echo "pid ${PID}; waiting ${RUN_SECONDS}s for the menu to render on worker threads ..."
sleep "${RUN_SECONDS}"

FIRED=0
grep -q "\[repro\] forcing VK_ERROR" "${LOG}" && FIRED=1

if ! kill -0 "${PID}" 2>/dev/null; then
	echo "RESULT: process exited (no hang). Injection fired: ${FIRED}."
	echo "        See ${LOG}."
	exit 1
fi

echo "== process still alive; sampling stacks =="
sample "${PID}" 2 -file "${SAMPLE}" >/dev/null 2>&1
kill -9 "${PID}" 2>/dev/null

# Verdict: a render worker is blocked with AllocateVulkanMemory calling into
# FinishRenderThreads (i.e. the recovery re-entered the frame loop off-thread).
DEADLOCK=0
if grep -q "AllocateVulkanMemory" "${SAMPLE}" \
	&& grep -q "FinishRenderThreads" "${SAMPLE}" \
	&& grep -q "__psynch_mutexwait" "${SAMPLE}"; then
	# Confirm AllocateVulkanMemory and FinishRenderThreads appear in the same
	# stack (worker recovery), not just anywhere in the sample.
	if awk '/AllocateVulkanMemory/{a=1} a&&/FinishRenderThreads/{print; exit}' "${SAMPLE}" | grep -q FinishRenderThreads; then
		DEADLOCK=1
	fi
fi

echo
echo "==================== VERDICT ===================="
echo "injection fired on a worker thread : $([ ${FIRED} -eq 1 ] && echo YES || echo NO)"
echo "renderer deadlocked in recovery    : $([ ${DEADLOCK} -eq 1 ] && echo YES || echo NO)"
echo "log    : ${LOG}"
echo "sample : ${SAMPLE}"
echo "================================================="

if [ ${FIRED} -eq 1 ] && [ ${DEADLOCK} -eq 1 ]; then
	echo "REPRODUCED: OOM recovery on a render worker thread deadlocked the renderer."
	exit 0
fi
echo "NOT reproduced as expected — inspect the log and sample above."
exit 1
