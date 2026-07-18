# Issue #12128: Failed to swap framebuffers. Try to update your GPU drivers.

No code change made.

## Findings

- The report is a Vulkan "device lost" crash when opening the scoreboard (TAB),
  surfaced through the generic `GFX_ERROR_TYPE_SWAP_FAILED` path in
  `src/engine/client/backend_sdl.cpp` (`HandleError`). That error message is
  only the client reporting that the GPU/driver died; it does not point at a
  specific DDNet rendering defect, and nothing in the scoreboard rendering code
  (`src/game/client/components/scoreboard.cpp`) shows an obvious degenerate
  draw that would explain a driver crash.
- The issue thread contains the actual diagnosis: maintainer Robyt3 identified
  from the attached dumps that the reporter is on the latest AMD Radeon driver
  2.0.388, "for which multiple bugs with other games have apparently already
  been reported", and advised downgrading the driver. AssassinTee noted the
  reporter is on Windows 10.0.26100 (24H2), which is known for widespread
  driver issues.
- The issue is labeled `3rd party` and `Vulkan` by the maintainers, is still
  open awaiting the reporter's feedback after a driver downgrade, and has no
  linked PR.

## Conclusion

This is a third-party GPU driver fault, not a reproducible DDNet bug. There is
no identified defect in the DDNet code to fix, and any speculative change to
the scoreboard or Vulkan backend would be a guess. Per the maintainers'
triage, the resolution lies with the AMD driver / OS update, so the correct
outcome here is no code change.
