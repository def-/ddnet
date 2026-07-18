# Issue #12155: UI color ingame does not match UI color in the main menu

No code change made. This is intended (if arguably confusing) behavior inherited
from vanilla Teeworlds, and changing it as the issue requests is a design
decision / feature request, not a mechanical bug fix.

## What the code does

`ui_color` (`g_Config.m_UiColor`, "Interface color") is consumed in
`src/game/client/components/menus.cpp`:

- **Outgame (main menu):** `RenderBackground()` fills the screen with
  `ms_GuiColor.WithAlpha(1.0f)` — the setting colors the *background*, and
  alpha is deliberately discarded (there is nothing behind the background to
  blend with). The menu panels themselves use the hard-coded
  `ms_ColorTabbarInactiveOutgame` / `ms_ColorTabbarActiveOutgame` /
  `ms_ColorTabbarHoverOutgame` colors (fixed black/white translucents set in
  `UpdateColors()`), plus ~50 more directly hard-coded
  `ColorRGBA(0, 0, 0, 0.x)` panel colors throughout the menus code.

- **Ingame:** `UpdateColors()` derives `ms_ColorTabbarInactiveIngame` /
  `ms_ColorTabbarActiveIngame` from `ms_GuiColor`, including its alpha, so the
  setting tints the menu panels over the game view.

This exactly matches the reported "current behavior" — it is how the feature
was designed (the scheme dates back to vanilla Teeworlds).

## Why no fix

Making the main-menu UI follow `ui_color` (the issue's "expected behavior")
would require:

1. Retheming the outgame menu: the panels are not driven by one variable but by
   dozens of hard-coded colors across `menus*.cpp` — a broad visual change.
2. Deciding what colors the background instead — the reporter explicitly wants
   UI color and background color to be *separately* configurable, i.e. a new
   setting (feature request).
3. Changing the default main-menu appearance for all users.

All three need a maintainer/design decision; there is no small "correct" patch
that fixes the inconsistency without redesigning the menu theming. The reporter
themselves asks "Is this intended?" — the accurate answer is yes.
