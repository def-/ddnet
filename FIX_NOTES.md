# Issue #12288: ShowDistance sends double the size — no code change

## What the bug is

`CNetMsg_Cl_ShowDistance` (`show-distance@netmsg.ddnet.tw`) carries the client's
**full** visible screen size in world units: the client fills `m_X`/`m_Y` from
`IGraphics::CalcScreenParams()` (src/game/client/gameclient.cpp:2401), which
returns full width/height (~1430×804 at zoom 1, 16:9).

The server stores this verbatim (`OnShowDistanceNetMessage`,
src/game/server/gamecontext.cpp:2800) and then compares it against the
distance from the view center — a **half**-extent — in
`NetworkClipped()` (src/game/server/entity.cpp:101,105). So entities are
networked in an area 2× the screen size in each dimension. This has been the
behavior since the message was introduced in 2020 (commit 1782d95d9); commit
23c145d76 later removed an additional client-side 1.25× factor but kept the
full-size semantics. Every deployed client sends full size; every deployed
server interprets it as a half-extent.

## Why I made no change

The issue is labeled **Needs Discussion** upstream and doesn't state the
desired resolution; every candidate fix changes protocol semantics or gameplay
behavior in a way that needs a maintainer decision:

1. **Client sends half the size instead.** Breaks new clients against every
   existing server: clipping would happen exactly at the visible edge, with no
   slack for the dyncam camera offset, entity sprite size, or interpolation —
   visible pop-in.

2. **Server halves the received value.** Cuts the range for all existing
   clients to exactly the visible extent. This breaks dyncam users: the server
   clips relative to the character position (`m_ViewPos`,
   src/game/server/player.cpp:240), but the client camera center can be offset
   from the character by up to `350 * zoom` (src/game/client/components/camera.cpp:266).
   The reach needed from the character is roughly `half_extent + 350*zoom`:
   in x that's ~1065×zoom (vs. halved range 715×zoom), in y ~752×zoom (vs.
   402×zoom). Notably, in the y direction the current "doubled" value
   (804×zoom) is only ~6% above what max-offset dyncam actually needs — the
   current behavior is not simply 2× too generous, and the correct margin
   formula (fixed vs. zoom/aspect-proportional, whether to use
   `m_CameraInfo`) is exactly the open design question.

3. **New message / renamed fields** (e.g. carry the actual per-axis distance,
   or rename to width/height). A protocol addition — maintainer territory.

Related code that would need to move in lockstep with any semantic change:
the spectator-count heuristic divides the stored value by 2.5/2.3 to
approximate the visible area (src/game/server/player.cpp:420), and
`NetworkClippedLine` uses `max(x, y)` of it (src/game/server/entity.cpp:126).
The default `m_ShowDistance = vec2(1200, 800)` (src/game/server/player.cpp:120,
used for clients that never send the message, e.g. vanilla 0.6) is already in
half-extent-with-slack units, i.e. a *different* unit than the message payload —
another symptom of the same confusion.

## Suggested direction (for the discussion, not implemented)

The only side that can change without breaking cross-version behavior is the
server's interpretation, since all clients (old and new) send the same full
size. A server-only fix would convert to a half-extent plus an explicit
margin covering the dyncam offset, e.g. clip at
`size/2 + 350 * (size derived zoom scale)` per axis, and adjust
player.cpp:420 accordingly — but picking that margin (and whether to instead
derive the range from `Cl_CameraInfo`) is the decision the issue's
"Needs Discussion" label is asking for.
