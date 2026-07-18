# Issue #11930 — AIR_JUMP missing in sixup 0.7→0.6 snapshot translation

## Conclusion: no code change

The missing `COREEVENTFLAG_AIR_JUMP → NETEVENTTYPE_SOUNDWORLD(SOUND_PLAYER_AIRJUMP)`
mapping is not a behavior bug. Adding it as the issue suggests would make the
air-jump sound play **twice** for every non-local player's air jump.

## Why

The DDNet (0.6) client never relies on a server-sent sound event for air jumps.
It derives them client-side from the snapshot character core:

- An air jump sets `m_Jumped |= 3` in the core (`src/game/gamecore.cpp:246-252`),
  and `m_Jumped` is a networked field in both the 0.6 and 0.7
  `CNetObj_CharacterCore` (`datasrc/network.py:194`, `datasrc/seven/network.py:163`).
- The client detects the bit-2 transition of `m_Jumped` between consecutive
  snapshots for every player and calls `m_Effects.AirJump()` — which plays
  `SOUND_PLAYER_AIRJUMP` and spawns the particles
  (`src/game/client/gameclient.cpp:2441-2466`, "detect air jump for other
  players"; `src/game/client/components/effects.cpp:26-50`).
- The sixup translation copies the 0.7 character core into the 0.6 item
  unchanged, including `m_Jumped`
  (`src/game/client/sixup_translate_snapshot.cpp:221`), so this detection works
  identically when connected to a 0.7 server.

Consistently, the DDNet 0.6 **server** also sends no sound event for
`COREEVENT_AIR_JUMP` (`src/game/server/entities/character.cpp:908-918` creates
sounds only for GROUND_JUMP and the HOOK_* events) — air jump is the one core
event the 0.6 client handles purely from the snapshot.

The GROUND_JUMP/HOOK_* translations in `sixup_translate_snapshot.cpp` exist
because those sounds normally arrive as server-sent `CNetEvent_SoundWorld`
items in 0.6 and the client has no client-side detection for them for
non-local players. AIR_JUMP does not need (and must not get) the same
treatment: with the suggested mapping, each non-local air jump would be played
once by the translated sound event and a second time by the `m_Jumped`
detection.

## Caveat

The client-side detection uses a `!Grounded`-in-previous-snapshot heuristic to
distinguish air jumps from single-jump ground jumps (which also set bit 2), so
a rare edge case (air jump immediately after leaving the ground within one
snapshot interval) can be missed — but that behavior is identical on native
DDNet 0.6 servers and is inherent to the detection design, not a sixup
translation gap. Answering the issue's question: yes, the omission is
effectively intentional; the correct resolution is a clarifying comment/reply
on the issue, not a code change.
