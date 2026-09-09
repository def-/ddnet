# DDNet Teehistorian Replayer

A browser page that replays teehistorian recordings: upload a `.teehistorian`
file, the map is fetched from `https://maps.ddnet.org` automatically, the
recording is converted to a demo in the browser and played back with the
Emscripten build of the DDNet client.

## How it works

1. The page parses the teehistorian header (JSON) to find `map_name` and
   `map_sha256` and downloads the map (`maps.ddnet.org` sends
   `access-control-allow-origin: *`, so this works cross-origin). Recordings
   can be uploaded as local files or streamed directly from a URL: the
   converter reads strictly sequentially and stops downloading as soon as the
   requested time range has been converted, so only the needed prefix of a
   multi-GiB recording is transferred and it is never held in memory.
   `archive-server.py` serves a teehistorian archive this way (e.g.
   `/media/teehistorian2/data`), decompressing `.xz` recordings on the fly and
   sending the required CORS headers, making the whole archive replayable
   without pre-decompressing anything. Note that the stream URL must be
   reachable via https when the page itself is served via https.
2. `teehistorian2demo` (see `src/tools/teehistorian2demo.cpp`), compiled to
   WebAssembly, converts the recording to a `.demo` file with the map embedded.
   Positions are taken directly from the recorded position stream; hooks,
   velocities and angles come from replaying the recorded inputs through the
   shared physics (`CCharacterCore`), snapped to the recorded positions every
   tick so they cannot desync. The recording is parsed in a streaming manner,
   so files larger than memory work. An optional time window (`start`/`end`,
   as seconds, `M:SS` or `H:MM:SS`) limits the conversion; a single
   teehistorian file can cover days of server time and convert to a demo
   roughly 2-3x its size, so the page requires a time window for recordings
   above 400 MiB.
3. The demo is written into the filesystem of the running Emscripten DDNet
   client and played via the file-drop callback, with the full demo player UI
   (seeking, speed, pause).

## Watching ranks

The page has a second mode that plays the demo of a single rank from the map
pages: `index.html?uuid=<game uuid>&time=<seconds>&ts=<finish epoch>&name=<player>`
(repeat `name` for team ranks, `server` overrides the demo service base URL).

The heavy lifting happens server-side on the archive host, because
teehistorian is delta-coded: reaching a run deep into a recording requires
parsing from the start, and the raw recording must not be sent to viewers
(it contains every player's inputs). `record_race.GameID` is the game uuid
and the archive file name, so the recording is found with a handful of
stats. `teehistorian2demo --rank` then finds the exact `player-finish` /
`team-finish` event matching the rank's time and player names (recordings
older than April 2024 have no finish events; the run is located by its
wall-clock offset instead), converts only the run's time window and hides
every other team: their snapshot items and messages are simply never written
to the demo, so the viewer cannot switch to them. The demo starts a few
seconds before the run and the page focuses the camera on the finishing
player via the client's `spectate` console command
(`EmscriptenCallbackConsoleExecute`).

Demos are never published as they come out of the converter: a demo is a
tick-exact recording of the run's inputs and could be turned into a bot that
reproduces the rank. `demo_scramble` (see `src/tools/demo_scramble.cpp`) noises
every demo first, under a key derived from a secret that never leaves the
archive host. The key of a demo is the same every time it is scrambled: two
publications of one run under two keys average out to the run itself. It moves
the tees
by two of the 32 units a tile is wide sideways and one in height (the client
reads the ground under a tee from its position), turns the aim of every tee,
the hook it flies and the shots it fires by up to 1.1 degrees, publishes the
velocity of a moving tee as the difference of the positions it publishes
rather than the one that was recorded, and holds every change of a movement
key back by up to two ticks, every second hook by one tick and every change of
the jump bits by one tick. That stays below what a viewer can
see and is far above what the physics tolerate, so replaying the recorded
inputs drifts off within seconds.

What it does not do is hide the run. The trajectory is what makes the demo
worth watching, and the physics can be run backwards from a trajectory, so
someone who writes a solver that follows the demo tick by tick can still work
out inputs that reproduce it. The scrambling stops the demo from being an
input recording, it does not stop the map from being solved by someone with
the demo in front of them. The key is 128 bits, no field is published
finer than the physics rounds it, and no quantity is published twice through
two different noises, because both of those hand the noise back: a red team
recovered a 32 bit key from eight samples of a velocity that was published
finer than a whole unit, in one second, and undid every channel exactly, and
the velocity as recorded was a second view of the trajectory that gave back
96 % of the true per tick movement. What stays readable is the tick a
weapon was fired in, which the projectile carries, and everything the
trajectory itself gives away.

- `archive-server.py` serves `/rankdemo` and `/rankmeta` (conversion on
  demand, cached with LRU pruning).
- `top-ranks.py` (database host) dumps the #1 solo and team rank of every map
  as a jsonl manifest.
- `pregen.py` (archive host) pre-generates the demos for the manifest so the
  map page links never wait for a conversion, and writes `watchable.jsonl`
  with the outcome per rank, which decides which ranks get a watch link. A
  rank that was published once stays in the manifest and on the web host even
  after it is beaten, so a link that was shared keeps working, and its time
  stays clickable on the map page next to the newer record.

A link is `/watch/?uuid=<game uuid>`, and the page reads the rest of the run
out of `watchable.jsonl`. One recording can hold more than one published rank
(a team rank and a solo rank of the same game, or two runs of a long server
session), and the link then carries the rank time as well.

Players who joined the server before the recording started only have their
name in earlier recordings (until the 2024-04 `player-name` chunks); the
service follows the `prev_game_uuid` chain (recorded since 2023-08) and
seeds names from previous recordings via `--prev`. Ranks from recordings
older than 2023-08 whose player joined on an earlier map cannot be located
and stay unwatchable.

## Checking the controls

The replay controls have to fit two rows on a phone with nothing overlapping
and the playback controls centered in the bar. `check-controls.js` measures
that at the widths phones and desktops actually have, against the file or
against the deployment:

```sh
npm install playwright && npx playwright install chromium
NODE_PATH=$(npm root) node check-controls.js index.html
```

## Building

Follow `docs/BUILDING-emscripten.md` for the emsdk setup, then:

```sh
mkdir build-wasm && cd build-wasm
emcmake cmake .. -G "Unix Makefiles" -DVIDEORECORDER=OFF -DVULKAN=OFF -DSERVER=OFF -DTOOLS=ON -DPREFER_BUNDLED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make DDNet teehistorian2demo -j$(nproc)
```

## Deploying

Copy these files into one directory on the web server:

- `other/teehistorian-replayer/index.html` (the watch page) and
  `other/teehistorian-replayer/list.html` (the overview)
- `build-wasm/DDNet.js`, `build-wasm/DDNet.wasm`, `build-wasm/DDNet.data`
- `build-wasm/teehistorian2demo.js`, `build-wasm/teehistorian2demo.wasm`

The client uses pthreads (SharedArrayBuffer), so the page must be served with
cross-origin isolation headers:

```
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy: same-origin
```

For local testing `other/emscripten/server.py` sets these headers:

```sh
python3 ../other/emscripten/server.py 8080  # in the deploy/build directory
```

## Limitations

- Plain chat is not part of teehistorian recordings (only chat commands like
  `/team`, `/pause` are, via console command chunks) and therefore cannot be
  shown.
- The recording holds the map, the positions and the inputs, nothing the
  server built on top of them. The world around the run is replayed by running
  the server's own code on the map: pickups, doors, freeze lasers, draggers,
  plasma and shotgun turrets, the projectiles and lasers of every shot, the
  switchers, the tune zones and the tiles that freeze or take a weapon away.
  Where that leaves gaps:
  - Weapons are fired once per tick, while the server fires once per input it
    receives, so a player sending several inputs in one tick loses shots.
  - A weapon that teleports (tele gun tiles) is not modelled, and a projectile
    entering a teleporter takes its first exit instead of a random one: the
    server's prng cannot be reconstructed from the recording.
  - Settings a player chooses (ninja jetpack) are not in the recording, so the
    defaults are assumed.
- Hooks are re-simulated from inputs and can be slightly off around teleporters
  and switch doors.
- `.teehistorian.xz` files must be decompressed before uploading.
- 0.7 (sixup) players may show up without name/skin if the recording lacks
  their translated start info; DDNet servers additionally record
  `teehistorian-player-name` chunks which are used as fallback.
