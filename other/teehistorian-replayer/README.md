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

## Building

Follow `docs/BUILDING-emscripten.md` for the emsdk setup, then:

```sh
mkdir build-wasm && cd build-wasm
emcmake cmake .. -G "Unix Makefiles" -DVIDEORECORDER=OFF -DVULKAN=OFF -DSERVER=OFF -DTOOLS=ON -DPREFER_BUNDLED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make DDNet teehistorian2demo -j$(nproc)
```

## Deploying

Copy these files into one directory on the web server:

- `other/teehistorian-replayer/index.html`
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
- Weapon switches are reconstructed from inputs (wanted weapon), server-forced
  weapon changes (e.g. shields, ninja) are not visible.
- Freeze, weapon pickups, projectiles and other server-side entities are not
  part of the recording and are not reconstructed.
- Hooks are re-simulated from inputs and can be slightly off around teleporters
  and switch doors.
- `.teehistorian.xz` files must be decompressed before uploading.
- 0.7 (sixup) players may show up without name/skin if the recording lacks
  their translated start info; DDNet servers additionally record
  `teehistorian-player-name` chunks which are used as fallback.
