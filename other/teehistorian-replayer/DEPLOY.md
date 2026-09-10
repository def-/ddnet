# Deploying the rank watch feature

The demos of all #1 ranks are pre-generated on the archive host (li) and
served as **static files from ddnet.org**, li is only needed for the nightly
generation, an outage means stale demos instead of a broken feature. All #1
solo + team demos together are ~3 GB (~0.85 MB average per demo).

A test deployment of exactly this layout runs behind an unlisted path while
the feature is being tried out (`list.html` lists all watchable ranks). The
scripts here default to `/var/www/watch/`, so a test deployment needs its
path passed to `sync-ranks.py --target` and to the nginx location below.

## ddnet.org

Static directory `/var/www/watch/` (owned by teeworlds so li can rsync into
it):

- `index.html` (the watch page), `list.html` (the overview of every
  watchable rank), from other/teehistorian-replayer/
- `DDNet.js`, `DDNet.wasm`, `DDNet.data`, `teehistorian2demo.{js,wasm}`,
  emscripten build (see README.md), ~64 MB one-time cacheable page load
- `demos/` + `watchable.jsonl`, rsynced from li nightly

nginx location in `/etc/nginx/shared/ddnet` (the client needs
cross-origin-isolation for SharedArrayBuffer, and mime.types has no wasm
entry). The watch page is the directory index, so a rank link is
`/watch/?uuid=...`, and the overview of everything watchable is
`/watch/list.html`:

```nginx
location /watch/ {
  add_header Cross-Origin-Embedder-Policy require-corp;
  add_header Cross-Origin-Opener-Policy same-origin;
  types {
    application/wasm wasm;
    text/html html;
    application/javascript js;
    application/octet-stream demo;
    text/plain data;
  }
}

# The client is 63 MB and the page asks for it with the build tag of the
# deployment, so it never changes under a name it is asked for. add_header
# does not reach into this location from the one above, so the isolation
# headers the client needs are repeated.
location ~ ^/watch/.+\.(wasm|data|js)$ {
  add_header Cross-Origin-Embedder-Policy require-corp;
  add_header Cross-Origin-Opener-Policy same-origin;
  add_header Cache-Control "public, max-age=31536000, immutable";
  types {
    application/wasm wasm;
    application/javascript js;
    text/plain data;
  }
}

# Cloudflare caches by file extension and .demo is not on its list, .gz is,
# so a demo is asked for under its gzip name and the encoding header lets the
# browser unpack it on the way in. A demo keeps its name when a rank is
# regenerated, so it is cached for a day rather than forever, and the page
# asks for it under the revision the manifest names.
location ~ ^/watch/demos/.+\.demo\.gz$ {
  add_header Content-Encoding gzip;
  add_header Cross-Origin-Embedder-Policy require-corp;
  add_header Cross-Origin-Opener-Policy same-origin;
  add_header Cache-Control "public, max-age=86400";
  default_type application/octet-stream;
}

# Demos of earlier runs are plain files, and gzip_static serves a .demo from
# its .demo.gz when one is there. gunzip answers the rare client that does not
# take gzip, which keeps curl and wget honest.
location ~ ^/watch/demos/ {
  gzip_static always;
  gunzip on;
  gzip_vary on;
  add_header Cross-Origin-Embedder-Policy require-corp;
  add_header Cross-Origin-Opener-Policy same-origin;
  add_header Cache-Control "public, max-age=86400";
  default_type application/octet-stream;
}

location = /watch/watchable.jsonl {
  add_header Cross-Origin-Embedder-Policy require-corp;
  add_header Cross-Origin-Opener-Policy same-origin;
  add_header Cache-Control "no-cache";
  default_type application/json;
}
```

Cloudflare sits in front and caches by file extension. Demos are published as
`.demo.gz`, which is on its list, and measure `cf-cache-status: MISS` then
`HIT` with the run coming back decompressed. `.wasm` and `.data` are not on
the list and answer `DYNAMIC` even with the header above, so a first visit
still pulls 63 MB through the origin (a second visit loads nothing, the
browser has them for a year). A Cache Rule on `ddnet.org/watch/*` with
"eligible for cache" and "respect origin TTL" is what would make Cloudflare
keep those two as well.

Demos are stored gzipped everywhere: in the cache on the archive host, in the
upload and on the web host, which is about 30 % off all three (measured over
25 demos, `gzip -9` leaves 70.4 %, brotli would leave 65.4 % and nginx here
has no brotli module). A regenerated demo keeps its name, so every manifest
entry carries a `rev`, a short hash of the file, that the page appends to the
demo URL: the new demo reaches the browser under a name no cache holds yet.
`sync-ranks.py` takes it from the files it uploads, so a re-scramble of the
cache is published by an upload and needs no manifest run.

Manifest generation, root's crontab (top-ranks.py reads
`/etc/mysql/debian.cnf`, so it has to run as root; it writes into the web
directory because the archive host logs in as teeworlds and cannot read
/root). One query per map and kind, all of them served by an index:

```
30 5 * * * python3 /root/top-ranks.py > /var/www/watch/top-ranks.jsonl.new 2>/dev/null && mv /var/www/watch/top-ranks.jsonl.new /var/www/watch/top-ranks.jsonl
```

It dumps `--candidates` (20) rank rows per map and kind, so the archive host
can walk down them when a recording is missing, and leaves out finishes
younger than `--min-age-days` (14): a fresh rank can still turn out to be
cheated and be deleted again.

The index the pages read is the `record_watch` table (`watch-index.sql`),
filled from the uploaded manifest by `import-watchable.py`, which
`sync-ranks.py` runs over ssh right after the upload. Both live in
`/home/teeworlds/servers/scripts/` next to the pages, together with
`watchlinks.py`, the lookup the pages import. It keeps its own MariaDB
connection and swallows its errors: a page that cannot reach the index shows
the times it always showed.

Clickable times, all four patches against the versions of 2026-09-09:

| patch | file | page |
| --- | --- | --- |
| `ddnet-watch.patch` | `ddnet.py` | the record tables of every page below |
| `maps-mz-watch.patch` | `maps_mz.py` | ddnet.org/maps (Materialize, uwsgi 9033) |
| `maps-watch.patch` | `maps.py` | ddnet.org/maps2 (MariaDB, uwsgi 9003) |
| `ranks-watch.patch` | `ranks.py` | ddnet.org/ranks (static, regenerated) |

`ddnet-watch.patch` gives `printExactSoloRecords`, `printSoloRecords` and
`printTeamRecords` an optional `watch` argument, `{time in milliseconds:
url}`, and prints the times it holds as links. The pages look their map up in
`record_watch` and pass it on, `ranks.py` loads the whole index once because
it writes every map in one run. Roll the maps uwsgi apps after patching, the
rank pages pick it up with their next regeneration.

## Archive host (li)

Converter build (once; redo after updating ~/git/ddnet, branch "pr-replayer").
`demo_scramble` runs on every demo before it is published, the pipeline
refuses to start without it. Its key comes from `scramble.secret` in the cache
directory, written on first use: one run published twice under two keys can be
averaged back to the run itself, so a re-scramble has to reproduce the noise.
Losing the secret costs nothing but a new noise on the next re-scramble.

```sh
cmake -S ~/git/ddnet -B ~/git/ddnet/build-tools -GNinja -DCMAKE_BUILD_TYPE=Release -DCLIENT=OFF -DSERVER=OFF -DTOOLS=ON
ninja -C ~/git/ddnet/build-tools teehistorian2demo demo_scramble
```

Nightly pre-generation + push, crontab of deen (after the 6:00 archive
download). `sync-ranks.py` fetches the manifest, converts what is missing and
uploads it. Re-runs are cheap: cached demos are skipped instantly and failed
conversions are carried forward from the previous watchable.jsonl, use
`--retry-failed` after tool fixes, which makes every failed rank convert
again:

```
0 12 * * * cd ~/git/ddnet/other/teehistorian-replayer && nice -n19 ionice -c3 python3 sync-ranks.py > ~/teehistorian-demos/sync.log 2>&1
```

`--no-generate` uploads the manifest in the cache as it is, for a
pre-generation that was run separately (sharded over several `pregen.py`, or
after `rescramble.py` re-scrambled the cache).

`--ranks N` publishes the N best ranks per map and kind instead of only the
first one. It converts the candidates in rank order and stops after N of them
worked, so a map whose #1 recording is gone still gets its best watchable
rank (the overview and the map pages show which one it is).

(li's `ddnet` ssh alias logs in as teeworlds, which owns the web directory.
The upload deletes demos the manifest no longer names, so beaten and deleted
ranks do not pile up there. The local cache prunes itself by age above
`--cache-limit-gb`, which is why an upload always goes through the manifest
and not through "everything in the cache directory".)

`archive-server.py` (on-demand conversion service) is not part of this
static deployment; it remains useful for local testing and for a possible
later "watch any rank" feature.

## Moderator teehistorian access

`/var/www/bep/teehistorian2demo.php` on the web host converts a whole
recording for a moderator, unscrambled: it is the tool for looking into a
suspicious run and it is behind the moderator login. It pipes the recording
through `/home/teeworlds/bin/teehistorian2demo-pipe`
(`teehistorian2demo-pipe.py` here), a wrapper that reads the
map name and hash out of the recording's own header, downloads the map into
`/var/tmp/th2demo-maps` once and runs the converter on it. The wrapper exists
because the converter needs the map: it replays the entities of the recording,
which the recording itself does not hold.

Recording and demo both stream through the wrapper, so a conversion of a large
recording keeps sending instead of going silent for minutes. The demo is
therefore sent while it is written and its header still says length 0, the
same as the `/dev/stdout` conversion before it. The player reads the length
from the file itself.

`/home/teeworlds/bin/teehistorian2demo` has to be built on a host with the web
host's glibc (ssh dev, Debian 13, `cmake -DCLIENT=OFF -DSERVER=OFF -DTOOLS=ON`),
a build from the archive host does not run there.

## Scale

2424 maps, so `--ranks 1` is up to ~4400 demos (solo + team) at ~1.2 MB each,
a few GB on the web host. The cost is dominated by the recordings: they are xz
files of up to 1.4 GB that have to be decompressed whole (a day of ger10 is
3.6 GB raw, ~6 minutes end to end for one rank), so the first full run takes
many hours. Every later run only touches ranks that are new, were beaten or
previously failed.

The conversion itself replays the map's entities from the first tick of the
recording, because a bouncing shotgun bullet, a rotating freeze laser and the
weapons a player carries all depend on everything that happened before the
run. That is a few seconds per hour of recording on an entity-heavy map.

## Coverage notes

- Ranks are watchable when their `GameID` recording is in the archive.
  Failures per rank are recorded in `watchable.jsonl` (`status`/`message`).
- The archive sync (`/media/teehistorian/download.sh`, daily 6:00) had a gap
  2026-07-13 to 2026-07-26 and the disk is 99% full, so recent ranks stay
  unwatchable until synced.
- Recordings before 2024-04 have no finish events (timestamp approximation
  with wider margins) and before 2023-08 no `prev_game_uuid` (players who
  joined before the recording started cannot be identified, such ranks stay
  unwatchable).

## The moderators' converter

`https://ddnet.org/teehistorian2demo.php` converts whole recordings and single
runs for moderators with `/home/teeworlds/bin/teehistorian2demo`, without
scrambling. It has to be the converter of this checkout, built for the web
host: `deploy-moderator-tool.sh` on the archive host builds it in a Debian 13
container and renames it into place on the web host. Run it after every
converter change, the rank demos and the moderators' demos then come from the
same code.
