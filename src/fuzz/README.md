# Coverage-guided fuzzing of the DDNet server

Round 3 fuzzed the net protocol by firing UDP at a whole server process: 1.33M packets
across five fuzzers, a server restart per segment, no coverage feedback — and it came back
clean. These targets run the same parsers **in-process under libFuzzer**, which is a
different regime: `fz_unpack_packet` does **86 000 exec/s**, i.e. it exceeds round 3's
entire packet volume every 15 seconds, with coverage guidance steering the input.

The complementary half is `../fuzz/fuzz_session.py`, which fuzzes the *stateful* side —
session lifecycles across several clients. That split is deliberate: round 3's only
memory-safety bug came from protocol **state confusion**, not from malformed bytes.

## Why Homebrew LLVM

Apple clang ships no libFuzzer runtime — linking `-fsanitize=fuzzer` fails with
`libclang_rt.fuzzer_osx.a not found`. Everything here is built with
`/opt/homebrew/opt/llvm/bin/clang++`.

## Setup

```sh
# a source tree you can dirty; do not use the one you develop in
cp -Rc ~/ddnet-source /path/to/ddnet-src        # APFS clone, near-instant
cd /path/to/ddnet-src

cmake -B build-fuzz -GNinja \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DCLIENT=OFF -DSERVER=ON -DTOOLS=ON -DDOWNLOAD_GTEST=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -g -O1" \
  -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -g -O1" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
ninja -C build-fuzz twping        # builds the instrumented object set

DDNET_FUZZ_SRC=/path/to/ddnet-src /path/to/tools/libfuzzer/build.sh
```

`build.sh` links each target by reusing `twping`'s link line with our object swapped in —
the same trick `../build_fakeclient.sh` uses, so no CMakeLists changes are needed.

## Targets

| target | covers |
|---|---|
| `fz_unpack_packet` | `CNetBase::UnpackPacket` — header parse, size/flag checks, huffman, for both 0.6 and 0.7. Byte 0 of the input picks the protocol |
| `fz_snap_delta` | `CSnapshotDelta::CreateDelta` + `UnpackDelta`, over two snapshots built by the **real** `CSnapshotBuilder` so both are structurally valid. Also a **correctness oracle** — see below |
| `fz_unpack_msg` | `CNetObjHandler::SecureUnpackMsg` for both 0.6 and 0.7 — the funnel every client message passes through |
| `fz_savetee` | `CSaveTee::FromString` / `CSaveTeam::FromString`, the save-code parser |
| `fz_console` | `CConsole::ExecuteLine` under `CFGFLAG_CHAT` — the chat/rcon command parser. **Found C2** |
| `fz_sevensix` | `CTeeInfo::ToSixup`/`FromSixup` and the generated 0.6↔0.7 translation tables |
| `fz_chunk` | `CPacketChunkUnpacker::UnpackNextChunk` — the chunk layer, both header splits |
| `fz_nameban` | `CNameBans::IsBanned` and the confusables/skeleton machinery under it |
| `fz_fuzzystr` | `sqlstr::EscapeLike` on an exactly-sized destination, plus `sqlstr::FuzzyString` on a NUL-terminated exactly-sized buffer. The terminator is not optional: an earlier version left it off, which broke the callers' precondition and produced the one-byte over-read filed as C3 — a false positive, since the loop's `if(!pString[i]) break;` already covers the lookahead |
| `fz_gamemsg` | `CGameContext::OnMessage` against a **real** `CGameContext` — the whole server message dispatch, including `PreProcessMsg` (0.7→0.6 translation) and `Whisper` |
| `fz_netserver` | `CNetServer::Recv` — the unauthenticated packet ingress path, driven by real datagrams on loopback |
| `fz_serverpkt` | `CServer::ProcessClientPacket` — the server's own system-message layer: password check, rcon auth, map-download chunk index, the input ack, and the session state transitions |
| `fz_tiles` | the tile handlers — `CCharacter::HandleTiles`, `HandleSkippableTiles`, `CGameControllerDDNet::HandleCharacterTiles`. Writes the map's own layer arrays from the input and places the tee in them, instead of trying to drive a tee there |

The four whole-server targets link against the **server** object set rather than twping's, so
they must stub `IsInterrupted()` (which `main.cpp` defined). `build.sh` picks the right link line per target.

## Oracles beat crash-only targets

`fz_snap_delta` originally only watched for faults, which means it would sail straight past
*silent* corruption — and C1 was exactly that on a release build. It now asserts the
round-trip contract the server relies on (`server.cpp:1113`, `if(DeltaSize)`):

* `DeltaSize == 0` → "nothing changed" and nothing is sent, so `pFrom` must equal `pTo`
* `DeltaSize > 0` → `UnpackDelta` must succeed **and reproduce `pTo`**

Two things this immediately caught, neither of which a crash-only target could see:

1. **The C1 fix is incomplete.** The same-key/different-size case now returns `-206`/`-207`:
   the server emits a delta its own decoder rejects. Written up in FINDINGS.md under C1.
2. Two harness bugs of my own — feeding `DeltaSize == 0` to `UnpackDelta` (the API means
   "send nothing"), and deduping snapshot items on the *requested* key when sixup mode
   rewrites the type through `Obj_SixToSeven` before storing it.

Comparison is **by key, order-insensitive**. `CSnapshotBuilder::Finish` does not sort, and
`UnpackDelta` rebuilds as `[surviving pFrom items][delta items]`, so a perfectly correct
result routinely has a different item order. A `memcmp` oracle fires on every reorder and
buries the real signal.

Deliberately **not** covered here: the chunk layer (`CPacketChunkUnpacker`). It needs a
live `CNetConnection` and `FeedPacket` has `dbg_assert`s on its preconditions, so feeding
it raw fuzz output would abort on harness misuse rather than on a defect. It is exercised
with real connection state by `../fuzz/fuzz_session.py`.

## UBSan must be made FATAL

UBSan is compiled in, but by default it only **prints and continues** — libFuzzer runs
straight past the error and records nothing, so a campaign can report "clean" while
tripping undefined behaviour on every input. The builds therefore pass
`-fno-sanitize-recover=undefined`, and it is worth also setting

```sh
export UBSAN_OPTIONS=halt_on_error=1:abort_on_error=1:print_stacktrace=1
```

Verified with a deliberate canary (`float`→`int` overflow, the same shape as the server's
known UB): default config printed `runtime error` and kept fuzzing; with the flag it aborts
and libFuzzer records a crash.

**The AFL full-server build has no UBSan at all** (`-fsanitize=address` only). Adding it is
worthwhile but not free: the known float→int sites (`collision.cpp:911/912`, `math.h:19`,
`gamecontext.cpp:451/486`, `player.cpp:376/401`) fire constantly and would swamp the
campaign, so it needs a suppressions file for those specific sites — not a blanket
`-fno-sanitize=float-cast-overflow`, which would discard a real bug class.

## The three server targets, and why they are layered

They cover three consecutive layers of the same path, and the boundaries are deliberate:

| target | entry | what only it can reach |
|---|---|---|
| `fz_netserver` | `CNetServer::Recv` | pre-connection handling: tokens, `OnPreConnMsg`, `TryAcceptClient`, ban checks. **No auth at all** |
| `fz_serverpkt` | `CServer::ProcessClientPacket` | the password check, rcon auth, map download, `m_LastAckedSnapshot`, `READY`/`ENTERGAME` transitions |
| `fz_gamemsg` | `CGameContext::OnMessage` | the game handlers, `PreProcessMsg`, `Whisper`, the world |

Measured edges: **13 100** (`fz_gamemsg`), **7 200** (`fz_serverpkt`), 560 (`fz_netserver`),
against ~800 for the component console harness.

Two design choices carry most of that:

**Sequences, not single messages.** A game server is a pile of state machines: `/team` then
`/practice` then `/save`, call-vote then vote, pause, kill, timeout. None of it is reachable
one message at a time, because each message is judged against what the previous ones left
behind. One input therefore encodes a sequence, and the tick is advanced between records so
time-dependent code (spam protection, vote timers, respawn delays) progresses. Adding this
took `fz_gamemsg` from 9 334 to 11 927 edges on its own.

**Write the tiles, do not travel to them.** `fz_tiles` exists because `HandleTiles` is the
single biggest uncovered block in the server and no amount of packet mutation reaches it: a
tee has to physically cross a tile before its handler runs, and sixteen records of network
input do not walk one across a map. So that target inverts the problem — it writes a 5×5 patch
of every map layer (game, front, tele, tune, switch, speedup) straight from the fuzz input,
re-runs `CCollision::Init` so the derived state matches, drops the tee in the middle with a
chosen velocity and input, and ticks. Five minutes from a single random seed reached
**853/1912 edges in `HandleTiles`**, against **552/1912** for the `fz_serverpkt` corpus grown
over a ten hour campaign.

Two things keep it honest. Every byte written is a byte a real `.map` can contain, and maps
are attacker-supplied (a server downloads one from a vote, a client from the server), so a
crash here is a crash a hand-made map causes. And `CCollision::Init` is re-run after every
repaint, because it is what derives the teleporter target lists, the highest switch number and
the door tiles from the tiles — painting a tele-in whose number has no tele-out *without*
re-deriving would be a crash no map can cause. `CGameWorld::Init` goes with it, since the
switcher vector is sized from `m_HighestSwitchNumber`.

The per-input tick budget is 8, measured rather than guessed: ticking is where the time goes
(an input costs 33 ms at 32 ticks, of which painting and re-deriving are under 1 ms together),
and at 8 it runs 4.6× faster for 5% less coverage in the same wall clock. At the velocities
the input can set, a tee crosses the whole patch well inside 8 ticks; beyond that it is flying
through the unmodified map, which is what `fz_serverpkt` already does.

**A snapshot oracle, not just crash detection.** Faults *inside* a message handler are the
easy case. The damage a hostile message does is usually **stored** — a bad position, a
dangling id, an inconsistent team — and only surfaces when the server serialises that state
for other clients. So each iteration ends by running the real `CServer::DoSnapshot`, which
is precisely where C1 manifested. The acked tick that picks the delta base is taken from the
fuzz input, exactly as `ProcessClientPacket` takes it from `NETMSG_INPUT` with no check that
the server ever sent that tick.

`m_CurrentGameTick` is protected in `IServer` with no setter, and `CServer::Run` — which
normally advances it — is never called. The fixture reaches it by forming a
pointer-to-member through a derived class, which is legal for a protected base member, and
then applying it to the real object. That is well-defined, unlike `static_cast`ing the
server pointer to a type the object is not.

## A score backend, or scoreworker stays cold

`CGameContext::OnInit` constructs `CScore` against the `DbPool`, but `CServer::Run`
registers the database — and these targets never call `Run`, so without help the pool has no
connection and every rank, `/top5`, `/times`, name-lookup and save/load path returns
immediately. The fixture registers a SQLite database **before `OnInit`**, and the file name
carries the pid because campaign workers run concurrently and would otherwise contend on one
database. Confirmed live by the tables `CScoreWorker` creates: `record_race`,
`record_teamrace`, `record_maps`, `record_saves`, `record_points` and their backups.

## Standing up a real server in-process

`fz_gamemsg` is the one that matters most: it reaches **~9 300 edges** where the component
console harness reaches ~800, because it drives the real dispatch with a live world,
players, controller and score backend. Construction follows `src/test/gameworld_test.cpp`,
which is the project's own recipe, with four deviations that are all load-bearing:

1. **`CreateStorage` needs `argv`.** It asserts "Expected at least one argument", so the
   whole setup lives in `LLVMFuzzerInitialize`, which is handed libFuzzer's own argv.
2. **The working directory must contain `data/`.** Storage resolves `$DATADIR` relative to
   the CWD and `LoadMap` needs `data/maps/coverage.map`; from anywhere else it aborts. The
   campaign script symlinks `data` into each target's log directory rather than sharing one
   CWD, because libFuzzer writes `fuzz-<job>.log` into the CWD and two targets sharing one
   would overwrite each other's logs.
3. **The net server must be opened.** `gameworld_test` never sends; a message harness does,
   on nearly every input. Opening it on loopback gives the queued responses a valid socket,
   and the peer addresses stay zeroed so nothing leaves the machine.
4. **Static teardown must be skipped** (`atexit(_Exit)`). The harness deliberately never
   runs the server's shutdown sequence, so at exit the async logger joins an
   already-joined thread and ASan aborts — recorded as a crash on the *empty input*, which
   stops the campaign for a reason that has nothing to do with the target.

`fz_netserver` has one non-obvious requirement: **bind to a fixed port, not 0.**
`CNetServer::Address()` reports the address it was given rather than the one the OS bound,
so with port 0 every datagram goes to port 0 and is dropped. That fails silently — the
harness still runs, it just never reaches the server. It also has to wait for the datagram
to land (`net_socket_read_wait`) before draining, or the send and the processing straddle
iterations and libFuzzer attributes each input's coverage to the *next* input. Both
mistakes were made and measured here:

| | edges | corpus after 10M execs |
|---|---|---|
| port 0, no wait | 79 | 2 entries |
| fixed port + wait | **558** | **45 entries** |

## Build defines must match the project

`build.sh` extracts the `-D` flags from the project's own compile command rather than
hardcoding them. This is not cosmetic. A Debug build defines `_GLIBCXX_DEBUG`, which makes
`std::vector` into `std::__debug::vector` — a different type with a different layout. A
harness built without it hands the library a differently-shaped object, and the symptoms
look exactly like target defects:

* a **stack-buffer-overflow inside `vector::begin()`** when the harness constructs a
  `CNameBans` the library then reads, and
* a **`-fsanitize=function` type mismatch** on a callback whose signature involves
  `std::vector`, which cost an earlier session a working `fz_console` victim callback that
  was removed as "harness disagreeing with the build".

Both vanished once the defines matched. Extract them; do not guess them. Note the extractor
must match a **C++** compile — the first `-c` command in the ninja output is a C dependency
that carries none of the project's defines, and picking it silently yields an empty list.

## Running

One script does everything — minify the corpus, run every target, report, and keep the
corpus for next time:

```sh
src/fuzz/run.sh              # all targets, 1 hour
src/fuzz/run.sh -t 600       # ten minutes
src/fuzz/run.sh -b           # rebuild the harnesses first
src/fuzz/run.sh fz_serverpkt # one target
src/fuzz/run.sh -t 300 -j 2 fz_gamemsg fz_console
```

It prints a live line every 30 s and a summary at the end:

```
minifying corpora (keeping only what preserves coverage)
target                 before      after
fz_chunk                  415        116
fz_snap_delta             826        187

[  30s left, 6 procs] savetee=232 chunk=260 snap_delta=544

================================ summary ================================
target                   cov       ft    corpus artifacts
fz_savetee               232     1445       263         0
fz_chunk                 260      954       116         0
```

State lives in one directory (`~/ddnet-fuzz`, override with `FUZZ_RUN`) and is **reused**:
`corpus/<target>`, `artifacts/<target>`, `log/<target>.log`, and `repro/<target>/<artifact>`
for any artifact the summary managed to turn into a runnable sequence.

**Minify, or every run gets slower than the last.** A libFuzzer corpus grows without bound
— measured here at 230 MB for `fz_gamemsg` after a few hours — and the whole corpus is
re-executed at startup. `run.sh` runs `-merge=1` before every campaign, which keeps only the
inputs needed to preserve coverage; in the run above that was 415 → 116 entries with
identical coverage. The checked-in seeds are re-added afterwards, because a merge will drop
a seed that adds no *new* edge even though it is the thing documenting a reachable shape.

**What a seed is for: an ORDER, not a vocabulary.** The `chatcmd_*` seeds already list every
team command and the campaign fires all of them, yet the team machinery sits at a few percent
of its edges — `CGameTeams::ProcessSaveTeam` is entered thousands of times for 5 of its 376
edges. The commands are not what is missing, the order that gets past their preconditions is:
`/team 1` before `/save`, and `/save` never while the team is in practice mode, because it
refuses that outright. The four `seq_team_*` seeds spell those orders out. They are verified
to *work* — the server logs both tees joining team 1, and the commands execute — but see the
warning below before claiming a coverage number for them.

Two traps in writing more of them. A command needs the tee to exist, so the sequence has to
tick before its first `/team`, and the async half of a save only advances while ticks keep
coming, which is what the per-input tick budget is for. And slot 1 is the **0.7** client in
`fz_gamemsg`: the same chat line needs the 0.7 `Cl_Say` id and a `(mode, target, text)` body,
not the 0.6 id and `(team, text)`. A seed that forgets this is silently a no-op, which is how
the first draft of these was written and it cost an afternoon to notice.

**Always pass `-seed=1` to a `-print_coverage=1` replay.** Without it libFuzzer shuffles the
corpus before executing it, and because these targets are stateful the order changes the
result: three identical replays of the *same* corpus gave 797, 803 and 821 covered functions,
with `CGameTeams::ProcessSaveTeam` swinging between 5/376 and 53/376. That is not a property
of the input, it is the shuffle. With `-seed=1`, `fz_tiles` reproduces exactly (379 functions,
four runs out of four).

A second, smaller source survives that on the two targets that register a database:
`fz_gamemsg` with `-seed=1` still moves by ±3 functions, because `CDbConnectionPool`'s
constructor starts a worker thread and whether an async save or load result lands inside the
run is a race. `fz_tiles` builds its fixture with `WithSqlite=false` for exactly this reason.

So: a difference of a few functions or a few edges between two replays means nothing. Judge a
harness change with a **campaign A/B** instead — two runs of the same duration from the same
starting corpus, side by side, compared on libFuzzer's own accumulated `cov:` counter.

**The summary replays every artifact for you**, and prints the reproduce command, the input
(a printable preview plus base64, so it can be recreated anywhere) and the report the replay
produced. That is deliberate: an artifact is worth nothing until it has been replayed, and
having to fetch it off the machine that ran the campaign first is what used to stop that
from happening.

**The artifact kinds do not mean the same thing.** `crash-`, `leak-`, `oom-` and `timeout-`
are failures and have to fail again. `slow-unit-` is **not a failure**: libFuzzer saves one
when a single execution exceeds `-report_slow_units` (10 s), so it exits 0 on replay by
definition, and the exit code says nothing. Judge it on the two per-execution costs the
summary measures instead — the same input run 20 times and then a few thousand times, in one
process. The fixtures keep server state across inputs on purpose, so what hides behind a slow
unit is state that *grows* every input until some per-input scan turns quadratic, and that
shows as a per-execution cost that climbs. A flat one is not a finding: libFuzzer times a
unit by wall clock, and on a box running every target at once a unit that is descheduled long
enough crosses 10 s on its own. The first one this campaign produced was exactly that —
3 ms standalone against the 10 s that saved it, and 7.2 → 1.9 µs/exec from 20 to 5000
repetitions, so nothing was growing. Do not go looking for the campaign's own `Slow unit:`
line either: in `-fork` mode the child prints it into a temp directory libFuzzer deletes on
exit, and the parent never echoes it, so `log/<target>.log` has no trace of it.

**An artifact that should fail and does not is not a reproducer**, because the state it
needed came from inputs that ran before it in the same process. libFuzzer will run several
*file* arguments back to back in one process without resetting anything between them (hand it
a *directory* instead and it starts fuzzing), so the summary replays the whole corpus and
then the artifact, halves the prefix down while a half still reproduces, and stores what
survives under `repro/<target>/<artifact>/` as numbered inputs plus a `replay.sh` that it
then runs from there to prove the stored copy works. That is the reproducer to check in. When
it does not reproduce even with the whole corpus in front of it, the input that built the
state was never saved — a corpus only keeps what added coverage — and there is nothing to
store; the summary says so rather than guessing. In this project a state-dependent artifact
has meant a defect in the harness as often as one in the server.

## Results so far

* **`fz_snap_delta` rediscovers the known `CreateDelta` out-of-bounds read in seconds**,
  blind, with the identical stack to the one it took a multi-client cross-protocol takeover
  to find out-of-process:

  ```
  ERROR: AddressSanitizer: heap-buffer-overflow  READ of size 4
      #0 CSnapshotDelta::DiffItem   snapshot.cpp:238
      #1 CSnapshotDelta::CreateDelta snapshot.cpp:355
  0 bytes after 16-byte region
  ```

  `crash-createdelta-oob` is that input. Treat it as the target's **self-test**: if it ever
  stops reproducing, the bug was fixed upstream — pick a new canary rather than deleting
  the check.
* **`fz_console` found C2 blind**, from an empty corpus, in under ten minutes: a reachable
  `dbg_assert` in `CConsole::CResult::GetVictim()` that kills a **Release** server with
  SIGILL, plus an unauthenticated validation bypass. Full write-up in FINDINGS.md.
* **`fz_snap_delta` with the oracle: 16.7 M executions, zero round-trip violations** once the
  known `-206`/`-207` case is classified rather than trapped.
* **`fz_sevensix`: 67.1 M executions at ~86 000/s, clean.** No memory-safety or UB finding in
  `CTeeInfo` conversion or the translation tables; the tables are bounds-guarded inline
  lookups and the harness drives them over the full int range including `INT_MIN`.
* **`fz_unpack_msg`: 49.0 M executions at ~162 700/s, clean.**
* **`fz_unpack_packet`: 10 510 141 executions in 121 s, clean.** No ASan, no UBSan. That is
  ~8× round 3's entire out-of-process packet volume, in two minutes, and it corroborates the
  earlier negative far more strongly than the UDP campaigns did.

## Known false crashes — check these before filing anything

The in-process fixture creates its client slots directly rather than through a real socket
handshake, so **their peer address is all zeroes**. `net_addr_str` ends in
`dbg_assert_failed("unknown NETADDR type %d")`, so *any* server path that formats a client's
address aborts — and that is a property of the harness, not of DDNet, because a real slot
always carries the address of the accepted connection.

If a stack contains `net_addr_str`, it is almost certainly this. Seen so far:

| reached via | why it is not a bug |
|---|---|
| `OnNetMsgRconAuth` → `BanAddr` → `CNetBan::MakeBanInfo` | the target sets `sv_rcon_bantime 0` to take the drop path instead |
| rcon `unmute <index>` → `CMutes::UnmuteIndex` | the index *is* validated; it aborts formatting the zeroed address of a mute that legitimately exists |

The real fix is to give the fixture genuinely accepted connections, the way `fz_netserver`
now does with a vanilla handshake — that would also un-stub the send path and make the
address-keyed mute and ban logic real. It is the largest remaining item in this suite.

## Fixture limitations that look like findings

`fz_serverpkt` reported a `dbg_assert` in `net_addr_str` on its first run, reached from
`OnNetMsgRconAuth` → `BanAddr` → `CNetBan::MakeBanInfo`. It is **not** a defect: after
`sv_rcon_max_tries` wrong passwords the server bans `ClientAddr(ClientId)`, and the
fixture's slots have no real connection, so their peer address is zeroed and `net_addr_str`
asserts on the unknown type. A real slot always carries the address of the accepted
connection. `CNetServer::m_aSlots` is private, so the fixture cannot install one; the target
sets `sv_rcon_bantime 0` to take the drop path instead. **`CNetBan` therefore still needs
its own target** — it is a self-contained class that takes client-derived strings, so it is
cheap to write.

## Well-supported negatives

Recorded so the next session does not spend budget re-deriving them.

* **`CPacketChunkUnpacker::UnpackNextChunk` — no runaway read.** The `// TODO: add checking
  here so we don't read too far` above the chunk-skip walk looks alarming, and
  `CNetChunkHeader::Unpack` does read 2–3 bytes before the caller's only bounds check. But
  the check `if(pData + Header.m_Size > pEnd)` sits **before** both `continue` paths and
  before `return true`, so every chunk that a later iteration skips over was already
  validated to fit. `pData` therefore never passes `pEnd` by more than one header, and that
  lands inside `m_aExtraData`, the adjacent member of the same struct — intra-object, which
  ASan does not detect and which cannot fault or escape. 7.5M executions, clean.
* **`CNetServer::OnPreConnMsg` — bounded despite the unchecked length.** `Unpacker.Reset(
  pData, h.m_Size)` really does use 10 attacker bits that are never compared against
  `m_DataSize`, but `pData` is at most `m_aChunkData + 3` and `h.m_Size` is at most 1023,
  so the range stays inside the 1397-byte `m_aChunkData`. The consequence is reading stale
  bytes of the previous packet, which the attacker supplied anyway — a correctness wart,
  not a memory-safety bug. Covered by `fz_netserver` regardless.
* **`CConsoleNetConnection::Recv` (econ) — bounded by inspection.** `Update` refuses to read
  when `sizeof(m_aBuffer) <= m_BufferOffset` and caps the read at the remaining space; both
  the start and end walks in `Recv` are bounded by `m_BufferOffset`; the copy out is guarded
  by `MaxLength`; and both `mem_move`s use offsets that are provably `<= m_BufferOffset`.
  Econ is also **off by default** (`ec_port` and `ec_password` must both be set). Not worth
  a socketpair harness ahead of anything above.
* **Not server-reachable at all:** `CServerInfo2::FromJson` and `CStun` (client-only — the
  server has no references), demo playback and teehistorian (the server only *writes*),
  `CTuningParams` (no client message sets it), and map/datafile loading (map bytes come from
  disk; the only client-driven map change is an admin-defined vote option).

## Adding a target

Copy either `.cpp`, add its name to the `TARGETS` default in `build.sh`. Two rules that
keep results honest:

1. **Model what the server can actually reach.** `fz_unpack_packet` truncates its input at
   `NET_MAX_PACKETSIZE` because `net_udp_recv` does; without that you "find" overflows the
   network stack makes impossible. `fz_console` gates on `str_utf8_check` for the same
   reason — `CUnpacker::GetString` rejects invalid UTF-8, so a crash reached with raw bytes
   is not a crash a client can cause. It produced exactly one such false lead; see the
   dismissed variant under C2 in FINDINGS.md.
1b. **Honour the callee's contract, or you report yourself.** `fz_console`'s first two
   crashes were the harness calling `GetVictim()` on commands whose format declares no `v`,
   which asserts by design. Real callbacks only call it for `v` commands; the harness now
   does the same, and it also installs a `ClientsForVictim` equivalent so `me`/`all` resolve
   the way they do on a server.
2. **Build structured inputs with the real builders.** `fz_snap_delta` constructs both
   snapshots through `CSnapshotBuilder` and checks `IsValid()` before calling `CreateDelta`,
   so a finding is about `CreateDelta` rather than about garbage bytes.
