#!/usr/bin/env python3
# Pre-generates rank demos for a top-ranks.jsonl manifest (from top-ranks.py
# on the database host) so the watch page never waits for a conversion. Writes
# a watchable.jsonl with the outcome for every rank, which decides which ranks
# get watch links on the map pages. Runs on the archive host, safe to re-run:
# already cached demos are skipped instantly.
#
# The manifest holds several rank candidates per map and kind. Recordings of
# old runs are often gone, so the candidates are converted in rank order until
# --ranks of them worked.
#
# Usage: pregen.py top-ranks.jsonl watchable.jsonl

import argparse
import concurrent.futures
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rankdemo import Converter, DiskFullError, RankDemoError

parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument("output")
parser.add_argument("--root", default="/media/teehistorian/data")
parser.add_argument("--tool", default=str(pathlib.Path.home() / "git/ddnet/build-tools/teehistorian2demo"))
parser.add_argument("--cache", default=str(pathlib.Path.home() / "teehistorian-demos"))
parser.add_argument("--cache-limit-gb", type=float, default=40)
parser.add_argument("--ranks", type=int, default=1, help="ranks to publish per map and kind")
parser.add_argument("--jobs", type=int, default=4,
    help="conversions in flight at once, the archive disk answers several readers faster than one")
parser.add_argument("--reconvert-map", action="append", default=[], metavar="MAP",
    help="convert the ranks of this map again (repeatable), for a tool fix that concerns a few maps")
parser.add_argument("--reconvert-before", metavar="DATE",
    help="make the demos of published ranks that finished before this date (YYYY-MM-DD) again, "
        "for a fix that only changes recordings of that age")
parser.add_argument("--reconvert", action="store_true",
    help="convert every published rank again, to bring demos made by an older converter up to date")
parser.add_argument("--partial", action="store_true",
    help="the manifest is a slice of the whole one (sync-ranks.py --maps and --runs), so the cache keeps "
        "the demos of every other map instead of being pruned to what this output names")
parser.add_argument("--retry-failed", action="store_true",
    help="retry ranks whose conversion failed in the previous output (by default only \"not in the archive\" failures are retried, e.g. after a tool fix)")
args = parser.parse_args()

converter = Converter(args.tool, args.root, args.cache, int(args.cache_limit_gb * 1024**3))

RECONVERT_BEFORE = datetime.datetime.strptime(args.reconvert_before, "%Y-%m-%d").timestamp() \
    if args.reconvert_before else None


def remake(entry):
    """Whether a rank that is published already is converted again"""
    return args.reconvert or entry["map"] in args.reconvert_map or \
        (RECONVERT_BEFORE is not None and (entry.get("ts") or 0) < RECONVERT_BEFORE)


# What a run of pregen decides about a rank, the rest of a line comes from the
# manifest and is refreshed on every run
OUTCOME_FIELDS = ("status", "code", "message", "cid", "team", "finishers", "demo", "rev")


def outcome(entry):
    return {field: entry[field] for field in OUTCOME_FIELDS if field in entry}


def ok_result(demo_path, meta):
    # finish_cids are the players of the team that were there at the finish,
    # which is who the rank belongs to (names are not unique)
    return {"status": "ok", "cid": meta["cid"], "team": meta["team"],
        "finishers": meta.get("finish_cids", []), "demo": demo_path.name,
        "rev": meta.get("rev", "")}


def generate(entry, reconvert=False):
    try:
        demo_path, meta = converter.convert(entry["uuid"], entry["time"], entry["names"], entry.get("ts"), reconvert,
            entry.get("recording"), entry.get("aliases"))
        return ok_result(demo_path, meta)
    except DiskFullError:
        raise  # not the rank's failure, the run stops
    except RankDemoError as error:
        # The finish can sit outside the scan window around ts (DST-ambiguous
        # timestamps, servers whose tick fell far behind wall-clock time), so
        # retry with a full scan of the recording.
        if error.status == 404 and entry.get("ts") and "No finish" in str(error):
            try:
                demo_path, meta = converter.convert(entry["uuid"], entry["time"], entry["names"], None,
                    recording_uuid=entry.get("recording"), aliases=entry.get("aliases"))
                return ok_result(demo_path, meta)
            except DiskFullError:
                raise
            except RankDemoError as retry_error:
                # The first attempt knew the rank's timestamp and says more
                if "No finish" not in str(retry_error):
                    error = retry_error
        return {"status": "error", "code": error.status, "message": str(error)}
    except Exception as error:  # one corrupt recording must not end the run
        return {"status": "error", "code": 500, "message": f"{type(error).__name__}: {error}"}


def entry_key(entry):
    # The recording is part of the key: a rank that the manifest points at
    # another recording than last time is converted again
    return json.dumps([entry.get(field) for field in ("uuid", "time", "ts", "names", "recording")])


def run_key(entry):
    # The run a rank is of: the recording it is in, its time and when it
    # finished. The members of a team finish share all three, and a bot that
    # replays one run under changing names gets the same time to the
    # hundredth every time but a different timestamp. A team rank saved
    # under a stale game id names the recording separately.
    return entry.get("recording", entry["uuid"]), entry["time"], entry.get("ts")


def main():
    # Conversions that failed on an archived recording fail the same way every
    # night (a full scan each, the expensive class), carry them forward.
    # "Not in the archive" is retried: the archive syncs daily.
    # A rank that was published once is not converted again, its demo is
    # there. It stays published only while it is one of the wanted ranks.
    previous = {}
    previous_aliases = {}
    published = {}
    try:
        with open(args.output) as previous_output:
            for line in previous_output:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("status") == "ok":
                    published[entry_key(entry)] = entry
                elif not args.retry_failed and "not in the archive" not in entry.get("message", ""):
                    previous[entry_key(entry)] = outcome(entry)
                    previous_aliases[entry_key(entry)] = entry.get("aliases")
    except OSError:
        pass

    groups = {}
    with open(args.manifest) as manifest:
        for line in manifest:
            # The candidate list is written by an ssh that can die mid-line,
            # and one torn line must not cost the whole run
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            groups.setdefault((entry["map"], entry["kind"]), []).append(entry)

    # One pass over the archive indexes, so the candidates whose recording is
    # long gone are answered from memory instead of a stat in every location
    # directory
    uuids = {entry.get("recording", entry["uuid"]) for entries in groups.values() for entry in entries}
    print(f"{converter.load_index(uuids)} of {len(uuids)} recordings in the archive index", file=sys.stderr, flush=True)

    ok = errors = 0
    written = set()
    written_ok = set()
    # Team groups first: a solo rank of a member of a team run carries the same
    # time and is the same run, the loop below hands it the team run's demo
    # once it has seen the team run. A solo rank of another run in the same
    # recording is a demo of its own.
    # The other way round as well: a team rank that was not among the wanted
    # ranks of its map is still linked when a member's solo rank was, and it
    # is made as a team demo, which shows the whole team.
    groups = dict(sorted(groups.items(), key=lambda item: item[0][1] != "team"))
    team_runs = {}
    unwanted_teams = {}
    # The demo every run has, by map and run, for the ranks of the run that
    # get no demo of their own
    run_demos = {}
    flat = [(map_name, kind, entry) for (map_name, kind), entries in groups.items() for entry in entries]

    # Whether a candidate is converted, taken from the last run or skipped is
    # decided in order below, but the conversions themselves run a few
    # candidates ahead on a pool: a conversion is mostly waiting for the
    # archive disk, and several readers get more out of it than one. A
    # conversion the order then turns out not to need is simply not read.
    # A failed rank is tried again when the manifest names its players by
    # names it did not know last time
    def carried(entry):
        key = entry_key(entry)
        return key in previous and previous_aliases[key] == entry.get("aliases")

    def work_of(entry):
        key = entry_key(entry)
        if key in published:
            return "reconvert" if remake(entry) else None
        if carried(entry):
            return None
        return "generate"

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs))
    futures = {}
    # One candidate of a group at a time: the next one is only worth
    # converting when this one fails, and mostly it does not. Groups run
    # side by side.
    inflight_groups = set()
    submitted = 0

    def submit_ahead(upto):
        nonlocal submitted
        while submitted < min(upto, len(flat)):
            map_name, kind, entry = flat[submitted]
            work = work_of(entry)
            if work is not None:
                if (map_name, kind) in inflight_groups:
                    return
                futures[submitted] = pool.submit(generate, entry, work == "reconvert")
                inflight_groups.add((map_name, kind))
            submitted += 1

    with open(args.output + ".new", "w") as output:
        for index, (map_name, kind, entry) in enumerate(flat):
            submit_ahead(index + 2 * max(1, args.jobs))
            future = futures.pop(index, None)
            if future is not None:
                inflight_groups.discard((map_name, kind))
            if index == 0 or flat[index - 1][:2] != (map_name, kind):
                group_wanted = args.ranks
            # A run a report names is published whatever rank it holds, and it
            # does not take a place away from the map's own ranks
            if group_wanted == 0 and not entry.get("extra"):
                if future is not None:
                    future.cancel()
                # A rank that was published and has been beaten since keeps the
                # demo it was published with, and with it the links that were
                # shared for it. The demo exists, so this costs nothing, and
                # the pass below writes the rank out again.
                published_entry = published.get(entry_key(entry))
                if published_entry is not None:
                    run_demos.setdefault((map_name, run_key(entry)), outcome(published_entry))
                if kind == "team":
                    unwanted_teams.setdefault(map_name, {})[run_key(entry)] = entry
                continue
            key = entry_key(entry)
            team_result = team_runs.get(map_name, {}).get(run_key(entry))
            team_entry = unwanted_teams.get(map_name, {}).pop(run_key(entry), None) if kind == "solo" else None
            if team_entry is not None and team_result is None:
                team_key = entry_key(team_entry)
                if team_key in published:
                    result = outcome(published[team_key])
                else:
                    result = generate(team_entry)
                if result["status"] == "ok":
                    team_result = result
                    team_runs.setdefault(map_name, {})[run_key(team_entry)] = result
                    run_demos.setdefault((map_name, run_key(team_entry)), result)
                    written.add(team_key)
                    written_ok.add(team_key)
                    output.write(json.dumps({**team_entry, **result}, ensure_ascii=False) + "\n")
                    output.flush()
                else:
                    print(f"{map_name} (team #{team_entry.get('rank', '?')}): {result['message']}", file=sys.stderr, flush=True)
            if kind == "solo" and team_result is not None:
                # The solo rank of a member of a team run links the team run's
                # demo, which is the same run, instead of keeping a copy of its
                # own from an earlier conversion. It is one of the wanted ranks
                # all the same, otherwise every member of every team run of the
                # map has its team converted, 16 demos for Adrenaline 5.
                if future is not None:
                    future.cancel()
                group_wanted -= 1
                written.add(key)
                written_ok.add(key)
                output.write(json.dumps({**entry, **team_result}, ensure_ascii=False) + "\n")
                output.flush()
                continue
            published_entry = published.get(key)
            if published_entry and remake(entry):
                # A demo that is already published is made again. It stays
                # as it is when the recording is gone or the converter
                # fails, there is nothing to make it from then. A recording
                # that IS there and no longer yields the run, or yields one
                # that does not show it (422), means the demo that was
                # published is of something else, and it goes.
                result = future.result() if future is not None else generate(entry, reconvert=True)
                if result["status"] != "ok" and "No finish" not in result["message"] and result.get("code") != 422:
                    print(f"{map_name} ({kind} #{entry.get('rank', '?')}): kept the published demo: "
                        f"{result['message']}", file=sys.stderr, flush=True)
                    result = outcome(published_entry)
                elif result["status"] != "ok":
                    print(f"{map_name} ({kind} #{entry.get('rank', '?')}): dropped the published demo, "
                        f"the recording no longer yields this run: {result['message']}", file=sys.stderr, flush=True)
            elif published_entry:
                result = outcome(published_entry)
            elif carried(entry):
                result = previous[key]
            else:
                result = future.result() if future is not None else generate(entry)
            if result["status"] == "ok":
                ok += 1
                if not entry.get("extra"):
                    group_wanted -= 1
                if kind == "team":
                    team_runs.setdefault(map_name, {})[run_key(entry)] = result
                run_demos.setdefault((map_name, run_key(entry)), result)
                written_ok.add(key)
            else:
                errors += 1
                print(f"{map_name} ({kind} #{entry.get('rank', '?')}): {result['message']}", file=sys.stderr, flush=True)
            written.add(key)
            output.write(json.dumps({**entry, **result}, ensure_ascii=False) + "\n")
            output.flush()
        pool.shutdown(wait=False, cancel_futures=True)
        # Every rank of a run links the run's demo, whichever rank it was made
        # for: a team rank whose own conversion failed links a member's solo
        # demo (it shows the whole team as well), the ranks beyond the wanted
        # ones of a run that has a demo are linked too, and so are the ranks
        # that were published before and have been beaten since
        linked = 0
        for (map_name, kind), entries in groups.items():
            for entry in entries:
                key = entry_key(entry)
                result = run_demos.get((map_name, run_key(entry)))
                if key not in written_ok and result is not None:
                    linked += 1
                    written.add(key)
                    written_ok.add(key)
                    output.write(json.dumps({**entry, **result}, ensure_ascii=False) + "\n")
                    output.flush()
        # A rank that was published once keeps its line for good, whether it
        # has been beaten, has fallen out of the candidate list or is no
        # longer a rank at all: its demo is on the web host and the link that
        # was shared for it has to go on working. "kept" says the line is
        # there for the link alone and not because the map still counts the
        # rank among its candidates. sync-ranks.py drops the line when the
        # demo really is gone from both hosts.
        kept = 0
        for key, entry in published.items():
            if key not in written:
                kept += 1
                output.write(json.dumps({**entry, "kept": True}, ensure_ascii=False) + "\n")
    pathlib.Path(args.output + ".new").replace(args.output)
    print(f"{ok} demos ready, {linked} further ranks link them, {kept} kept for their links, "
        f"{errors} candidates failed", file=sys.stderr)
    # Only a run over the whole manifest knows which demos nothing names any
    # more. A slice of it names a handful and would take the rest of the cache
    # with it.
    if not args.partial:
        converter.drop_unnamed({entry.get("demo") for entry in map(json.loads, open(args.output, encoding="utf-8"))})


if __name__ == "__main__":
    try:
        main()
    except DiskFullError as error:
        # The output so far stays as .new, the demos it made are in the cache
        # and the next run picks them up without converting them again
        sys.exit(f"stopped: {error}")
