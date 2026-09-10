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
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rankdemo import Converter, RankDemoError

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
parser.add_argument("--reconvert", action="store_true",
    help="convert every published rank again, to bring demos made by an older converter up to date")
parser.add_argument("--retry-failed", action="store_true",
    help="retry ranks whose conversion failed in the previous output (by default only \"not in the archive\" failures are retried, e.g. after a tool fix)")
args = parser.parse_args()

converter = Converter(args.tool, args.root, args.cache, int(args.cache_limit_gb * 1024**3))


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
        demo_path, meta = converter.convert(entry["uuid"], entry["time"], entry["names"], entry.get("ts"), reconvert)
        return ok_result(demo_path, meta)
    except RankDemoError as error:
        # The finish can sit outside the scan window around ts (DST-ambiguous
        # timestamps, servers whose tick fell far behind wall-clock time), so
        # retry with a full scan of the recording.
        if error.status == 404 and entry.get("ts") and "No finish" in str(error):
            try:
                demo_path, meta = converter.convert(entry["uuid"], entry["time"], entry["names"], None)
                return ok_result(demo_path, meta)
            except RankDemoError as retry_error:
                error = retry_error
        return {"status": "error", "code": error.status, "message": str(error)}
    except Exception as error:  # one corrupt recording must not end the run
        return {"status": "error", "code": 500, "message": f"{type(error).__name__}: {error}"}


def entry_key(entry):
    return json.dumps([entry.get(field) for field in ("uuid", "time", "ts", "names")])


def main():
    # Conversions that failed on an archived recording fail the same way every
    # night (a full scan each, the expensive class), carry them forward.
    # "Not in the archive" is retried: the archive syncs daily.
    # Ranks that were published once are carried forward as they are, whether
    # or not they are still the best rank of their map: a link that was shared
    # keeps working after the rank was beaten.
    previous = {}
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
    except OSError:
        pass

    groups = {}
    with open(args.manifest) as manifest:
        for line in manifest:
            entry = json.loads(line)
            groups.setdefault((entry["map"], entry["kind"]), []).append(entry)

    # One pass over the archive indexes, so the candidates whose recording is
    # long gone are answered from memory instead of a stat in every location
    # directory
    uuids = {entry["uuid"] for entries in groups.values() for entry in entries}
    print(f"{converter.load_index(uuids)} of {len(uuids)} recordings in the archive index", file=sys.stderr, flush=True)

    ok = errors = 0
    written = set()
    # Team groups first: a solo rank of a member of a team run carries the same
    # time and would be a second copy of that demo, which the loop below skips
    # by having seen the team run already
    groups = dict(sorted(groups.items(), key=lambda item: item[0][1] != "team"))
    # A team run's demo already contains the whole team, so the solo rank of a
    # member, which carries the same time, would be a second copy of the same
    # file. A solo rank of another run in the same recording is a demo of its
    # own and is kept.
    team_runs = {}
    flat = [(map_name, kind, entry) for (map_name, kind), entries in groups.items() for entry in entries]

    # Whether a candidate is converted, taken from the last run or skipped is
    # decided in order below, but the conversions themselves run a few
    # candidates ahead on a pool: a conversion is mostly waiting for the
    # archive disk, and several readers get more out of it than one. A
    # conversion the order then turns out not to need is simply not read.
    def work_of(entry):
        key = entry_key(entry)
        if key in published:
            return "reconvert" if args.reconvert else None
        if key in previous:
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
            if group_wanted == 0 or (kind == "solo" and (entry["uuid"], entry["time"]) in team_runs.get(map_name, set())):
                if future is not None:
                    future.cancel()
                continue
            key = entry_key(entry)
            published_entry = published.get(key)
            if published_entry and args.reconvert:
                # A demo that is already published is made again. It stays
                # as it is when the recording is gone, there is nothing to
                # make it from then. A recording that IS there and no
                # longer yields the run means the demo that was published
                # is of something else, and it goes.
                result = future.result() if future is not None else generate(entry, reconvert=True)
                if result["status"] != "ok" and "not in the archive" in result["message"]:
                    print(f"{map_name} ({kind} #{entry.get('rank', '?')}): kept the published demo, "
                        f"its recording is gone: {result['message']}", file=sys.stderr, flush=True)
                    result = outcome(published_entry)
                elif result["status"] != "ok":
                    print(f"{map_name} ({kind} #{entry.get('rank', '?')}): dropped the published demo, "
                        f"the recording no longer yields this run: {result['message']}", file=sys.stderr, flush=True)
            elif published_entry:
                result = outcome(published_entry)
            elif key in previous:
                result = previous[key]
            else:
                result = future.result() if future is not None else generate(entry)
            if result["status"] == "ok":
                ok += 1
                group_wanted -= 1
                if kind == "team":
                    team_runs.setdefault(map_name, set()).add((entry["uuid"], entry["time"]))
            else:
                errors += 1
                print(f"{map_name} ({kind} #{entry.get('rank', '?')}): {result['message']}", file=sys.stderr, flush=True)
            written.add(key)
            output.write(json.dumps({**entry, **result}, ensure_ascii=False) + "\n")
            output.flush()
        pool.shutdown(wait=False, cancel_futures=True)
        # The ranks of earlier runs that no map still names, their demos are
        # on the web host and their links are out there
        kept = 0
        for key, entry in published.items():
            if key not in written:
                kept += 1
                output.write(json.dumps(entry, ensure_ascii=False) + "\n")
    pathlib.Path(args.output + ".new").replace(args.output)
    print(f"{ok} demos ready, {kept} kept from earlier runs, {errors} candidates failed", file=sys.stderr)


if __name__ == "__main__":
    main()
