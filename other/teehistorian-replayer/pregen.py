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


def generate(entry):
    try:
        demo_path, meta = converter.convert(entry["uuid"], entry["time"], entry["names"], entry.get("ts"))
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
    with open(args.output + ".new", "w") as output:
        for (map_name, kind), entries in groups.items():
            wanted = args.ranks
            for entry in entries:
                if wanted == 0:
                    break
                if kind == "solo" and (entry["uuid"], entry["time"]) in team_runs.get(map_name, set()):
                    continue
                key = entry_key(entry)
                published_entry = published.get(key)
                result = outcome(published_entry) if published_entry else previous.get(key) or generate(entry)
                if result["status"] == "ok":
                    ok += 1
                    wanted -= 1
                    if kind == "team":
                        team_runs.setdefault(map_name, set()).add((entry["uuid"], entry["time"]))
                else:
                    errors += 1
                    print(f"{map_name} ({kind} #{entry.get('rank', '?')}): {result['message']}", file=sys.stderr, flush=True)
                written.add(key)
                output.write(json.dumps({**entry, **result}, ensure_ascii=False) + "\n")
                output.flush()
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
