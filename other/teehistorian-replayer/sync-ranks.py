#!/usr/bin/env python3
# One command for the rank demo sync, run on the archive host (li): takes the
# rank manifest of the database host, pre-generates the demos of the ranks it
# names and copies them and the manifest to the web host. Safe to re-run,
# everything already converted and already uploaded is skipped.
#
# The candidate list comes from top-ranks.py, which runs on the database host
# (deployed there as watch-candidates.py) and is run at the start of every
# sync: a list from the last nightly is up to a day old, and every rank set
# or deleted since is then missed. Pass --manifest-source - to use an
# existing local file.
#
# --maps refreshes a few maps instead of all of them, which is what the watch
# lane runs when a deleted top rank leaves a map without a demo. --runs does
# the same for the single runs a moderation report links, whatever rank they
# hold and whether or not they have been deleted since.
#
# Usage: sync-ranks.py [--ranks 1] [--maps MAP... | --runs UUID=TIME...] [--dry-run]

import argparse
import fcntl
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--manifest-host", default="ddnet",
    help="the database host, where the candidate list is made")
parser.add_argument("--manifest-script", default="/home/teeworlds/servers/scripts/watch-candidates.py",
    help="the candidate list generator on that host")
parser.add_argument("--manifest-source",
    help="scp source of a ready candidate list instead of making one, - to use --manifest as it is")
parser.add_argument("--manifest", help="where the candidate list is kept (default: <cache>/top-ranks.jsonl)")
parser.add_argument("--cache", default=str(pathlib.Path.home() / "teehistorian-demos"))
parser.add_argument("--target", default="ddnet:/var/www/watch",
    help="ssh destination of the web directory holding demos/ and watchable.jsonl")
parser.add_argument("--ranks", type=int, default=1, help="ranks to publish per map and kind")
parser.add_argument("--runs", nargs="+", metavar="UUID=TIME",
    help="convert these runs and add them to the manifest, for the ranks a moderation report links: they "
        "are published whatever rank they hold and deleted ones are found too")
parser.add_argument("--maps", nargs="+", metavar="MAP",
    help="refresh only these maps, for a top rank that was deleted: their candidates are fetched and "
        "converted and the result is merged into the manifest, minutes instead of the half hour a whole one takes")
parser.add_argument("--import-script", default="/home/teeworlds/servers/scripts/import-watchable.py",
    help="loads the uploaded manifest into the record_watch table on the web host")
parser.add_argument("--retry-failed", action="store_true", help="passed on to pregen.py")
parser.add_argument("--reconvert-map", action="append", default=[], metavar="MAP",
    help="convert the ranks of this map again (repeatable)")
parser.add_argument("--reconvert-before", metavar="DATE",
    help="passed on to pregen.py: redo the published ranks that finished before this date")
parser.add_argument("--reconvert", action="store_true",
    help="passed on to pregen.py, converts every published rank again after a converter fix")
parser.add_argument("--prune", action="store_true",
    help="delete demos of ranks the manifest no longer names, which breaks the links that were shared for them")
parser.add_argument("--dry-run", action="store_true", help="report what would be uploaded and deleted")
parser.add_argument("--no-generate", action="store_true",
    help="upload the manifest as it is, for a pregen that was run separately")
args = parser.parse_args()
# Next to the demos, not in /tmp: the list of the last run is what a shrunken
# fresh one is compared against
args.manifest = args.manifest or str(pathlib.Path(args.cache) / "top-ranks.jsonl")

HERE = pathlib.Path(__file__).resolve().parent
CACHE = pathlib.Path(args.cache)
WATCHABLE = CACHE / "watchable.jsonl"
# What a --maps run works on, kept beside the whole ones so a refresh that
# went wrong can be looked at
PART_MANIFEST = CACHE / "top-ranks-maps.jsonl"
PART_WATCHABLE = CACHE / "watchable-maps.jsonl"
# Whether this run works on a slice of the manifest rather than the whole one
PART = bool(args.maps or args.runs)
HOST, _, REMOTE = args.target.partition(":")


def run(command, **kwargs):
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    return subprocess.run(command, check=True, **kwargs)


def fetch_manifest():
    if args.manifest_source == "-":
        return
    if args.manifest_source:
        run(["scp", "-q", args.manifest_source, args.manifest])
        return
    if args.maps or args.runs:
        # ssh hands the remote shell one string, and a map name has spaces in
        # it. A few hundred lines have nothing for the guard below to compare.
        select = ["--maps"] + args.maps if args.maps else ["--runs"] + args.runs
        remote = " ".join(shlex.quote(part)
            for part in ["python3", args.manifest_script] + select)
        with open(PART_MANIFEST, "w") as out:
            run(["ssh", args.manifest_host, remote], stdout=out)
        return
    fresh = args.manifest + ".new"
    with open(fresh, "w") as out:
        run(["ssh", args.manifest_host, "python3", args.manifest_script], stdout=out)
    lines = sum(1 for _ in open(fresh, encoding="utf-8", errors="replace"))
    before = sum(1 for _ in open(args.manifest, encoding="utf-8", errors="replace")) \
        if pathlib.Path(args.manifest).exists() else 0
    # A query that fails halfway must not shrink the list: the ranks it leaves
    # out would lose their demo and the links that were shared for them
    if lines < 1000 or lines < before * 0.9:
        sys.exit(f"the fresh candidate list has {lines} lines, the one before had {before}, refusing")
    pathlib.Path(fresh).replace(args.manifest)
    print(f"{lines} rank candidates", file=sys.stderr, flush=True)


def part_key(line):
    entry = json.loads(line)
    return json.dumps([entry.get(field) for field in ("uuid", "time", "kind", "names")])


def seed_part():
    """pregen carries a rank it published before and one whose conversion
    failed over from its previous output. A refresh of a few maps gets the
    lines of those maps to carry, so it converts what is new and nothing else."""
    lines = []
    if WATCHABLE.is_file() and args.maps:
        lines = [line for line in open(WATCHABLE, encoding="utf-8")
            if json.loads(line).get("map") in set(args.maps)]
    PART_WATCHABLE.write_text("".join(lines), encoding="utf-8")


def merge_part():
    """The manifest keeps every map it had, with the refreshed ones replaced by
    what the run made of them. A rank that was deleted is in no candidate list
    any more, so it drops out here and stops being linked.

    Named runs replace their own line and nothing else: the rest of their map
    was never converted in this pass and would be lost."""
    lines = list(open(WATCHABLE, encoding="utf-8")) if WATCHABLE.is_file() else []
    fresh = list(open(PART_WATCHABLE, encoding="utf-8"))
    if args.maps:
        kept = [line for line in lines if json.loads(line).get("map") not in set(args.maps)]
    else:
        replaced = {part_key(line) for line in fresh}
        kept = [line for line in lines if part_key(line) not in replaced]
    temp = WATCHABLE.with_suffix(".jsonl.new")
    temp.write_text("".join(kept + fresh), encoding="utf-8")
    temp.replace(WATCHABLE)
    print(f"{len(fresh)} manifest lines of {len(args.maps or args.runs)} refreshed "
        f"{'maps' if args.maps else 'runs'} merged in, {len(kept)} kept", file=sys.stderr, flush=True)


def generate():
    # Best effort at the lowest priority, not the idle class: the archive disk
    # is never idle (hourly rsyncs from every game server, the daily archive
    # and index runs) and an idle-class reader makes no progress at all
    if PART:
        seed_part()
    command = ["nice", "-n19", "ionice", "-c2", "-n7", sys.executable, str(HERE / "pregen.py"),
        str(PART_MANIFEST) if PART else args.manifest,
        str(PART_WATCHABLE) if PART else str(WATCHABLE),
        "--cache", str(CACHE), "--ranks", str(args.ranks)] + \
        (["--partial"] if PART else []) + \
        (["--retry-failed"] if args.retry_failed else []) + \
        (["--reconvert"] if args.reconvert else []) + \
        [arg for map_name in args.reconvert_map for arg in ("--reconvert-map", map_name)] + \
        (["--reconvert-before", args.reconvert_before] if args.reconvert_before else [])
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    # A refresh of a few maps is done in minutes, so it is waited out whole
    # and the manifest is only touched once it worked
    if PART:
        if subprocess.run(command).returncode != 0:
            sys.exit("pregen failed")
        merge_part()
        return
    process = subprocess.Popen(command)
    # A run takes hours, what it has finished goes up every few minutes so
    # that a fixed demo is watched as soon as it exists, not when the last one
    # is done
    uploaded = set()
    while True:
        try:
            process.wait(timeout=UPLOAD_EVERY_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                upload_partial(uploaded)
            except Exception as error:
                print(f"partial upload failed, next try in {UPLOAD_EVERY_SECONDS} s: {error}", file=sys.stderr, flush=True)
            continue
        break
    if process.returncode != 0:
        sys.exit(f"pregen failed with {process.returncode}")


UPLOAD_EVERY_SECONDS = 300


def upload_partial(uploaded):
    """The demos of the ranks the running pre-generation has finished, and the
    run files that name them. The run file of a recording holds the lines the
    run has for it and the earlier lines for its other ranks, so a link keeps
    working while its rank is still in the queue."""
    partial = pathlib.Path(str(WATCHABLE) + ".new")
    if not partial.is_file():
        return
    fresh = {}
    for line in partial.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("status") == "ok" and entry.get("demo"):
            fresh.setdefault(entry["uuid"], {})[(entry["kind"], entry["time"])] = line
    demos = sorted({json.loads(line)["demo"] for lines in fresh.values() for line in lines.values()}
        - uploaded)
    demos = [name for name in demos if (CACHE / "demos" / name).is_file()]
    if not demos:
        return
    if WATCHABLE.is_file():
        for line in WATCHABLE.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("status") == "ok" and entry.get("demo") and entry["uuid"] in fresh:
                fresh[entry["uuid"]].setdefault((entry["kind"], entry["time"]), line)
    runs = CACHE / "runs"
    runs.mkdir(exist_ok=True)
    changed = []
    for uuid, lines in fresh.items():
        path = runs / f"{uuid}.jsonl"
        text = "".join(line + "\n" for line in lines.values())
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
            changed.append(path.name)
    run(["rsync", "-a", "--files-from=-", str(CACHE / "demos"), f"{args.target}/demos/"],
        input="\n".join(demos), text=True)
    if changed:
        run(["rsync", "-a", "--files-from=-", str(runs), f"{args.target}/runs/"],
            input="\n".join(changed), text=True)
    uploaded.update(demos)
    print(f"{len(demos)} demos and {len(changed)} run files uploaded while the run goes on", file=sys.stderr, flush=True)


def wanted_demos():
    demos = set()
    with open(WATCHABLE, encoding="utf-8") as watchable:
        for line in watchable:
            entry = json.loads(line)
            if entry.get("status") == "ok":
                demos.add(entry["demo"])
    return demos


def refresh_revisions():
    """The page asks for a demo under its revision, so that a re-scrambled one
    is not served out of a CDN cache holding the previous one. The converter
    writes it beside the demo, and a re-scramble updates it there without
    touching the manifest: an upload of a re-scrambled cache needs no
    pre-generation run."""
    if not WATCHABLE.is_file():
        sys.exit(f"{WATCHABLE} does not exist, run the pre-generation first")
    lines = []
    changed = 0
    with open(WATCHABLE, encoding="utf-8") as watchable:
        for line in watchable:
            entry = json.loads(line)
            meta = CACHE / "demos" / (entry.get("demo", "x")[:-len(".demo.gz")] + ".json")
            if entry.get("status") == "ok" and meta.is_file():
                try:
                    rev = json.loads(meta.read_text()).get("rev", "")
                except ValueError:
                    rev = ""
                if rev and rev != entry.get("rev"):
                    entry["rev"] = rev
                    changed += 1
            lines.append(json.dumps(entry, ensure_ascii=False))
    if changed and not args.dry_run:
        temp = WATCHABLE.with_suffix(".jsonl.new")
        temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temp.replace(WATCHABLE)
    print(f"{changed} demo revisions refreshed", file=sys.stderr)


def write_runs():
    """One small file per recording next to the demos, holding the manifest
    lines of its ranks: the page opens a run with that instead of the whole
    manifest, which is 400 KB on the wire on every visit."""
    runs = CACHE / "runs"
    runs.mkdir(exist_ok=True)
    lines = {}
    with open(WATCHABLE, encoding="utf-8") as watchable:
        for line in watchable:
            entry = json.loads(line)
            if entry.get("status") == "ok" and entry.get("demo"):
                lines.setdefault(entry["uuid"], []).append(line)
    for path in runs.glob("*.jsonl"):
        if path.stem not in lines:
            path.unlink()
    for uuid, entries in lines.items():
        path = runs / f"{uuid}.jsonl"
        text = "".join(entries)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    return len(lines)


def drop_from_manifest(demos):
    lines = [line for line in open(WATCHABLE, encoding="utf-8")
        if json.loads(line).get("demo") not in demos]
    temp = WATCHABLE.with_suffix(".jsonl.new")
    temp.write_text("".join(lines), encoding="utf-8")
    temp.replace(WATCHABLE)


def deploy(demos):
    if not demos:
        sys.exit("no demo in the manifest, refusing to empty the web directory")
    listing = subprocess.run(["ssh", HOST, f"mkdir -p {REMOTE}/demos && ls -1 {REMOTE}/demos"],
        check=True, capture_output=True, text=True).stdout
    present = {name for name in listing.split("\n") if name.endswith(".demo.gz")}
    # A demo the cache dropped to stay under its size limit stays on the web
    # host, the manifest still names it. Everything else is offered to rsync,
    # which skips what is already there and re-uploads what the converter
    # produces differently after a fix.
    cached = sorted(name for name in demos if (CACHE / "demos" / name).is_file())
    gone = sorted(name for name in demos - present if not (CACHE / "demos" / name).is_file())
    # A demo the manifest no longer names is a rank that was beaten. Its link
    # is out there and its file is small, so it stays unless --prune says
    # otherwise.
    stale = sorted(present - demos) if args.prune else []
    total = sum((CACHE / "demos" / name).stat().st_size for name in cached if name not in present)
    if gone:
        # Their link would go to a demo that is on neither host, so the page
        # would report the rank as missing from the archive, which it is not
        print(f"{len(gone)} demos of the manifest are neither cached nor uploaded and are "
            f"dropped from it, first: {gone[0]}", file=sys.stderr)
        if not args.dry_run:
            drop_from_manifest(set(gone))
    print(f"{len(demos)} demos referenced, {len(cached)} offered to rsync "
        f"({total / 1024**2:.0f} MiB not on the web host yet), {len(stale)} to delete", file=sys.stderr)
    if args.dry_run:
        return

    # rsync takes the list on stdin, so a run of thousands of demos does not
    # end up as one huge command line
    if cached:
        run(["rsync", "-a", "--files-from=-", str(CACHE / "demos"), f"{args.target}/demos/"],
            input="\n".join(cached), text=True)
    # The manifest decides which ranks the map pages link, so it goes up only
    # after every demo it names is there, and the index the pages read is
    # loaded from it right after
    run(["rsync", "-a", str(WATCHABLE), f"{args.target}/watchable.jsonl"])
    print(f"{write_runs()} runs written", file=sys.stderr)
    run(["rsync", "-a", "--delete", str(CACHE / "runs") + "/", f"{args.target}/runs/"])
    run(["ssh", HOST, f"python3 {args.import_script} {REMOTE}/watchable.jsonl"])
    # Demos of ranks that were beaten or deleted, the web host is not a cache
    if stale:
        run(["ssh", HOST, f"cd {REMOTE}/demos && xargs -0 rm -f --"],
            input="\0".join(stale) + "\0", text=True)


def main():
    # One run at a time: the nightly run and one started by hand would both
    # write the manifest and upload
    lock = open(CACHE / "sync.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another sync-ranks.py is running")
    # A dry run reports, it does not convert several thousand recordings
    if not args.no_generate and not args.dry_run:
        fetch_manifest()
        generate()
    refresh_revisions()
    deploy(wanted_demos())


if __name__ == "__main__":
    main()
