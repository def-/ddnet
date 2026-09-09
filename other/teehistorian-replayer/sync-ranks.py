#!/usr/bin/env python3
# One command for the rank demo sync, run on the archive host (li): takes the
# rank manifest of the database host, pre-generates the demos of the ranks it
# names and copies them and the manifest to the web host. Safe to re-run,
# everything already converted and already uploaded is skipped.
#
# The manifest comes from top-ranks.py, which has to run on the database host
# as root (it reads /etc/mysql/debian.cnf), so it is fetched instead of run
# from here. Pass --manifest-source - to use an existing local file.
#
# Usage: sync-ranks.py [--ranks 1] [--dry-run]

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--manifest-source", default="ddnet:/var/www/watch/top-ranks.jsonl",
    help="scp source of the rank manifest, - to use --manifest as it is")
parser.add_argument("--manifest", default="/tmp/top-ranks.jsonl")
parser.add_argument("--cache", default=str(pathlib.Path.home() / "teehistorian-demos"))
parser.add_argument("--target", default="ddnet:/var/www/watch",
    help="ssh destination of the web directory holding demos/ and watchable.jsonl")
parser.add_argument("--ranks", type=int, default=1, help="ranks to publish per map and kind")
parser.add_argument("--import-script", default="/home/teeworlds/servers/scripts/import-watchable.py",
    help="loads the uploaded manifest into the record_watch table on the web host")
parser.add_argument("--retry-failed", action="store_true", help="passed on to pregen.py")
parser.add_argument("--prune", action="store_true",
    help="delete demos of ranks the manifest no longer names, which breaks the links that were shared for them")
parser.add_argument("--dry-run", action="store_true", help="report what would be uploaded and deleted")
parser.add_argument("--no-generate", action="store_true",
    help="upload the manifest as it is, for a pregen that was run separately")
args = parser.parse_args()

HERE = pathlib.Path(__file__).resolve().parent
CACHE = pathlib.Path(args.cache)
WATCHABLE = CACHE / "watchable.jsonl"
HOST, _, REMOTE = args.target.partition(":")


def run(command, **kwargs):
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    return subprocess.run(command, check=True, **kwargs)


def fetch_manifest():
    if args.manifest_source == "-":
        return
    run(["scp", "-q", args.manifest_source, args.manifest])


def generate():
    run(["nice", "-n19", "ionice", "-c3", sys.executable, str(HERE / "pregen.py"),
        args.manifest, str(WATCHABLE), "--cache", str(CACHE), "--ranks", str(args.ranks)] +
        (["--retry-failed"] if args.retry_failed else []))


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
    run(["ssh", HOST, f"python3 {args.import_script} {REMOTE}/watchable.jsonl"])
    # Demos of ranks that were beaten or deleted, the web host is not a cache
    if stale:
        run(["ssh", HOST, f"cd {REMOTE}/demos && xargs -0 rm -f --"],
            input="\0".join(stale) + "\0", text=True)


def main():
    # A dry run reports, it does not convert several thousand recordings
    if not args.no_generate and not args.dry_run:
        fetch_manifest()
        generate()
    refresh_revisions()
    deploy(wanted_demos())


if __name__ == "__main__":
    main()
