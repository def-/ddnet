#!/usr/bin/env python3
# Makes the ghosts of the cached demos that were published before there were
# any, from the published demo itself. --all makes them again for every demo,
# after a change to demo_ghost.
#
# Usage: ghosts.py [--cache ~/teehistorian-demos] [--jobs 4] [--all]

import argparse
import json
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rankdemo import Converter

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default=str(pathlib.Path.home() / "teehistorian-demos"))
parser.add_argument("--tool", default=str(pathlib.Path.home() / "git/ddnet/build-tools/teehistorian2demo"))
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--all", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

converter = Converter(args.tool, "/", args.cache, 1 << 60)
if not converter.ghost_tool.is_file():
    sys.exit(f"{converter.ghost_tool} is not built")
demos = []
for path in sorted(converter.demos.glob("*.demo.gz")):
    meta_path = converter.siblings(path)[1]
    if path.name.endswith(".raw.demo.gz") or not meta_path.is_file():
        continue
    if args.all or "ghosts" not in json.loads(meta_path.read_text()):
        demos.append(path)
print(f"{len(demos)} cached demos to make ghosts of", file=sys.stderr)
if args.dry_run:
    sys.exit(0)

made = without = failed = 0
counted = threading.Lock()


def add(demo_path):
    global made, without, failed
    error = None
    try:
        meta = converter.add_ghosts(demo_path)
    except Exception as failure:  # one broken demo must not end the run
        error = f"{type(failure).__name__}: {failure}"
    else:
        error = meta.get("ghost_error")
    with counted:
        if error is not None:
            failed += 1
            print(f"{demo_path.name}: {error}", file=sys.stderr, flush=True)
        elif meta.get("ghosts"):
            made += 1
        else:
            without += 1
        if (made + without + failed) % 200 == 0:
            print(f"{made + without + failed} of {len(demos)}", file=sys.stderr, flush=True)


with ThreadPoolExecutor(max_workers=args.jobs) as pool:
    list(pool.map(add, demos))
print(f"{made} demos have ghosts now, {without} hold no whole run, {failed} failed", file=sys.stderr)
