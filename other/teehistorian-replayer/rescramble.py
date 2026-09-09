#!/usr/bin/env python3
# Scrambles every cached demo again from the converter output kept beside it.
# The expensive half of the pipeline is reading the recordings, so a change to
# demo_scramble costs minutes here instead of the hours a full run over the
# archive takes.
#
# Usage: rescramble.py [--cache ~/teehistorian-demos] [--jobs 4]

import argparse
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
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

converter = Converter(args.tool, "/", args.cache, 1 << 60)
raws = sorted(converter.demos.glob("*.raw.demo.gz"))
print(f"{len(raws)} cached demos to scramble again", file=sys.stderr)
if args.dry_run:
    sys.exit(0)

done = failed = 0
counted = threading.Lock()


def rescramble(raw_path):
    global done, failed
    demo_path = raw_path.with_name(raw_path.name.replace(".raw.demo.gz", ".demo.gz"))
    error = None
    try:
        converter.scramble_cached(raw_path, demo_path)
    except Exception as failure:  # one broken demo must not end the run
        error = failure
    with counted:
        if error is None:
            done += 1
        else:
            failed += 1
            print(f"{demo_path.name}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if (done + failed) % 200 == 0:
            print(f"{done + failed} of {len(raws)}", file=sys.stderr, flush=True)


with ThreadPoolExecutor(max_workers=args.jobs) as pool:
    list(pool.map(rescramble, raws))
print(f"{done} demos scrambled again, {failed} failed", file=sys.stderr)
