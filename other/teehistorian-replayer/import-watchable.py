#!/usr/bin/env python3
# Loads the watchable.jsonl of the rank demo pre-generation into the
# record_watch table (see watch-index.sql), which is what the map and rank
# pages look up to make a rank time clickable. Runs on the web host, as the
# user the web scripts run as, after the archive host uploaded the demos.
#
# --gone reports instead: the maps of the manifest whose rank has been
# deleted since, which is what the watch lane on the archive host refreshes.
#
# Usage: import-watchable.py [--gone] [/var/www/watch/watchable.jsonl]

import argparse
import json
import sys

sys.path.insert(0, "/home/teeworlds/servers/scripts")
from mysql import mysqlConnect

parser = argparse.ArgumentParser()
parser.add_argument("manifest", nargs="?", default="/var/www/watch/watchable.jsonl")
parser.add_argument("--gone", action="store_true",
    help="print the maps whose rank is no longer a rank and change nothing")
args = parser.parse_args()


def rows(path, candidates_only=False):
    """candidates_only leaves out the lines that are not among their map's
    candidates: the ones the pre-generation keeps for their links alone, and
    the runs a report named, where a deleted rank is the point."""
    for line in open(path, encoding="utf-8"):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("status") != "ok" or "demo" not in entry:
            continue
        if candidates_only and (entry.get("kept") or entry.get("extra")):
            continue
        yield (entry["map"], entry["kind"], round(float(entry["time"]) * 1000), entry["uuid"])


def existing(con, watchable):
    """Ranks are deleted when they turn out to be cheated, and the manifest is
    a day old at most but still older than that. A rank that is no longer in
    the rank tables is not linked, whatever the manifest says.

    The game uuid has to match, not only the time: two runs of a map can carry
    the same time, and a demo that belongs to a deleted run would otherwise be
    handed to the rank that replaced it, which is then shown under the wrong
    names. The time comes along to reach the rank through an index (Time is a
    float, so it is matched as a range), the uuid decides."""
    cur = con.cursor()
    kept = []
    for row in watchable:
        map_name, kind, milli, game_id = row
        table = "record_teamrace" if kind == "team" else "record_race"
        cur.execute(f"SELECT 1 FROM {table} WHERE Map = %s AND Time BETWEEN %s AND %s AND GameID = %s LIMIT 1",
            (map_name, milli / 1000 - 0.05, milli / 1000 + 0.05, game_id))
        if cur.fetchone():
            kept.append(row)
    cur.close()
    if len(kept) != len(watchable):
        print(f"{len(watchable) - len(kept)} watchable ranks are gone from the rank tables", file=sys.stderr)
    return kept


def gone_maps(con, watchable):
    """The maps of the manifest whose rank is no longer a rank, which is what
    tells the archive host to make a demo of the one that took its place.

    The manifest, not record_watch: the import below leaves a deleted rank out
    of the table, so the table forgets it ever had one, while the manifest
    keeps the line until the map is converted again."""
    kept = set(existing(con, watchable))
    return sorted({row[0] for row in watchable if row not in kept})


def main():
    # Two ranks of the same map and kind can carry the same time, a tie that
    # the manifest keeps both of. The table has one row per time, and the
    # first of them is the one the page links.
    # A rank that is gone and whose line is only kept for its link needs no
    # refresh, its map has one already: asking for it again would name the same
    # map on every tick for good.
    seen = set()
    watchable = []
    for row in rows(args.manifest, candidates_only=args.gone):
        if row[:3] not in seen:
            seen.add(row[:3])
            watchable.append(row)
    if not watchable:
        sys.exit(f"no watchable rank in {args.manifest}")

    con = mysqlConnect()
    if args.gone:
        for map_name in gone_maps(con, watchable):
            print(map_name)
        con.close()
        return
    watchable = existing(con, watchable)
    if not watchable:
        sys.exit("no watchable rank is still a rank, refusing to empty the table")
    cur = con.cursor()
    cur.execute(open(__file__.replace("import-watchable.py", "watch-index.sql")).read())
    # One transaction, so a page never sees the table half filled
    cur.execute("START TRANSACTION")
    cur.execute("DELETE FROM record_watch")
    cur.executemany(
        "INSERT INTO record_watch (Map, Kind, TimeMilli, GameID) VALUES (%s, %s, %s, %s)", watchable)
    con.commit()
    cur.close()
    con.close()
    print(f"{len(watchable)} watchable ranks in record_watch", file=sys.stderr)


if __name__ == "__main__":
    main()
