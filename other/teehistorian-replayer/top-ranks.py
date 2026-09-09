#!/usr/bin/env python3
# Dumps the top solo and team ranks of every map as a jsonl manifest for the
# teehistorian rank demo pre-generation (pregen.py on the archive host). More
# candidates than wanted ranks are dumped: recordings older than the archive
# or from a server that was not recording are gone, so pregen walks down the
# list until a rank can be converted. Runs on the database host.
#
# Usage: top-ranks.py > top-ranks.jsonl

import argparse
import json
import sys
from zoneinfo import ZoneInfo

import MySQLdb

TZ = ZoneInfo("Europe/Berlin")

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", type=int, default=20,
    help="rank rows dumped per map and kind, pregen walks down them while recordings are missing")
parser.add_argument("--min-age-days", type=int, default=14,
    help="skip finishes younger than this: a fresh rank can still turn out to be cheated and be deleted")
parser.add_argument("--maps", nargs="+", help="only these maps (default: every map with a page)")
args = parser.parse_args()


def epoch(timestamp):
    return int(timestamp.replace(tzinfo=TZ).timestamp())


def rank_of(time, times):
    """The rank a time has in a sorted list, ties share the better rank."""
    return 1 + sum(1 for other in times if other < time)


def solo_ranks(cur, map_name):
    # (Map, Time, Name) covers the ordering, so this reads the first rows of
    # the index and nothing else
    cur.execute("SELECT Name, Time, Timestamp, GameID FROM record_race WHERE Map = %s ORDER BY Time LIMIT %s",
        (map_name, args.candidates))
    rows = cur.fetchall()
    times = [row[1] for row in rows]
    return [{"kind": "solo", "map": map_name, "names": [name], "time": str(time), "ts": epoch(timestamp),
        "uuid": game_id, "rank": rank_of(time, times)}
        for name, time, timestamp, game_id in rows if game_id]


def team_ranks(cur, map_name):
    # The rows of one team run share an ID, and all of them carry the same
    # time and game
    cur.execute("SELECT MIN(Time), MIN(Timestamp), MIN(GameID), GROUP_CONCAT(Name SEPARATOR 0x09) "
        "FROM record_teamrace WHERE Map = %s GROUP BY ID ORDER BY MIN(Time) LIMIT %s",
        (map_name, args.candidates))
    rows = cur.fetchall()
    times = [row[0] for row in rows]
    ranks = []
    for time, timestamp, game_id, names in rows:
        if isinstance(names, bytes):
            names = names.decode()
        names = sorted(names.split("\t"))
        # A team of one is the same run as the solo rank, and the tool would
        # produce the same demo twice
        if not game_id or len(names) < 2:
            continue
        ranks.append({"kind": "team", "map": map_name, "names": names, "time": str(time), "ts": epoch(timestamp),
            "uuid": game_id, "rank": rank_of(time, times)})
    return ranks


def main():
    conn = MySQLdb.connect(read_default_file="/etc/mysql/debian.cnf", db="teeworlds", charset="utf8mb4")
    cur = conn.cursor()
    # The global cap would kill the per-map queries on the largest maps
    cur.execute("SET SESSION max_statement_time=0")
    # The default of 1024 bytes silently truncates the roster of a large team
    cur.execute("SET SESSION group_concat_max_len = 1000000")

    if args.maps:
        maps = args.maps
    else:
        cur.execute("SELECT Map FROM record_maps ORDER BY Map")
        maps = [row[0] for row in cur.fetchall()]

    cur.execute("SELECT UNIX_TIMESTAMP(NOW() - INTERVAL %s DAY)", (args.min_age_days,))
    newest_ts = int(cur.fetchone()[0])

    count = skipped = 0
    for map_name in maps:
        # Team ranks first, they are the more interesting replays
        for entry in team_ranks(cur, map_name) + solo_ranks(cur, map_name):
            if entry["ts"] > newest_ts:
                skipped += 1
                continue
            print(json.dumps(entry, ensure_ascii=False))
            count += 1
    print(f"{count} rank candidates over {len(maps)} maps, {skipped} skipped as younger than "
        f"{args.min_age_days} days", file=sys.stderr)


if __name__ == "__main__":
    main()
