#!/usr/bin/env python3
# Loads the watchable.jsonl of the rank demo pre-generation into the
# record_watch table (see watch-index.sql), which is what the map and rank
# pages look up to make a rank time clickable. Runs on the web host, as the
# user the web scripts run as, after the archive host uploaded the demos.
#
# Usage: import-watchable.py [/var/www/watch/watchable.jsonl]

import json
import sys

sys.path.insert(0, "/home/teeworlds/servers/scripts")
from mysql import mysqlConnect

MANIFEST = sys.argv[1] if len(sys.argv) > 1 else "/var/www/watch/watchable.jsonl"


def rows(path):
    for line in open(path, encoding="utf-8"):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("status") != "ok" or "demo" not in entry:
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


def main():
    # Two ranks of the same map and kind can carry the same time, a tie that
    # the manifest keeps both of. The table has one row per time, and the
    # first of them is the one the page links.
    seen = set()
    watchable = []
    for row in rows(MANIFEST):
        if row[:3] not in seen:
            seen.add(row[:3])
            watchable.append(row)
    if not watchable:
        sys.exit(f"no watchable rank in {MANIFEST}, refusing to empty the table")

    con = mysqlConnect()
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
