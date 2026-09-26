# The ranks that can be watched in the browser, for the map and rank pages:
# every rank whose demo was pre-generated from the teehistorian archive is in
# record_watch (see watch-index.sql), and its time is shown as a link to the
# replayer.
#
# Copied to /home/teeworlds/servers/scripts/ next to the pages that import it.
# The lookup is its own connection and never raises: a page that cannot reach
# the index shows the times it always showed.

import sys
from urllib.parse import quote

sys.path.insert(0, "/home/teeworlds/servers/scripts")
from mysql import mysqlConnect

WATCH_URL = "/watch/"
COLUMNS = "Map, Kind, TimeMilli, GameID"

connection = None


def watchUrl(gameId, milli, shared):
    """The game uuid is the whole link, the page reads the run it belongs to
    out of the manifest that sits next to the demos. One recording can hold
    more than one published rank, and then the time says which."""
    url = WATCH_URL + "?uuid=" + quote(gameId)
    if shared:
        # Three decimals, the page matches the time against the manifest with
        # a tolerance of five thousandths and two would sit on that boundary
        url += "&time=%.3f" % (milli / 1000.0)
    return url


def fetch(mapName):
    global connection
    if connection is None:
        connection = mysqlConnect()
        connection.autocommit(True)
    cur = connection.cursor()
    if mapName is None:
        cur.execute("SELECT %s FROM record_watch" % COLUMNS)
    else:
        cur.execute("SELECT %s FROM record_watch WHERE Map = %%s" % COLUMNS, (mapName,))
    rows = cur.fetchall()
    cur.close()
    return rows


def centi(milli):
    """A rank time in hundredths, which is the resolution the game records.
    The keys are matched against times the pages read out of a MySQL float or
    a Materialize real, both single precision: 22354.2 seconds comes back as
    22354.19921875 and would miss a key in milliseconds by one."""
    return round(milli / 10)


def nameKeys(cur, mapName, rows, links):
    """The same links under (time in hundredths, player) keys, for a page
    whose rank rows carry no game uuid (the Materialize backed /maps/). A run
    is the rank of every player that finished it, so each of them is a key.
    One query per published rank, through the (Map, Time) index: a lookup by
    game uuid alone walks every finish of the map and a busy map has more of
    them than max_statement_time allows."""
    for kind, table in (("solo", "record_race"), ("team", "record_teamrace")):
        for _, rowKind, milli, gameId in rows:
            if rowKind != kind:
                continue
            url = links.get((mapName, kind), {}).get((gameId, centi(milli)))
            if url is None:
                continue
            cur.execute(f"SELECT Name FROM {table} WHERE Map = %s AND Time BETWEEN %s AND %s AND GameID = %s",
                (mapName, milli / 1000.0 - 0.05, milli / 1000.0 + 0.05, gameId))
            for (name,) in cur.fetchall():
                links[(mapName, kind)][(centi(milli), name)] = url


def watchLinks(mapName=None):
    """{(map, kind): {(game uuid, time in hundredths): url}} of what can be
    watched, for one map or for all of them. A page hands the inner dict of
    the records it is about to print to printSoloRecords and its siblings.

    Keyed by the run and its time: several ranks of a map can carry the same
    time, and only the one the demo was made from may be a link, while one
    recording can hold several published ranks, each with a link of its own."""
    global connection
    try:
        rows = fetch(mapName)
    except Exception:
        # A connection that timed out in a long lived worker, or no index yet
        connection = None
        try:
            rows = fetch(mapName)
        except Exception:
            connection = None
            return {}
    shared = set()
    seen = set()
    for _, _, _, gameId in rows:
        (shared if gameId in seen else seen).add(gameId)
    links = {}
    for map, kind, milli, gameId in rows:
        links.setdefault((map, kind), {})[(gameId, centi(milli))] = watchUrl(gameId, milli, gameId in shared)
    if mapName is not None and links:
        try:
            cur = connection.cursor()
            nameKeys(cur, mapName, rows, links)
            cur.close()
        except Exception:
            # The uuid keys are there either way, a page that has them is
            # unaffected by a lookup that failed here
            connection = None
    return links


def playerLinks(name):
    """{(map, kind): url} of the player's ranks that can be watched, for the
    player page. It shows one run a map, the best solo and the best team run,
    so a demo of a run the player has since beaten gets no link there. A join
    per kind through the (Map, Name) index of the rank table, one row per
    published rank the player is part of, then the best times of those maps."""
    global connection
    links = watchLinks()
    if not links:
        return {}
    found = {}
    try:
        cur = connection.cursor()
        for kind, table in (("solo", "record_race"), ("team", "record_teamrace")):
            cur.execute(f"SELECT w.Map, w.TimeMilli, w.GameID FROM record_watch w JOIN {table} r ON r.Map = w.Map AND r.Name = %s AND r.GameID = w.GameID AND r.Time BETWEEN w.TimeMilli / 1000 - 0.05 AND w.TimeMilli / 1000 + 0.05 WHERE w.Kind = %s",
                (name, kind))
            rows = cur.fetchall()
            if not rows:
                continue
            maps = sorted({map for map, _, _ in rows})
            cur.execute(f"SELECT Map, MIN(Time) FROM {table} WHERE Name = %s AND Map IN ({', '.join(['%s'] * len(maps))}) GROUP BY Map",
                (name, *maps))
            best = {map: round(time * 100) for map, time in cur.fetchall()}
            for map, milli, gameId in rows:
                url = links.get((map, kind), {}).get((gameId, centi(milli)))
                if url is not None and centi(milli) == best.get(map):
                    found[(map, kind)] = url
        cur.close()
    except Exception:
        connection = None
        return {}
    return found
