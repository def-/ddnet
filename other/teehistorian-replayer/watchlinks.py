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


def nameKeys(cur, mapName, rows, links):
    """The same links under (time in milliseconds, player) keys, for a page
    whose rank rows carry no game uuid (the Materialize backed /maps/). A run
    is the rank of every player that finished it, so each of them is a key.
    One query per published rank, through the (Map, Time) index: a lookup by
    game uuid alone walks every finish of the map and a busy map has more of
    them than max_statement_time allows."""
    for kind, table in (("solo", "record_race"), ("team", "record_teamrace")):
        for _, rowKind, milli, gameId in rows:
            if rowKind != kind:
                continue
            url = links.get((mapName, kind), {}).get((gameId, milli))
            if url is None:
                continue
            cur.execute(f"SELECT Name FROM {table} WHERE Map = %s AND Time BETWEEN %s AND %s AND GameID = %s",
                (mapName, milli / 1000.0 - 0.05, milli / 1000.0 + 0.05, gameId))
            for (name,) in cur.fetchall():
                links[(mapName, kind)][(milli, name)] = url


def watchLinks(mapName=None):
    """{(map, kind): {(game uuid, time in milliseconds): url}} of what can be
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
        links.setdefault((map, kind), {})[(gameId, milli)] = watchUrl(gameId, milli, gameId in shared)
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
