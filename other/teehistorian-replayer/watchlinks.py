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
    is the rank of every player that finished it, so each of them is a key."""
    for kind, table in (("solo", "record_race"), ("team", "record_teamrace")):
        watched = {(gameId, milli) for _, rowKind, milli, gameId in rows if rowKind == kind and gameId}
        ids = sorted({gameId for gameId, _ in watched})
        if not ids:
            continue
        cur.execute("SELECT Name, ROUND(Time * 1000), GameID FROM %s WHERE Map = %%s AND GameID IN (%s)"
            % (table, ",".join(["%s"] * len(ids))), [mapName] + ids)
        for name, milli, gameId in cur.fetchall():
            # One recording holds a whole game and every rank set in it, so
            # only the rank that was published is a key, not its neighbours
            if (gameId, int(milli)) not in watched:
                continue
            url = links.get((mapName, kind), {}).get(gameId)
            if url is not None:
                links[(mapName, kind)][(int(milli), name)] = url


def watchLinks(mapName=None):
    """{(map, kind): {game uuid: url}} of what can be watched, for one map or
    for all of them. A page hands the inner dict of the records it is about to
    print to printSoloRecords and its siblings.

    Keyed by the run and not by its time: several ranks of a map can carry the
    same time, and only the one the demo was made from may be a link."""
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
        links.setdefault((map, kind), {})[gameId] = watchUrl(gameId, milli, gameId in shared)
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
