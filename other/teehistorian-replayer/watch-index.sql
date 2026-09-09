-- The ranks that have a pre-generated demo, so the map and rank pages can
-- turn their time into a link. Written by import-watchable.py from the
-- watchable.jsonl the archive host uploads, read by watchlinks.py.
CREATE TABLE IF NOT EXISTS record_watch (
  Map VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  Kind VARCHAR(4) NOT NULL,
  -- The time in milliseconds is what a displayed rank is matched on, a float
  -- of the same run reaches the page through several conversions
  TimeMilli INT NOT NULL,
  -- The whole link: everything else about the run is in watchable.jsonl,
  -- which the watch page reads
  GameID VARCHAR(36) NOT NULL,
  PRIMARY KEY (Map, Kind, TimeMilli)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
