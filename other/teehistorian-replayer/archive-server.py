#!/usr/bin/env python3
# Serves rank demos converted on the fly from the teehistorian archive.
#
#   GET /rankdemo?uuid=<game_uuid>&time=<seconds>&ts=<finish epoch>&name=<player>
#   GET /rankmeta?<same parameters>
#
# The rank's game uuid (record_race.GameID) is the archive file name, so the
# recording is found with a handful of stats. The teehistorian2demo tool's
# --rank mode locates the exact finish event, converts only the run's time
# window and hides all other teams. Results are cached, the first request for
# an uncached rank blocks until the conversion is done (seconds, up to minutes
# for runs deep into multi-GiB recordings). <ts> is optional but lets the scan
# stop early instead of parsing the whole recording. For team ranks, repeat
# the name parameter for every team member.
#
# With --raw, GET /<region>/<game_uuid> additionally streams whole recordings
# (decompressing .xz on the fly) for the manual upload/URL page. Do not expose
# --raw publicly: raw teehistorian files contain every player's inputs.
#
# Usage: archive-server.py [--port 8140] [--root /media/teehistorian/data]
#                          [--tool ~/git/ddnet/build-tools/teehistorian2demo]
#                          [--cache ~/teehistorian-demos] [--raw]

import argparse
import json
import pathlib
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rankdemo import Converter, RankDemoError

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8140)
parser.add_argument("--root", default="/media/teehistorian/data")
parser.add_argument("--tool", default=str(pathlib.Path.home() / "git/ddnet/build-tools/teehistorian2demo"))
parser.add_argument("--cache", default=str(pathlib.Path.home() / "teehistorian-demos"))
parser.add_argument("--cache-limit-gb", type=float, default=20)
parser.add_argument("--raw", action="store_true", help="also stream raw recordings")
args = parser.parse_args()

ROOT = pathlib.Path(args.root).resolve()
CONVERTER = Converter(args.tool, ROOT, args.cache, int(args.cache_limit_gb * 1024**3))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")

    def rank_params(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        uuid = query.get("uuid", [""])[0].lower()
        time_str = query.get("time", [""])[0]
        names = query.get("name", [])
        ts_epoch = None
        if query.get("ts", [""])[0]:
            try:
                ts_epoch = int(query["ts"][0])
            except ValueError:
                raise RankDemoError(400, "Invalid ts") from None
        return CONVERTER.convert(uuid, time_str, names, ts_epoch)

    def do_GET(self):
        route = urllib.parse.urlsplit(self.path).path
        try:
            if route == "/rankdemo":
                demo_path, meta = self.rank_params()
                self.send_response(200)
                self.send_cors_headers()
                self.send_header("Content-Type", "application/octet-stream")
                # The cache holds demos gzipped, the browser unpacks them
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(demo_path.stat().st_size))
                self.send_header("X-Rank-Meta", json.dumps(meta))
                self.send_header("Access-Control-Expose-Headers", "X-Rank-Meta")
                self.end_headers()
                with open(demo_path, "rb") as file:
                    while chunk := file.read(1024 * 1024):
                        self.wfile.write(chunk)
            elif route == "/rankmeta":
                _, meta = self.rank_params()
                body = json.dumps(meta).encode()
                self.send_response(200)
                self.send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif args.raw:
                self.serve_raw()
            else:
                self.send_error(404)
        except RankDemoError as error:
            self.send_error(error.status, explain=str(error))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def resolve_raw(self):
        try:
            route = urllib.parse.urlsplit(self.path).path
            path = (ROOT / urllib.parse.unquote(route).lstrip("/")).resolve()
            path.relative_to(ROOT)
        except (ValueError, OSError):
            return None
        for candidate in (path, path.with_name(path.name + ".teehistorian"), path.with_name(path.name + ".teehistorian.xz")):
            if candidate.is_file():
                return candidate
        return None

    def serve_raw(self):
        path = self.resolve_raw()
        if path is None:
            self.send_error(404)
            return
        compressed = path.suffix == ".xz"
        self.send_response(200)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/octet-stream")
        if not compressed:
            self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        try:
            if compressed:
                with subprocess.Popen(["xz", "-dc", str(path)], stdout=subprocess.PIPE) as process:
                    while chunk := process.stdout.read(1024 * 1024):
                        self.wfile.write(chunk)
            else:
                with open(path, "rb") as file:
                    while chunk := file.read(1024 * 1024):
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Client stopped reading (e.g. time range fully converted)
            pass

    def log_message(self, format, *args):
        print(f"{self.address_string()} {format % args}", flush=True)


if __name__ == "__main__":
    print(f"Serving {ROOT} on port {args.port}", flush=True)
    ThreadingHTTPServer(("", args.port), Handler).serve_forever()
