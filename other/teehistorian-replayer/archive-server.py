#!/usr/bin/env python3
# Streams teehistorian recordings to the teehistorian replayer web page,
# decompressing .xz archives on the fly and sending CORS headers.
#
# Usage: archive-server.py [port] [root]
#
# GET /<subdir>/<name> serves <root>/<subdir>/<name>, also trying the
# .teehistorian and .teehistorian.xz suffixes, so the archive layout
# /media/teehistorian2/data/<region>/<game_uuid>.teehistorian.xz can be
# fetched as /<region>/<game_uuid>.
#
# The replayer reads the stream sequentially and closes the connection as soon
# as the requested time range has been converted, which also stops the xz
# decompression, so serving only the needed prefix of a multi-GiB recording is
# cheap.

import pathlib
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8140
ROOT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/media/teehistorian2/data").resolve()


class Handler(BaseHTTPRequestHandler):
	protocol_version = "HTTP/1.0"

	def resolve(self):
		try:
			path = (ROOT / self.path.lstrip("/")).resolve()
			path.relative_to(ROOT)
		except (ValueError, OSError):
			return None
		for candidate in (path, path.with_name(path.name + ".teehistorian"), path.with_name(path.name + ".teehistorian.xz")):
			if candidate.is_file():
				return candidate
		return None

	def do_GET(self):
		path = self.resolve()
		if path is None:
			self.send_error(404)
			return
		compressed = path.suffix == ".xz"
		self.send_response(200)
		self.send_header("Access-Control-Allow-Origin", "*")
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
	print(f"Serving {ROOT} on port {PORT}", flush=True)
	ThreadingHTTPServer(("", PORT), Handler).serve_forever()
