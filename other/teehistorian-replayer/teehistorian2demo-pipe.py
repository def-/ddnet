#!/usr/bin/env python3
# Converts a teehistorian recording to a demo for the moderator page, which
# pipes the recording in and takes the demo out. The converter needs the map
# the recording was made on, which is what this adds: the map is looked up in
# the recording's own header and downloaded once into a cache. Recording and
# demo both stream, so a long conversion keeps sending.
#
# Usage: teehistorian2demo-pipe < recording > demo

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

TOOL = os.environ.get("TEEHISTORIAN2DEMO", "/home/teeworlds/bin/teehistorian2demo")
# Not in /var/tmp: every local account can create a directory there, and the
# maps this reads are what the converter replays
MAP_CACHE = pathlib.Path(os.environ.get("TEEHISTORIAN_MAP_CACHE", "/var/lib/th2demo/maps"))
CONVERT_TIMEOUT = 30 * 60
MAP_URL = "https://maps.ddnet.org"
# The header is 16 magic bytes and a json object terminated by a null byte
HEADER_MAX_SIZE = 1024 * 1024


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def read_header(stream):
    """Returns the header and the bytes that were read to find it."""
    data = b""
    while len(data) < HEADER_MAX_SIZE:
        chunk = stream.read(64 * 1024)
        if not chunk:
            break
        data += chunk
        end = data.find(b"\0", 16)
        if end >= 0:
            try:
                return json.loads(data[16:end]), data
            except (UnicodeDecodeError, ValueError) as error:
                fail(f"Corrupt teehistorian header: {error}")
    fail("Unparsable teehistorian header")


def fetch_map(map_name, map_sha256, map_crc):
    """maps.ddnet.org names a map after when it was released: <name>_<sha256>
    since 2020, <name>_<crc>_<sha256> for the years before, <name>_<crc>
    before the server recorded hashes at all. The hashes come out of the
    recording, so they have to look like hashes before they are used as file
    names, and what comes back is checked against them: a cached map is never
    looked at again."""
    if map_sha256 and not re.fullmatch(r"[0-9a-f]{64}", map_sha256):
        fail("The recording names a map hash that is not one")
    if map_crc and not re.fullmatch(r"[0-9a-f]{8}", map_crc):
        fail("The recording names a map crc that is not one")
    if not map_sha256 and not map_crc:
        fail("The recording has neither map_sha256 nor map_crc in its header, the map cannot be looked up")
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    path = MAP_CACHE / f"{map_sha256 or 'crc-' + map_crc}.map"
    if path.is_file():
        return path
    name = urllib.parse.quote(map_name)
    if map_sha256:
        names = [f"{name}_{map_sha256}.map"] + ([f"{name}_{map_crc}_{map_sha256}.map"] if map_crc else [])
    else:
        names = [f"{name}_{map_crc}.map"]
    data = None
    for file_name in names:
        try:
            with urllib.request.urlopen(f"{MAP_URL}/{file_name}", timeout=60) as response:
                data = response.read()
            break
        except urllib.error.HTTPError as error:
            if error.code != 404:
                fail(f"Failed to download the map {map_name}: {error}")
        except OSError as error:
            fail(f"Failed to download the map {map_name}: {error}")
    if data is None:
        fail(f"The map server has none of {', '.join(names)}")
    if map_sha256 and hashlib.sha256(data).hexdigest() != map_sha256:
        fail(f"The map {map_name} does not hash to what the recording says")
    if not map_sha256 and f"{zlib.crc32(data) & 0xffffffff:08x}" != map_crc:
        fail(f"The map {map_name} does not hash to what the recording says")
    temp = path.with_name(path.name + ".new")
    temp.write_bytes(data)
    temp.replace(path)
    return path


def feed(path, head):
    """Writes the recording into the pipe the converter reads."""
    try:
        with open(path, "wb") as pipe:
            pipe.write(head)
            shutil.copyfileobj(sys.stdin.buffer, pipe, 1024 * 1024)
    except BrokenPipeError:
        pass  # the converter stopped early, it says why itself


def main():
    header, head = read_header(sys.stdin.buffer)
    if not isinstance(header, dict):
        fail("The teehistorian header is not an object")
    map_path = fetch_map(header.get("map_name", "unknown"), header.get("map_sha256", ""), header.get("map_crc", ""))

    workdir = tempfile.mkdtemp(prefix="th2demo-")
    try:
        recording = pathlib.Path(workdir) / "in.teehistorian"
        os.mkfifo(recording)
        writer = threading.Thread(target=feed, args=(recording, head), daemon=True)
        writer.start()
        # The recorder writes through the client storage, which only takes a
        # relative path, so the converter runs in the directory it writes to
        demo = pathlib.Path(workdir) / "out.demo"
        process = subprocess.Popen([TOOL, str(recording), str(map_path), "out.demo"],
            cwd=workdir, stdout=sys.stderr, stderr=sys.stderr)
        # Send the demo while it is being written, a long conversion would
        # otherwise sit silent for minutes. A reader that goes away, or a
        # conversion that never ends, must not leave the converter running.
        deadline = time.monotonic() + CONVERT_TIMEOUT
        sent = 0
        try:
            while True:
                running = process.poll() is None
                if demo.is_file():
                    with open(demo, "rb") as file:
                        file.seek(sent)
                        while True:
                            chunk = file.read(1024 * 1024)
                            if not chunk:
                                break
                            sys.stdout.buffer.write(chunk)
                            sent += len(chunk)
                    sys.stdout.buffer.flush()
                if not running:
                    break
                if time.monotonic() > deadline:
                    fail("The conversion took too long")
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        if process.returncode != 0:
            fail(f"The conversion failed with code {process.returncode}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)  # the reader went away, the converter was killed above
