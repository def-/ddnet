"""Locate teehistorian recordings by game uuid and convert single rank runs to
demos via the teehistorian2demo tool's --rank mode. Shared between
archive-server.py (on-demand) and pregen.py (nightly pre-generation)."""

import gzip
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

UUID_RE = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
MAP_DOWNLOAD_URL = "https://maps.ddnet.org"
CONVERT_TIMEOUT = 15 * 60
HEADER_MAX_SIZE = 1024 * 1024


class RankDemoError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Converter:
    def __init__(self, tool, archive_root, cache_dir, cache_limit_bytes, scramble_tool=None):
        self.tool = Path(tool)
        # Every published demo goes through the scrambler, a raw one is an
        # input recording of the run and could be replayed by a bot
        self.scramble_tool = Path(scramble_tool) if scramble_tool else self.tool.with_name("demo_scramble")
        if not self.scramble_tool.is_file():
            raise FileNotFoundError(f"Scrambler not built: {self.scramble_tool}")
        self.root = Path(archive_root)
        self.cache = Path(cache_dir)
        self.demos = self.cache / "demos"
        self.maps = self.cache / "maps"
        self.tmp = self.cache / "tmp"
        for directory in (self.demos, self.maps, self.tmp):
            directory.mkdir(parents=True, exist_ok=True)
        self.cache_limit_bytes = cache_limit_bytes
        self.locks = {}
        self.locks_mutex = threading.Lock()
        self.index = None
        self.unindexed = []

    def load_index(self, uuids):
        """Which location directory holds a game uuid, read from the archive
        indexes that archive.sh appends to when a recording arrives. Proving a
        recording absent otherwise costs a stat in every location directory,
        seconds each on the archive disk, and most ranks old enough to be a
        record have no recording left."""
        wanted = {uuid.encode() for uuid in uuids}
        self.index = {}
        self.unindexed = []
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir():
                continue
            index = sub / "index.txt.gz"
            opener = gzip.open
            if not index.is_file():
                # A location that started recording after the last daily
                # gzip run, its index is still the plain file
                index = sub / "index.txt"
                opener = open
            if not index.is_file():
                self.unindexed.append(sub)
                continue
            with opener(index, "rb") as file:
                for line in file:
                    if line[:36] in wanted:
                        self.index[line[:36].decode()] = sub
        return len(self.index)

    def find_recording(self, uuid):
        """Exact-path stats only: the per-region directories are too large to
        list, but the rank's game uuid is the file name."""
        if self.index is None:
            directories = [sub for sub in sorted(self.root.iterdir()) if sub.is_dir()]
        else:
            directories = self.unindexed[:]
            if uuid in self.index:
                directories.insert(0, self.index[uuid])
        for sub in directories:
            for ext in (".teehistorian", ".teehistorian.xz"):
                path = sub / (uuid + ext)
                if path.is_file():
                    return path
        return None

    def read_header(self, path):
        """The file starts with 16 magic bytes, then the json header terminated
        by a null byte."""
        if path.suffix == ".xz":
            with subprocess.Popen(["xz", "-dc", str(path)], stdout=subprocess.PIPE) as process:
                data = process.stdout.read(HEADER_MAX_SIZE)
                process.kill()
        else:
            with open(path, "rb") as file:
                data = file.read(HEADER_MAX_SIZE)
        end = data.find(b"\0", 16)
        if len(data) < 17 or end < 0:
            raise RankDemoError(500, "Unparsable teehistorian header")
        try:
            return json.loads(data[16:end])
        except (UnicodeDecodeError, ValueError) as error:
            raise RankDemoError(500, f"Corrupt teehistorian header: {error}") from error

    def fetch_map(self, map_name, map_sha256):
        # The hash is the file name and half the URL, and it comes out of the
        # recording, so it has to look like a hash before it is used as either
        if not map_sha256:
            raise RankDemoError(500, "The recording has no map_sha256 in its header, the map cannot be looked up")
        if not re.fullmatch(r"[0-9a-f]{64}", map_sha256):
            raise RankDemoError(500, "The recording names a map hash that is not one")
        path = self.maps / f"{map_sha256}.map"
        if path.is_file():
            return path
        url = f"{MAP_DOWNLOAD_URL}/{urllib.parse.quote(map_name)}_{map_sha256}.map"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except OSError as error:
            raise RankDemoError(500, f"Failed to download map: {error}") from error
        if hashlib.sha256(data).hexdigest() != map_sha256:
            raise RankDemoError(500, f"The map {map_name} does not hash to what the recording says")
        # Moved into place, a kill during the write would otherwise leave a
        # short map in the cache that is never downloaded again
        # Named per thread, two conversions may fetch the same map at once
        temp = path.with_name(f"{path.name}.{threading.get_ident()}.new")
        temp.write_bytes(data)
        temp.replace(path)
        return path

    def demo_paths(self, uuid, time_str, names):
        """The published demo, the converter output it was scrambled from, and
        the metadata. Demos are kept gzipped, which is a third off the disk
        here, off the upload and off every download: nginx serves the file as
        it is and the browser unpacks it.

        The converter output stays beside the published demo and is never
        uploaded. It is what makes a change to the scrambler a re-scramble of
        minutes (rescramble.py) instead of converting every recording again."""
        key = hashlib.sha1("|".join([uuid, time_str] + names).encode()).hexdigest()[:16]
        base = self.demos / f"{uuid}-{key}"
        return base.with_suffix(".demo.gz"), base.with_suffix(".raw.demo.gz"), base.with_suffix(".json")

    def prev_chain(self, recording, header, depth=3):
        """Previous recordings of the same server (oldest first), needed to
        learn the names of players that joined before the recording started."""
        chain = []
        current = header
        while depth > 0:
            prev_uuid = current.get("prev_game_uuid", "")
            if not UUID_RE.match(prev_uuid):
                break
            prev = None
            for ext in (".teehistorian", ".teehistorian.xz"):
                candidate = recording.parent / (prev_uuid + ext)
                if candidate.is_file():
                    prev = candidate
                    break
            if prev is None:
                break
            chain.append(prev)
            current = self.read_header(prev)
            depth -= 1
        chain.reverse()
        return chain

    def materialize(self, recording, workdir, name):
        """Decompress .xz recordings into the workdir, the tool parses a
        recording twice and needs a seekable plain file."""
        if recording.suffix != ".xz":
            return recording
        path = workdir / f"{name}.teehistorian"
        with open(path, "wb") as file:
            subprocess.run(["xz", "-dc", str(recording)], stdout=file, check=True, timeout=CONVERT_TIMEOUT)
        return path

    def convert(self, uuid, time_str, names, ts_epoch=None, reconvert=False):
        """Returns (demo_path, meta_dict), converting and caching on demand."""
        if not UUID_RE.match(uuid):
            raise RankDemoError(400, "Invalid game uuid")
        if not names or not all(names):
            raise RankDemoError(400, "Missing player name")
        try:
            if float(time_str) <= 0:
                raise ValueError
        except ValueError:
            raise RankDemoError(400, "Invalid rank time") from None

        demo_path, raw_path, meta_path = self.demo_paths(uuid, time_str, names)
        with self.locks_mutex:
            lock = self.locks.setdefault(demo_path.name, threading.Lock())
        with lock:
            # reconvert reads the recording again, for a demo that was made by
            # an older converter. Everything is written to a work directory and
            # moved into place, so a conversion that fails leaves the demo that
            # is already published alone.
            # A reconvert brings demos up to date with the converter, one that
            # was made after the tools were last built is up to date already,
            # so a run that was stopped picks up where it was
            if reconvert and demo_path.is_file() and meta_path.is_file():
                tools_built = max(self.tool.stat().st_mtime, self.scramble_tool.stat().st_mtime)
                if meta_path.stat().st_mtime >= tools_built:
                    reconvert = False
            if not reconvert:
                if demo_path.is_file() and meta_path.is_file():
                    return demo_path, json.loads(meta_path.read_text())
                if raw_path.is_file() and meta_path.is_file():
                    # Converted before, only the scrambling is missing
                    self.scramble_cached(raw_path, demo_path)
                    return demo_path, json.loads(meta_path.read_text())

            recording = self.find_recording(uuid)
            if recording is None:
                raise RankDemoError(404, "Recording not in the archive (yet)")
            header = self.read_header(recording)
            map_name = header.get("map_name", "unknown")
            map_sha256 = header.get("map_sha256", "")
            map_path = self.fetch_map(map_name, map_sha256)

            offset = "-"
            if ts_epoch is not None and "start_time" in header:
                start = datetime.strptime(header["start_time"], "%Y-%m-%dT%H:%M:%S%z")
                seconds = int(ts_epoch - start.timestamp())
                if seconds >= 0:
                    offset = str(seconds)

            workdir = Path(tempfile.mkdtemp(dir=self.tmp))
            try:
                input_path = self.materialize(recording, workdir, "input")
                try:
                    meta = self.run_tool(workdir, input_path, map_path, [], time_str, offset, names)
                except RankDemoError as error:
                    # The names of players that joined before the recording
                    # started are only in the previous recordings, retry with
                    # them seeded. Only helps recordings between 2023-08
                    # (prev_game_uuid added) and 2024-04 (player-name chunks
                    # added); older recordings have no chain pointer at all.
                    chain = self.prev_chain(recording, header) if error.status == 404 else []
                    if not chain:
                        raise
                    prev_args = []
                    for index, prev in enumerate(chain):
                        prev_args += ["--prev", str(self.materialize(prev, workdir, f"prev{index}"))]
                    meta = self.run_tool(workdir, input_path, map_path, prev_args, time_str, offset, names)
                self.scramble(workdir, "out.demo", "watch.demo", self.scramble_key(demo_path))
                for name in ("out.demo", "watch.demo"):
                    with open(workdir / (name + ".gz"), "wb") as compressed:
                        subprocess.run(["gzip", "-9", "-c", name], cwd=workdir,
                            stdout=compressed, check=True, timeout=CONVERT_TIMEOUT)
                meta.update(uuid=uuid, time=time_str, names=names, map_name=map_name, map_sha256=map_sha256,
                    rev=self.revision(workdir / "watch.demo.gz"))
                self.write_meta(meta_path, meta)
                shutil.move(workdir / "out.demo.gz", raw_path)
                shutil.move(workdir / "watch.demo.gz", demo_path)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
            self.prune()
            return demo_path, meta

    def run_tool(self, workdir, input_path, map_path, prev_args, time_str, offset, names):
        result = subprocess.run(
            [str(self.tool), str(input_path), str(map_path), "out.demo"] + prev_args + ["--rank", time_str, offset] + names,
            cwd=workdir, capture_output=True, text=True, timeout=CONVERT_TIMEOUT)
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip().splitlines()
            message = output[-1] if output else "conversion failed"
            status = 404 if "No finish" in message else 500
            raise RankDemoError(status, message)
        meta = None
        for line in result.stdout.splitlines():
            if line.startswith("{"):
                meta = json.loads(line)
        if meta is None:
            raise RankDemoError(500, "Tool reported no rank metadata")
        return meta

    def scramble_key(self, demo_path):
        """The key of one demo's noise. It has to be the same every time that
        demo is scrambled: two publications of one run under two keys average
        out to the run itself, and rescramble.py exists to scramble the whole
        cache again after a change to the tool.

        The secret is written once and kept out of everything that leaves the
        host. Losing it costs nothing but a new noise on the next re-scramble,
        and the demos already published stay as they are."""
        secret_path = self.cache / "scramble.secret"
        if not secret_path.is_file():
            with open(secret_path, "xb", opener=lambda path, flags: os.open(path, flags, 0o600)) as secret:
                secret.write(os.urandom(32))
        secret = secret_path.read_bytes()
        return hmac.new(secret, demo_path.name.encode(), "sha256").hexdigest()[:32]

    def scramble(self, workdir, source, target, key):
        """Noise the run below what a viewer can see, so that the demo cannot
        be turned back into the inputs that produced the rank."""
        result = subprocess.run([str(self.scramble_tool), source, target, "--key", key],
            cwd=workdir, capture_output=True, text=True, timeout=CONVERT_TIMEOUT)
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip().splitlines()
            raise RankDemoError(500, output[-1] if output else "scrambling failed")

    def write_meta(self, meta_path, meta):
        """Written beside the file and moved into place, a reader that finds
        the metadata must find all of it."""
        temp = meta_path.with_suffix(".json.new")
        temp.write_text(json.dumps(meta))
        temp.replace(meta_path)

    def revision(self, path):
        """What the demo currently holds, so that a re-scrambled demo reaches
        the page through a URL the CDN has not cached yet."""
        return hashlib.sha1(path.read_bytes()).hexdigest()[:12]

    def scramble_cached(self, raw_path, demo_path):
        """Scramble a converter output that is already in the cache. The demo
        is written under a temporary name and moved into place, so a reader
        never sees half a file."""
        workdir = Path(tempfile.mkdtemp(dir=self.tmp))
        try:
            with open(workdir / "out.demo", "wb") as raw:
                subprocess.run(["gzip", "-dc", str(raw_path)], stdout=raw, check=True, timeout=CONVERT_TIMEOUT)
            self.scramble(workdir, "out.demo", "watch.demo", self.scramble_key(demo_path))
            with open(workdir / "watch.demo.gz", "wb") as compressed:
                subprocess.run(["gzip", "-9", "-c", "watch.demo"], cwd=workdir,
                    stdout=compressed, check=True, timeout=CONVERT_TIMEOUT)
            _, meta_path = self.siblings(demo_path)
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                meta["rev"] = self.revision(workdir / "watch.demo.gz")
                self.write_meta(meta_path, meta)
            shutil.move(workdir / "watch.demo.gz", demo_path)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def siblings(self, demo_path):
        """The converter output and the metadata beside a published demo.
        Path.with_suffix only replaces ".gz", so the names are built here."""
        base = self.demos / demo_path.name[:-len(".demo.gz")]
        return base.with_name(base.name + ".raw.demo.gz"), base.with_suffix(".json")

    def prune(self):
        """Drop the oldest cached demos above the cache size limit."""
        demos = sorted((path for path in self.demos.glob("*.demo.gz") if not path.name.endswith(".raw.demo.gz")),
            key=lambda path: path.stat().st_mtime)
        size = lambda path: path.stat().st_size if path.is_file() else 0
        total = sum(size(path) + size(self.siblings(path)[0]) for path in demos)
        while demos and total > self.cache_limit_bytes:
            oldest = demos.pop(0)
            raw, meta = self.siblings(oldest)
            total -= size(oldest) + size(raw)
            oldest.unlink(missing_ok=True)
            raw.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
