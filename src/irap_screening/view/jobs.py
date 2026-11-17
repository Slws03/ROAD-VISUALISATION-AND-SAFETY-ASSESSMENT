"""Receive an uploaded route and drive the detectors over it.

The viewer's upload control makes the page the front of the whole chain: a zip
of a prepared route comes in, the two detectors run over it, and the route
joins the ones already on the page. Everything a job produces lands in the
route's own directory, in the layout a manual run leaves behind, so
`irap report` works on an uploaded route exactly as on any other.

The two detectors run as subprocesses. CLRerNet lives in a separate
environment, and a process that exits gives its model's memory back, so the
two never hold the GPU at once.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from irap_screening.config import Settings, flatten, load_config
from irap_screening.view.export import add_route, is_route_dir

Log = Callable[[str], None]
Step = tuple[str, Callable[[Log], None]]

_ROUTE_ID = re.compile(r"[A-Za-z0-9_-]+")
_LOG_LINES = 20


class UploadError(ValueError):
    """The upload itself is unusable: not a zip, or not a route."""


class Conflict(Exception):
    """The upload cannot run now: a job is running, or the name is taken."""


def valid_route_id(name: str) -> bool:
    """A route id becomes a directory name and a URL segment, so it stays plain."""
    return _ROUTE_ID.fullmatch(name) is not None


def extract_route(zip_path: Path, route_dir: Path) -> None:
    """Unpack a route zip into `route_dir`, which must not exist yet.

    A zip is input from outside: a member may name a path that climbs out of
    the target, so every member's resolved destination is checked before a
    byte is written. Whatever goes wrong, nothing half-extracted is left.
    """
    if route_dir.exists():
        raise Conflict(
            f"a route named {route_dir.name} already exists; delete {route_dir} to replace it"
        )
    root = route_dir.resolve()
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            paths = [member.filename.split("/") for member in members]
            # Zipping a folder from Explorer nests the route one level down.
            wrapped = len({parts[0] for parts in paths}) == 1 and all(
                len(parts) > 1 for parts in paths
            )
            skip = 1 if wrapped else 0
            for member, parts in zip(members, paths, strict=True):
                target = root.joinpath(*parts[skip:])
                if not target.resolve().is_relative_to(root):
                    raise UploadError(f"zip member escapes the route: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        if not is_route_dir(root) or not any((root / "frames").glob("*")):
            raise UploadError(
                "the zip is not a route: it needs frames.csv, gps.csv, imu.csv "
                "and a non-empty frames/"
            )
    except Exception as error:
        shutil.rmtree(root, ignore_errors=True)
        if isinstance(error, zipfile.BadZipFile):
            raise UploadError("not a zip file") from error
        raise


def run_command(command: list[str], log: Log) -> None:
    """Run one detector to completion, feeding its output to the job log."""
    # A child Python otherwise buffers its output until it exits, and encodes
    # it in the console's code page rather than UTF-8.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    last = ""
    for line in process.stdout:
        if line.strip():
            last = line.strip()
            log(last)
    if process.wait() != 0:
        raise RuntimeError(last or f"exit code {process.returncode}")


class JobRunner:
    """Runs one job at a time on a background thread and reports where it is."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._log: deque[str] = deque(maxlen=_LOG_LINES)
        self._status: dict = {"state": "idle"}
        self._thread: threading.Thread | None = None

    def status(self) -> dict:
        with self._lock:
            return {**self._status, "log": list(self._log)}

    def start(self, route_id: str, steps: list[Step], kind: str = "upload") -> None:
        with self._lock:
            # ponytail: one job at a time; a queue if routes ever arrive
            # faster than they finish.
            if self._status["state"] == "running":
                raise Conflict("a job is already running")
            self._log.clear()
            self._status = {
                "state": "running",
                "kind": kind,
                "route_id": route_id,
                "step": 0,
                "steps": len(steps),
                "label": "",
                "error": None,
            }
            self._thread = threading.Thread(target=self._run, args=(steps,), daemon=True)
            self._thread.start()

    def _update(self, **changes: object) -> None:
        with self._lock:
            self._status.update(changes)

    def _append(self, line: str) -> None:
        with self._lock:
            self._log.append(line)

    def _run(self, steps: list[Step]) -> None:
        try:
            for number, (label, run) in enumerate(steps, start=1):
                with self._lock:
                    self._status.update(step=number, label=label)
                    self._log.clear()
                run(self._append)
        except Exception as error:  # noqa: BLE001 (whatever stopped the step is the report)
            self._update(state="failed", error=str(error) or type(error).__name__)
        else:
            self._update(state="done")


@dataclass
class Upload:
    """What the upload control needs to turn a zip into a served route."""

    routes_root: Path
    out_dir: Path
    cfg: Settings
    lane_command: list[str]
    sign_command: list[str]
    # The server's live route table. Publishing adds to it, which is what
    # makes a new route's frames reachable without a restart.
    route_dirs: dict[str, Path] = field(default_factory=dict)
    runner: JobRunner = field(default_factory=JobRunner)


def pipeline_steps(upload: Upload, route_id: str, zip_path: Path) -> list[Step]:
    """Everything between a received zip and a route on the page, in order."""
    route = upload.routes_root / route_id

    def extract(log: Log) -> None:
        try:
            extract_route(zip_path, route)
        finally:
            zip_path.unlink(missing_ok=True)

    def lanes(log: Log) -> None:
        run_command([*upload.lane_command, "--route", str(route)], log)

    def lane_import(log: Log) -> None:
        from irap_screening.runners.lane import export

        export(route / "clrernet_raw.json", route, upload.cfg)

    def signs(log: Log) -> None:
        run_command([*upload.sign_command, "--route", str(route)], log)

    def report(log: Log) -> None:
        add_route(route_id, route, upload.route_dirs, upload.cfg, upload.out_dir)

    return [
        ("Extracting", extract),
        ("Lane detection", lanes),
        ("Lane import", lane_import),
        ("Sign detection", signs),
        ("Report", report),
    ]


# Read by the detectors, not the report: their values are baked into the cached
# detections, so changing them here would change nothing until the detectors
# run again.
DETECTOR_KEYS = ("sign.runner_conf", "sign.imgsz")


@dataclass
class Rebuild:
    """What the viewer's config panel needs to re-run the report under new thresholds.

    Only the CPU layer runs again, over each route's cached detections, so a
    rebuild needs no model and takes seconds per route. Panel changes are
    layered over the configuration the server started with and kept in memory:
    restarting `irap view` returns to that configuration.
    """

    out_dir: Path
    cfg: Settings
    profile: Path | None = None
    overrides: list[str] = field(default_factory=list)
    upload: Upload | None = None
    # The server's live route table, shared with the upload control, so a
    # route uploaded this session is rebuilt too and stays in routes.json.
    route_dirs: dict[str, Path] = field(default_factory=dict)
    runner: JobRunner = field(default_factory=JobRunner)
    panel: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.upload is not None:
            # One runner: an upload and a rebuild both write the output directory.
            self.runner = self.upload.runner

    def state(self) -> dict:
        """Every threshold now in force, and where each value came from."""
        values = flatten(self.cfg.model_dump(mode="json"))
        sources = {key: self.cfg.provenance.get(key, "default") for key in values}
        sources.update(dict.fromkeys(self.panel, "viewer"))
        return {"values": values, "sources": sources, "detector_keys": list(DETECTOR_KEYS)}

    def apply(self, changes: dict[str, str]) -> None:
        """Validate the changes and start re-running every route under them.

        Raises ValueError, naming the problem, if a key is unknown, belongs to
        a detector, or its value fails the configuration's own validation.
        """
        known = flatten(self.cfg.model_dump(mode="json"))
        for key in changes:
            if key not in known:
                raise ValueError(f"unknown setting: {key}")
            if key in DETECTOR_KEYS:
                raise ValueError(f"{key} is read by the detector; re-run it to change this")
        panel = {**self.panel, **{key: str(value) for key, value in changes.items()}}
        # The same loader as the command line, so every range check applies.
        cfg = load_config(
            self.profile, [*self.overrides, *(f"{k}={v}" for k, v in panel.items())]
        )
        steps = [
            (f"Report {route_id}", self._step(route_id, cfg))
            for route_id in sorted(self.route_dirs)
        ]
        self.runner.start("config", steps, kind="config")
        self.cfg, self.panel = cfg, panel
        if self.upload is not None:
            self.upload.cfg = cfg

    def _step(self, route_id: str, cfg: Settings) -> Callable[[Log], None]:
        def run(log: Log) -> None:
            add_route(route_id, self.route_dirs[route_id], self.route_dirs, cfg, self.out_dir)

        return run
