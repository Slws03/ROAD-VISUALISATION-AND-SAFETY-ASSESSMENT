"""Receive an uploaded route and drive the detectors over it.

The viewer's upload control makes the page the front of the whole chain: a zip
of a prepared route comes in, the two detectors run over it, and the route
joins the ones already on the page. Everything a job produces lands in the
route's own directory, in the layout a manual run leaves behind, so
`irap report` works on an uploaded route exactly as on any other.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from irap_screening.view.export import is_route_dir

_ROUTE_ID = re.compile(r"[A-Za-z0-9_-]+")


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
