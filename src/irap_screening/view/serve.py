"""Serve the built viewer directory over HTTP.

The page fetches data.json, which a browser refuses to do from a file:// URL, so
the viewer needs a server even though nothing about it is dynamic. The standard
library is enough: this only hands out static files, and adding a web framework
for that would pull a dependency into a project whose CPU layer deliberately has
almost none.

The viewer's own output directory does not hold the route's frames or its raw
detection files -- those stay where the route wrote them, one level up. Rather
than duplicate them into the output directory, two URL prefixes are forwarded
straight to the route directory instead.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import webbrowser
from collections.abc import Callable
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from irap_screening.decision.report import SCORING_MODULES
from irap_screening.view.review import save_review_entry

Resolver = Callable[[str, Path], Path]
WriteTarget = Callable[[str], "Path | None"]
_VERDICTS = ("ok", "flag")

# The only route_dir subdirectories the viewer is allowed to read. Anything
# else the route directory holds (gps.csv, imu.csv, ...) stays unreachable
# over HTTP.
_ROUTE_PREFIXES = ("frames", "detections")


def _safe_segments(request_path: str) -> list[str]:
    """Split a request path into path segments safe to rejoin under a base directory.

    Mirrors SimpleHTTPRequestHandler.translate_path's own traversal handling:
    query/fragment stripped, percent-decoded, and each "/"-split segment is
    dropped rather than honoured if it is "." or ".." *or* if it
    contains an embedded path separator of its own (e.g. a literal backslash,
    which os.path.dirname treats as a separator on Windows) -- otherwise a
    single URL segment could smuggle a traversal past the "." / ".." filter.
    """
    path = request_path.split("?", 1)[0].split("#", 1)[0]
    path = urllib.parse.unquote(path)
    return [
        word
        for word in path.split("/")
        if word and word not in (".", "..") and not os.path.dirname(word)
    ]


def _resolve_path(request_path: str, out_dir: Path, route_dir: Path) -> Path:
    """Map one HTTP request path to the file it should be served from.

    With the traversal guards in _safe_segments, forwarding /frames/ and
    /detections/ to route_dir is exactly as safe as stock static-file
    serving -- a crafted path cannot walk out of the directory it resolves into.
    """
    words = _safe_segments(request_path)
    base = route_dir if words and words[0] in _ROUTE_PREFIXES else out_dir
    result = base
    for word in words:
        result = result / word
    return result


def _review_write_target(request_path: str, route_dir: Path) -> Path | None:
    """Where a POST /review should write, for a single-route server."""
    return route_dir if _safe_segments(request_path) == ["review"] else None


def _review_write_target_multi(
    request_path: str, route_dirs: dict[str, Path]
) -> Path | None:
    """Where a POST /<route_id>/review should write, for a multi-route server."""
    words = _safe_segments(request_path)
    if len(words) == 2 and words[1] == "review" and words[0] in route_dirs:
        return route_dirs[words[0]]
    return None


def _resolve_multi_path(
    request_path: str, out_dir: Path, route_dirs: dict[str, Path]
) -> Path:
    """Like _resolve_path, but for a page serving several routes at once.

    Each route's frames/detections are scoped under its own id segment
    (/<route_id>/frames/...), since one shared page can no longer tell routes
    apart by the /frames//detections/ prefix alone. Everything else --
    including /<route_id>/data.json -- resolves as an ordinary file under
    out_dir, which already holds each route's exported subdirectory.
    """
    words = _safe_segments(request_path)
    if len(words) >= 2 and words[0] in route_dirs and words[1] in _ROUTE_PREFIXES:
        base, remainder = route_dirs[words[0]], words[1:]
    else:
        base, remainder = out_dir, words
    result = base
    for word in remainder:
        result = result / word
    return result


class _QuietHandler(SimpleHTTPRequestHandler):
    """Static handler without the per-request logging, which drowns the console.

    Also the only place this project accepts a write: POST /review (or
    /<route_id>/review under the multi-route page) records one human review
    verdict. Nothing else is writable -- everything else stays plain static
    file serving.
    """

    def __init__(
        self, *args: object, resolve: Resolver, write_target: WriteTarget, **kwargs: object
    ) -> None:
        self._resolve = resolve
        self._write_target = write_target
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        return str(self._resolve(path, Path(self.directory)))

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's own naming)
        route_dir = self._write_target(self.path)
        if route_dir is None:
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            segment_id = str(payload["segment_id"])
            module = str(payload["module"])
            verdict = str(payload["verdict"])
            note = str(payload.get("note", ""))
        except (ValueError, KeyError, TypeError):
            self.send_error(400, "Malformed review payload")
            return
        if module not in SCORING_MODULES or verdict not in _VERDICTS:
            self.send_error(400, "Unknown module or verdict")
            return

        entry = save_review_entry(route_dir, segment_id, module, verdict, note)
        body = json.dumps(entry).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def end_headers(self) -> None:
        # Re-running `irap view` rewrites data.json and refreshes the route's
        # frames/detections in place. Without this the browser would keep
        # showing the previous run.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def _run(
    directory: Path,
    resolve: Resolver,
    write_target: WriteTarget,
    port: int,
    open_browser: bool,
) -> None:
    """Serve `directory` until interrupted, resolving requests via `resolve`. Blocks."""
    handler = partial(
        _QuietHandler, directory=str(directory), resolve=resolve, write_target=write_target
    )
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"Serving {directory} at {url}")
        print("Press Ctrl+C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def serve_directory(
    directory: Path, route_dir: Path, port: int = 8000, open_browser: bool = True
) -> None:
    """Serve one route's viewer directory until interrupted. Blocks."""
    directory = Path(directory).resolve()
    route_dir = Path(route_dir).resolve()
    _run(
        directory,
        lambda p, out: _resolve_path(p, out, route_dir),
        lambda p: _review_write_target(p, route_dir),
        port,
        open_browser,
    )


def serve_routes(
    directory: Path,
    route_dirs: dict[str, Path],
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Serve a multi-route viewer directory until interrupted. Blocks."""
    directory = Path(directory).resolve()
    route_dirs = {route_id: Path(path).resolve() for route_id, path in route_dirs.items()}
    _run(
        directory,
        lambda p, out: _resolve_multi_path(p, out, route_dirs),
        lambda p: _review_write_target_multi(p, route_dirs),
        port,
        open_browser,
    )
