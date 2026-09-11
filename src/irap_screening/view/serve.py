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

import os
import urllib.parse
import webbrowser
from collections.abc import Callable
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

Resolver = Callable[[str, Path], Path]

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
    """Static handler without the per-request logging, which drowns the console."""

    def __init__(self, *args: object, resolve: Resolver, **kwargs: object) -> None:
        self._resolve = resolve
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        return str(self._resolve(path, Path(self.directory)))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def end_headers(self) -> None:
        # Re-running `irap view` rewrites data.json and refreshes the route's
        # frames/detections in place. Without this the browser would keep
        # showing the previous run.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def _run(directory: Path, resolve: Resolver, port: int, open_browser: bool) -> None:
    """Serve `directory` until interrupted, resolving requests via `resolve`. Blocks."""
    handler = partial(_QuietHandler, directory=str(directory), resolve=resolve)
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
    _run(directory, lambda p, out: _resolve_path(p, out, route_dir), port, open_browser)


def serve_routes(
    directory: Path,
    route_dirs: dict[str, Path],
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Serve a multi-route viewer directory until interrupted. Blocks."""
    directory = Path(directory).resolve()
    route_dirs = {route_id: Path(path).resolve() for route_id, path in route_dirs.items()}
    _run(directory, lambda p, out: _resolve_multi_path(p, out, route_dirs), port, open_browser)
