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

import urllib.parse
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The only route_dir subdirectories the viewer is allowed to read. Anything
# else the route directory holds (gps.csv, imu.csv, ...) stays unreachable
# over HTTP.
_ROUTE_PREFIXES = ("frames", "detections")


def _resolve_path(request_path: str, out_dir: Path, route_dir: Path) -> Path:
    """Map one HTTP request path to the file it should be served from.

    Mirrors SimpleHTTPRequestHandler.translate_path's own traversal handling
    (query/fragment stripped, percent-decoded, normalized, any "." or ".."
    segment dropped rather than honoured) so forwarding /frames/ and
    /detections/ to route_dir is exactly as safe as stock static-file
    serving -- a crafted path cannot walk out of the directory it resolves
    into.
    """
    path = request_path.split("?", 1)[0].split("#", 1)[0]
    path = urllib.parse.unquote(path)
    words = [word for word in path.split("/") if word and word not in (".", "..")]

    base = route_dir if words and words[0] in _ROUTE_PREFIXES else out_dir
    result = base
    for word in words:
        result = result / word
    return result


class _QuietHandler(SimpleHTTPRequestHandler):
    """Static handler without the per-request logging, which drowns the console."""

    def __init__(self, *args: object, route_dir: Path, **kwargs: object) -> None:
        self._route_dir = Path(route_dir)
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        return str(_resolve_path(path, Path(self.directory), self._route_dir))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def end_headers(self) -> None:
        # Re-running `irap view` rewrites data.json and refreshes the route's
        # frames/detections in place. Without this the browser would keep
        # showing the previous run.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve_directory(
    directory: Path, route_dir: Path, port: int = 8000, open_browser: bool = True
) -> None:
    """Serve a directory until interrupted. Blocks."""
    directory = Path(directory).resolve()
    route_dir = Path(route_dir).resolve()
    handler = partial(_QuietHandler, directory=str(directory), route_dir=route_dir)

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
