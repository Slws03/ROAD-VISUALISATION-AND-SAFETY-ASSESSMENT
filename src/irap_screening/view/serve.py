"""Serve the built viewer directory over HTTP.

The page fetches data.json, which a browser refuses to do from a file:// URL, so
the viewer needs a server even though nothing about it is dynamic. The standard
library is enough: this only hands out static files, and adding a web framework
for that would pull a dependency into a project whose CPU layer deliberately has
almost none.
"""

from __future__ import annotations

import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _QuietHandler(SimpleHTTPRequestHandler):
    """Static handler without the per-request logging, which drowns the console."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def end_headers(self) -> None:
        # Re-running `irap view` rewrites data.json and the evidence images in
        # place. Without this the browser would keep showing the previous run.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve_directory(directory: Path, port: int = 8000, open_browser: bool = True) -> None:
    """Serve a directory until interrupted. Blocks."""
    directory = Path(directory).resolve()
    handler = partial(_QuietHandler, directory=str(directory))

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
