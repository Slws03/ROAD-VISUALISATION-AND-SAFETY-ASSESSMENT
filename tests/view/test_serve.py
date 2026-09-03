"""Request-path mapping for the viewer's static server.

The server has to reach two directories the viewer's own output directory
does not contain: the route's frames and its raw detection files. These
tests pin exactly which prefixes escape the output directory and where they
land, since a mistake here is a silent 404 or, worse, a path escaping the
route directory entirely.
"""

from pathlib import Path

from irap_screening.view.serve import _resolve_path

OUT_DIR = Path("/srv/route/view")
ROUTE_DIR = Path("/srv/route")


def test_an_ordinary_path_resolves_under_out_dir():
    assert _resolve_path("/data.json", OUT_DIR, ROUTE_DIR) == OUT_DIR / "data.json"


def test_the_root_path_resolves_to_out_dir_itself():
    assert _resolve_path("/", OUT_DIR, ROUTE_DIR) == OUT_DIR


def test_frames_prefix_resolves_under_route_dir():
    assert (
        _resolve_path("/frames/f_000000.jpg", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "frames" / "f_000000.jpg"
    )


def test_detections_prefix_resolves_under_route_dir():
    assert (
        _resolve_path("/detections/lane.jsonl", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "detections" / "lane.jsonl"
    )


def test_query_and_fragment_are_stripped():
    assert (
        _resolve_path("/frames/f_000000.jpg?t=1#x", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "frames" / "f_000000.jpg"
    )


def test_parent_segments_cannot_escape_the_exposed_directory():
    """A crafted path must not walk out of frames/ or route_dir."""
    assert (
        _resolve_path("/frames/../../secret.txt", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "frames" / "secret.txt"
    )
    assert (
        _resolve_path("/../secret.txt", OUT_DIR, ROUTE_DIR) == OUT_DIR / "secret.txt"
    )


def test_a_segment_with_an_embedded_separator_cannot_escape_route_dir():
    """A single URL segment can smuggle a path separator (e.g. a literal
    backslash on Windows, where os.path.dirname treats it as one), which
    would otherwise let a crafted request walk out of frames/ even though
    "." and ".." alone are filtered. _resolve_path receives the path after
    percent-decoding (that happens inside this function), so the raw
    encoded request "/frames%5C..%5C..%5Cgps.csv" arrives here already
    decoded as a literal backslash sequence.
    """
    assert (
        _resolve_path("/frames/..\\..\\gps.csv", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "frames"
    )


def test_a_segment_with_an_embedded_separator_cannot_escape_out_dir():
    """Same trick, but aimed at escaping out_dir entirely."""
    assert _resolve_path("/..\\..\\secret.txt", OUT_DIR, ROUTE_DIR) == OUT_DIR
