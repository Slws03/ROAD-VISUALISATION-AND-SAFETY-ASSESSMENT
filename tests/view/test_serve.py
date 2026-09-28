"""Request-path mapping for the viewer's static server.

The server has to reach two directories the viewer's own output directory
does not contain: the route's frames and its raw detection files. These
tests pin exactly which prefixes escape the output directory and where they
land, since a mistake here is a silent 404 or, worse, a path escaping the
route directory entirely.
"""

from pathlib import Path

from irap_screening.view.serve import (
    _resolve_multi_path,
    _resolve_path,
    _review_write_target,
    _review_write_target_multi,
)

OUT_DIR = Path("/srv/route/view")
ROUTE_DIR = Path("/srv/route")
ROUTE_DIRS = {"route_a": Path("/data/route_a"), "route_b": Path("/data/route_b")}


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


def test_multi_route_data_json_resolves_under_out_dir():
    """<route_id>/data.json is an ordinary file export_routes already wrote there."""
    assert (
        _resolve_multi_path("/route_a/data.json", OUT_DIR, ROUTE_DIRS)
        == OUT_DIR / "route_a" / "data.json"
    )


def test_multi_route_frames_prefix_resolves_under_that_routes_dir():
    assert (
        _resolve_multi_path("/route_a/frames/f_000000.jpg", OUT_DIR, ROUTE_DIRS)
        == ROUTE_DIRS["route_a"] / "frames" / "f_000000.jpg"
    )


def test_multi_route_detections_prefix_resolves_under_that_routes_dir():
    assert (
        _resolve_multi_path("/route_b/detections/lane.jsonl", OUT_DIR, ROUTE_DIRS)
        == ROUTE_DIRS["route_b"] / "detections" / "lane.jsonl"
    )


def test_multi_route_unknown_route_id_falls_back_to_out_dir():
    """A path that merely looks like /<id>/frames/... but names no known route
    is just an ordinary path under out_dir, not an error."""
    assert (
        _resolve_multi_path("/nope/frames/x.jpg", OUT_DIR, ROUTE_DIRS)
        == OUT_DIR / "nope" / "frames" / "x.jpg"
    )


def test_multi_route_root_resolves_to_out_dir_itself():
    assert _resolve_multi_path("/", OUT_DIR, ROUTE_DIRS) == OUT_DIR
    assert _resolve_multi_path("/routes.json", OUT_DIR, ROUTE_DIRS) == OUT_DIR / "routes.json"


def test_review_write_target_accepts_only_the_review_path():
    assert _review_write_target("/review", ROUTE_DIR) == ROUTE_DIR


def test_review_write_target_rejects_anything_else():
    assert _review_write_target("/data.json", ROUTE_DIR) is None
    assert _review_write_target("/frames/review", ROUTE_DIR) is None
    assert _review_write_target("/", ROUTE_DIR) is None


def test_multi_review_write_target_resolves_the_named_route():
    assert (
        _review_write_target_multi("/route_a/review", ROUTE_DIRS)
        == ROUTE_DIRS["route_a"]
    )


def test_multi_review_write_target_rejects_an_unknown_route():
    assert _review_write_target_multi("/nope/review", ROUTE_DIRS) is None


def test_multi_review_write_target_rejects_a_bare_route_id():
    assert _review_write_target_multi("/route_a", ROUTE_DIRS) is None
