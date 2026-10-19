"""Building the local viewer directory from a route.

The viewer's job is to let a flag be checked against pixels, so the tests care
about the link between the two staying intact: every segment must be able to
reach its frames, and every frame the route path the browser will fetch it
from.
"""

import json
from pathlib import Path

from irap_screening.config import load_config
from irap_screening.view.export import (
    TEMPLATE,
    add_route,
    discover_routes,
    export_routes,
    export_view,
)
from irap_screening.view.review import save_review_entry
from tests.test_pipeline import _write_route

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "route_demo"


def _export(tmp_path, **kwargs):
    out = tmp_path / "view"
    export_view(FIXTURE, load_config(), out, **kwargs)
    return out


def test_writes_the_two_pieces_the_page_needs(tmp_path):
    out = _export(tmp_path)

    assert (out / "index.html").exists()
    assert (out / "data.json").exists()


def test_no_evidence_directory_is_written(tmp_path):
    """Rendering moved to the browser; nothing should be rasterized any more."""
    out = _export(tmp_path)

    assert not (out / "evidence").exists()


def test_every_referenced_frame_has_a_path_the_browser_can_fetch(tmp_path):
    """A missing map entry reads as a frame the pipeline never saw."""
    out = _export(tmp_path)
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))

    referenced = {fid for segment in data["segments"] for fid in segment["frame_ids"]}
    assert referenced
    assert referenced == set(data["frames"])
    for frame_id in referenced:
        path = data["frames"][frame_id]
        assert path.startswith("frames/")
        assert (FIXTURE / path).exists()


def test_segments_carry_the_frame_ids_the_report_omits(tmp_path):
    """report.json has only a representative frame; the viewer steps through all."""
    out = _export(tmp_path)
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))

    first = data["segments"][0]
    assert len(first["frame_ids"]) > 1
    assert first["representative_frame_id"] == first["frame_ids"][0]


def test_the_report_travels_intact(tmp_path):
    """The viewer must not become a second, divergent summary of the run."""
    out = _export(tmp_path)
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))

    assert data["route_id"] == "R01"
    assert "coverage" in data
    assert "config" in data
    for segment in data["segments"]:
        for module in ("curvature", "grade", "delineation", "condition"):
            assert "flag" in segment[module]


def test_the_page_loads_its_data_at_runtime(tmp_path):
    """Data stays a separate file so a re-run refreshes without rebuilding the page."""
    out = _export(tmp_path)
    html = (out / "index.html").read_text(encoding="utf-8")

    assert "data.json" in html


def test_a_route_without_sign_detections_still_exports(tmp_path):
    """The committed fixture has no sign.jsonl; that is a valid run, not an error."""
    assert not (FIXTURE / "detections" / "sign.jsonl").exists()

    out = _export(tmp_path)
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))

    assert data["frames"]


def test_coverage_uses_total_and_not_assessed(tmp_path):
    """Pins the shape the page reads; it derives assessed from these two.

    The page cannot be unit tested here, so the contract it depends on is
    asserted on the Python side instead.
    """
    out = _export(tmp_path)
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))

    for module in ("curvature", "grade", "delineation", "condition"):
        entry = data["coverage"][module]
        assert set(entry) == {"total", "not_assessed"}
    assert data["coverage"]["condition"]["not_assessed"] == data["coverage"]["condition"]["total"]


def test_segments_carry_an_empty_review_when_none_was_ever_saved(tmp_path):
    out = _export(tmp_path)
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))

    for segment in data["segments"]:
        assert segment["review"] == {}


def test_a_saved_review_merges_into_the_matching_segment(tmp_path):
    """review.json lives beside the route, not in the export output -- it must
    still show up in data.json, or the viewer would have no way to display it."""
    route = tmp_path / "route"
    _write_route(route)
    save_review_entry(route, "R01-0000", "delineation", "ok", "checked against the frame")

    out = tmp_path / "view"
    export_view(route, load_config(), out)
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))

    segment = next(s for s in data["segments"] if s["segment_id"] == "R01-0000")
    assert segment["review"]["delineation"]["verdict"] == "ok"
    other = next(s for s in data["segments"] if s["segment_id"] != "R01-0000")
    assert other["review"] == {}


def test_the_template_ships_with_the_package():
    """It is data, not code, so a packaging change could drop it silently."""
    assert TEMPLATE.exists()


def test_discover_routes_treats_a_route_directory_as_one_route():
    assert discover_routes(FIXTURE) == {FIXTURE.name: FIXTURE}


def test_discover_routes_scans_subdirectories_of_a_parent(tmp_path):
    _write_route(tmp_path / "route_a")
    _write_route(tmp_path / "route_b")
    (tmp_path / "not_a_route").mkdir()  # no frames.csv/gps.csv/imu.csv: skipped

    routes = discover_routes(tmp_path)

    assert routes == {"route_a": tmp_path / "route_a", "route_b": tmp_path / "route_b"}


def test_export_routes_writes_one_shared_page_and_per_route_data(tmp_path):
    _write_route(tmp_path / "route_a")
    _write_route(tmp_path / "route_b")
    out = tmp_path / "view"

    out_dir = export_routes(discover_routes(tmp_path), load_config(), out)

    assert (out_dir / "index.html").exists()
    routes = json.loads((out_dir / "routes.json").read_text(encoding="utf-8"))
    assert {r["id"] for r in routes} == {"route_a", "route_b"}
    for route_id in ("route_a", "route_b"):
        assert (out_dir / route_id / "data.json").exists()
        # Only the shared page at out_dir is served; a nested one would be dead weight.
        assert not (out_dir / route_id / "index.html").exists()


def test_add_route_joins_an_already_exported_directory(tmp_path):
    _write_route(tmp_path / "route_a")
    _write_route(tmp_path / "route_b")
    out = tmp_path / "view"
    routes = {"route_a": tmp_path / "route_a"}
    export_routes(routes, load_config(), out)

    add_route("route_b", tmp_path / "route_b", routes, load_config(), out)

    assert routes["route_b"] == (tmp_path / "route_b").resolve()
    listed = json.loads((out / "routes.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in listed] == ["route_a", "route_b"]
    assert (out / "route_b" / "data.json").exists()
    assert not (out / "route_b" / "index.html").exists()


def test_the_page_carries_the_upload_control_hidden_by_default():
    page = TEMPLATE.read_text(encoding="utf-8")

    assert 'id="upload" hidden' in page
    assert 'fetch("job")' in page
    assert "upload?name=" in page


def test_profile_holds_the_samples_the_flags_were_decided_on(tmp_path):
    """The evidence chart must plot the flag's own numbers, not a re-derivation."""
    out = _export(tmp_path)
    text = (out / "data.json").read_text(encoding="utf-8")
    assert "NaN" not in text  # JSON.parse rejects it; gated samples must be null
    profile = json.loads(text)["profile"]

    lengths = {len(series) for series in profile.values()}
    assert len(lengths) == 1 and lengths.pop() > 0
    radii = set(profile["radius_m"])
    grades = set(profile["grade_pct"])
    for segment in json.loads(text)["segments"]:
        curvature, grade = segment["curvature"], segment["grade"]
        if "radius_m" in curvature:
            assert round(curvature["radius_m"], 1) in radii
        if "grade_pct" in grade:
            assert round(grade["grade_pct"], 2) in grades


def test_grade_window_points_at_the_sample_the_grade_flag_used(tmp_path):
    """The viewer shades this window and marks this sample, so it must be the flag's own."""
    data = json.loads((_export(tmp_path) / "data.json").read_text(encoding="utf-8"))
    profile = data["profile"]
    for segment in data["segments"]:
        grade, window = segment["grade"], segment["grade_window"]
        if "grade_pct" not in grade:
            continue
        assert profile["grade_pct"][window["index"]] == round(grade["grade_pct"], 2)
        assert window["start_m"] <= profile["dist_m"][window["index"]] <= window["end_m"]
        if not window["fallback"]:
            assert window["end_m"] - window["start_m"] >= 10.0 - 0.1  # 0.1: rounding
