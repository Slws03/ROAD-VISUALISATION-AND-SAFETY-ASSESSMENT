"""Building the local viewer directory from a route.

The viewer's job is to let a flag be checked against pixels, so the tests care
about the link between the two staying intact: every segment must be able to
reach its frames, and every frame the route path the browser will fetch it
from.
"""

import json
from pathlib import Path

from irap_screening.config import load_config
from irap_screening.view.export import TEMPLATE, export_view

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


def test_the_template_ships_with_the_package():
    """It is data, not code, so a packaging change could drop it silently."""
    assert TEMPLATE.exists()
