"""The report is the contract with the future dashboard; these tests pin it."""

import json
from pathlib import Path

from irap_screening.config import load_config
from irap_screening.decision.report import (
    SCHEMA_VERSION,
    build_report,
    to_geojson,
    to_markdown,
    write_report,
)
from irap_screening.io.schema import (
    DoubleLineRecord,
    ExistenceResult,
    Flag,
    ModuleResult,
    Recommendation,
    Segment,
    SignRecord,
)

CFG = load_config()


def _fixture():
    segments = [
        Segment(
            segment_id="R01-0000",
            start_m=0.0, end_m=100.0,
            start_latlon=(1.5500, 110.3500), end_latlon=(1.5509, 110.3500),
        ),
        Segment(
            segment_id="R01-0001",
            start_m=100.0, end_m=200.0,
            start_latlon=(1.5509, 110.3500), end_latlon=(1.5518, 110.3500),
        ),
    ]
    module_results = {
        "curvature": [
            ModuleResult(module="curvature", segment_id="R01-0000", flag=Flag.CRITICAL,
                         metrics={"radius_m": 312.5}),
            ModuleResult(module="curvature", segment_id="R01-0001",
                         flag=Flag.NOT_ASSESSED, notes="Below minimum speed."),
        ],
        "grade": [
            ModuleResult(module="grade", segment_id=s.segment_id, flag=Flag.NORMAL,
                         metrics={"grade_pct": 1.0})
            for s in segments
        ],
        "lane": [
            ModuleResult(module="lane", segment_id=s.segment_id, flag=Flag.WATCH,
                         metrics={"weber_contrast": 0.28},
                         evidence=[f"{s.segment_id}#lane#0"])
            for s in segments
        ],
        "condition": [
            ModuleResult(module="condition", segment_id=s.segment_id, flag=Flag.NORMAL,
                         metrics={"pothole_count": 0.0})
            for s in segments
        ],
    }
    existence = [
        ExistenceResult(segment_id=s.segment_id,
                        double_line=DoubleLineRecord(present=False))
        for s in segments
    ]
    recommendations = {
        "R01-0000": [
            Recommendation(
                rule="curve_warning_incomplete",
                trigger={"module": "curvature", "radius_m": 312.5},
                missing=["chevron"], present=["advance_bend_sign"],
                priority="high", text="Curve is missing chevron markers.",
            )
        ],
        "R01-0001": [],
    }
    return segments, module_results, existence, recommendations


def _report():
    return build_report("R01", *_fixture(), cfg=CFG)


def test_report_carries_schema_version_and_config_snapshot():
    report = _report()
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["config"]["curvature"]["threshold_m"] == 400


def test_raw_metrics_travel_alongside_flags():
    """A dashboard needs the continuous value to plot a profile, not just a colour."""
    segment = _report()["segments"][0]
    assert segment["curvature"]["radius_m"] == 312.5
    assert segment["curvature"]["flag"] == "critical"


def test_recommendation_names_the_missing_mitigation():
    entry = _report()["segments"][0]["recommendations"][0]
    assert entry["missing"] == ["chevron"]
    assert entry["present"] == ["advance_bend_sign"]


def test_review_field_is_reserved_and_null():
    assert _report()["segments"][0]["review"] is None


def test_not_assessed_is_reported_as_uncovered_not_clean():
    """An unevaluated segment must not look identical to a confirmed-safe one."""
    segment = _report()["segments"][1]
    assert segment["curvature"]["flag"] == "not_assessed"
    assert segment["coverage"] == "uncovered"
    assert _report()["segments"][0]["coverage"] == "covered"


def test_coverage_statistics_are_reported_per_module():
    coverage = _report()["coverage"]
    assert coverage["curvature"]["not_assessed"] == 1
    assert coverage["curvature"]["total"] == 2
    assert coverage["grade"]["not_assessed"] == 0


def test_cv_modules_carry_instance_references():
    assert _report()["segments"][0]["lane"]["instances"] == ["R01-0000#lane#0"]


def test_geojson_has_one_linestring_feature_per_segment():
    geo = to_geojson(_report())
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == 2
    assert geo["features"][0]["geometry"]["type"] == "LineString"


def test_geojson_coordinates_are_lon_lat_order():
    """GeoJSON is x,y which means longitude first; swapping puts Sarawak in Somalia."""
    first = to_geojson(_report())["features"][0]["geometry"]["coordinates"][0]
    assert first[0] > 100.0
    assert first[1] < 10.0


def test_geojson_properties_embed_the_segment_record():
    properties = to_geojson(_report())["features"][0]["properties"]
    assert properties["segment_id"] == "R01-0000"
    assert properties["curvature"]["flag"] == "critical"


def test_markdown_lists_the_missing_mitigation():
    text = to_markdown(_report())
    assert "R01-0000" in text
    assert "chevron" in text


def test_write_report_produces_all_three_files(tmp_path: Path):
    write_report(tmp_path, _report())
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "segments.geojson").exists()
    assert (tmp_path / "report.md").exists()
    loaded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_condition_is_carried_even_though_it_is_a_stub():
    """Road Condition is a declared stub, and the report must still carry it.

    The stub exists precisely because "condition" is one of the four
    SCORING_MODULES here. Dropping it would hide an unimplemented attribute
    instead of surfacing it, which is what not_assessed is for.
    """
    report = _report()

    for record in report["segments"]:
        assert "condition" in record
        assert "flag" in record["condition"]

    assert "condition" in report["coverage"]
    assert report["coverage"]["condition"]["total"] == len(report["segments"])


def test_signs_double_line_and_frame_reference_are_rendered():
    """Nothing else in the suite fills these three dashboard fields.

    Without this test, deleting the lines that populate signs, double_line and
    representative_frame_id would leave every other test green.
    """
    segments, module_results, existence, recommendations = _fixture()
    segments[0].frame_ids = ["f_000001", "f_000002"]
    existence[0] = ExistenceResult(
        segment_id="R01-0000",
        signs=[SignRecord(id="f_000001#sign#0", cls="Bend", conf=0.91, at_m=70.0)],
        double_line=DoubleLineRecord(
            present=True, confidence=0.8, evidence=["f_000001#lane#0"]
        ),
    )

    report = build_report(
        "R01", segments, module_results, existence, recommendations, cfg=CFG
    )
    first = report["segments"][0]

    assert first["signs"][0]["cls"] == "Bend"
    assert first["signs"][0]["at_m"] == 70.0
    assert first["double_line"]["present"] is True
    assert first["representative_frame_id"] == "f_000001"
