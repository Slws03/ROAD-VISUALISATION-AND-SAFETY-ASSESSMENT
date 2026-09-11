"""The committed fixture is the reproducibility guarantee; it must stay runnable."""

from pathlib import Path

from irap_screening.config import load_config
from irap_screening.pipeline import run_pipeline

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "route_demo"


def test_fixture_exists_and_contains_images():
    """A JSONL alone cannot reproduce contrast measurements; pixels are required."""
    assert FIXTURE.exists()
    assert list((FIXTURE / "frames").glob("*.jpg"))


def test_fixture_runs_end_to_end_without_a_gpu():
    report = run_pipeline(FIXTURE, load_config(), route_id="DEMO")
    assert report["segments"]
    assert report["schema_version"] == "1.0"


def test_fixture_exercises_every_scoring_module():
    report = run_pipeline(FIXTURE, load_config(), route_id="DEMO")
    first = report["segments"][0]
    for module in ("curvature", "grade", "delineation", "condition"):
        assert "flag" in first[module]


def test_fixture_produces_at_least_one_recommendation():
    """A fixture where nothing fires would not exercise the decision layer."""
    report = run_pipeline(FIXTURE, load_config(), route_id="DEMO")
    assert any(record["recommendations"] for record in report["segments"])


def test_fixture_stays_small_enough_to_commit():
    total_bytes = sum(f.stat().st_size for f in FIXTURE.rglob("*") if f.is_file())
    assert total_bytes < 50 * 1024 * 1024
