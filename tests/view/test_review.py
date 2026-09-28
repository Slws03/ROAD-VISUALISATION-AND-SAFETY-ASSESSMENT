"""The human-review sidecar file: read, write, overwrite. It never touches
report.json, so there is nothing here about flags or coverage."""

from pathlib import Path

from irap_screening.view.review import load_review, save_review_entry


def test_missing_review_file_loads_as_empty(tmp_path: Path):
    assert load_review(tmp_path) == {}


def test_save_then_load_round_trips(tmp_path: Path):
    entry = save_review_entry(tmp_path, "R01-0004", "delineation", "ok", "sky glare")
    assert entry["verdict"] == "ok"
    assert entry["note"] == "sky glare"
    assert "reviewed_at" in entry

    loaded = load_review(tmp_path)
    assert loaded["R01-0004"]["delineation"] == entry


def test_a_second_module_on_the_same_segment_does_not_clobber_the_first(tmp_path: Path):
    save_review_entry(tmp_path, "R01-0004", "delineation", "ok", "")
    save_review_entry(tmp_path, "R01-0004", "condition", "flag", "pothole near f_000023")

    loaded = load_review(tmp_path)
    assert loaded["R01-0004"]["delineation"]["verdict"] == "ok"
    assert loaded["R01-0004"]["condition"]["verdict"] == "flag"


def test_saving_again_overwrites_the_prior_verdict_for_that_module(tmp_path: Path):
    save_review_entry(tmp_path, "R01-0004", "delineation", "flag", "first look")
    save_review_entry(tmp_path, "R01-0004", "delineation", "ok", "second look, it's fine")

    loaded = load_review(tmp_path)
    assert loaded["R01-0004"]["delineation"]["verdict"] == "ok"
    assert loaded["R01-0004"]["delineation"]["note"] == "second look, it's fine"
