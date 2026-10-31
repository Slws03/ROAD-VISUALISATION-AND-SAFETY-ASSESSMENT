"""The human-review sidecar file: read, write, overwrite. It never touches
report.json, so there is nothing here about coverage."""

from pathlib import Path

from irap_screening.view.review import (
    load_review,
    save_recommendation_edit,
    save_review_entry,
)


def test_missing_review_file_loads_as_empty(tmp_path: Path):
    assert load_review(tmp_path) == {}


def test_save_then_load_round_trips(tmp_path: Path):
    entry = save_review_entry(tmp_path, "R01-0004", "delineation", 2, "sky glare")
    assert (entry["category"], entry["flag"]) == (2, "critical")
    assert entry["note"] == "sky glare"
    assert "reviewed_at" in entry

    loaded = load_review(tmp_path)
    assert loaded["R01-0004"]["delineation"] == entry


def test_a_second_module_on_the_same_segment_does_not_clobber_the_first(tmp_path: Path):
    save_review_entry(tmp_path, "R01-0004", "delineation", 1, "")
    save_review_entry(tmp_path, "R01-0004", "condition", 3, "pothole near f_000023")

    loaded = load_review(tmp_path)
    assert loaded["R01-0004"]["delineation"]["flag"] == "normal"
    assert loaded["R01-0004"]["condition"]["flag"] == "critical"


def test_saving_again_overwrites_the_prior_review_for_that_module(tmp_path: Path):
    save_review_entry(tmp_path, "R01-0004", "delineation", 2, "first look")
    save_review_entry(tmp_path, "R01-0004", "delineation", 1, "second look, it's fine")

    loaded = load_review(tmp_path)
    assert loaded["R01-0004"]["delineation"]["category"] == 1
    assert loaded["R01-0004"]["delineation"]["note"] == "second look, it's fine"


def test_a_recommendation_edit_sits_beside_the_module_reviews(tmp_path: Path):
    save_review_entry(tmp_path, "R01-0004", "curvature", 1, "")
    entry = save_recommendation_edit(tmp_path, "R01-0004", "  Add chevrons at 120 m.\n")

    loaded = load_review(tmp_path)["R01-0004"]
    assert loaded["recommendation"] == entry
    assert entry["text"] == "Add chevrons at 120 m."
    assert loaded["curvature"]["category"] == 1


def test_each_category_implies_its_flag(tmp_path: Path):
    flag = lambda module, category: save_review_entry(  # noqa: E731
        tmp_path, "R01-0004", module, category, "")["flag"]
    assert flag("curvature", 1) == "normal"
    assert flag("curvature", 2) == "critical"  # follows threshold_m = 800
    assert flag("grade", 4) == "critical"
    assert flag("condition", 2) == "normal"  # Medium: no watch band to map to
    assert flag("quality_of_curve", 3) == "normal"
    assert flag("delineation", None) == "not_assessed"


def test_blank_text_drops_the_edit_so_the_generated_one_shows_again(tmp_path: Path):
    save_recommendation_edit(tmp_path, "R01-0004", "Add chevrons.")

    assert save_recommendation_edit(tmp_path, "R01-0004", "   ") is None
    assert "recommendation" not in load_review(tmp_path)["R01-0004"]
