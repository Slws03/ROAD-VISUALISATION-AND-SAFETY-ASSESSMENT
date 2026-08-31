"""The CLRerNet dump conversion, which is where lane coordinates can go wrong.

Nothing here imports mmcv or torch. The Colab side only serialises numbers; all
the arithmetic that could be plausibly wrong is in this module, so all of it is
tested here.
"""

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from irap_screening.config import load_config
from irap_screening.io.detections import load_detections
from irap_screening.io.schema import Frame, LaneInstance
from irap_screening.modules.lane import weber_contrast_at
from irap_screening.modules.marking import scan_near_lane_point
from irap_screening.runners.lane import (
    RawFrame,
    RawLane,
    RawLaneMeta,
    build_record,
    crop_geometry,
    export,
    to_original_points,
)

# The geometry the demo notebook actually ran with.
META = {
    "crop_top": 0.25,
    "crop_bottom": 0.20,
    "cut_height": 94,
    "ori_img_h": 376,
    "ori_img_w": 1241,
    "conf_threshold": 0.1,
}
SOURCE_WH = (1920, 1080)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "route_demo"


def _meta(**overrides) -> RawLaneMeta:
    return RawLaneMeta(**{**META, **overrides})


def _frame(index: int, name: str) -> Frame:
    return Frame(
        frame_id=f"f_{index:06d}",
        path=f"frames/{name}",
        timestamp_utc=datetime(2026, 8, 31, tzinfo=UTC) + timedelta(seconds=index),
        sha256="abc123def4567890",
    )


def test_crop_geometry_truncates_like_the_notebook():
    """int() truncation, not rounding: the mapping was validated against that."""
    assert crop_geometry(SOURCE_WH, _meta()) == (270, 594)


def test_crop_bottom_of_zero_keeps_the_full_lower_half():
    """The notebook special-cases crop_bottom == 0 rather than slicing to -0."""
    crop_top_px, canvas_h = crop_geometry(SOURCE_WH, _meta(crop_bottom=0.0))
    assert (crop_top_px, canvas_h) == (270, 810)


def test_normalised_extremes_map_to_the_crop_boundaries():
    """The mapping's two fixed points, which pin the whole affine transform.

    y_norm at the cut ratio is the top row of the crop; y_norm at 1.0 is its
    last row. Anything else in between follows from these two.
    """
    points = to_original_points(
        [(0.0, 0.25), (1.0, 1.0), (0.5, 0.625)], SOURCE_WH, _meta()
    )

    assert points[0] == pytest.approx((0.0, 270.0))
    assert points[1] == pytest.approx((1920.0, 864.0))
    # Midway in y_norm lands midway down the canvas.
    assert points[2] == pytest.approx((960.0, 567.0))


def test_y_stays_within_the_crop_for_the_full_normalised_range():
    """Every y CLRerNet can emit lands inside the crop, never elsewhere in the frame."""
    ys = [0.25 + 0.05 * step for step in range(16)]
    points = to_original_points([(0.5, y) for y in ys], SOURCE_WH, _meta())

    assert all(270.0 <= y <= 864.0 for _, y in points)


def test_off_frame_x_is_passed_through_unclamped():
    """CLRerNet extends curved lanes past the image edge; those points are kept.

    Clamping would invent a position on the frame border and feed it to the
    contrast sampler as if a marking were there. Both consumers --
    lane.weber_contrast_at and marking.scan_near_lane_point -- bounds-check and
    return None, so an out-of-frame point is dropped rather than measured. The
    exporter's job is to report where the detector put the point.
    """
    points = to_original_points([(-0.1, 0.9), (1.2, 0.9)], SOURCE_WH, _meta())

    assert points[0][0] == pytest.approx(-192.0)
    assert points[1][0] == pytest.approx(2304.0)


def test_off_frame_points_are_dropped_by_the_consumers_not_measured():
    """Pins the assumption the previous test relies on, in the consumers themselves."""
    gray = np.full((1080, 1920), 100, dtype=np.uint8)
    cfg = load_config()

    assert weber_contrast_at(gray, -192.0, 500.0, cfg.lane.bg_window) is None
    assert weber_contrast_at(gray, 2304.0, 500.0, cfg.lane.bg_window) is None
    assert scan_near_lane_point(gray, -192.0, 500.0, cfg) is None
    assert scan_near_lane_point(gray, 2304.0, 500.0, cfg) is None


def test_dump_above_the_config_threshold_is_rejected(tmp_path):
    """A dump floor above lane.clrernet_conf makes the config knob inert.

    Sweeping clrernet_conf below the dump's floor would then report no change
    while in fact changing nothing -- the same inversion as sign.runner_conf.
    """
    route = tmp_path / "route"
    _write_route(route, ["a.png"])
    _write_dump(tmp_path / "raw.json", ["a.png"])
    raw = json.loads((tmp_path / "raw.json").read_text(encoding="utf-8"))
    raw["meta"]["conf_threshold"] = 0.6
    (tmp_path / "raw.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        export(tmp_path / "raw.json", route, load_config())

    assert "could never bind" in str(excinfo.value)


def test_model_wh_comes_from_the_dump_not_a_hardcoded_constant():
    """The notebook's cv2.resize is an editable literal; Transform must track it."""
    raw = RawFrame(
        filename="a.png",
        source_wh=SOURCE_WH,
        lanes=[RawLane(conf=0.75, points_norm=[(0.3, 0.5)])],
    )

    record = build_record(_frame(0, "a.png"), raw, _meta(model_wh=(1640, 590)))

    assert record.transform.model_wh == (1640, 590)


def test_the_wrong_reading_of_y_norm_would_be_caught():
    """Guards against the competing interpretation of CLRerNet's output.

    If y_norm were normalised over the whole original frame, y_norm = 1.0 would
    map to row 1080 rather than 864 -- the bottom of the image instead of the
    bottom of the crop. The two readings differ by 0.73, which is why the
    overlay check was decisive.
    """
    (_, y_bottom), = to_original_points([(0.5, 1.0)], SOURCE_WH, _meta())

    assert y_bottom == pytest.approx(864.0)
    assert y_bottom != pytest.approx(1080.0)


def test_meta_rejects_a_cut_height_that_disagrees_with_the_crop():
    """The one coupling that makes a wrong mapping look entirely plausible."""
    with pytest.raises(ValidationError) as excinfo:
        _meta(cut_height=0)

    assert "stretches every lane" in str(excinfo.value)


def test_record_carries_frame_identity_and_a_truthful_transform():
    frame = _frame(3, "shot_003.png")
    raw = RawFrame(
        filename="shot_003.png",
        source_wh=SOURCE_WH,
        lanes=[RawLane(conf=0.75, points_norm=[(0.3, 0.5), (0.32, 0.8)])],
    )

    record = build_record(frame, raw, _meta())

    assert record.frame_id == "f_000003"
    assert record.image_path == "frames/shot_003.png"
    assert record.sha256 == frame.sha256
    assert record.type == "lane"
    assert record.transform.crop_top == 0.25
    assert record.transform.crop_bottom == 0.20
    assert record.transform.source_wh == SOURCE_WH
    assert isinstance(record.instances[0], LaneInstance)
    assert record.instances[0].id == "f_000003#lane#0"


def _write_route(root, names):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_id", "path", "timestamp_utc", "sha256"]
        )
        writer.writeheader()
        for index, name in enumerate(names):
            frame = _frame(index, name)
            writer.writerow(
                {
                    "frame_id": frame.frame_id,
                    "path": frame.path,
                    "timestamp_utc": frame.timestamp_utc.isoformat(),
                    "sha256": frame.sha256,
                }
            )


def _write_dump(path, names, lanes_per_frame=1):
    payload = {
        "meta": META,
        "frames": [
            {
                "filename": name,
                "source_wh": list(SOURCE_WH),
                "lanes": [
                    {"conf": 0.75, "points_norm": [[0.3, 0.5], [0.32, 0.8]]}
                ]
                * lanes_per_frame,
            }
            for name in names
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_writes_records_the_fusion_layer_can_load(tmp_path):
    names = ["a.png", "b.png", "c.png"]
    route = tmp_path / "route"
    _write_route(route, names)
    _write_dump(tmp_path / "raw.json", names)

    out = export(tmp_path / "raw.json", route, load_config())

    loaded = load_detections(out)
    assert set(loaded) == {"f_000000", "f_000001", "f_000002"}
    assert all(record.type == "lane" for record in loaded.values())


def test_export_rejects_a_dump_missing_route_frames(tmp_path):
    """A Colab run that stopped halfway would read as a stretch of unmarked road."""
    route = tmp_path / "route"
    _write_route(route, ["a.png", "b.png", "c.png"])
    _write_dump(tmp_path / "raw.json", ["a.png", "b.png"])

    with pytest.raises(ValueError) as excinfo:
        export(tmp_path / "raw.json", route, load_config())

    assert "c.png" in str(excinfo.value)


def test_export_rejects_a_dump_with_unknown_frames(tmp_path):
    route = tmp_path / "route"
    _write_route(route, ["a.png"])
    _write_dump(tmp_path / "raw.json", ["a.png", "stray.png"])

    with pytest.raises(ValueError) as excinfo:
        export(tmp_path / "raw.json", route, load_config())

    assert "stray.png" in str(excinfo.value)


def test_export_rejects_a_dump_with_no_lanes_anywhere(tmp_path):
    """Zero lanes everywhere grades the whole route critical with full confidence.

    At conf_threshold 0.1 CLRerNet does not legitimately return nothing across a
    route, so this is a broken export, not a finding.
    """
    route = tmp_path / "route"
    _write_route(route, ["a.png", "b.png"])
    _write_dump(tmp_path / "raw.json", ["a.png", "b.png"], lanes_per_frame=0)

    with pytest.raises(ValueError) as excinfo:
        export(tmp_path / "raw.json", route, load_config())

    assert "critical" in str(excinfo.value)


def test_export_reproduces_the_committed_fixture_artifact(tmp_path):
    """The exporter must be able to produce exactly what the pipeline consumes.

    The committed fixture's lane.jsonl was synthesised directly by
    scripts/make_fixture.py, never by this code path. Rebuilding the dump that
    CLRerNet would have had to emit for those points and running it through the
    exporter checks the two agree instance for instance -- which is the only
    available end-to-end check until a real Colab dump exists.
    """
    fixture = FIXTURE
    committed = [
        json.loads(line)
        for line in (fixture / "detections" / "lane.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    by_id = {record["frame_id"]: record for record in committed}

    rows = list(csv.DictReader((fixture / "frames.csv").open(encoding="utf-8")))
    dump = {
        "meta": {
            "crop_top": 0.0,
            "crop_bottom": 0.0,
            "cut_height": 0,
            "ori_img_h": 480,
            "ori_img_w": 640,
            "conf_threshold": 0.1,
        },
        "frames": [],
    }
    for row in rows:
        record = by_id[row["frame_id"]]
        width, height = record["transform"]["source_wh"]
        dump["frames"].append(
            {
                "filename": Path(row["path"]).name,
                "source_wh": [width, height],
                "lanes": [
                    {
                        "conf": instance["conf"],
                        "points_norm": [
                            [x / width, y / height] for x, y in instance["points"]
                        ],
                    }
                    for instance in record["instances"]
                ],
            }
        )

    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(dump), encoding="utf-8")
    out = export(raw, fixture, load_config(), out_path=tmp_path / "lane.jsonl")

    regenerated = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert len(regenerated) == len(committed)
    for expected, actual in zip(committed, regenerated, strict=True):
        assert actual["frame_id"] == expected["frame_id"]
        assert actual["instances"] == expected["instances"]
        assert actual["transform"] == expected["transform"]
