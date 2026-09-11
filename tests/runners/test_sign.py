"""The sign runner's contract with the fusion layer.

These tests deliberately never import ultralytics: the runner's torch-touching
part is one lazily imported call, and everything the rest of the pipeline
depends on -- identity, class vocabulary, schema -- is pure and testable here.
"""

import json
from datetime import UTC, datetime

import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.detections import DetectionSet, load_detections
from irap_screening.io.schema import Frame
from irap_screening.modules.marking import analyse_marking
from irap_screening.runners.sign import (
    EXPECTED_CLASSES,
    build_record,
    letterbox_wh,
    validate_class_names,
)
from irap_screening.segment import build_segments
from tests.factories import straight_route


def _frame(index: int = 0) -> Frame:
    return Frame(
        frame_id=f"f_{index:06d}",
        path=f"frames/f_{index:06d}.jpg",
        timestamp_utc=datetime(2026, 8, 31, 0, 0, index, tzinfo=UTC),
        sha256="abc123def4567890",
    )


def test_accepts_the_trained_weight_vocabulary():
    validate_class_names(dict(enumerate(EXPECTED_CLASSES)))


def test_rejects_drifted_class_names():
    """A retrained weight using 'bend' would match no gap-check rule, silently."""
    names = dict(enumerate(EXPECTED_CLASSES))
    names[0] = "bend"

    with pytest.raises(ValueError) as excinfo:
        validate_class_names(names)

    message = str(excinfo.value)
    assert "bend" in message and "Bend" in message


def test_rejects_reordered_class_indices():
    """Index order is part of the contract: box.cls is resolved through it."""
    names = dict(enumerate(EXPECTED_CLASSES))
    names[0], names[1] = names[1], names[0]

    with pytest.raises(ValueError):
        validate_class_names(names)


def test_record_carries_frame_identity_verbatim():
    """Identity must come from frames.csv, or the fusion layer looks up nothing."""
    frame = _frame(7)
    record = build_record(frame, [((10.0, 20.0, 30.0, 40.0), 0.8, "Bend")], (1920, 1080), 1280)

    assert record.frame_id == frame.frame_id
    assert record.image_path == frame.path
    assert record.sha256 == frame.sha256
    assert record.type == "sign"


def test_instance_ids_are_traceable_and_unique():
    frame = _frame()
    boxes = [
        ((10.0, 20.0, 30.0, 40.0), 0.8, "Bend"),
        ((50.0, 60.0, 70.0, 80.0), 0.4, "Chevron"),
    ]
    record = build_record(frame, boxes, (1920, 1080), 1280)

    assert [instance.id for instance in record.instances] == [
        "f_000000#sign#0",
        "f_000000#sign#1",
    ]
    assert [instance.cls for instance in record.instances] == ["Bend", "Chevron"]
    assert record.instances[1].conf == pytest.approx(0.4)


def test_a_frame_with_no_signs_still_gets_a_record():
    """"Looked, found nothing" and "never ran" must not be the same artifact.

    Coverage statistics depend on the difference, and so does any later
    detection-rate check that decides what to escalate for review.
    """
    record = build_record(_frame(), [], (1920, 1080), 1280)

    assert record.instances == []


def test_records_round_trip_through_the_fusion_layer_loader(tmp_path):
    """The end the runner exists for: what it writes, load_detections must read."""
    frames = [_frame(0), _frame(1)]
    out = tmp_path / "sign.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for frame in frames:
            record = build_record(
                frame, [((1.0, 2.0, 3.0, 4.0), 0.9, "No overtaking")], (1280, 720), 1280
            )
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")

    loaded = load_detections(out)

    assert set(loaded) == {"f_000000", "f_000001"}
    assert loaded["f_000001"].instances[0].cls == "No overtaking"


@pytest.mark.parametrize(
    ("source_wh", "imgsz", "expected"),
    [
        ((4592, 3448), 1280, (1280, 992)),
        ((2445, 2445), 1280, (1280, 1280)),
        ((1349, 486), 1280, (1280, 480)),
        ((1280, 720), 1280, (1280, 736)),
        ((640, 480), 1000, (1000, 776)),
    ],
)
def test_letterbox_shape_matches_ultralytics(source_wh, imgsz, expected):
    """model_wh must record the shape the model saw, not the requested budget.

    Every expected value here was read off ultralytics.data.augment.LetterBox
    directly. The last case pins that a non-stride-multiple imgsz is not padded
    on the long side, which the naive round-up would get wrong.
    """
    assert letterbox_wh(source_wh, imgsz) == expected


def test_runner_output_reaches_gap_check_as_a_sign_record(tmp_path):
    """The hop the round-trip test does not cover: marking's own filters.

    A record can be schema-valid, load cleanly, and still be dropped by
    analyse_marking -- on a missing cls, or on the confidence threshold. This
    walks a real runner record all the way to the SignRecord gap-check reads.
    """
    frames, gps, imu = straight_route(length_m=200.0)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    segments = build_segments(route, cfg)

    first = route.frames[0]
    accepted = build_record(first, [((10.0, 20.0, 30.0, 40.0), 0.93, "Bend")], (1280, 720), 1280)
    marginal_frame = route.frames[1]
    marginal = build_record(
        marginal_frame, [((10.0, 20.0, 30.0, 40.0), 0.15, "Chevron")], (1280, 720), 1280
    )

    results = analyse_marking(
        route,
        segments,
        DetectionSet(sign={first.frame_id: accepted, marginal_frame.frame_id: marginal}),
        cfg,
        image_root=tmp_path,
        grade_results=[],
    )

    signs = [sign for result in results for sign in result.signs]
    assert [sign.cls for sign in signs] == ["Bend"], (
        "the 0.93 Bend must arrive as a SignRecord; the 0.15 Chevron sits between "
        "sign.runner_conf and sign.conf and must be held back for review"
    )
    assert signs[0].id == f"{first.frame_id}#sign#0"
