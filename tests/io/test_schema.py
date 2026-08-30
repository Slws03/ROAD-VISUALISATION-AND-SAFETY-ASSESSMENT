"""The schema is the contract between processes; these tests pin it exactly."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from irap_screening.io.schema import (
    BoxInstance,
    DetectionRecord,
    Flag,
    GpsFix,
    ImuSample,
    LaneInstance,
    ModuleResult,
    Transform,
)


def test_flag_values_are_exactly_the_four_allowed():
    assert {f.value for f in Flag} == {"normal", "watch", "critical", "not_assessed"}


def test_flag_rejects_irap_source_vocabulary():
    """Poor/Medium are iRAP citation words, never internal flag values."""
    with pytest.raises(ValueError):
        Flag("Poor")


def test_gps_fix_speed_is_optional():
    fix = GpsFix(
        timestamp_utc=datetime(2026, 8, 31, tzinfo=UTC),
        lat=1.55,
        lon=110.35,
    )
    assert fix.speed_mps is None


def test_imu_sample_requires_all_six_axes():
    with pytest.raises(ValidationError):
        ImuSample(
            timestamp_utc=datetime(2026, 8, 31, tzinfo=UTC),
            ax=0.0,
            ay=0.0,
            az=9.81,
            gx=0.0,
            gy=0.0,
        )


def test_detection_record_parses_lane_instances():
    record = DetectionRecord(
        frame_id="f_000001",
        image_path="frames/f_000001.jpg",
        sha256="a3f2b1c0d9e8f7a6",
        type="lane",
        transform=Transform(
            source_wh=(1920, 1080), crop_top=0.25, crop_bottom=0.20, model_wh=(800, 320)
        ),
        instances=[LaneInstance(id="f_000001#lane#0", points=[(100.0, 700.0)], conf=0.87)],
    )
    assert isinstance(record.instances[0], LaneInstance)
    assert record.instances[0].id == "f_000001#lane#0"


def test_detection_record_parses_box_instances_for_sign():
    record = DetectionRecord(
        frame_id="f_000002",
        image_path="frames/f_000002.jpg",
        sha256="b4e3c2d1a0f9e8b7",
        type="sign",
        transform=Transform(
            source_wh=(1920, 1080), crop_top=0.0, crop_bottom=0.0, model_wh=(640, 640)
        ),
        instances=[
            BoxInstance(
                id="f_000002#sign#0", bbox=(10.0, 20.0, 60.0, 80.0), conf=0.91, cls="Bend"
            )
        ],
    )
    assert record.instances[0].cls == "Bend"


def test_detection_record_rejects_unknown_type():
    with pytest.raises(ValidationError):
        DetectionRecord(
            frame_id="f_000003",
            image_path="frames/f_000003.jpg",
            sha256="c5d4e3f2a1b0c9d8",
            type="guardrail",
            transform=Transform(
                source_wh=(1920, 1080), crop_top=0.0, crop_bottom=0.0, model_wh=(640, 640)
            ),
            instances=[],
        )


def test_module_result_defaults_evidence_to_empty():
    result = ModuleResult(
        module="curvature",
        segment_id="R01-0000",
        flag=Flag.CRITICAL,
        metrics={"radius_m": 312.5},
    )
    assert result.evidence == []
    assert result.notes is None


def test_lane_record_rejects_box_instances():
    """A mislabelled runner output must fail at the boundary, not downstream.

    Without this check the box instance parses cleanly, the lane module's
    isinstance filter yields nothing, and a frame with no lane detected grades
    as the worst contrast band -- a confident wrong answer instead of an error.
    """
    with pytest.raises(ValidationError):
        DetectionRecord(
            frame_id="f_000004",
            image_path="frames/f_000004.jpg",
            sha256="d6e5f4a3b2c1d0e9",
            type="lane",
            transform=Transform(
                source_wh=(1920, 1080), crop_top=0.0, crop_bottom=0.0, model_wh=(800, 320)
            ),
            instances=[
                BoxInstance(id="f_000004#lane#0", bbox=(1.0, 2.0, 3.0, 4.0), conf=0.9)
            ],
        )


def test_sign_record_rejects_lane_instances():
    with pytest.raises(ValidationError):
        DetectionRecord(
            frame_id="f_000005",
            image_path="frames/f_000005.jpg",
            sha256="e7f6a5b4c3d2e1f0",
            type="sign",
            transform=Transform(
                source_wh=(1920, 1080), crop_top=0.0, crop_bottom=0.0, model_wh=(640, 640)
            ),
            instances=[
                LaneInstance(id="f_000005#sign#0", points=[(1.0, 2.0)], conf=0.9)
            ],
        )
