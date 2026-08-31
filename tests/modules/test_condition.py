"""Road Condition is a declared stub in this build.

The tests pin that it is deliberately unimplemented rather than accidentally
unwired: it must report not_assessed for every segment, explain why, and stay
inert even when pothole detections are present.
"""

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.detections import DetectionSet
from irap_screening.io.schema import BoxInstance, DetectionRecord, Flag, Transform
from irap_screening.modules.condition import analyse_condition
from irap_screening.segment import build_segments
from tests.factories import straight_route


def _setup():
    frames, gps, imu = straight_route(length_m=300.0)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    return route, build_segments(route, cfg), cfg


def test_every_segment_is_not_assessed():
    """An unimplemented attribute must not masquerade as a clean one."""
    route, segments, cfg = _setup()
    results = analyse_condition(route, segments, DetectionSet(), cfg)

    assert len(results) == len(segments)
    assert all(result.flag is Flag.NOT_ASSESSED for result in results)


def test_each_result_explains_why():
    """Coverage statistics are only useful if the reason travels with them."""
    route, segments, cfg = _setup()
    results = analyse_condition(route, segments, DetectionSet(), cfg)

    assert all(result.notes for result in results)
    assert all("not implemented" in result.notes.lower() for result in results)


def test_stub_is_inert_even_with_pothole_detections_present():
    """Distinguishes a deliberate stub from a module someone forgot to wire up.

    Feeding it real detections must not change the verdict. If it did, the module
    would be half-implemented: neither honest about the gap nor correct about the
    potholes.
    """
    route, segments, cfg = _setup()
    pothole = {
        frame.frame_id: DetectionRecord(
            frame_id=frame.frame_id,
            image_path=frame.path,
            sha256="0" * 16,
            type="pothole",
            transform=Transform(
                source_wh=(1280, 720), crop_top=0.0, crop_bottom=0.0, model_wh=(640, 640)
            ),
            instances=[
                BoxInstance(
                    id=f"{frame.frame_id}#pothole#0",
                    bbox=(600.0, 500.0, 660.0, 540.0),
                    conf=0.9,
                )
            ],
        )
        for frame in route.frames
    }
    results = analyse_condition(route, segments, DetectionSet(pothole=pothole), cfg)

    assert all(result.flag is Flag.NOT_ASSESSED for result in results)
