"""Sign existence facts for gap-check.

This module never produces a Flag. It supplies existence facts only.
"""

from __future__ import annotations

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute
from irap_screening.io.detections import DetectionSet
from irap_screening.io.schema import BoxInstance, ExistenceResult, Segment, SignRecord

MODULE_NAME = "marking"


def analyse_marking(
    route: AlignedRoute,
    segments: list[Segment],
    detections: DetectionSet,
    cfg: Settings,
) -> list[ExistenceResult]:
    """Collect the accepted sign records of each segment."""
    distance_by_id = {
        frame.frame_id: float(distance)
        for frame, distance in zip(route.frames, route.frame_dist_m, strict=False)
    }

    results: list[ExistenceResult] = []
    for segment in segments:
        signs: list[SignRecord] = []
        for frame_id in segment.frame_ids:
            sign_record = detections.sign.get(frame_id)
            if sign_record is None:
                continue
            for instance in sign_record.instances:
                if not isinstance(instance, BoxInstance) or instance.cls is None:
                    continue
                if instance.conf < cfg.sign.conf:
                    continue
                signs.append(
                    SignRecord(
                        id=instance.id,
                        cls=instance.cls,
                        conf=instance.conf,
                        at_m=distance_by_id.get(frame_id, 0.0),
                    )
                )
        results.append(ExistenceResult(segment_id=segment.segment_id, signs=signs))
    return results
