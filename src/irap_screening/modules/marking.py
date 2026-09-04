"""Sign and double-line existence facts for gap-check.

Double lines are found by a CLRerNet-guided Weber contrast scan: each scan is
anchored on a lane coordinate the detector already reported, rather than
sweeping whole image rows blindly. Perspective makes the two strokes crowd
together near the vanishing point and spread apart near the camera, so the
search window widens in steps until a valid pair is found.

This module never produces a Flag. It supplies existence facts only.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute
from irap_screening.io.detections import DetectionSet
from irap_screening.io.schema import (
    BoxInstance,
    DoubleLineRecord,
    ExistenceResult,
    Flag,
    LaneInstance,
    ModuleResult,
    Segment,
    SignRecord,
)

MODULE_NAME = "marking"


def weber_blobs(row: np.ndarray, cfg: Settings) -> list[tuple[int, int]]:
    """Segment bright spans in a 1D intensity row using Weber contrast.

    Returns (start, end) index pairs. Spans narrower than min_width are dropped;
    spans separated by less than min_gap_merge are merged, which absorbs the
    single-pixel dropouts common in worn paint.
    """
    values = row.astype(float)
    background = float(np.median(values))
    if background <= 1e-6:
        return []

    contrast = (values - background) / background
    mask = contrast >= cfg.doubleline.contrast_thresh
    if not mask.any():
        return []

    spans: list[list[int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append([start, index])
            start = None
    if start is not None:
        spans.append([start, len(mask)])

    merged: list[list[int]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= cfg.doubleline.min_gap_merge:
            merged[-1][1] = span[1]
        else:
            merged.append(span)

    return [
        (lo, hi) for lo, hi in merged if (hi - lo) >= cfg.doubleline.min_width
    ]


def scan_near_lane_point(
    gray: np.ndarray, x: float, y: float, cfg: Settings
) -> float | None:
    """Look for a double-line pair around one lane coordinate.

    Widens the half-window through the configured steps and stops at the first
    valid pair. Returns the gap in pixels between the two stroke centres, or
    None when no pair qualifies.
    """
    height, width = gray.shape[:2]
    row_index = int(round(y))
    anchor = int(round(x))
    if not (0 <= row_index < height and 0 <= anchor < width):
        return None

    for half in cfg.doubleline.window_steps:
        lo = max(0, anchor - half)
        hi = min(width, anchor + half + 1)
        blobs = weber_blobs(gray[row_index, lo:hi], cfg)
        if len(blobs) < 2:
            continue

        centres = [lo + (blob[0] + blob[1]) / 2.0 for blob in blobs]
        max_gap = min(half * 2 * cfg.doubleline.max_gap_fraction, cfg.doubleline.max_gap_cap_px)

        best: float | None = None
        for left, right in zip(centres, centres[1:], strict=False):
            gap = right - left
            if not (cfg.doubleline.min_gap_px <= gap <= max_gap):
                continue
            # The pair must sit near the anchor, so a wider window cannot pair up
            # with an unrelated marking further away.
            near_anchor = (
                abs(left - anchor) <= cfg.doubleline.anchor_tol_px
                or abs(right - anchor) <= cfg.doubleline.anchor_tol_px
                or left <= anchor <= right
            )
            if near_anchor and (best is None or gap < best):
                best = gap
        if best is not None:
            return best
    return None


def analyse_marking(
    route: AlignedRoute,
    segments: list[Segment],
    detections: DetectionSet,
    cfg: Settings,
    image_root: Path,
    grade_results: list[ModuleResult],
) -> list[ExistenceResult]:
    """Collect sign records and double-line presence for each segment.

    The double-line scan only runs on a critical-grade segment: gap-check's
    only use of double_line.present is as an alternative to a No-overtaking
    sign on a critical grade (decision/gapcheck.py rule 2). Scanning anywhere
    else spends the Weber-contrast pass -- inherently noisy against roadside
    clutter such as a guardrail -- on segments where its answer is never
    consulted and can only mislead someone reading the viewer's evidence.
    """
    image_root = Path(image_root)
    frames_by_id = {frame.frame_id: frame for frame in route.frames}
    distance_by_id = {
        frame.frame_id: float(distance)
        for frame, distance in zip(route.frames, route.frame_dist_m, strict=False)
    }
    grade_flag_by_segment = {result.segment_id: result.flag for result in grade_results}

    results: list[ExistenceResult] = []
    for segment in segments:
        signs: list[SignRecord] = []
        gaps: list[float] = []
        evidence: list[str] = []
        scan_double_line = grade_flag_by_segment.get(segment.segment_id) is Flag.CRITICAL

        for frame_id in segment.frame_ids:
            sign_record = detections.sign.get(frame_id)
            if sign_record is not None:
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

            if not scan_double_line:
                continue

            lane_record = detections.lane.get(frame_id)
            frame = frames_by_id.get(frame_id)
            if lane_record is None or frame is None:
                continue
            image = cv2.imread(str(image_root / frame.path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            for instance in lane_record.instances:
                if not isinstance(instance, LaneInstance):
                    continue
                for px, py in instance.points:
                    gap = scan_near_lane_point(image, px, py, cfg)
                    if gap is not None:
                        gaps.append(gap)
                        evidence.append(instance.id)

        present = len(gaps) > 0
        confidence = min(1.0, len(gaps) / max(1, len(segment.frame_ids)))
        results.append(
            ExistenceResult(
                segment_id=segment.segment_id,
                signs=signs,
                double_line=DoubleLineRecord(
                    present=present,
                    assessed=scan_double_line,
                    confidence=confidence if present else 0.0,
                    evidence=sorted(set(evidence)),
                ),
            )
        )
    return results
