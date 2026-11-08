"""Lane marking detectability from Weber contrast at CLRerNet lane points.

Severity comes from contrast, not from detector confidence. The synthetic
degradation experiment showed CLRerNet confidence is a threshold-type response
rather than a graded one, so confidence is used only for binary presence and
corridor location, while contrast carries the severity banding.

Note the comparison direction: lower contrast means a worse marking, the
opposite of the curvature and grade thresholds.

MODULE_NAME is "delineation", not "lane": this is the report-facing key
gap-check and the viewer key their output on, and iRAP's Delineation
attribute is what a reader will look for it under. It follows AiRAP, whose
Delineation is "Determined based on recognition of road lines/markings/lanes":
the Coding Manual's hazard-warning-sign half ("Signs warning of severe
hazards, and centre and edge markings are generally present and visible") is
not assessed, and the viewer says so next to it. Everything internal to this file (the
functions, cfg.lane.*, LaneInstance) keeps the more specific "lane"
naming, since that is what it actually computes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute
from irap_screening.io.detections import DetectionSet
from irap_screening.io.schema import Flag, LaneInstance, ModuleResult, Segment

MODULE_NAME = "delineation"  # external/report key -- see module docstring;
                              # internal names (analyse_lane, LaneInstance,
                              # cfg.lane.*) stay lane-marking-specific.


# Sides the background window can be taken from: the full strip around the
# point, or only the pixels to its right or left, past the peak window.
BOTH, RIGHT, LEFT = "both", "right", "left"
_MIN_SIDE_PX = 8  # fewer pixels than this on the chosen side is no background


def weber_contrast_at(
    gray: np.ndarray,
    x: float,
    y: float,
    bg_window: int,
    peak_frac: float = 0.0,
    side: str = BOTH,
) -> float | None:
    """Weber contrast C = (I - I_bg) / I_bg at one lane point.

    I is the peak intensity in a small neighbourhood around the point; I_bg is
    the median of the surrounding horizontal window, which is dominated by road
    surface rather than by the marking itself. The neighbourhood is
    peak_frac of the image width either side, never under 4 px: CLRerNet's
    points can sit several pixels off the paint, and a fixed width misses it
    on a large frame where it would not on a small one.

    side RIGHT or LEFT takes I_bg from that half of the window only, beyond
    the peak neighbourhood: for an edge line, whose other half is verge.
    """
    height, width = gray.shape[:2]
    col = int(round(x))
    row = int(round(y))
    if not (0 <= col < width and 0 <= row < height):
        return None

    half = max(1, bg_window // 2)
    reach = max(4, int(round(peak_frac * width)))
    if side == RIGHT:
        strip = gray[row, min(width, col + reach + 1):min(width, col + half + 1)]
    elif side == LEFT:
        strip = gray[row, max(0, col - half):max(0, col - reach)]
    else:
        strip = gray[row, max(0, col - half):min(width, col + half + 1)]
    if strip.size < (_MIN_SIDE_PX if side != BOTH else 1):
        return None

    local_lo = max(0, col - reach)
    local_hi = min(width, col + reach + 1)
    marking = float(gray[row, local_lo:local_hi].astype(float).max())
    background = float(np.median(strip.astype(float)))
    if background <= 1e-6:
        return None
    return (marking - background) / background


def lane_contrast(
    gray: np.ndarray, lane: LaneInstance, cfg: Settings, side: str = BOTH
) -> float | None:
    """One line's contrast: a high percentile of its points' Weber contrast.

    A high percentile, not the median: see contrast_percentile in the
    configuration for why a dashed marking demands it. None when no point
    lands inside the frame.
    """
    samples = [
        value
        for value in (
            weber_contrast_at(
                gray, x, y, cfg.lane.bg_window, cfg.lane.peak_window_frac, side
            )
            for x, y in lane.points
        )
        if value is not None
    ]
    return float(np.percentile(samples, cfg.lane.contrast_percentile)) if samples else None


def _bottom_x(lane: LaneInstance, height: int, width: int) -> float:
    """Where the line, fitted straight, meets the bottom row, as a share of the width."""
    points = np.asarray(lane.points, dtype=float)
    if len(points) < 2 or np.ptp(points[:, 1]) == 0:
        return float(points[:, 0].mean()) / width
    slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
    return float(slope * (height - 1) + intercept) / width


def frame_contrasts(
    gray: np.ndarray, lanes: list[LaneInstance], cfg: Settings
) -> dict[str, float]:
    """Each line's contrast in one frame, keyed by lane id.

    The outermost line on each side of a frame with two or more lines is an
    edge line: half its window is verge (grass, soil, kerb, shoulder), which
    pulls the median off the road surface, up or down. Its background comes
    from the road-centre side only, the side facing the frame centre. Lines
    between have road on both sides and keep the full window, where one side
    alone would be thrown by a vehicle or a shadow (delineation_review §8).
    """
    height, width = gray.shape[:2]
    sides = {lane.id: BOTH for lane in lanes}
    if len(lanes) >= 2:
        bottoms = [_bottom_x(lane, height, width) for lane in lanes]
        for index in {int(np.argmin(bottoms)), int(np.argmax(bottoms))}:
            sides[lanes[index].id] = RIGHT if bottoms[index] < 0.5 else LEFT
    out: dict[str, float] = {}
    for lane in lanes:
        contrast = lane_contrast(gray, lane, cfg, sides[lane.id])
        if contrast is not None:
            out[lane.id] = contrast
    return out


def lane_contrasts(
    route: AlignedRoute,
    detections: DetectionSet,
    cfg: Settings,
    image_root: Path,
    frame_ids: set[str],
) -> dict[str, float]:
    """The contrast of every accepted line in the given frames, keyed by lane id.

    The same numbers analyse_lane grades with, for the viewer to print beside
    each line.
    """
    paths = {frame.frame_id: frame.path for frame in route.frames}
    out: dict[str, float] = {}
    for frame_id in frame_ids:
        record = detections.lane.get(frame_id)
        lanes = [
            i for i in (record.instances if record else [])
            if isinstance(i, LaneInstance) and i.conf >= cfg.lane.clrernet_conf
        ]
        if not lanes or frame_id not in paths:
            continue
        image = cv2.imread(str(Path(image_root) / paths[frame_id]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        out.update(frame_contrasts(image, lanes, cfg))
    return out


def _grade(visible_share: float, cfg: Settings) -> Flag:
    """Adequate when markings are "generally present and visible" (iRAP Coding
    Manual): at least min_visible_share of the detected lines clear
    weber_adequate. The worst line alone used to decide, so one worn edge line
    beside road works condemned a segment whose centre and other edge were
    clear. Two bands, matching iRAP's Adequate/Poor; no watch band.
    """
    return Flag.NORMAL if visible_share >= cfg.lane.min_visible_share else Flag.CRITICAL


def irap_delineation_category(flag: Flag) -> int:
    """iRAP Delineation descriptive category. Display only, mirrors the flag:
    1 (Adequate) for Flag.NORMAL, 2 (Poor) for Flag.CRITICAL -- unlike
    curvature/grade there is no independent numeric scale to derive from.
    """
    return 1 if flag is Flag.NORMAL else 2


def analyse_lane(
    route: AlignedRoute,
    segments: list[Segment],
    detections: DetectionSet,
    cfg: Settings,
    image_root: Path,
) -> list[ModuleResult]:
    """One ModuleResult per segment, graded by the share of visible lines."""
    image_root = Path(image_root)
    frames_by_id = {frame.frame_id: frame for frame in route.frames}

    results: list[ModuleResult] = []
    for segment in segments:
        if not segment.frame_ids:
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    notes="No frames fall within this segment.",
                )
            )
            continue

        contrasts: list[float] = []
        evidence: list[str] = []
        detected_frames = 0

        for frame_id in segment.frame_ids:
            record = detections.lane.get(frame_id)
            lanes = (
                [i for i in record.instances if isinstance(i, LaneInstance)]
                if record
                else []
            )
            lanes = [lane for lane in lanes if lane.conf >= cfg.lane.clrernet_conf]
            if not lanes:
                # A frame the detector could not resolve is not evidence that the
                # road is unmarked. Scoring it as zero contrast and aggregating by
                # the worst frame let one miss condemn a whole segment; sparse
                # detection is handled once, per segment, below.
                continue

            detected_frames += 1
            frame = frames_by_id.get(frame_id)
            if frame is None:
                continue
            image = cv2.imread(str(image_root / frame.path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            for lane_id, contrast in frame_contrasts(image, lanes, cfg).items():
                contrasts.append(contrast)
                evidence.append(lane_id)

        total_frames = len(segment.frame_ids)
        detection_rate = detected_frames / total_frames if total_frames else 0.0
        if contrasts and detection_rate < cfg.lane.min_detection_rate:
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    metrics={
                        "detection_rate": detection_rate,
                        "detected_frames": float(detected_frames),
                        "total_frames": float(total_frames),
                    },
                    notes=(
                        f"Lane detection rate {detection_rate:.0%} is below the "
                        f"{cfg.lane.min_detection_rate:.0%} floor; too little evidence "
                        "to grade this segment either way."
                    ),
                    evidence=evidence,
                )
            )
            continue
        if not contrasts:
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    metrics={
                        "detection_rate": detection_rate,
                        "detected_frames": float(detected_frames),
                        "total_frames": float(total_frames),
                    },
                    notes=(
                        "No lane was detected anywhere in this segment. That is "
                        "not the same as the road being unmarked, and CLRerNet's "
                        "output cannot tell the two apart."
                    ),
                )
            )
            continue

        worst_contrast = float(min(contrasts))
        lines_visible = sum(c >= cfg.lane.weber_adequate for c in contrasts)
        flag = _grade(lines_visible / len(contrasts), cfg)
        results.append(
            ModuleResult(
                module=MODULE_NAME,
                segment_id=segment.segment_id,
                flag=flag,
                metrics={
                    "weber_contrast": worst_contrast,
                    "lines_visible": float(lines_visible),
                    "lines_total": float(len(contrasts)),
                    "detection_rate": detection_rate,
                    "detected_frames": float(detected_frames),
                    "total_frames": float(total_frames),
                    "irap_category": float(irap_delineation_category(flag)),
                },
                evidence=evidence,
            )
        )
    return results
