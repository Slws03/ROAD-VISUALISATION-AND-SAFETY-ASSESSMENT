"""Draw detections back onto the frame they were found in.

The viewer exists to check a flag against the pixels behind it, so the drawing
must land exactly where the detector said it did. That is why this runs in
Python rather than on a canvas in the browser: the coordinate arithmetic already
has one tested implementation, and a second one in JavaScript would be a second
chance to get it wrong -- invisibly, since a misplaced lane line still looks
like a lane line.

The frame is scaled first and the points scaled with it, rather than drawing at
full resolution and shrinking afterwards, which would thin a two-pixel line into
nothing.
"""

from __future__ import annotations

import cv2
import numpy as np

from irap_screening.io.schema import BoxInstance, DetectionRecord, LaneInstance

EVIDENCE_WIDTH = 640

# BGR, matching OpenCV's channel order.
LANE_COLOR = (0, 255, 0)
SIGN_COLOR = (0, 200, 255)
LABEL_COLOR = (0, 0, 0)

LINE_THICKNESS = 2
POINT_RADIUS = 3


def scale_factor(source_wh: tuple[int, int], target_width: int) -> float:
    """Return the downscale ratio, never enlarging a frame beyond its own size."""
    return min(1.0, target_width / float(source_wh[0]))


def _scale_point(x: float, y: float, ratio: float) -> tuple[int, int]:
    return int(round(x * ratio)), int(round(y * ratio))


def _in_frame(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _draw_lanes(canvas: np.ndarray, record: DetectionRecord, ratio: float) -> None:
    height, width = canvas.shape[:2]
    for instance in record.instances:
        if not isinstance(instance, LaneInstance):
            continue
        # Off-frame points are dropped rather than clamped. CLRerNet extends
        # curved lanes past the image edge, and a clamped point would draw a
        # marking along the border that was never detected there.
        points = [
            point
            for point in (_scale_point(x, y, ratio) for x, y in instance.points)
            if _in_frame(point[0], point[1], width, height)
        ]
        if len(points) == 1:
            cv2.circle(canvas, points[0], POINT_RADIUS, LANE_COLOR, -1)
        elif len(points) > 1:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False,
                          LANE_COLOR, LINE_THICKNESS)


def _draw_signs(canvas: np.ndarray, record: DetectionRecord, ratio: float) -> None:
    height, width = canvas.shape[:2]
    for instance in record.instances:
        if not isinstance(instance, BoxInstance):
            continue
        x1, y1, x2, y2 = instance.bbox
        top_left = _scale_point(x1, y1, ratio)
        bottom_right = _scale_point(x2, y2, ratio)
        cv2.rectangle(canvas, top_left, bottom_right, SIGN_COLOR, LINE_THICKNESS)

        # The class string is what gap-check matches on, so show it with the
        # confidence that decided whether it was accepted at all.
        label = f"{instance.cls or '?'} {instance.conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        label_y = max(text_h + baseline, top_left[1])
        label_x = min(top_left[0], max(0, width - text_w))
        cv2.rectangle(
            canvas,
            (label_x, label_y - text_h - baseline),
            (min(width, label_x + text_w), label_y),
            SIGN_COLOR,
            -1,
        )
        cv2.putText(canvas, label, (label_x, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, LABEL_COLOR, 1, cv2.LINE_AA)


def render_evidence(
    image: np.ndarray,
    lane: DetectionRecord | None,
    sign: DetectionRecord | None,
    target_width: int = EVIDENCE_WIDTH,
) -> np.ndarray:
    """Return a downscaled copy of the frame with its detections drawn on it.

    A frame with no detections is a real state, not an error: it renders clean.
    """
    height, width = image.shape[:2]
    ratio = scale_factor((width, height), target_width)
    if ratio < 1.0:
        canvas = cv2.resize(
            image,
            (int(round(width * ratio)), int(round(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        canvas = image.copy()

    if lane is not None:
        _draw_lanes(canvas, lane, ratio)
    if sign is not None:
        _draw_signs(canvas, sign, ratio)
    return canvas
