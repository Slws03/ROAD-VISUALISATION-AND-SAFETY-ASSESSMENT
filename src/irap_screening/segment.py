"""Route segmentation into fixed-length iRAP coding units.

iRAP codes attributes every 100 m and computes a Star Rating Score per 100 m
segment, so aligning to that unit makes this project's output directly
comparable with iRAP's. A trailing partial segment is dropped rather than
coded, because a short segment is not a valid coding unit.
"""

from __future__ import annotations

import numpy as np

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute
from irap_screening.io.schema import Segment


def build_segments(
    route: AlignedRoute, cfg: Settings, route_id: str = "R01"
) -> list[Segment]:
    """Cut the route into contiguous fixed-length segments along cumulative distance."""
    length_m = float(cfg.segment.length_m)
    total = route.total_length_m
    count = int(total // length_m)
    if count == 0:
        return []

    segments: list[Segment] = []
    for index in range(count):
        start_m = index * length_m
        end_m = start_m + length_m

        start_lat = float(np.interp(start_m, route.gps_dist_m, [f.lat for f in route.gps]))
        start_lon = float(np.interp(start_m, route.gps_dist_m, [f.lon for f in route.gps]))
        end_lat = float(np.interp(end_m, route.gps_dist_m, [f.lat for f in route.gps]))
        end_lon = float(np.interp(end_m, route.gps_dist_m, [f.lon for f in route.gps]))

        # Half-open interval so a frame or sample on a boundary belongs to
        # exactly one segment.
        frame_mask = (route.frame_dist_m >= start_m) & (route.frame_dist_m < end_m)
        frame_ids = [
            frame.frame_id
            for frame, inside in zip(route.frames, frame_mask, strict=False)
            if inside
        ]

        imu_indices = np.flatnonzero(
            (route.imu_dist_m >= start_m) & (route.imu_dist_m < end_m)
        )
        imu_start = int(imu_indices[0]) if imu_indices.size else 0
        imu_end = int(imu_indices[-1]) + 1 if imu_indices.size else 0

        segments.append(
            Segment(
                segment_id=f"{route_id}-{index:04d}",
                start_m=start_m,
                end_m=end_m,
                start_latlon=(start_lat, start_lon),
                end_latlon=(end_lat, end_lon),
                frame_ids=frame_ids,
                imu_start=imu_start,
                imu_end=imu_end,
            )
        )
    return segments
