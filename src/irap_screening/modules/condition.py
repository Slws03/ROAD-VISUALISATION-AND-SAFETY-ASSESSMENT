"""Road Condition - declared stub.

The full implementation estimates defect width by projecting image points onto the
road plane (intrinsics, mounting height and pitch), then classifies by segment-wide
defect count against the iRAP Road Condition table. It is deferred: the camera
parameters it needs are not available, so ipm.camera_height_m is still null in the
configuration.

Reporting not_assessed for every segment is deliberate. It keeps the attribute
visible in the report's coverage statistics instead of letting a missing fourth
attribute pass unnoticed, and it keeps this module's signature identical to the full
implementation so restoring it later touches no caller.
"""

from __future__ import annotations

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute
from irap_screening.io.detections import DetectionSet
from irap_screening.io.schema import Flag, ModuleResult, Segment

MODULE_NAME = "condition"

_STUB_NOTE = (
    "Road Condition analysis is not implemented in this build. Defect sizing "
    "requires camera intrinsics and mounting height, which are not yet available."
)


def analyse_condition(
    route: AlignedRoute,
    segments: list[Segment],
    detections: DetectionSet,
    cfg: Settings,
    pitch_rad: float = 0.0,
) -> list[ModuleResult]:
    """Return not_assessed for every segment, with the reason attached."""
    return [
        ModuleResult(
            module=MODULE_NAME,
            segment_id=segment.segment_id,
            flag=Flag.NOT_ASSESSED,
            notes=_STUB_NOTE,
        )
        for segment in segments
    ]
