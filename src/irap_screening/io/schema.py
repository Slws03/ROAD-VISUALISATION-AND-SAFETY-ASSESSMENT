"""Canonical data models shared across the whole pipeline.

These types are the contract between the GPU detection processes and the CPU
fusion layer. Detection coordinates are always in original-image pixels; the
runners are responsible for inverting any crop or resize they applied.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]


class Flag(StrEnum):
    """Severity of one module on one segment.

    NOT_ASSESSED is a real state meaning "this segment could not be evaluated"
    (for example curvature below the minimum speed). It must never be collapsed
    into NORMAL downstream.
    """

    NORMAL = "normal"
    WATCH = "watch"
    CRITICAL = "critical"
    NOT_ASSESSED = "not_assessed"


class Frame(BaseModel):
    frame_id: str
    path: str
    timestamp_utc: datetime
    sha256: str


class GpsFix(BaseModel):
    timestamp_utc: datetime
    lat: Latitude
    lon: Longitude
    speed_mps: float | None = None
    altitude: float | None = None
    hdop: float | None = None
    # Degrees clockwise from north, as the receiver reports its course.
    bearing_deg: float | None = None


class ImuSample(BaseModel):
    timestamp_utc: datetime
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class Transform(BaseModel):
    """Records the crop and resize a runner applied, keeping the mapping auditable."""

    source_wh: tuple[int, int]
    crop_top: float
    crop_bottom: float
    model_wh: tuple[int, int]


class LaneInstance(BaseModel):
    id: str
    points: list[tuple[float, float]]
    conf: float


class BoxInstance(BaseModel):
    id: str
    bbox: tuple[float, float, float, float]
    conf: float
    cls: str | None = None


class DetectionRecord(BaseModel):
    frame_id: str
    image_path: str
    sha256: str
    type: Literal["lane", "sign"]
    transform: Transform
    instances: list[LaneInstance | BoxInstance]

    @model_validator(mode="after")
    def _instances_match_type(self) -> DetectionRecord:
        """Reject instances that do not match the record's declared type.

        The union is not discriminated, so pydantic would otherwise accept a
        record declaring "lane" whose instances carry bbox. That parses cleanly,
        yields zero lanes at the isinstance filter downstream, and a frame with
        no lane detected grades as the worst contrast band -- turning a
        malformed file into a confident critical verdict.
        """
        expected = LaneInstance if self.type == "lane" else BoxInstance
        for instance in self.instances:
            if not isinstance(instance, expected):
                raise ValueError(
                    f"record type {self.type!r} requires {expected.__name__}, "
                    f"got {type(instance).__name__} for instance {instance.id!r}"
                )
        return self


class Segment(BaseModel):
    """A 100 m stretch of route: the shared coordinate system for every module."""

    segment_id: str
    start_m: float
    end_m: float
    start_latlon: tuple[float, float]
    end_latlon: tuple[float, float]
    frame_ids: list[str] = Field(default_factory=list)
    imu_start: int = 0
    imu_end: int = 0


class ModuleResult(BaseModel):
    """Output of one scoring module for one segment."""

    module: str
    segment_id: str
    flag: Flag
    metrics: dict[str, float] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    notes: str | None = None


class SignRecord(BaseModel):
    id: str
    cls: str
    conf: float
    at_m: float


class ExistenceResult(BaseModel):
    """Sign and Marking output. Carries existence facts, never a flag.

    This module does not score. Giving it a flag would invite folding it into
    the per-segment rating, which the design explicitly forbids.
    """

    segment_id: str
    signs: list[SignRecord] = Field(default_factory=list)


class Recommendation(BaseModel):
    rule: str
    trigger: dict[str, float | str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    present: list[str] = Field(default_factory=list)
    # Mitigations the route cannot show either way, e.g. a sign position
    # before the first photo: neither missing nor present.
    unchecked: list[str] = Field(default_factory=list)
    text: str = ""
