"""Threshold configuration.

Layering: packaged default.yaml -> optional profile file -> CLI --set overrides.
Every resolved value records where it came from so `irap config show` can
explain itself. Validation happens at load time: a bad threshold must fail
before any analysis runs, not after producing a wrong report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class _Section(BaseModel):
    """Base for config sections.

    extra=forbid makes a typo in a --set key fail at load time rather than being
    silently ignored and leaving the default in force.
    """

    model_config = {"extra": "forbid"}


class AggregationSettings(_Section):
    lane: str = "max_risk"
    curvature: str = "max_risk"
    grade: str = "max_risk"
    condition: str = "count_then_classify"


class SegmentSettings(_Section):
    length_m: PositiveFloat = 100.0
    aggregation: AggregationSettings = Field(default_factory=AggregationSettings)


class SamplingSettings(_Section):
    target_spacing_m: PositiveFloat = 20.0


class CurvatureSettings(_Section):
    threshold_m: PositiveFloat = 400.0
    v_min_mps: PositiveFloat = 5.0
    stationary_speed_mps: PositiveFloat = 0.5
    stationary_accel_var: PositiveFloat = 0.05


class GradeSettings(_Section):
    threshold_pct: PositiveFloat = 7.5
    comp_filter_alpha: float = Field(default=0.98, gt=0.0, lt=1.0)


class LaneSettings(_Section):
    clrernet_conf: float = Field(default=0.41, ge=0.0, le=1.0)
    weber_watch: float = Field(default=0.35, ge=0.0)
    weber_urgent: float = Field(default=0.20, ge=0.0)
    bg_window: PositiveInt = 81


class DoubleLineSettings(_Section):
    window_steps: list[PositiveInt] = Field(default_factory=lambda: [50, 90, 130, 180, 240])
    bg_window: PositiveInt = 81
    contrast_thresh: float = Field(default=0.4, ge=0.0)
    min_width: PositiveInt = 3
    min_gap_merge: PositiveInt = 2
    min_gap_px: PositiveInt = 3
    max_gap_fraction: float = Field(default=0.9, gt=0.0, le=1.0)
    max_gap_cap_px: PositiveInt = 150
    anchor_tol_px: PositiveInt = 20


class SignSettings(_Section):
    runner_conf: float = Field(default=0.10, ge=0.0, le=1.0)
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    imgsz: PositiveInt = 1280

    @model_validator(mode="after")
    def _runner_is_the_more_permissive_floor(self) -> SignSettings:
        """Reject an inverted pair, which silently disables the CPU-side threshold.

        The runner writes its floor into the cached artifact, so a runner floor
        above conf makes conf unreachable: the low-confidence band the design
        reserves for review is discarded before the CPU layer ever sees it.
        """
        if self.runner_conf > self.conf:
            raise ValueError(
                f"sign.runner_conf ({self.runner_conf}) must not exceed sign.conf "
                f"({self.conf}); the runner's floor is baked into the cached "
                "detections and cannot be relaxed afterwards"
            )
        return self


class IpmSettings(_Section):
    """Ground-plane projection parameters. No monocular depth network involved.

    camera_height_m is a hard prerequisite for width estimation. When it is None
    the condition module still counts and locates defects but reports no width.
    """

    camera_height_m: PositiveFloat | None = None
    fx: PositiveFloat | None = None
    fy: PositiveFloat | None = None
    cx: float | None = None
    cy: float | None = None
    max_range_m: PositiveFloat = 15.0

    def is_complete(self) -> bool:
        return None not in (self.camera_height_m, self.fx, self.fy, self.cx, self.cy)


class ConditionSettings(_Section):
    yolo_conf: float = Field(default=0.35, ge=0.0, le=1.0)
    width_threshold_cm: PositiveFloat = 20.0
    count_poor: PositiveInt = 5
    count_medium: PositiveInt = 1
    ipm: IpmSettings = Field(default_factory=IpmSettings)


class GapCheckSettings(_Section):
    curve_sign_distance_m: PositiveFloat = 230.0
    curve_sign_tolerance_m: PositiveFloat = 50.0


class Settings(_Section):

    segment: SegmentSettings = Field(default_factory=SegmentSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    curvature: CurvatureSettings = Field(default_factory=CurvatureSettings)
    grade: GradeSettings = Field(default_factory=GradeSettings)
    lane: LaneSettings = Field(default_factory=LaneSettings)
    doubleline: DoubleLineSettings = Field(default_factory=DoubleLineSettings)
    sign: SignSettings = Field(default_factory=SignSettings)
    condition: ConditionSettings = Field(default_factory=ConditionSettings)
    gapcheck: GapCheckSettings = Field(default_factory=GapCheckSettings)

    provenance: dict[str, str] = Field(default_factory=dict, exclude=True)


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"Override must be key=value, got: {item}")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def _assign(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set {dotted}: {part} is not a section")
    node[parts[-1]] = value


def load_config(
    profile: Path | None = None,
    overrides: list[str] | None = None,
    default_path: Path = DEFAULT_CONFIG_PATH,
) -> Settings:
    """Load thresholds with layered overrides, recording where each value came from."""
    with default_path.open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}

    provenance = dict.fromkeys(_flatten(base), "default")
    merged = base

    if profile is not None:
        with Path(profile).open(encoding="utf-8") as handle:
            profile_data = yaml.safe_load(handle) or {}
        merged = _deep_merge(merged, profile_data)
        provenance.update(dict.fromkeys(_flatten(profile_data), "profile"))

    for item in overrides or []:
        dotted, value = _parse_override(item)
        _assign(merged, dotted, value)
        provenance[dotted] = "cli"

    settings = Settings(**merged)
    settings.provenance = provenance
    return settings
