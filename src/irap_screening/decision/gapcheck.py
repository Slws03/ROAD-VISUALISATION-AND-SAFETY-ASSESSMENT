"""Cross-module gap-check.

The design philosophy is: state the road fact, check whether the matching
mitigation already exists, and recommend only where there is a gap. A module
flag and a recommendation are therefore different things — a critical curve
with its warning sign and chevrons in place keeps the critical flag and yields
no recommendation.

The three mitigations occupy different spaces and are checked differently:

- Advance bend sign: a point 230 m before the curve entry (JKR), with tolerance.
- Chevron: runs along the curve, so it is checked by overlap with the segment.
- No-overtaking control: covers the hazardous stretch, also checked by overlap.

Checking chevrons or double lines upstream would be wrong; they belong inside
the hazardous stretch, not before it.
"""

from __future__ import annotations

from irap_screening.config import Settings
from irap_screening.io.schema import (
    ExistenceResult,
    Flag,
    ModuleResult,
    Recommendation,
    Segment,
    SignRecord,
)

MITIGATIONS = ("advance_bend_sign", "chevron", "no_overtaking_sign", "double_line")

ADVANCE_BEND_CLASSES = frozenset({"Bend", "Double Bend"})
CHEVRON_CLASS = "Chevron"
NO_OVERTAKING_CLASS = "No overtaking"

_PRIORITY_LADDER = ["low", "medium", "high", "highest"]


def _raise_priority(current: str) -> str:
    index = _PRIORITY_LADDER.index(current)
    return _PRIORITY_LADDER[min(index + 1, len(_PRIORITY_LADDER) - 1)]


def _by_segment(results: list[ModuleResult]) -> dict[str, ModuleResult]:
    return {result.segment_id: result for result in results}


def _flag_of(
    results: dict[str, list[ModuleResult]], module: str, segment_id: str
) -> Flag:
    entry = _by_segment(results.get(module, [])).get(segment_id)
    return entry.flag if entry is not None else Flag.NOT_ASSESSED


def _metric_of(
    results: dict[str, list[ModuleResult]], module: str, segment_id: str, key: str
) -> float | None:
    entry = _by_segment(results.get(module, [])).get(segment_id)
    return entry.metrics.get(key) if entry is not None else None


def run_gapcheck(
    segments: list[Segment],
    module_results: dict[str, list[ModuleResult]],
    existence: list[ExistenceResult],
    cfg: Settings,
) -> dict[str, list[Recommendation]]:
    """Apply rules 1 to 4 and return recommendations keyed by segment id."""
    existence_by_segment = {entry.segment_id: entry for entry in existence}
    all_signs: list[SignRecord] = [sign for entry in existence for sign in entry.signs]

    recommendations: dict[str, list[Recommendation]] = {s.segment_id: [] for s in segments}

    for segment in segments:
        curvature_flag = _flag_of(module_results, "curvature", segment.segment_id)
        grade_flag = _flag_of(module_results, "grade", segment.segment_id)
        lane_flag = _flag_of(module_results, "lane", segment.segment_id)
        entry = existence_by_segment.get(segment.segment_id)

        # Rule 1: a critical curve needs an advance warning sign AND chevrons.
        # not_assessed is skipped explicitly: it means unevaluated, not safe.
        if curvature_flag is Flag.CRITICAL:
            window_centre = segment.start_m - cfg.gapcheck.curve_sign_distance_m
            tolerance = cfg.gapcheck.curve_sign_tolerance_m
            advance_present = any(
                sign.cls in ADVANCE_BEND_CLASSES
                and abs(sign.at_m - window_centre) <= tolerance
                for sign in all_signs
            )
            chevron_present = any(
                sign.cls == CHEVRON_CLASS and segment.start_m <= sign.at_m < segment.end_m
                for sign in all_signs
            )

            missing = [
                name
                for name, ok in (
                    ("advance_bend_sign", advance_present),
                    ("chevron", chevron_present),
                )
                if not ok
            ]
            present = [
                name
                for name, ok in (
                    ("advance_bend_sign", advance_present),
                    ("chevron", chevron_present),
                )
                if ok
            ]
            if missing:
                radius = _metric_of(module_results, "curvature", segment.segment_id, "radius_m")
                recommendations[segment.segment_id].append(
                    Recommendation(
                        rule="curve_warning_incomplete",
                        trigger={"module": "curvature", "radius_m": radius or 0.0},
                        missing=missing,
                        present=present,
                        priority="high",
                        text=(
                            f"Curve (R={radius:.0f} m) is missing: {', '.join(missing)}."
                            if radius
                            else f"Critical curve is missing: {', '.join(missing)}."
                        ),
                    )
                )

        # Rule 2: a critical grade needs no-overtaking control, sign or paint.
        if grade_flag is Flag.CRITICAL:
            sign_present = any(
                sign.cls == NO_OVERTAKING_CLASS
                and segment.start_m <= sign.at_m < segment.end_m
                for sign in all_signs
            )
            paint_present = bool(entry and entry.double_line.present)

            if not (sign_present or paint_present):
                grade_pct = _metric_of(module_results, "grade", segment.segment_id, "grade_pct")
                recommendations[segment.segment_id].append(
                    Recommendation(
                        rule="grade_overtaking_control_missing",
                        trigger={"module": "grade", "grade_pct": grade_pct or 0.0},
                        missing=["no_overtaking_sign", "double_line"],
                        present=[],
                        priority="high",
                        text=(
                            f"Steep grade ({grade_pct:.1f} percent) has no overtaking "
                            "control: neither a No overtaking sign nor a double line."
                            if grade_pct
                            else "Steep grade has no overtaking control."
                        ),
                    )
                )

        # Rule 3: a curve on a steep grade compounds the risk.
        if curvature_flag is Flag.CRITICAL and grade_flag is Flag.CRITICAL:
            recommendations[segment.segment_id].append(
                Recommendation(
                    rule="compound_curve_on_grade",
                    trigger={"module": "curvature+grade"},
                    missing=[],
                    present=[],
                    priority="highest",
                    text="Critical curve on a critical grade; inspect as a priority.",
                )
            )

        # Rule 4: worst-band lane marking raises priority but adds nothing new.
        # It corresponds to no iRAP countermeasure, so turning it into a
        # recommendation of its own would overclaim.
        if lane_flag is Flag.CRITICAL:
            for recommendation in recommendations[segment.segment_id]:
                recommendation.priority = _raise_priority(recommendation.priority)
                recommendation.priority_raised_by.append("lane_marking_worst_band")

    return recommendations
