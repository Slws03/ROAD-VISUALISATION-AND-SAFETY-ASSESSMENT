"""Cross-module gap-check.

The design philosophy is: state the road fact, check whether the matching
mitigation already exists, and recommend only where there is a gap. A module
flag and a recommendation are therefore different things -- a critical curve
with its warning sign and chevrons in place keeps the critical flag and yields
no recommendation.

The three mitigations occupy different spaces and are checked differently:

- Advance warning sign: one per series of curves (JKR ATJ 2B/85 cl. 2.3,
  curves closer than 180 m), 230 m before the first curve's tangent point,
  with tolerance. The series also decides which sign JKR calls for.
- Chevron: runs along a curve, so it is checked on each curve of the series.
- Lane marking: the delineation module's own finding. Adjacent segments it
  flags critical are one stretch to repaint, with one recommendation.

Checking chevrons upstream would be wrong; they belong inside the curve, not
before it.

A sign is placed at the photo it was detected in (SignRecord.at_m), which is
before the sign itself: the camera sees it ahead. Searches therefore reach
curve_sign_lookahead_m further back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from irap_screening.config import Settings
from irap_screening.io.schema import (
    ExistenceResult,
    Flag,
    ModuleResult,
    Recommendation,
    Segment,
    SignRecord,
)
from irap_screening.modules.curvature import RIGHT, Curve

MITIGATIONS = ("advance_bend_sign", "chevron", "visible_marking")

ADVANCE_BEND_CLASSES = frozenset({"Bend", "Double Bend"})
CHEVRON_CLASS = "Chevron"


def _is_critical(result: ModuleResult) -> bool:
    return result.flag is Flag.CRITICAL


def curve_spans(
    segments: list[Segment],
    module_results: dict[str, list[ModuleResult]],
    is_curve: Callable[[ModuleResult], bool] = _is_critical,
) -> dict[str, tuple[float, float]]:
    """Map each curve segment to the extent of the curve it belongs to.

    is_curve picks which curvature results count as curve; by default the
    critical ones, which is what rule 1 warns about.

    The 100 m grid is iRAP's reporting unit, so a 300 m bend necessarily spans
    three of them, and each of those units genuinely carries a critical
    curvature attribute. Signage adequacy is not a property of the unit though:
    a bend is warned of once, at its entry. Reading each unit as its own curve
    demanded one advance sign per unit, at a different upstream point each time,
    which no real installation would satisfy.

    A run is broken by any segment that is not critical, including one that was
    never assessed -- continuity cannot be claimed across a unit whose curvature
    is unknown.
    """
    by_id = _by_segment(module_results.get("curvature", []))
    member = [
        (result := by_id.get(s.segment_id)) is not None
        and result.flag is not Flag.NOT_ASSESSED
        and is_curve(result)
        for s in segments
    ]
    spans: dict[str, tuple[float, float]] = {}
    start = 0
    while start < len(segments):
        if not member[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(segments) and member[end + 1]:
            end += 1
        extent = (segments[start].start_m, segments[end].end_m)
        for segment in segments[start : end + 1]:
            spans[segment.segment_id] = extent
        start = end + 1
    return spans


def _by_segment(results: list[ModuleResult]) -> dict[str, ModuleResult]:
    return {result.segment_id: result for result in results}


def _flag_of(
    results: dict[str, list[ModuleResult]], module: str, segment_id: str
) -> Flag:
    entry = _by_segment(results.get(module, [])).get(segment_id)
    return entry.flag if entry is not None else Flag.NOT_ASSESSED


def _overlaps(segment: Segment, start_m: float, end_m: float) -> bool:
    return segment.start_m < end_m and segment.end_m > start_m


def curve_series(curves: list[Curve], cfg: Settings) -> list[list[Curve]]:
    """Group consecutive curves closer than curve_series_gap_m into one series."""
    series: list[list[Curve]] = []
    for curve in curves:
        if series and curve.start_m - series[-1][-1].end_m < cfg.gapcheck.curve_series_gap_m:
            series[-1].append(curve)
        else:
            series.append([curve])
    return series


def jkr_warning_sign(series: list[Curve]) -> str:
    """The warning sign ATJ 2B/85 cl. 2.3 names for a series of curves.

    One curve: Curve sign WD.10a (right) / WD.10b (left). Two curves turning
    opposite ways: Reverse Curve WD.2a (right first) / WD.2b (left first).
    Anything else, including two curves the same way: Winding Road WD.11.
    """
    first = series[0].direction
    if len(series) == 1:
        return "WD.10a" if first == RIGHT else "WD.10b"
    if len(series) == 2 and series[1].direction != first:
        return "WD.2a" if first == RIGHT else "WD.2b"
    return "WD.11"


PRESENT, MISSING, UNCHECKED = "present", "missing", "unchecked"


def _advance_sign_status(
    series_start_m: float, signs: list[SignRecord], frame_m: Sequence[float], cfg: Settings
) -> str:
    """Whether an advance warning sign was found near its JKR position.

    Any bend warning class counts: the detector cannot tell WD.10a from WD.10b
    and has no Winding Road class. Unchecked when the photos do not cover the
    search stretch, e.g. it lies before the route starts.
    """
    point = series_start_m - cfg.gapcheck.curve_sign_distance_m
    lo = point - cfg.gapcheck.curve_sign_tolerance_m - cfg.gapcheck.curve_sign_lookahead_m
    hi = point + cfg.gapcheck.curve_sign_tolerance_m
    if any(sign.cls in ADVANCE_BEND_CLASSES and lo <= sign.at_m <= hi for sign in signs):
        return PRESENT
    covered = len(frame_m) > 0 and min(frame_m) <= lo and any(lo <= m <= hi for m in frame_m)
    return MISSING if covered else UNCHECKED


def _curve_recommendation(
    number: int,
    series: list[Curve],
    signs: list[SignRecord],
    frame_m: Sequence[float],
    cfg: Settings,
) -> Recommendation | None:
    """Rule 1 for one series: an advance warning sign, and chevrons on every curve."""
    start_m, end_m = series[0].start_m, series[-1].end_m
    lookahead = cfg.gapcheck.curve_sign_lookahead_m
    without_chevrons = sum(
        not any(
            sign.cls == CHEVRON_CLASS and curve.start_m - lookahead <= sign.at_m <= curve.end_m
            for sign in signs
        )
        for curve in series
    )
    status = {
        "advance_bend_sign": _advance_sign_status(start_m, signs, frame_m, cfg),
        "chevron": MISSING if without_chevrons else PRESENT,
    }
    missing = [name for name, state in status.items() if state == MISSING]
    unchecked = [name for name, state in status.items() if state == UNCHECKED]
    if not missing and not unchecked:
        return None

    code = jkr_warning_sign(series)
    radius = min(curve.radius_m for curve in series)
    sign_at_m = start_m - cfg.gapcheck.curve_sign_distance_m
    text = (
        f"Curve series S{number:02d} ({len(series)} curve(s), {start_m:.0f}-{end_m:.0f} m, "
        f"sharpest R={radius:.0f} m): JKR calls for {code} at {sign_at_m:.0f} m."
    )
    if missing:
        text += f" Missing: {', '.join(missing)}."
    if unchecked:
        text += f" Could not be checked: {', '.join(unchecked)}."
    return Recommendation(
        rule="curve_warning_incomplete",
        trigger={
            "module": "curvature",
            "series_id": f"S{number:02d}",
            "curve_start_m": start_m,
            "curve_end_m": end_m,
            "radius_m": radius,
            "curves": float(len(series)),
            "first_direction": series[0].direction,
            "sign_code": code,
            "sign_at_m": sign_at_m,
            "curves_without_chevrons": float(without_chevrons),
        },
        missing=missing,
        present=[name for name, state in status.items() if state == PRESENT],
        unchecked=unchecked,
        text=text,
    )


def _marking_recommendation(
    number: int,
    stretch: list[Segment],
    module_results: dict[str, list[ModuleResult]],
    cfg: Settings,
) -> Recommendation:
    """Rule 3 for one stretch of adjacent segments with poor delineation.

    The detector cannot tell a worn line from an unmarked road edge, so the
    recommendation does not say which: repaint, or mark where there is none.
    """
    def summed(metric: str) -> float:
        return sum(
            _metric_of(module_results, "delineation", s.segment_id, metric) or 0.0
            for s in stretch
        )

    visible, total = summed("lines_visible"), summed("lines_total")
    start_m, end_m = stretch[0].start_m, stretch[-1].end_m
    return Recommendation(
        rule="lane_marking_poor",
        trigger={
            "module": "delineation",
            "series_id": f"D{number:02d}",
            "start_m": start_m,
            "end_m": end_m,
            "segments": float(len(stretch)),
            "lines_visible": visible,
            "lines_total": total,
            "contrast_threshold": cfg.lane.weber_adequate,
        },
        missing=["visible_marking"],
        present=[],
        text=(
            f"Stretch D{number:02d} ({start_m:.0f}-{end_m:.0f} m): {visible:.0f} of "
            f"{total:.0f} lines detected across its photos clearly visible. Repaint, or mark where "
            "there is none, after a site check."
        ),
    )


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
    curves: list[Curve],
    frame_m: Sequence[float],
) -> dict[str, list[Recommendation]]:
    """Apply rules 1 to 3 and return recommendations keyed by segment id.

    curves are find_curves' bends; frame_m is where each photo was taken.
    """
    all_signs: list[SignRecord] = [sign for entry in existence for sign in entry.signs]

    recommendations: dict[str, list[Recommendation]] = {s.segment_id: [] for s in segments}

    # Rule 1: a series of critical curves needs an advance warning sign AND
    # chevrons on each curve. A curve counts where it overlaps a segment the
    # curvature module flags critical, so the two never disagree; not_assessed
    # is not critical: it means unevaluated, not safe. One recommendation per
    # series, carried by every critical segment one of its curves crosses; a
    # normal segment a curve's tail reaches into does not carry it.
    critical = [
        s for s in segments
        if _flag_of(module_results, "curvature", s.segment_id) is Flag.CRITICAL
    ]
    kept = [c for c in curves if any(_overlaps(s, c.start_m, c.end_m) for s in critical)]
    for number, series in enumerate(curve_series(kept, cfg), start=1):
        recommendation = _curve_recommendation(number, series, all_signs, frame_m, cfg)
        if recommendation is None:
            continue
        for segment in critical:
            if any(_overlaps(segment, c.start_m, c.end_m) for c in series):
                recommendations[segment.segment_id].append(recommendation)

    # Rule 3: poor delineation needs the lines repainted. Adjacent poor
    # segments are one stretch; a normal or unassessed one ends it.
    stretches: list[list[Segment]] = []
    run: list[Segment] = []
    for segment in segments:
        if _flag_of(module_results, "delineation", segment.segment_id) is Flag.CRITICAL:
            run.append(segment)
        elif run:
            stretches.append(run)
            run = []
    if run:
        stretches.append(run)
    for number, stretch in enumerate(stretches, start=1):
        recommendation = _marking_recommendation(number, stretch, module_results, cfg)
        for segment in stretch:
            recommendations[segment.segment_id].append(recommendation)

    return recommendations
