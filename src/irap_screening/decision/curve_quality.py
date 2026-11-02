"""iRAP Quality of curve (attribute column 40), coded from what a route observes.

Coding Manual v3.10 p.51-53: 1 Adequate, 2 Poor, 3 Not applicable. The manual
judges how well signing, marking and sight distance let a driver read the
curve. Of those this pipeline sees signs and delineation only, so:

- Straight or gently curving (iRAP curvature category 1) is Not applicable,
  as the manual's consistency check requires.
- Poor delineation forces Poor, which the manual states outright.
- Otherwise a curve is Adequate when it carries an advance bend sign OR
  chevrons. The manual's indicative parameters accept either treatment on its
  own; this is deliberately looser than gap-check rule 1, which recommends
  both.

Sight distance, lighting, advisory speed plates and telemetry-based speed
behaviour are not observed, so a curve coded Adequate here may still be Poor
on site.

Each code carries a flag so the attribute colours the segment like a scored
module: Poor is critical on a curve the curvature module flags critical, and
normal on a milder one (a moderate curve is coded as the manual requires, but
is below curvature.threshold_m). Adequate and Not applicable are normal.
"""

from __future__ import annotations

from typing import Any

from irap_screening.config import Settings
from irap_screening.decision.gapcheck import (
    ADVANCE_BEND_CLASSES,
    CHEVRON_CLASS,
    _by_segment,
    curve_spans,
)
from irap_screening.io.schema import ExistenceResult, Flag, ModuleResult, Segment

ADEQUATE, POOR, NOT_APPLICABLE = 1, 2, 3
_FLAG = {
    None: Flag.NOT_ASSESSED,
    ADEQUATE: Flag.NORMAL,
    POOR: Flag.CRITICAL,
    NOT_APPLICABLE: Flag.NORMAL,
}


def _is_curving(result: ModuleResult) -> bool:
    """Anything sharper than straight or gently curving (radius below 800 m)."""
    return result.metrics.get("irap_category", 1.0) >= 2


def code_quality_of_curve(
    segments: list[Segment],
    module_results: dict[str, list[ModuleResult]],
    existence: list[ExistenceResult],
    cfg: Settings,
) -> dict[str, dict[str, Any]]:
    """Return {segment_id: {"code": 1|2|3|None, "basis": str, "flag": str}}."""
    curvature = _by_segment(module_results.get("curvature", []))
    delineation = _by_segment(module_results.get("delineation", []))
    spans = curve_spans(segments, module_results, is_curve=_is_curving)
    signs = [sign for entry in existence for sign in entry.signs]

    coded: dict[str, dict[str, Any]] = {}
    for segment in segments:
        sid = segment.segment_id
        result = curvature.get(sid)
        if result is None or result.flag is Flag.NOT_ASSESSED:
            coded[sid] = {"code": None, "basis": "Curvature was not assessed."}
            continue
        if sid not in spans:
            coded[sid] = {
                "code": NOT_APPLICABLE,
                "basis": "The road is straight or gently curving.",
            }
            continue
        lane = delineation.get(sid)
        if lane is not None and lane.flag is Flag.CRITICAL:
            coded[sid] = {
                "code": POOR,
                "basis": "Delineation is poor, so the curve is coded poor.",
            }
            continue

        # Same placement as rule 1: one sign at the curve's entry.
        window_centre = spans[sid][0] - cfg.gapcheck.curve_sign_distance_m
        advance = any(
            sign.cls in ADVANCE_BEND_CLASSES
            and abs(sign.at_m - window_centre) <= cfg.gapcheck.curve_sign_tolerance_m
            for sign in signs
        )
        chevron = any(
            sign.cls == CHEVRON_CLASS and segment.start_m <= sign.at_m < segment.end_m
            for sign in signs
        )
        if advance or chevron:
            found = " and ".join(
                name
                for name, ok in (
                    ("an advance bend warning sign", advance),
                    ("chevron markers", chevron),
                )
                if ok
            )
            coded[sid] = {"code": ADEQUATE, "basis": f"The curve has {found}."}
        else:
            coded[sid] = {
                "code": POOR,
                "basis": "No advance bend warning sign or chevron markers were detected.",
            }
    for sid, entry in coded.items():
        flag = _FLAG[entry["code"]]
        if flag is Flag.CRITICAL and curvature[sid].flag is not Flag.CRITICAL:
            flag = Flag.NORMAL
            entry["basis"] += (
                " Not flagged critical: the curve is no sharper than the "
                f"{cfg.curvature.threshold_m:.0f} m curvature threshold."
            )
        entry["flag"] = flag.value
    return coded
