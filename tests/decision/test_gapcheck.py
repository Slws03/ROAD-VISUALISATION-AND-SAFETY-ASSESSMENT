"""Gap-check: state the fact, check the mitigation, recommend only on a gap."""

from irap_screening.config import load_config
from irap_screening.decision.gapcheck import run_gapcheck
from irap_screening.io.schema import (
    DoubleLineRecord,
    ExistenceResult,
    Flag,
    ModuleResult,
    Segment,
    SignRecord,
)

CFG = load_config()


def _segments(count: int = 5) -> list[Segment]:
    return [
        Segment(
            segment_id=f"R01-{i:04d}",
            start_m=i * 100.0,
            end_m=(i + 1) * 100.0,
            start_latlon=(1.55, 110.35),
            end_latlon=(1.55, 110.35),
        )
        for i in range(count)
    ]


def _module_results(segments, curvature=Flag.NORMAL, grade=Flag.NORMAL, delineation=Flag.NORMAL):
    """Flag the fourth segment (start_m = 300) and leave the rest normal."""
    target = segments[3].segment_id

    def series(name, flagged):
        return [
            ModuleResult(
                module=name,
                segment_id=s.segment_id,
                flag=flagged if s.segment_id == target else Flag.NORMAL,
                metrics={"radius_m": 312.5} if name == "curvature" else {"grade_pct": 8.0},
            )
            for s in segments
        ]

    return {
        "curvature": series("curvature", curvature),
        "grade": series("grade", grade),
        "delineation": series("delineation", delineation),
        "condition": series("condition", Flag.NORMAL),
    }


def _existence(segments, signs=None, double_line_on=None):
    signs = signs or []
    return [
        ExistenceResult(
            segment_id=s.segment_id,
            signs=[sign for sign in signs if s.start_m <= sign.at_m < s.end_m],
            double_line=DoubleLineRecord(present=(s.segment_id == double_line_on)),
        )
        for s in segments
    ]


def _sign(cls: str, at_m: float) -> SignRecord:
    return SignRecord(id=f"x#sign#{at_m}", cls=cls, conf=0.9, at_m=at_m)


def test_critical_curve_with_no_mitigation_reports_both_missing():
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments), CFG,
    )
    entry = recs["R01-0003"][0]
    assert set(entry.missing) == {"advance_bend_sign", "chevron"}
    assert entry.present == []


def test_bend_sign_at_230m_upstream_satisfies_the_advance_warning():
    """Curve entry is 300 m; JKR places the sign 230 m before, so around 70 m."""
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments, signs=[_sign("Bend", 70.0)]), CFG,
    )
    entry = recs["R01-0003"][0]
    assert "advance_bend_sign" in entry.present
    assert entry.missing == ["chevron"]


def test_double_bend_substitutes_for_bend():
    """The two are mutually exclusive; either satisfies the advance warning."""
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments, signs=[_sign("Double Bend", 70.0)]), CFG,
    )
    assert "advance_bend_sign" in recs["R01-0003"][0].present


def test_bend_sign_far_from_the_placement_point_does_not_count():
    """A sign 300 m before the curve is outside the 230 m plus tolerance window."""
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments, signs=[_sign("Bend", 5.0)]), CFG,
    )
    assert "advance_bend_sign" in recs["R01-0003"][0].missing


def test_chevron_is_checked_by_overlap_not_upstream():
    """Chevrons run along the curve, so an upstream chevron does not satisfy it."""
    segments = _segments()
    overlapping = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments, signs=[_sign("Chevron", 350.0)]), CFG,
    )
    assert "chevron" in overlapping["R01-0003"][0].present

    upstream = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments, signs=[_sign("Chevron", 70.0)]), CFG,
    )
    assert "chevron" in upstream["R01-0003"][0].missing


def test_fully_mitigated_curve_produces_no_recommendation():
    """The critical flag stays; only the recommendation disappears."""
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments, signs=[_sign("Bend", 70.0), _sign("Chevron", 350.0)]), CFG,
    )
    assert recs.get("R01-0003", []) == []


def test_critical_grade_without_overtaking_control_is_flagged():
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, grade=Flag.CRITICAL),
        _existence(segments), CFG,
    )
    entry = recs["R01-0003"][0]
    assert entry.rule == "grade_overtaking_control_missing"
    assert set(entry.missing) == {"no_overtaking_sign", "double_line"}


def test_double_line_alone_satisfies_the_grade_rule():
    """Malaysian no-overtaking control is usually paint, not a sign."""
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, grade=Flag.CRITICAL),
        _existence(segments, double_line_on="R01-0003"), CFG,
    )
    assert recs.get("R01-0003", []) == []


def test_no_overtaking_sign_alone_satisfies_the_grade_rule():
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, grade=Flag.CRITICAL),
        _existence(segments, signs=[_sign("No overtaking", 350.0)]), CFG,
    )
    assert recs.get("R01-0003", []) == []


def test_compound_curve_and_grade_is_highest_priority():
    segments = _segments()
    recs = run_gapcheck(
        segments,
        _module_results(segments, curvature=Flag.CRITICAL, grade=Flag.CRITICAL),
        _existence(segments), CFG,
    )
    rules = {r.rule for r in recs["R01-0003"]}
    assert "compound_curve_on_grade" in rules
    compound = next(r for r in recs["R01-0003"] if r.rule == "compound_curve_on_grade")
    assert compound.priority == "highest"


def test_poor_lane_marking_raises_priority_without_adding_a_recommendation():
    segments = _segments()
    plain = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments), CFG,
    )
    raised = run_gapcheck(
        segments,
        _module_results(segments, curvature=Flag.CRITICAL, delineation=Flag.CRITICAL),
        _existence(segments), CFG,
    )
    assert len(raised["R01-0003"]) == len(plain["R01-0003"])
    assert raised["R01-0003"][0].priority_raised_by == ["lane_marking_worst_band"]
    assert raised["R01-0003"][0].priority == "highest"
    assert plain["R01-0003"][0].priority == "high"


def test_not_assessed_never_triggers_a_rule():
    """A low-speed curve is unevaluated, not confirmed safe."""
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.NOT_ASSESSED),
        _existence(segments), CFG,
    )
    assert recs.get("R01-0003", []) == []


def test_normal_segments_produce_nothing():
    segments = _segments()
    recs = run_gapcheck(segments, _module_results(segments), _existence(segments), CFG)
    assert all(not entries for entries in recs.values())


def test_placement_distance_is_configurable():
    cfg = load_config(overrides=["gapcheck.curve_sign_distance_m=100"])
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, curvature=Flag.CRITICAL),
        _existence(segments, signs=[_sign("Bend", 200.0)]), cfg,
    )
    assert "advance_bend_sign" in recs["R01-0003"][0].present


def _curve_module_results(segments, critical_ids, unassessed_ids=()):
    """Flag an explicit set of segments critical for curvature."""

    def series(name):
        results = []
        for s in segments:
            if name == "curvature" and s.segment_id in critical_ids:
                flag = Flag.CRITICAL
            elif name == "curvature" and s.segment_id in unassessed_ids:
                flag = Flag.NOT_ASSESSED
            else:
                flag = Flag.NORMAL
            results.append(
                ModuleResult(
                    module=name, segment_id=s.segment_id, flag=flag,
                    metrics={"radius_m": 45.0} if name == "curvature" else {},
                )
            )
        return results

    return {name: series(name) for name in ("curvature", "grade", "delineation", "condition")}


def _advance_present(recs, segment_id):
    """True when the segment reports no missing advance sign."""
    return all(
        "advance_bend_sign" not in entry.missing for entry in recs.get(segment_id, [])
    )


def test_one_advance_sign_serves_a_curve_spanning_several_segments():
    """A bend is signed once at its entry, not once per 100 m coding unit.

    The 100 m grid is iRAP's reporting unit, so a 300 m curve necessarily spans
    three of them. Reading each unit as its own curve demanded three advance
    signs at three different upstream points, which is not how signage works.
    """
    segments = _segments(count=9)
    critical = {"R01-0004", "R01-0005", "R01-0006"}
    # One sign at the curve's entry: 400 m - 230 m.
    signs = [_sign("Bend", 170.0)] + [
        _sign("Chevron", s.start_m + 50.0) for s in segments if s.segment_id in critical
    ]

    recs = run_gapcheck(
        segments, _curve_module_results(segments, critical), _existence(segments, signs), CFG
    )

    for segment_id in sorted(critical):
        assert _advance_present(recs, segment_id), f"{segment_id} still reports it missing"
    assert recs["R01-0004"] == [] and recs["R01-0006"] == []


def test_a_sign_at_a_later_segments_own_upstream_point_does_not_count():
    """Pins that the previous test is not simply accepting any Bend anywhere.

    500 - 230 = 270 m is where the old per-segment rule looked for segment 5's
    sign. It is not the curve's entry, so it must satisfy nothing.
    """
    segments = _segments(count=9)
    critical = {"R01-0004", "R01-0005", "R01-0006"}
    signs = [_sign("Bend", 270.0)]

    recs = run_gapcheck(
        segments, _curve_module_results(segments, critical), _existence(segments, signs), CFG
    )

    for segment_id in sorted(critical):
        assert not _advance_present(recs, segment_id)


def test_two_curves_separated_by_normal_road_keep_separate_entries():
    """Merging must not run through a straight stretch and join unrelated bends."""
    segments = _segments(count=9)
    critical = {"R01-0001", "R01-0002", "R01-0005", "R01-0006"}
    # Entry points: 100 - 230 = -130 (before the route) and 500 - 230 = 270.
    signs = [_sign("Bend", 270.0)] + [
        _sign("Chevron", s.start_m + 50.0) for s in segments if s.segment_id in critical
    ]

    recs = run_gapcheck(
        segments, _curve_module_results(segments, critical), _existence(segments, signs), CFG
    )

    assert _advance_present(recs, "R01-0005")
    assert _advance_present(recs, "R01-0006")
    # The first curve starts before the route, so its sign can never be observed.
    assert not _advance_present(recs, "R01-0001")


def test_an_unassessed_segment_breaks_the_run():
    """Continuity cannot be claimed through a segment that was never evaluated."""
    segments = _segments(count=9)
    critical = {"R01-0003", "R01-0005"}
    signs = [_sign("Bend", 270.0)]  # entry of the curve starting at 500 m

    recs = run_gapcheck(
        segments,
        _curve_module_results(segments, critical, unassessed_ids={"R01-0004"}),
        _existence(segments, signs),
        CFG,
    )

    assert _advance_present(recs, "R01-0005")
    assert not _advance_present(recs, "R01-0003")


def test_the_recommendation_names_the_curve_it_belongs_to():
    """Four findings on one bend must be legible as one feature, not four."""
    segments = _segments(count=9)
    critical = {"R01-0004", "R01-0005", "R01-0006"}

    recs = run_gapcheck(
        segments, _curve_module_results(segments, critical), _existence(segments), CFG
    )

    for segment_id in sorted(critical):
        trigger = recs[segment_id][0].trigger
        assert trigger["curve_start_m"] == 400.0
        assert trigger["curve_end_m"] == 700.0
