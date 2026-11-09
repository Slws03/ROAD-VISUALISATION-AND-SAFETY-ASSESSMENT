"""Gap-check: state the fact, check the mitigation, recommend only on a gap."""

from irap_screening.config import load_config
from irap_screening.decision.gapcheck import curve_series, jkr_warning_sign, run_gapcheck
from irap_screening.io.schema import (
    DoubleLineRecord,
    ExistenceResult,
    Flag,
    ModuleResult,
    Segment,
    SignRecord,
)
from irap_screening.modules.curvature import LEFT, RIGHT, Curve

CFG = load_config()
PHOTOS = [float(m) for m in range(0, 901, 20)]  # one photo every 20 m


def _segments(count: int = 9) -> list[Segment]:
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


def _module_results(
    segments, curvature=Flag.NORMAL, grade=Flag.NORMAL, delineation=Flag.NORMAL, targets=(5,)
):
    """Flag the segments at the target indices and leave the rest normal."""
    flagged_ids = {segments[i].segment_id for i in targets}

    def series(name, flagged):
        return [
            ModuleResult(
                module=name,
                segment_id=s.segment_id,
                flag=flagged if s.segment_id in flagged_ids else Flag.NORMAL,
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


# A left curve inside segment 5 (500-600 m). Its warning sign belongs at
# 510 - 230 = 280 m, searched from 280 - 50 - 50 = 180 m to 330 m.
CURVE = Curve(510.0, 590.0, LEFT, 312.5)


def _check(signs=(), curves=(CURVE,), photos=PHOTOS, cfg=CFG, **flags):
    segments = _segments()
    flags.setdefault("curvature", Flag.CRITICAL)
    return run_gapcheck(
        segments,
        _module_results(segments, **flags),
        _existence(segments, signs=list(signs)),
        cfg,
        list(curves),
        photos,
    )


def test_critical_curve_with_no_mitigation_reports_both_missing():
    entry = _check()["R01-0005"][0]
    assert set(entry.missing) == {"advance_bend_sign", "chevron"}
    assert entry.present == [] and entry.unchecked == []


def test_bend_sign_at_230m_before_the_curve_satisfies_the_advance_warning():
    entry = _check(signs=[_sign("Bend", 280.0)])["R01-0005"][0]
    assert "advance_bend_sign" in entry.present
    assert entry.missing == ["chevron"]


def test_a_sign_photographed_before_its_position_still_counts():
    """The camera sees a sign ahead, so its photo is taken before it: 200 m here."""
    entry = _check(signs=[_sign("Bend", 200.0)])["R01-0005"][0]
    assert "advance_bend_sign" in entry.present


def test_double_bend_substitutes_for_bend():
    """The detector cannot tell the JKR sign types apart; any bend warning counts."""
    entry = _check(signs=[_sign("Double Bend", 280.0)])["R01-0005"][0]
    assert "advance_bend_sign" in entry.present


def test_bend_sign_far_from_the_placement_point_does_not_count():
    entry = _check(signs=[_sign("Bend", 100.0)])["R01-0005"][0]
    assert "advance_bend_sign" in entry.missing


def test_chevron_is_checked_on_the_curve_not_upstream():
    """Chevrons run along the curve, so an upstream chevron does not satisfy it."""
    on_curve = _check(signs=[_sign("Chevron", 550.0)])["R01-0005"][0]
    assert "chevron" in on_curve.present
    seen_ahead = _check(signs=[_sign("Chevron", 480.0)])["R01-0005"][0]
    assert "chevron" in seen_ahead.present
    upstream = _check(signs=[_sign("Chevron", 300.0)])["R01-0005"][0]
    assert "chevron" in upstream.missing


def test_fully_mitigated_curve_produces_no_recommendation():
    """The critical flag stays; only the recommendation disappears."""
    recs = _check(signs=[_sign("Bend", 280.0), _sign("Chevron", 550.0)])
    assert recs["R01-0005"] == []


def test_a_sign_position_before_the_route_starts_is_unchecked_not_missing():
    """A curve 110 m in wants its sign at -120 m, which no photo can show."""
    recs = _check(curves=[Curve(110.0, 190.0, LEFT, 312.5)], targets=(1,))
    entry = recs["R01-0001"][0]
    assert entry.unchecked == ["advance_bend_sign"]
    assert "advance_bend_sign" not in entry.missing


def test_a_sign_position_with_no_photos_is_unchecked():
    entry = _check(photos=[m for m in PHOTOS if m >= 400.0])["R01-0005"][0]
    assert entry.unchecked == ["advance_bend_sign"]


def test_curves_closer_than_180m_form_one_series():
    near = [Curve(100.0, 150.0, LEFT, 300.0), Curve(320.0, 360.0, RIGHT, 300.0)]
    far = [Curve(100.0, 150.0, LEFT, 300.0), Curve(340.0, 380.0, RIGHT, 300.0)]
    assert len(curve_series(near, CFG)) == 1
    assert len(curve_series(far, CFG)) == 2


def test_the_jkr_sign_follows_the_series():
    left, right = Curve(0.0, 50.0, LEFT, 300.0), Curve(0.0, 50.0, RIGHT, 300.0)
    assert jkr_warning_sign([right]) == "WD.10a"
    assert jkr_warning_sign([left]) == "WD.10b"
    assert jkr_warning_sign([right, left]) == "WD.2a"
    assert jkr_warning_sign([left, right]) == "WD.2b"
    assert jkr_warning_sign([left, left]) == "WD.11"
    assert jkr_warning_sign([left, right, left]) == "WD.11"


def test_one_advance_sign_serves_a_series_of_curves():
    """A series is warned of once, before its first curve, not once per curve."""
    curves = [Curve(510.0, 560.0, LEFT, 300.0), Curve(650.0, 690.0, RIGHT, 300.0)]
    recs = _check(signs=[_sign("Bend", 280.0)], curves=curves, targets=(5, 6))
    for segment_id in ("R01-0005", "R01-0006"):
        entry = recs[segment_id][0]
        assert "advance_bend_sign" in entry.present
        assert entry.trigger["sign_code"] == "WD.2b"
        assert entry.trigger["curve_start_m"] == 510.0
        assert entry.trigger["curve_end_m"] == 690.0
    assert recs["R01-0005"][0] is recs["R01-0006"][0]


def test_chevrons_are_needed_on_every_curve_of_a_series():
    curves = [Curve(510.0, 560.0, LEFT, 300.0), Curve(650.0, 690.0, RIGHT, 300.0)]
    recs = _check(signs=[_sign("Chevron", 530.0)], curves=curves, targets=(5, 6))
    entry = recs["R01-0005"][0]
    assert "chevron" in entry.missing
    assert entry.trigger["curves_without_chevrons"] == 1.0


def test_a_curve_on_no_critical_segment_is_ignored():
    recs = _check(curves=[Curve(710.0, 790.0, LEFT, 312.5)])
    assert all(not entries for entries in recs.values())


def test_a_normal_segment_a_curve_reaches_into_carries_nothing():
    recs = _check(curves=[Curve(560.0, 640.0, LEFT, 312.5)])
    assert recs["R01-0005"] and recs["R01-0006"] == []


def test_critical_grade_without_overtaking_control_is_flagged():
    entry = _check(curvature=Flag.NORMAL, grade=Flag.CRITICAL)["R01-0005"][0]
    assert entry.rule == "grade_overtaking_control_missing"
    assert set(entry.missing) == {"no_overtaking_sign", "double_line"}


def test_double_line_alone_satisfies_the_grade_rule():
    """Malaysian no-overtaking control is usually paint, not a sign."""
    segments = _segments()
    recs = run_gapcheck(
        segments, _module_results(segments, grade=Flag.CRITICAL),
        _existence(segments, double_line_on="R01-0005"), CFG, [], PHOTOS,
    )
    assert recs["R01-0005"] == []


def test_no_overtaking_sign_alone_satisfies_the_grade_rule():
    recs = _check(signs=[_sign("No overtaking", 550.0)], curvature=Flag.NORMAL, grade=Flag.CRITICAL)
    assert recs["R01-0005"] == []


def test_a_critical_curve_on_a_critical_grade_adds_no_rule_of_its_own():
    """Only missing mitigations are recommended; the combination names none."""
    rules = sorted(r.rule for r in _check(grade=Flag.CRITICAL)["R01-0005"])
    assert rules == ["curve_warning_incomplete", "grade_overtaking_control_missing"]


def test_recommendations_carry_no_priority():
    """Priority was dropped: every rule's output is a gap, not a ranking."""
    dumped = _check(delineation=Flag.CRITICAL)["R01-0005"][0].model_dump()
    assert "priority" not in dumped
    assert "priority_raised_by" not in dumped


def test_not_assessed_never_triggers_a_rule():
    """A low-speed curve is unevaluated, not confirmed safe."""
    recs = _check(curvature=Flag.NOT_ASSESSED)
    assert all(not entries for entries in recs.values())


def test_normal_segments_produce_nothing():
    recs = _check(curvature=Flag.NORMAL, curves=())
    assert all(not entries for entries in recs.values())


def test_placement_distance_is_configurable():
    cfg = load_config(overrides=["gapcheck.curve_sign_distance_m=100"])
    entry = _check(signs=[_sign("Bend", 410.0)], cfg=cfg)["R01-0005"][0]
    assert "advance_bend_sign" in entry.present


def test_an_unchecked_sign_alone_still_recommends_a_look():
    """Chevrons all present, the sign position not photographed: flag it for a person."""
    recs = _check(
        signs=[_sign("Chevron", 150.0)],
        curves=[Curve(110.0, 190.0, LEFT, 312.5)],
        targets=(1,),
    )
    entry = recs["R01-0001"][0]
    assert entry.missing == [] and entry.unchecked == ["advance_bend_sign"]


def test_adjacent_poor_marking_is_one_stretch_with_one_recommendation():
    """Segments 4 and 5 touch, 7 stands apart: two stretches, not three."""
    recs = _check(curvature=Flag.NORMAL, delineation=Flag.CRITICAL, curves=(), targets=(4, 5, 7))
    first, second = recs["R01-0004"][0], recs["R01-0007"][0]
    assert first.rule == "lane_marking_poor" and first.missing == ["visible_marking"]
    assert recs["R01-0005"][0] is first
    assert (first.trigger["start_m"], first.trigger["end_m"]) == (400.0, 600.0)
    assert second.trigger["series_id"] == "D02" and recs["R01-0006"] == []


def test_poor_marking_on_a_critical_curve_gets_its_own_recommendation():
    rules = sorted(r.rule for r in _check(delineation=Flag.CRITICAL)["R01-0005"])
    assert rules == ["curve_warning_incomplete", "lane_marking_poor"]
