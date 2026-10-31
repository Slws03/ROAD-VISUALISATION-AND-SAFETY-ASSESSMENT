"""Quality of curve: iRAP column 40, coded 1/2/3 per Coding Manual v3.10 p.51-53."""

from irap_screening.config import load_config
from irap_screening.decision.curve_quality import code_quality_of_curve
from irap_screening.io.schema import ExistenceResult, Flag, ModuleResult, Segment, SignRecord

CFG = load_config()


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


def _results(segments, category=None, poor_delineation=(), unassessed=()):
    """category maps segment id -> iRAP curvature category; the rest are straight."""
    category = category or {}

    def curvature(s):
        if s.segment_id in unassessed:
            return ModuleResult(module="curvature", segment_id=s.segment_id, flag=Flag.NOT_ASSESSED)
        cat = category.get(s.segment_id, 1)
        return ModuleResult(
            module="curvature", segment_id=s.segment_id,
            flag=Flag.CRITICAL if cat >= 3 else Flag.NORMAL,
            metrics={"radius_m": 300.0, "irap_category": float(cat)},
        )

    def delineation(s):
        flag = Flag.CRITICAL if s.segment_id in poor_delineation else Flag.NORMAL
        return ModuleResult(module="delineation", segment_id=s.segment_id, flag=flag)

    return {
        "curvature": [curvature(s) for s in segments],
        "delineation": [delineation(s) for s in segments],
    }


def _existence(segments, signs=()):
    return [
        ExistenceResult(
            segment_id=s.segment_id,
            signs=[x for x in signs if s.start_m <= x.at_m < s.end_m],
        )
        for s in segments
    ]


def _sign(cls: str, at_m: float) -> SignRecord:
    return SignRecord(id=f"x#sign#{at_m}", cls=cls, conf=0.9, at_m=at_m)


def _codes(segments, results, signs=()):
    coded = code_quality_of_curve(segments, results, _existence(segments, signs), CFG)
    return {sid: entry["code"] for sid, entry in coded.items()}


def test_straight_road_is_not_applicable():
    segments = _segments()
    assert set(_codes(segments, _results(segments)).values()) == {3}


def test_unassessed_curvature_is_left_uncoded():
    segments = _segments()
    codes = _codes(segments, _results(segments, unassessed={"R01-0004"}))
    assert codes["R01-0004"] is None


def test_unsigned_curve_is_poor():
    segments = _segments()
    codes = _codes(segments, _results(segments, category={"R01-0004": 2}))
    assert codes["R01-0004"] == 2


def test_only_a_poor_curve_is_flagged_critical():
    segments = _segments()
    results = _results(segments, category={"R01-0004": 2}, unassessed={"R01-0007"})
    coded = code_quality_of_curve(segments, results, _existence(segments, ()), CFG)
    assert coded["R01-0004"]["flag"] == "critical"
    assert coded["R01-0000"]["flag"] == "normal"  # straight: not applicable
    assert coded["R01-0007"]["flag"] == "not_assessed"


def test_advance_sign_alone_is_adequate():
    """Manual condition b: a warning sign is enough; chevrons are not also required."""
    segments = _segments()
    codes = _codes(
        segments, _results(segments, category={"R01-0004": 3}), [_sign("Bend", 170.0)]
    )
    assert codes["R01-0004"] == 1


def test_chevron_alone_is_adequate():
    segments = _segments()
    codes = _codes(
        segments, _results(segments, category={"R01-0004": 4}), [_sign("Chevron", 450.0)]
    )
    assert codes["R01-0004"] == 1


def test_poor_delineation_forces_poor_even_when_signed():
    """Manual p.52: where delineation is poor, quality of curve is also poor."""
    segments = _segments()
    results = _results(segments, category={"R01-0004": 3}, poor_delineation={"R01-0004"})
    codes = _codes(segments, results, [_sign("Bend", 170.0), _sign("Chevron", 450.0)])
    assert codes["R01-0004"] == 2


def test_moderate_curve_spans_share_the_entry_sign():
    """A moderate (2) bend running into a sharp (3) one is one curve, signed once."""
    segments = _segments()
    results = _results(segments, category={"R01-0004": 2, "R01-0005": 3, "R01-0006": 2})
    codes = _codes(segments, results, [_sign("Bend", 170.0)])
    assert [codes[f"R01-000{i}"] for i in (4, 5, 6)] == [1, 1, 1]


def test_sign_at_a_later_segments_upstream_point_does_not_count():
    segments = _segments()
    results = _results(segments, category={"R01-0004": 2, "R01-0005": 3})
    codes = _codes(segments, results, [_sign("Bend", 270.0)])
    assert codes["R01-0004"] == 2 and codes["R01-0005"] == 2


def test_every_coded_segment_explains_itself():
    segments = _segments()
    coded = code_quality_of_curve(
        segments, _results(segments, category={"R01-0004": 2}), _existence(segments), CFG
    )
    assert all(entry["basis"] for entry in coded.values())
