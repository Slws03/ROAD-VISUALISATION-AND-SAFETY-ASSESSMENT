"""Report rendering: report.json, segments.geojson and report.md.

The output format is the only coupling point with the dashboard that will
consume it later, which makes it the most expensive thing to change after the
fact. Four rules follow from that:

- Raw metrics travel with flags, so a dashboard can draw a profile along the
  route and preview a threshold change instead of only colouring segments.
- Recommendations name the specific missing mitigation, which is what makes
  them actionable for a road authority.
- A config snapshot travels in the report, so any output can be traced to the
  thresholds that produced it.
- Segments that could not be evaluated are marked uncovered rather than clean.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from irap_screening.config import Settings
from irap_screening.io.schema import (
    ExistenceResult,
    Flag,
    ModuleResult,
    Recommendation,
    Segment,
)

SCHEMA_VERSION = "1.0"
SCORING_MODULES = ("curvature", "grade", "delineation", "condition")


def _index(results: list[ModuleResult]) -> dict[str, ModuleResult]:
    return {result.segment_id: result for result in results}


def _module_block(result: ModuleResult | None) -> dict[str, Any]:
    if result is None:
        return {"flag": Flag.NOT_ASSESSED.value}
    block: dict[str, Any] = {"flag": result.flag.value}
    block.update(result.metrics)
    if result.evidence:
        block["instances"] = result.evidence
    if result.notes:
        block["notes"] = result.notes
    return block


def build_report(
    route_id: str,
    segments: list[Segment],
    module_results: dict[str, list[ModuleResult]],
    existence: list[ExistenceResult],
    recommendations: dict[str, list[Recommendation]],
    cfg: Settings,
    quality_of_curve: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the full structured report."""
    indexed = {name: _index(results) for name, results in module_results.items()}
    existence_by_segment = {entry.segment_id: entry for entry in existence}

    coverage = {
        name: {"total": len(segments), "not_assessed": 0} for name in SCORING_MODULES
    }

    segment_records: list[dict[str, Any]] = []
    for segment in segments:
        record: dict[str, Any] = {
            "segment_id": segment.segment_id,
            "start_m": segment.start_m,
            "end_m": segment.end_m,
            "start_latlon": list(segment.start_latlon),
            "end_latlon": list(segment.end_latlon),
        }

        uncovered = False
        for name in SCORING_MODULES:
            result = indexed.get(name, {}).get(segment.segment_id)
            record[name] = _module_block(result)
            if record[name]["flag"] == Flag.NOT_ASSESSED.value:
                coverage[name]["not_assessed"] += 1
                uncovered = True

        # Derived attribute, not a scoring module: it has no flag and does not
        # count toward coverage.
        if quality_of_curve is not None:
            record["quality_of_curve"] = quality_of_curve.get(
                segment.segment_id, {"code": None, "basis": ""}
            )

        entry = existence_by_segment.get(segment.segment_id)
        record["signs"] = (
            [sign.model_dump() for sign in entry.signs] if entry else []
        )
        record["double_line"] = (
            entry.double_line.model_dump() if entry else {"present": False}
        )

        record["recommendations"] = [
            recommendation.model_dump()
            for recommendation in recommendations.get(segment.segment_id, [])
        ]
        record["representative_frame_id"] = (
            segment.frame_ids[0] if segment.frame_ids else None
        )
        # Marking a segment uncovered keeps it visually distinct from a segment
        # that was evaluated and found clean.
        record["coverage"] = "uncovered" if uncovered else "covered"

        segment_records.append(record)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "route_id": route_id,
        "config": cfg.model_dump(mode="json"),
        "coverage": coverage,
        "segments": segment_records,
    }


def to_geojson(report: dict[str, Any]) -> dict[str, Any]:
    """One LineString feature per segment, properties carrying the full record."""
    features = []
    for record in report["segments"]:
        start_lat, start_lon = record["start_latlon"]
        end_lat, end_lon = record["end_latlon"]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    # GeoJSON is x,y: longitude first.
                    "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
                },
                "properties": record,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def to_markdown(report: dict[str, Any]) -> str:
    """Human-readable summary listing only segments that carry a recommendation."""
    lines = [
        f"# Road Safety Screening Report - {report['route_id']}",
        "",
        f"Generated: {report['generated_at']}",
        f"Segments: {len(report['segments'])}",
        "",
        "## Coverage",
        "",
        "| Module | Assessed | Not assessed |",
        "| --- | --- | --- |",
    ]
    for name, stats in report["coverage"].items():
        lines.append(
            f"| {name} | {stats['total'] - stats['not_assessed']} | {stats['not_assessed']} |"
        )

    lines += ["", "## Recommendations", ""]
    flagged = [r for r in report["segments"] if r["recommendations"]]
    if not flagged:
        lines.append("No gaps found: every hazardous segment already has its mitigation.")
        return "\n".join(lines) + "\n"

    for record in flagged:
        seg_id = record["segment_id"]
        start = record["start_m"]
        end = record["end_m"]
        lines.append(f"### {seg_id} ({start:.0f}-{end:.0f} m)")
        lines.append("")
        for recommendation in record["recommendations"]:
            missing = ", ".join(recommendation["missing"]) or "none"
            # text already names the missing mitigations in a full sentence; fall
            # back to the bare list only when a rule supplies no text.
            detail = recommendation["text"] or f"missing: {missing}."
            lines.append(f"- {recommendation['rule']} - {detail}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, report: dict[str, Any]) -> None:
    """Write all three output files into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "segments.geojson").write_text(
        json.dumps(to_geojson(report), indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(to_markdown(report), encoding="utf-8")
