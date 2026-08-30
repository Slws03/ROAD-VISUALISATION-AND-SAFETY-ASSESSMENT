"""Adapter for the canonical CSV layout used by committed fixtures."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from irap_screening.io.schema import Frame, GpsFix, ImuSample
from irap_screening.io.sources.base import RouteSource


def _parse_ts(raw: str) -> datetime:
    """Accept ISO 8601 with a trailing Z, which fromisoformat rejects before 3.11."""
    return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


class FixtureSource(RouteSource):
    """Reads frames.csv, gps.csv and imu.csv from a route directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _rows(self, name: str) -> list[dict[str, str]]:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(f"Route file not found: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def frames(self) -> list[Frame]:
        return [
            Frame(
                frame_id=row["frame_id"],
                path=row["path"],
                timestamp_utc=_parse_ts(row["timestamp_utc"]),
                sha256=row["sha256"],
            )
            for row in self._rows("frames.csv")
        ]

    def gps(self) -> list[GpsFix]:
        return [
            GpsFix(
                timestamp_utc=_parse_ts(row["timestamp_utc"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                speed_mps=_optional_float(row, "speed_mps"),
                altitude=_optional_float(row, "altitude"),
                hdop=_optional_float(row, "hdop"),
            )
            for row in self._rows("gps.csv")
        ]

    def imu(self) -> list[ImuSample]:
        return [
            ImuSample(
                timestamp_utc=_parse_ts(row["timestamp_utc"]),
                ax=float(row["ax"]),
                ay=float(row["ay"]),
                az=float(row["az"]),
                gx=float(row["gx"]),
                gy=float(row["gy"]),
                gz=float(row["gz"]),
            )
            for row in self._rows("imu.csv")
        ]
