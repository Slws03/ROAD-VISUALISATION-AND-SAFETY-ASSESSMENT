"""Abstract route source.

The internal schema is owned by this project and stays stable. External
formats are translated by concrete adapters, so a provider changing their
column names touches exactly one file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from irap_screening.io.schema import Frame, GpsFix, ImuSample


class RouteSource(ABC):
    """One recorded drive: images plus GPS and IMU logs."""

    @abstractmethod
    def frames(self) -> list[Frame]:
        """Return frames ordered by timestamp."""

    @abstractmethod
    def gps(self) -> list[GpsFix]:
        """Return GPS fixes ordered by timestamp."""

    @abstractmethod
    def imu(self) -> list[ImuSample]:
        """Return IMU samples ordered by timestamp."""
