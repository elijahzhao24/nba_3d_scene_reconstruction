"""Data contracts returned by the court detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


Point: TypeAlias = tuple[float, float]
BoundingBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class CourtKeypoint:
    """One landmark from the model's nested court prediction."""

    landmark_id: int
    label: str
    image_xy: Point
    confidence: float


@dataclass(frozen=True)
class CourtDetection:
    """One court prediction with a dense, version-checked landmark tuple."""

    frame_idx: int
    confidence: float
    bbox_xyxy: BoundingBox
    keypoints: tuple[CourtKeypoint | None, ...]
    image_width: int | None = None
    image_height: int | None = None

    def confident_keypoints(
        self,
        confidence_threshold: float,
    ) -> tuple[CourtKeypoint, ...]:
        """Return present landmarks meeting a downstream fitting threshold."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return tuple(
            point
            for point in self.keypoints
            if point is not None and point.confidence >= confidence_threshold
        )
