"""Data contracts returned by the court detector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

Point: TypeAlias = tuple[float, float]
BoundingBox: TypeAlias = tuple[float, float, float, float]
Matrix3x3: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class CalibrationSource(str, Enum):
    """How a per-frame calibration record was produced."""

    DETECTED = "detected"
    HELD = "held"
    INVALID = "invalid"


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


@dataclass(frozen=True)
class CourtCalibration:
    """Image/court transforms and quality for one frame.

    Court coordinates use the centimeters defined by
    ``COURT_LANDMARK_POINTS_CM``. Conversion to viewer meters belongs at the
    export boundary.
    """

    segment_id: str
    frame_idx: int
    source_frame_idx: int | None
    valid: bool
    source: CalibrationSource
    image_to_court: Matrix3x3 | None
    court_to_image: Matrix3x3 | None
    keypoint_count: int
    inlier_count: int
    inlier_ratio: float
    court_coverage_ratio: float
    median_error_px: float | None
    age_frames: int | None
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlayerCourtPosition:
    """Raw floor-plane projection in canonical court centimeters."""

    segment_id: str
    frame_idx: int
    timestamp_seconds: float
    track_id: int
    footpoint_image_xy: Point | None
    raw_court_xy: Point | None
    calibration_source_frame_idx: int | None
    calibration_age_frames: int | None
    quality_flags: tuple[str, ...] = ()
