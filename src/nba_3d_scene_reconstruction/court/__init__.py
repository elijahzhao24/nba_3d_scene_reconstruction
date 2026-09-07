"""Basketball court detection and image-to-court calibration."""

from .calibration import (
    CourtCalibrator,
    CourtHomographyEstimator,
    HomographyEstimate,
    build_correspondences,
    project_points,
)
from .configuration import (
    COURT_LANDMARK_LABELS,
    COURT_LANDMARK_POINTS_CM,
    CourtCalibrationConfiguration,
    CourtDetectorConfiguration,
)
from .detector import CourtSchemaMismatchError, RoboflowCourtDetector
from .debug import CourtDebugRenderer, draw_source_overlay
from .projector import PlayerCourtProjector
from .schemas import (
    CalibrationSource, CourtCalibration, CourtDetection, CourtKeypoint,
    PlayerCourtPosition,
)

__all__ = [
    "COURT_LANDMARK_LABELS",
    "COURT_LANDMARK_POINTS_CM",
    "CalibrationSource",
    "CourtCalibration",
    "CourtCalibrationConfiguration",
    "CourtCalibrator",
    "CourtDetection",
    "CourtDetectorConfiguration",
    "CourtDebugRenderer",
    "CourtHomographyEstimator",
    "CourtKeypoint",
    "CourtSchemaMismatchError",
    "HomographyEstimate",
    "PlayerCourtPosition",
    "PlayerCourtProjector",
    "RoboflowCourtDetector",
    "build_correspondences",
    "draw_source_overlay",
    "project_points",
]
