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
from .debug import CourtDebugRenderer, draw_source_overlay
from .detector import CourtSchemaMismatchError, RoboflowCourtDetector
from .projector import PlayerCourtProjector
from .schemas import (
    CalibrationSource,
    CourtCalibration,
    CourtDetection,
    CourtKeypoint,
    PlayerCourtPosition,
)
from .smoothing import (
    TrajectorySmoothingConfiguration,
    clean_paths,
    clean_player_position_frames,
)

__all__ = [
    "COURT_LANDMARK_LABELS",
    "COURT_LANDMARK_POINTS_CM",
    "CalibrationSource",
    "CourtCalibration",
    "CourtCalibrationConfiguration",
    "CourtCalibrator",
    "CourtDebugRenderer",
    "CourtDetection",
    "CourtDetectorConfiguration",
    "CourtHomographyEstimator",
    "CourtKeypoint",
    "CourtSchemaMismatchError",
    "HomographyEstimate",
    "PlayerCourtPosition",
    "PlayerCourtProjector",
    "RoboflowCourtDetector",
    "TrajectorySmoothingConfiguration",
    "build_correspondences",
    "clean_paths",
    "clean_player_position_frames",
    "draw_source_overlay",
    "project_points",
]
