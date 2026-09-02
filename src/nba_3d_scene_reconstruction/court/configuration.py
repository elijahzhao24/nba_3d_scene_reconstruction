"""Version-pinned model settings and canonical NBA court landmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_COURT_MODEL_ID = "basketball-court-detection-2/22"
DEFAULT_COURT_API_URL = "https://serverless.roboflow.com"
DEFAULT_DETECTION_CONFIDENCE = 0.30
DEFAULT_KEYPOINT_CONFIDENCE = 0.50
DEFAULT_PARENT_CLASS_NAME = "court"
DEFAULT_PARENT_CLASS_ID = 0

# The labels intentionally contain gaps (based on roboflow dataset). Their tuple index is the dense
# landmark index returned by the version-pinned model skeleton.
COURT_LANDMARK_LABELS: tuple[str, ...] = (
    "01",
    "02",
    "04",
    "05",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "19",
    "21",
    "23",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "37",
    "38",
    "40",
    "41",
)

# Canonical court coordinates from Roboflow's basketball configuration.
# Origin: label 01. X runs baseline-to-baseline; Y runs sideline-to-sideline.
COURT_LANDMARK_POINTS_CM: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 91.0),
    (0.0, 518.0),
    (0.0, 1006.0),
    (0.0, 1433.0),
    (0.0, 1524.0),
    (160.0, 762.0),
    (424.0, 91.0),
    (424.0, 1433.0),
    (579.0, 518.0),
    (579.0, 762.0),
    (579.0, 1006.0),
    (835.0, 0.0),
    (884.0, 762.0),
    (835.0, 1524.0),
    (1432.0, 0.0),
    (1432.0, 762.0),
    (1432.0, 1524.0),
    (2030.0, 0.0),
    (1981.0, 762.0),
    (2030.0, 1524.0),
    (2286.0, 518.0),
    (2286.0, 762.0),
    (2286.0, 1006.0),
    (2441.0, 91.0),
    (2441.0, 1433.0),
    (2705.0, 762.0),
    (2865.0, 0.0),
    (2865.0, 91.0),
    (2865.0, 518.0),
    (2865.0, 1006.0),
    (2865.0, 1433.0),
    (2865.0, 1524.0),
)


def _environment_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    return float(raw_value) if raw_value is not None else default


@dataclass(frozen=True)
class CourtDetectorConfiguration:
    """Configuration kept together so model and skeleton cannot drift."""

    model_id: str = DEFAULT_COURT_MODEL_ID
    api_url: str = DEFAULT_COURT_API_URL
    detection_confidence: float = DEFAULT_DETECTION_CONFIDENCE
    keypoint_confidence: float = DEFAULT_KEYPOINT_CONFIDENCE
    parent_class_name: str = DEFAULT_PARENT_CLASS_NAME
    parent_class_id: int = DEFAULT_PARENT_CLASS_ID
    landmark_labels: tuple[str, ...] = COURT_LANDMARK_LABELS
    landmark_points_cm: tuple[tuple[float, float], ...] = COURT_LANDMARK_POINTS_CM

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.api_url:
            raise ValueError("api_url must not be empty")
        if not self.parent_class_name:
            raise ValueError("parent_class_name must not be empty")
        if self.parent_class_id < 0:
            raise ValueError("parent_class_id must be non-negative")
        for name, value in (
            ("detection_confidence", self.detection_confidence),
            ("keypoint_confidence", self.keypoint_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if len(self.landmark_labels) != len(self.landmark_points_cm):
            raise ValueError("landmark labels and points must have equal length")
        if len(set(self.landmark_labels)) != len(self.landmark_labels):
            raise ValueError("landmark labels must be unique")
        if not self.landmark_labels:
            raise ValueError("at least one landmark must be configured")

    @classmethod
    def from_environment(cls) -> CourtDetectorConfiguration:
        """Build configuration without reading or storing the API key."""
        return cls(
            model_id=os.environ.get(
                "COURT_MODEL_ID",
                DEFAULT_COURT_MODEL_ID,
            ),
            api_url=os.environ.get("COURT_API_URL", DEFAULT_COURT_API_URL),
            detection_confidence=_environment_float(
                "COURT_DETECTION_CONFIDENCE",
                DEFAULT_DETECTION_CONFIDENCE,
            ),
            keypoint_confidence=_environment_float(
                "COURT_KEYPOINT_CONFIDENCE",
                DEFAULT_KEYPOINT_CONFIDENCE,
            ),
        )
