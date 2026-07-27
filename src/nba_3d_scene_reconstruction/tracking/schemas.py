"""Data Schemas for the player tracking subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np


BoundingBox: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]


class TrackStatus(str, Enum):
    ACTIVE = "active"
    MISSING = "missing"
    RETIRED = "retired"


class ObservationSource(str, Enum):
    SAM2_PROPAGATION = "sam2_propagation"
    RFDETR_REPROMPT = "rfdetr_reprompt"
    MISSING = "missing"


@dataclass(frozen=True)
class VideoManifest:
    clip_id: str
    segment_id: str
    source_path: str
    frames_dir: str
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class PlayerDetection:
    frame_idx: int
    bbox_xyxy: BoundingBox
    confidence: float
    class_id: int


@dataclass(frozen=True)
class SamMaskPrediction:
    frame_idx: int
    track_id: int
    mask: np.ndarray
    score: float | None = None


@dataclass
class TrackState:
    track_id: int
    segment_id: str
    status: TrackStatus
    start_frame: int
    last_seen_frame: int
    missing_frame_count: int = 0
    end_frame: int | None = None
    latest_bbox_xyxy: BoundingBox | None = None


@dataclass(frozen=True)
class AssociationMatch:
    detection_index: int
    track_id: int
    score: float


@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[AssociationMatch, ...] = ()
    unmatched_detection_indices: tuple[int, ...] = ()
    unmatched_track_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlayerObservation:
    segment_id: str
    frame_idx: int
    timestamp_seconds: float
    track_id: int
    visible: bool
    source: ObservationSource
    bbox_xyxy: BoundingBox | None = None
    centroid_xy: Point | None = None
    footpoint_xy: Point | None = None
    mask_ref: str | None = None
    mask_area: int = 0
    detection_confidence: float | None = None
    quality_flags: tuple[str, ...] = ()
