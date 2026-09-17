"""Serializable pose records. Coordinates stay in their declared spaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseJoint2D:
    name: str
    image_xy: tuple[float, float]
    score: float
    observed: bool
    provenance: str = "model"


@dataclass(frozen=True)
class Pose2D:
    clip_id: str
    segment_id: str
    frame_idx: int
    timestamp_seconds: float
    track_id: int
    layout: str
    joints: tuple[PoseJoint2D, ...]
    visible: bool = True
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedPose2D:
    clip_id: str
    segment_id: str
    frame_idx: int
    timestamp_seconds: float
    track_id: int
    sequence_id: str
    layout: str
    joints: tuple[PoseJoint2D, ...]
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PoseJoint3D:
    name: str
    xyz: tuple[float, float, float]
    input_score: float
    input_observed: bool


@dataclass(frozen=True)
class Pose3D:
    clip_id: str
    segment_id: str
    frame_idx: int
    timestamp_seconds: float
    track_id: int
    sequence_id: str
    layout: str
    joints: tuple[PoseJoint3D, ...]
    coordinate_space: str = "pelvis_relative"
    units: str = "model_relative"
    axes: str = "motionbert_camera_axes"
    quality_flags: tuple[str, ...] = ()
