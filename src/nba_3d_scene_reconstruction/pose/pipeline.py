"""Model-independent orchestration used by workers and tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import cv2

from ..tracking.schemas import PlayerObservation, VideoManifest
from .estimator import Pose2DEstimator
from .lifter import PoseLifter
from .schemas import Pose2D, Pose3D, PreparedPose2D


def estimate_tracked_poses(
    manifest: VideoManifest,
    observations: Iterable[PlayerObservation],
    estimator: Pose2DEstimator,
) -> tuple[Pose2D, ...]:
    grouped: dict[int, list[PlayerObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.frame_idx].append(observation)
    output = []
    for frame_idx in sorted(grouped):
        frame_path = Path(manifest.frames_dir) / f"{frame_idx:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise OSError(f"could not read extracted frame: {frame_path}")
        output.extend(
            estimator.estimate(frame, grouped[frame_idx], clip_id=manifest.clip_id)
        )
    return tuple(output)


def lift_prepared_sequences(
    poses: Iterable[PreparedPose2D], lifter: PoseLifter
) -> tuple[Pose3D, ...]:
    grouped: dict[str, list[PreparedPose2D]] = defaultdict(list)
    for pose in poses:
        grouped[pose.sequence_id].append(pose)
    output = []
    for sequence_id in sorted(grouped):
        sequence = sorted(grouped[sequence_id], key=lambda pose: pose.frame_idx)
        output.extend(lifter.lift(sequence))
    return tuple(sorted(output, key=lambda pose: (pose.frame_idx, pose.track_id)))
