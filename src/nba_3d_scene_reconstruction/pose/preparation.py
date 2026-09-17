"""Confidence-aware interpolation and HALPE-to-H36M conversion."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise

from .schemas import Pose2D, PoseJoint2D, PreparedPose2D
from .skeletons import H36M_17


def prepare_pose_sequences(
    poses: tuple[Pose2D, ...],
    *,
    fps: float,
    confidence_threshold: float = 0.3,
    maximum_gap_seconds: float = 0.1,
) -> tuple[PreparedPose2D, ...]:
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be finite and positive")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    max_gap = math.floor(maximum_gap_seconds * fps + 1e-9)
    grouped: dict[tuple[str, str, int], list[Pose2D]] = defaultdict(list)
    for pose in poses:
        grouped[(pose.clip_id, pose.segment_id, pose.track_id)].append(pose)
    output: list[PreparedPose2D] = []
    for key, track in sorted(grouped.items()):
        track.sort(key=lambda item: item.frame_idx)
        halpe = [_threshold_pose(item, confidence_threshold) for item in track]
        halpe = _interpolate_joints(halpe, max_gap)
        runs = _continuous_runs(halpe, max_gap)
        for run_number, run in enumerate(runs):
            sequence_id = f"{key[0]}:{key[1]}:{key[2]}:{run_number}"
            for pose in run:
                output.append(_to_h36m(pose, sequence_id))
    return tuple(sorted(output, key=lambda p: (p.frame_idx, p.track_id)))


def _threshold_pose(pose: Pose2D, threshold: float) -> Pose2D:
    joints = tuple(
        joint
        if joint.observed and joint.score >= threshold
        else PoseJoint2D(
            joint.name,
            (0.0, 0.0),
            0.0,
            False,
            joint.provenance if joint.provenance == "missing" else "low_confidence",
        )
        for joint in pose.joints
    )
    return Pose2D(**{**pose.__dict__, "joints": joints})


def _interpolate_joints(track: list[Pose2D], max_gap: int) -> list[Pose2D]:
    if max_gap <= 0:
        return track
    frames = {pose.frame_idx: index for index, pose in enumerate(track)}
    mutable = [list(pose.joints) for pose in track]
    for joint_idx in range(len(track[0].joints) if track else 0):
        reliable = [
            i for i, pose in enumerate(track) if pose.joints[joint_idx].observed
        ]
        for left_i, right_i in pairwise(reliable):
            left, right = track[left_i], track[right_i]
            gap = right.frame_idx - left.frame_idx - 1
            if gap <= 0 or gap > max_gap:
                continue
            if any(
                frame not in frames
                for frame in range(left.frame_idx + 1, right.frame_idx)
            ):
                continue
            a, b = left.joints[joint_idx], right.joints[joint_idx]
            for frame in range(left.frame_idx + 1, right.frame_idx):
                ratio = (frame - left.frame_idx) / (right.frame_idx - left.frame_idx)
                xy = (
                    a.image_xy[0] + ratio * (b.image_xy[0] - a.image_xy[0]),
                    a.image_xy[1] + ratio * (b.image_xy[1] - a.image_xy[1]),
                )
                mutable[frames[frame]][joint_idx] = PoseJoint2D(
                    a.name, xy, 0.0, False, "interpolated"
                )
    return [
        Pose2D(**{**pose.__dict__, "joints": tuple(joints)})
        for pose, joints in zip(track, mutable)
    ]


def _continuous_runs(track: list[Pose2D], max_gap: int) -> list[list[Pose2D]]:
    runs: list[list[Pose2D]] = []
    for pose in track:
        if not any(
            joint.observed or joint.provenance == "interpolated"
            for joint in pose.joints
        ):
            continue
        if not runs or pose.frame_idx - runs[-1][-1].frame_idx > max_gap + 1:
            runs.append([pose])
        else:
            runs[-1].append(pose)
    return runs


def _derived(name: str, joints: dict[str, PoseJoint2D]) -> PoseJoint2D:
    if name == "spine":
        sources = (joints["neck"], joints["pelvis"])
    else:
        raise KeyError(name)
    if not all(item.observed or item.provenance == "interpolated" for item in sources):
        return PoseJoint2D(name, (0.0, 0.0), 0.0, False, "derived_missing")
    return PoseJoint2D(
        name,
        tuple(sum(item.image_xy[d] for item in sources) / len(sources) for d in (0, 1)),
        min(item.score for item in sources),
        all(item.observed for item in sources),
        "derived",
    )


def _to_h36m(pose: Pose2D, sequence_id: str) -> PreparedPose2D:
    source = {joint.name: joint for joint in pose.joints}
    mapped = []
    for name in H36M_17.joints:
        mapped.append(_derived(name, source) if name == "spine" else source[name])
    return PreparedPose2D(
        pose.clip_id,
        pose.segment_id,
        pose.frame_idx,
        pose.timestamp_seconds,
        pose.track_id,
        sequence_id,
        H36M_17.id,
        tuple(mapped),
        pose.quality_flags,
    )
