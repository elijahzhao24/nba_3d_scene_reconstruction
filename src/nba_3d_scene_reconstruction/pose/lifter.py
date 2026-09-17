"""Temporal 2D-to-3D lifting interface and MotionBERT adapter."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .schemas import Pose3D, PoseJoint3D, PreparedPose2D
from .skeletons import H36M_17


class PoseLifter(Protocol):
    def lift(self, sequence: Sequence[PreparedPose2D]) -> tuple[Pose3D, ...]: ...


def temporal_windows(
    length: int, *, window_size: int = 243, stride: int = 81
) -> tuple[range, ...]:
    if length <= 0:
        return ()
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if length <= window_size:
        return (range(length),)
    starts = list(range(0, length - window_size + 1, stride))
    tail = length - window_size
    if starts[-1] != tail:
        starts.append(tail)
    return tuple(range(start, start + window_size) for start in starts)


def normalize_sequence(values: np.ndarray) -> np.ndarray:
    """Apply MotionBERT's deterministic crop_scale(..., [1, 1]) once."""
    result = values.astype(np.float32, copy=True)
    valid = values[..., 2] != 0
    coordinates = values[..., :2][valid]
    if len(coordinates) < 4:
        result[...] = 0
        return result
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    # Reference implementation uses max(x range, y range), not the combined range.
    scale = float(max(maximum[0] - minimum[0], maximum[1] - minimum[1]))
    if scale == 0:
        result[...] = 0
        return result
    origin = (minimum + maximum - scale) / 2.0
    result[..., :2] = ((values[..., :2] - origin) / scale - 0.5) * 2.0
    result[..., :2] = np.clip(result[..., :2], -1.0, 1.0)
    return result


def make_root_relative(coordinates: np.ndarray) -> np.ndarray:
    """Return a copy with each frame's pelvis (joint zero) at the origin."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(H36M_17.joints), 3):
        raise ValueError("coordinates must have shape [frames, 17, 3]")
    return values - values[:, :1, :]


class MotionBertPoseLifter:
    """Use a checked-out MotionBERT release and its matching checkpoint."""

    def __init__(
        self, repository: str, config: str, checkpoint: str, *, device: str = "cuda"
    ) -> None:
        repository_path = str(Path(repository).resolve())
        if repository_path not in sys.path:
            sys.path.insert(0, repository_path)
        try:
            import torch
            from lib.utils.learning import load_backbone
            from lib.utils.tools import get_config
            from torch import nn
        except ImportError as error:
            raise RuntimeError(
                "install MotionBERT dependencies in the pose-3D environment"
            ) from error
        self.torch = torch
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        arguments = get_config(config)
        model = load_backbone(arguments)
        if self.device.type == "cuda":
            model = nn.DataParallel(model).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        model.load_state_dict(state["model_pos"], strict=True)
        self.model = model.eval()
        self.flip = bool(getattr(arguments, "flip", True))

    def lift(self, sequence: Sequence[PreparedPose2D]) -> tuple[Pose3D, ...]:
        if not sequence:
            return ()
        sequence = tuple(sequence)
        values = np.asarray(
            [
                [[*joint.image_xy, joint.score] for joint in pose.joints]
                for pose in sequence
            ],
            dtype=np.float32,
        )
        normalized = normalize_sequence(values)
        total = np.zeros((len(sequence), len(H36M_17.joints), 3), dtype=np.float64)
        counts = np.zeros(len(sequence), dtype=np.int32)
        with self.torch.no_grad():
            for indices in temporal_windows(len(sequence)):
                batch = (
                    self.torch.from_numpy(normalized[list(indices)])
                    .unsqueeze(0)
                    .to(self.device)
                )
                prediction = self.model(batch)
                if self.flip:
                    flipped = batch.clone()
                    flipped[..., 0] *= -1
                    left, right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
                    flipped[..., left + right, :] = flipped[..., right + left, :]
                    mirrored = self.model(flipped)
                    mirrored[..., 0] *= -1
                    mirrored[..., left + right, :] = mirrored[..., right + left, :]
                    prediction = (prediction + mirrored) / 2.0
                prediction = prediction.squeeze(0).detach().cpu().numpy()
                total[list(indices)] += prediction
                counts[list(indices)] += 1
        total /= counts[:, None, None]
        total = make_root_relative(total)
        return tuple(
            _pose3d(pose, coordinates) for pose, coordinates in zip(sequence, total)
        )


def _pose3d(pose: PreparedPose2D, coordinates: np.ndarray) -> Pose3D:
    joints = tuple(
        PoseJoint3D(
            name, tuple(float(value) for value in xyz), source.score, source.observed
        )
        for name, xyz, source in zip(H36M_17.joints, coordinates, pose.joints)
    )
    return Pose3D(
        pose.clip_id,
        pose.segment_id,
        pose.frame_idx,
        pose.timestamp_seconds,
        pose.track_id,
        pose.sequence_id,
        H36M_17.id,
        joints,
        quality_flags=pose.quality_flags,
    )
