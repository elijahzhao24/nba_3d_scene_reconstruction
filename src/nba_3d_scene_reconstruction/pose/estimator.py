"""2D estimator interface and MMPose RTMPose adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from ..tracking.schemas import PlayerObservation
from .schemas import Pose2D, PoseJoint2D
from .skeletons import HALPE_26


class Pose2DEstimator(Protocol):
    def estimate(
        self,
        frame: np.ndarray,
        observations: Iterable[PlayerObservation],
        *,
        clip_id: str,
    ) -> tuple[Pose2D, ...]: ...


class MMPosePose2DEstimator:
    """Load one top-down MMPose model and use finalized tracking boxes."""

    def __init__(self, config: str, checkpoint: str, *, device: str = "cuda:0") -> None:
        try:
            from mmpose.apis import init_model
            from mmpose.utils import register_all_modules
        except ImportError as error:
            raise RuntimeError("install MMPose in the pose-2D environment") from error
        register_all_modules()
        self.model = init_model(config, checkpoint, device=device)

    def estimate(
        self,
        frame: np.ndarray,
        observations: Iterable[PlayerObservation],
        *,
        clip_id: str,
    ) -> tuple[Pose2D, ...]:
        from mmpose.apis import inference_topdown

        observations = tuple(observations)
        usable = [item for item in observations if item.visible and item.bbox_xyxy]
        predictions: Sequence[object] = ()
        if usable:
            boxes = np.asarray([item.bbox_xyxy for item in usable], dtype=np.float32)
            predictions = inference_topdown(self.model, frame, bboxes=boxes)
            if len(predictions) != len(usable):
                raise RuntimeError("MMPose did not return one pose per tracked box")
        by_key = {id(item): prediction for item, prediction in zip(usable, predictions)}
        results = []
        for observation in observations:
            prediction = by_key.get(id(observation))
            if prediction is None:
                results.append(_missing_pose(observation, clip_id))
                continue
            instances = prediction.pred_instances
            points = np.asarray(instances.keypoints)[0]
            scores = np.asarray(instances.keypoint_scores)[0]
            if len(points) != len(HALPE_26.joints):
                raise ValueError(f"expected 26 HALPE joints, received {len(points)}")
            joints = tuple(
                PoseJoint2D(name, (float(xy[0]), float(xy[1])), float(score), True)
                for name, xy, score in zip(HALPE_26.joints, points, scores)
            )
            results.append(
                Pose2D(
                    clip_id,
                    observation.segment_id,
                    observation.frame_idx,
                    observation.timestamp_seconds,
                    observation.track_id,
                    HALPE_26.id,
                    joints,
                )
            )
        return tuple(results)


def _missing_pose(observation: PlayerObservation, clip_id: str) -> Pose2D:
    joints = tuple(
        PoseJoint2D(name, (0.0, 0.0), 0.0, False, "missing") for name in HALPE_26.joints
    )
    flags = tuple(dict.fromkeys((*observation.quality_flags, "pose_input_missing")))
    return Pose2D(
        clip_id,
        observation.segment_id,
        observation.frame_idx,
        observation.timestamp_seconds,
        observation.track_id,
        HALPE_26.id,
        joints,
        False,
        flags,
    )
