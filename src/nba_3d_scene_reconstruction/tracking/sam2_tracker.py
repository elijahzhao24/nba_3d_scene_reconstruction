"""Minimal SAM 2 adapter for player mask tracking."""

from __future__ import annotations

import math
import os
from typing import Any

import cv2
import numpy as np

from .schemas import BoundingBox, SamMaskPrediction

SAM2_MODEL_ID = "facebook/sam2.1-hiera-small"
DEFAULT_MASK_COMPONENT_RELATIVE_DISTANCE = 0.03


def filter_mask_segments_by_distance(
    mask: np.ndarray,
    *,
    relative_distance: float = DEFAULT_MASK_COMPONENT_RELATIVE_DISTANCE,
) -> np.ndarray:
    """Remove mask islands far from the largest connected component.

    This mirrors the reference notebook's
    ``filter_segments_by_distance(..., relative_distance=0.03, mode="edge")``
    call without making Supervision a runtime dependency.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if not 0.0 <= relative_distance <= 1.0:
        raise ValueError("relative_distance must be between 0 and 1")
    if not mask.any():
        return mask.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 2:
        return mask.copy()

    main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    maximum_distance = relative_distance * math.hypot(*mask.shape)
    distance_from_main = cv2.distanceTransform(
        (labels != main_label).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    keep_labels = np.zeros(count, dtype=bool)
    keep_labels[main_label] = True
    for label in range(1, count):
        if label == main_label:
            continue
        component = labels == label
        if (
            component.any()
            and float(distance_from_main[component].min()) <= maximum_distance
        ):
            keep_labels[label] = True
    return keep_labels[labels]


def _load_predictor() -> Any:
    # Import lazily so unit tests do not load PyTorch, CUDA, or model weights.
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    return SAM2VideoPredictor.from_pretrained(SAM2_MODEL_ID)


class Sam2PlayerTracker:
    """Track player masks in one continuous video segment."""

    def __init__(
        self,
        *,
        predictor: Any | None = None,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
    ) -> None:
        # Passing a predictor is only needed by tests. Production uses SAM 2.
        self.predictor = predictor or _load_predictor()
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self._state: object | None = None
        self._track_ids: set[int] = set()

    @property
    def track_ids(self) -> frozenset[int]:
        """Return the player IDs currently known to SAM 2."""
        return frozenset(self._track_ids)

    def start_segment(self, frames_dir: str | os.PathLike[str]) -> None:
        """Load an ordered directory of video frames into SAM 2."""
        if self._state is not None:
            self.predictor.reset_state(self._state)

        # SAM's defaults retain the whole resized video and tracking history on
        # the GPU. Store them in host RAM so longer clips leave room for inference.
        self._state = self.predictor.init_state(
            os.fspath(frames_dir),
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
        )
        self._track_ids.clear()

    def remove_player(self, track_id: int) -> None:
        """Release a retired object's SAM history without recomputing old masks."""
        state = self._require_state()
        if track_id not in self._track_ids:
            return
        self.predictor.remove_object(state, obj_id=track_id, need_output=False)
        self._track_ids.remove(track_id)

    # Existing ID's added to a set has no effect.
    # so prompt_player works for both addition and corrections
    def prompt_player(
        self,
        frame_idx: int,
        track_id: int,
        bbox_xyxy: BoundingBox,
    ) -> None:
        """Add a new player or correct an existing player with an RF-DETR box."""
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        if track_id < 0:
            raise ValueError("track_id must be non-negative")

        x1, y1, x2, y2 = bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive width and height")

        self.predictor.add_new_points_or_box(
            self._require_state(),
            frame_idx=frame_idx,
            obj_id=track_id,
            box=np.asarray(bbox_xyxy, dtype=np.float32),
        )
        self._track_ids.add(track_id)

    def propagate_frame(self, frame_idx: int) -> tuple[SamMaskPrediction, ...]:
        """Ask SAM 2 for every tracked player's mask on one frame."""
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        if not self._track_ids:
            return ()

        outputs = self.predictor.propagate_in_video(
            self._require_state(),
            start_frame_idx=frame_idx,
            max_frame_num_to_track=0,
        )
        for output_frame_idx, object_ids, mask_logits in outputs:
            masks = mask_logits.detach().cpu().numpy()
            if masks.ndim != 4 or masks.shape[1] != 1:
                raise ValueError("SAM 2 masks must have shape [objects, 1, H, W]")
            if masks.shape[0] != len(object_ids):
                raise ValueError(
                    "SAM 2 returned a different number of masks and object IDs"
                )

            return tuple(
                SamMaskPrediction(
                    frame_idx=output_frame_idx,
                    track_id=int(track_id),
                    mask=filter_mask_segments_by_distance(masks[index, 0] > 0.0),
                )
                for index, track_id in enumerate(object_ids)
            )

        return ()

    def _require_state(self) -> object:
        if self._state is None:
            raise RuntimeError("start_segment must be called before tracking")
        return self._state
