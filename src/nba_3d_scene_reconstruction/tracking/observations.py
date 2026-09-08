"""Convert original-resolution binary masks into per-frame player records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import cv2
import numpy as np

from .schemas import (
    BoundingBox,
    ObservationSource,
    PlayerObservation,
    Point,
    SamMaskPrediction,
)


def mask_footpoint(mask: np.ndarray) -> Point | None:
    """Return the robust footpoint of the largest connected mask region."""
    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype != np.bool_:
        raise ValueError("player masks must be 2D boolean arrays")
    if not value.any():
        return None
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        value.astype(np.uint8),
        connectivity=8,
    )
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    y, x = np.nonzero(labels == component)
    bottom = int(y.max())
    band_height = max(1, math.ceil((bottom - int(y.min()) + 1) * 0.05))
    feet_x = x[y > bottom - band_height]
    return (float(np.median(feet_x)), float(bottom))


def build_player_observations(
    masks: Iterable[SamMaskPrediction],
    *,
    segment_id: str,
    frame_idx: int,
    fps: float,
    track_ids: Iterable[int],
    detection_confidences: Mapping[int, float] | None = None,
    fallback_bboxes: Mapping[int, BoundingBox | None] | None = None,
) -> tuple[PlayerObservation, ...]:
    """Emit one record per live track, including tracks without a current mask.

    Keep the largest connected component to discard detached mask noise. The
    footpoint is the median x of the lowest 5% of its height, at its bottom y.
    Boxes use exclusive right/bottom bounds; points use pixel coordinates.
    If a live track has no usable mask, its most recent detector box supplies a
    provisional bottom-center footpoint. The record remains marked missing so
    downstream consumers can distinguish that fallback from current mask data.
    """
    if not segment_id or frame_idx < 0:
        raise ValueError(
            "segment_id must not be empty and frame_idx must be non-negative"
        )
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    by_track: dict[int, SamMaskPrediction] = {}
    for prediction in masks:
        if prediction.frame_idx != frame_idx:
            raise ValueError("mask frame_idx must match observation frame_idx")
        if prediction.track_id in by_track:
            raise ValueError("duplicate mask track_id")
        by_track[prediction.track_id] = prediction

    observations = []
    for track_id in sorted(set(track_ids)):
        prediction = by_track.get(track_id)
        geometry = {}
        flags: list[str] = []
        visible = False
        if prediction is None:
            flags.append("mask_missing")
        else:
            mask = np.asarray(prediction.mask)
            if mask.ndim != 2 or mask.dtype != np.bool_:
                raise ValueError("player masks must be 2D boolean arrays")
            if not mask.any():
                flags.append("empty_mask")
            else:
                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8),
                    connectivity=8,
                )
                component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                y, x = np.nonzero(labels == component)
                if count > 2:
                    flags.append("disconnected_mask_regions_removed")
                bottom = int(y.max())
                band_height = max(1, math.ceil((bottom - int(y.min()) + 1) * 0.05))
                feet_x = x[y > bottom - band_height]
                if bottom == mask.shape[0] - 1:
                    flags.append("mask_touches_image_bottom")
                geometry = {
                    "bbox_xyxy": (
                        float(x.min()),
                        float(y.min()),
                        float(x.max() + 1),
                        float(bottom + 1),
                    ),
                    "centroid_xy": (float(x.mean()), float(y.mean())),
                    "footpoint_xy": (float(np.median(feet_x)), float(bottom)),
                    "mask_area": len(x),
                }
                visible = True
        if not visible:
            fallback_bbox = (fallback_bboxes or {}).get(track_id)
            if fallback_bbox is not None:
                x1, y1, x2, y2 = fallback_bbox
                if x2 > x1 and y2 > y1 and np.isfinite(fallback_bbox).all():
                    geometry = {
                        "bbox_xyxy": fallback_bbox,
                        "centroid_xy": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        "footpoint_xy": ((x1 + x2) / 2.0, y2),
                        "mask_area": 0,
                    }
                    flags.append("bbox_footpoint_fallback")
        observations.append(
            PlayerObservation(
                segment_id=segment_id,
                frame_idx=frame_idx,
                timestamp_seconds=frame_idx / fps,
                track_id=track_id,
                visible=visible,
                source=(
                    ObservationSource.SAM2_PROPAGATION
                    if visible
                    else ObservationSource.MISSING
                ),
                detection_confidence=(detection_confidences or {}).get(track_id),
                quality_flags=tuple(flags),
                **geometry,
            )
        )
    return tuple(observations)
