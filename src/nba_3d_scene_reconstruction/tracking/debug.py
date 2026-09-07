"""Player mask rendering shared by tracking and court diagnostics."""

from __future__ import annotations

import cv2
import numpy as np

from .schemas import SamMaskPrediction


TRACK_COLORS = (
    (66, 135, 245),
    (80, 200, 120),
    (235, 90, 90),
    (80, 210, 230),
    (210, 110, 220),
    (230, 170, 70),
)


def draw_tracking_overlay(
    frame: np.ndarray,
    masks: tuple[SamMaskPrediction, ...],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Draw colored masks, outlines, and track IDs on one video frame."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    result = frame.copy()
    expected_shape = frame.shape[:2]

    for prediction in masks:
        mask = np.asarray(prediction.mask, dtype=np.bool_)
        if mask.shape != expected_shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match frame shape {expected_shape}"
            )
        if not mask.any():
            continue

        color = TRACK_COLORS[prediction.track_id % len(TRACK_COLORS)]
        color_layer = result.copy()
        color_layer[mask] = color
        result = cv2.addWeighted(color_layer, alpha, result, 1.0 - alpha, 0)

        mask_image = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(result, contours, -1, color, 2)

        y_coordinates, x_coordinates = np.nonzero(mask)
        label_position = (
            int(x_coordinates.min()),
            max(18, int(y_coordinates.min()) - 6),
        )
        cv2.putText(
            result,
            f"ID {prediction.track_id}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return result


