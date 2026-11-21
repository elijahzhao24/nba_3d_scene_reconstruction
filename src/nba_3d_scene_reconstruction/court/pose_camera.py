"""Estimate a segment's source camera direction from saved court homographies."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .configuration import COURT_LANDMARK_POINTS_CM
from .schemas import Matrix3x3

_LENGTH_CM = max(x for x, _ in COURT_LANDMARK_POINTS_CM)
_WIDTH_CM = max(z for _, z in COURT_LANDMARK_POINTS_CM)
_COURT_CORNERS = np.array(
    [(0, 0, 1), (_LENGTH_CM, 0, 1), (0, _WIDTH_CM, 1), (_LENGTH_CM, _WIDTH_CM, 1)]
).T


@dataclass(frozen=True)
class PoseCamera:
    """Angles of optical forward, measured from world −Z and down from level."""

    yaw_deg: float
    pitch_deg: float


def _camera_forward(
    homography: Matrix3x3, image_width: float, image_height: float
) -> np.ndarray | None:
    try:
        matrix = np.asarray(homography, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    magnitude = np.abs(matrix).max()
    if not magnitude:
        return None
    matrix = matrix / magnitude

    depths = matrix[2] @ _COURT_CORNERS
    if depths.max() < 0:
        matrix *= -1
    elif depths.min() <= 0:
        return None

    # H = K [r_x, r_z, t] on the X/Z court plane. For centered, square pixels,
    # the corrected floor columns are orthogonal and have equal length.
    center = np.array([[image_width / 2], [image_height / 2]])
    matrix[:2] -= center * matrix[2]
    x_depth, z_depth = matrix[2, :2]
    x_img, z_img = matrix[:2, :2].T
    denominator = x_depth * z_depth
    if abs(denominator) > 1e-6 * max(x_depth**2, z_depth**2):
        focal_squared = -float(x_img @ z_img) / denominator
    else:
        # One vanishing point can be at infinity in a front-on sideline view.
        difference = x_depth**2 - z_depth**2
        if not difference:
            return None
        focal_squared = -float(x_img @ x_img - z_img @ z_img) / difference

    if not (0.2 * image_width) ** 2 <= focal_squared <= (5 * image_width) ** 2:
        return None
    matrix[:2] /= math.sqrt(focal_squared)
    x_axis, z_axis = matrix[:, :2].T
    x_length, z_length = np.linalg.norm(matrix[:, :2], axis=0)
    if min(x_length, z_length) <= 0.75 * max(x_length, z_length):
        return None
    x_axis = x_axis / x_length
    z_axis = z_axis / z_length
    overlap = x_axis @ z_axis
    if abs(overlap) > 0.12:
        return None
    z_axis = (z_axis - overlap * x_axis) / math.sqrt(1 - overlap**2)
    y_axis = np.cross(z_axis, x_axis)
    # The camera center is −Rᵀt. Only its height sign is needed here.
    if y_axis @ matrix[:, 2] >= 0:
        return None
    return np.array([x_axis[2], y_axis[2], z_axis[2]])


def estimate_pose_camera(
    homographies: Iterable[Matrix3x3], image_width: float, image_height: float
) -> PoseCamera | None:
    if not all(math.isfinite(size) and size > 0 for size in (image_width, image_height)):
        return None

    directions = [
        forward for matrix in homographies
        if (forward := _camera_forward(matrix, image_width, image_height)) is not None
    ]
    if not directions:
        return None
    values = np.stack(directions)
    center = np.median(values, axis=0)
    length = np.linalg.norm(center)
    if length < 0.5:
        return None
    # Use a group of consistent detected frames. Discard isolated plausible
    # fits from noisy landmarks without averaging views across a camera cut.
    angles = np.degrees(np.arccos(np.clip(values @ (center / length), -1, 1)))
    inliers = values[angles <= np.clip(3 * np.median(angles), 5, 20)]
    if len(inliers) * 2 <= len(values):
        return None
    x, _, z = np.median(inliers, axis=0)
    if math.hypot(x, z) < 0.05:
        return None
    horizontal = np.hypot(inliers[:, 0], inliers[:, 2])
    pitch = np.median(np.arctan2(-inliers[:, 1], horizontal))
    return PoseCamera(
        yaw_deg=math.degrees(math.atan2(x, -z)),
        pitch_deg=math.degrees(pitch),
    )
