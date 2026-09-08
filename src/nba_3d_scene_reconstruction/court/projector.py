"""Project player footpoints into canonical court centimeters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ..tracking.schemas import PlayerObservation
from .configuration import COURT_LANDMARK_POINTS_CM
from .schemas import CourtCalibration, PlayerCourtPosition, Point


class PlayerCourtProjector:
    """Preserve missing positions and calibration provenance without clamping.

    A 100 cm margin permits players and provisional box anchors just outside
    the playing surface.
    This is a raw floor-plane estimate; jumping/occlusion need later cleanup.
    """

    def __init__(
        self,
        *,
        court_margin_cm: float = 100.0,
        landmark_points_cm: tuple[Point, ...] = COURT_LANDMARK_POINTS_CM,
    ) -> None:
        if not math.isfinite(court_margin_cm) or court_margin_cm < 0:
            raise ValueError("court_margin_cm must be finite and non-negative")
        points = np.asarray(landmark_points_cm, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 4
                or not np.isfinite(points).all()):
            raise ValueError("landmark_points_cm must contain finite court points")
        self._minimum = points.min(axis=0) - court_margin_cm
        self._maximum = points.max(axis=0) + court_margin_cm

    def contains_image_point(
        self,
        point: Point,
        calibration: CourtCalibration,
    ) -> bool:
        """Return whether an image point projects within the court margin."""
        if not calibration.valid or calibration.image_to_court is None:
            return False
        court_xy, failure = self._project_point(point, calibration.image_to_court)
        return court_xy is not None and failure is None

    def project(
        self,
        observations: Iterable[PlayerObservation],
        calibration: CourtCalibration,
    ) -> tuple[PlayerCourtPosition, ...]:
        """Join only observations and calibrations from the same segment/frame."""
        positions = []
        for observation in observations:
            if (observation.segment_id != calibration.segment_id
                    or observation.frame_idx != calibration.frame_idx):
                raise ValueError("observation and calibration segment/frame must match")
            flags = list(observation.quality_flags + calibration.quality_flags)
            court_xy = None
            if not calibration.valid or calibration.image_to_court is None:
                flags.append("calibration_invalid")
            if not observation.visible:
                flags.append("player_missing")
            if observation.footpoint_xy is None:
                flags.append("footpoint_missing")
            elif calibration.valid and calibration.image_to_court is not None:
                court_xy, failure = self._project_point(
                    observation.footpoint_xy, calibration.image_to_court,
                )
                if failure:
                    flags.append(failure)
            positions.append(PlayerCourtPosition(
                segment_id=observation.segment_id,
                frame_idx=observation.frame_idx,
                timestamp_seconds=observation.timestamp_seconds,
                track_id=observation.track_id,
                footpoint_image_xy=observation.footpoint_xy,
                raw_court_xy=court_xy,
                calibration_source_frame_idx=calibration.source_frame_idx,
                calibration_age_frames=calibration.age_frames,
                quality_flags=tuple(dict.fromkeys(flags)),
            ))
        return tuple(positions)

    def _project_point(self, point: Point, homography) -> tuple[Point | None, str | None]:
        matrix = np.asarray(homography, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return None, "invalid_homography"
        if not np.isfinite(point).all():
            return None, "invalid_footpoint"
        scale = float(np.max(np.abs(matrix)))
        if scale == 0:
            return None, "invalid_homography"
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = (matrix / scale) @ np.asarray((*point, 1.0))
            if not np.isfinite(transformed).all() or abs(transformed[2]) < 1e-12:
                return None, "projection_at_infinity"
            xy = transformed[:2] / transformed[2]
        if not np.isfinite(xy).all():
            return None, "projection_nonfinite"
        if np.any(xy < self._minimum) or np.any(xy > self._maximum):
            return None, "outside_court"
        return (float(xy[0]), float(xy[1])), None
