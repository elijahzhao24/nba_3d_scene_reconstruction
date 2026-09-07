"""Render synchronized source and bird's-eye diagnostics without model calls."""

from __future__ import annotations

import math
import textwrap
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..tracking.debug import TRACK_COLORS, draw_tracking_overlay
from ..tracking.schemas import PlayerObservation, SamMaskPrediction
from .configuration import CourtDetectorConfiguration
from .schemas import CourtCalibration, CourtDetection, PlayerCourtPosition

if TYPE_CHECKING:
    from ..pipeline import SceneFrame


BACKGROUND = (25, 23, 20)
TEXT = (236, 235, 230)
MUTED = (166, 159, 147)
DETECTED = (65, 215, 255)  # BGR: yellow
REPROJECTED = (245, 220, 60)  # cyan
INVALID = (110, 130, 250)


def _text(canvas, value, xy, color=TEXT, scale=0.55, thickness=1):
    cv2.putText(canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _pixel(point, shape):
    """Skip nonfinite/offscreen markers before converting to OpenCV integers."""
    if point is None or not np.isfinite(point).all():
        return None
    x, y = point
    height, width = shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None
    return int(round(x)), int(round(y))


def draw_source_overlay(
    frame: np.ndarray,
    masks: tuple[SamMaskPrediction, ...],
    observations: tuple[PlayerObservation, ...],
    detection: CourtDetection | None,
    calibration: CourtCalibration,
    *,
    configuration: CourtDetectorConfiguration | None = None,
) -> np.ndarray:
    """Draw in source pixels; raw landmarks appear only on their actual frame."""
    config = configuration or CourtDetectorConfiguration()
    for record in (*masks, *observations):
        if record.frame_idx != calibration.frame_idx:
            raise ValueError("overlay records must match calibration frame")
    if any(o.segment_id != calibration.segment_id for o in observations):
        raise ValueError("overlay observations must match calibration segment")
    if detection is not None and detection.frame_idx != calibration.frame_idx:
        raise ValueError("court detection must match calibration frame")

    result = draw_tracking_overlay(frame, masks)
    scale = max(1.0, frame.shape[1] / 1280)
    size = max(5, round(6 * scale))
    thickness = max(1, round(2 * scale))
    projected = {}
    detected_ids = {
        point.landmark_id
        for point in (() if detection is None else detection.keypoints)
        if point is not None
    }
    if calibration.valid and calibration.court_to_image is not None:
        matrix = np.asarray(calibration.court_to_image, dtype=np.float64)
        if matrix.shape == (3, 3) and np.isfinite(matrix).all():
            for index, point in enumerate(config.landmark_points_cm):
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    homogeneous = matrix @ np.asarray((*point, 1.0))
                    if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-12:
                        continue
                    pixel = _pixel(homogeneous[:2] / homogeneous[2], frame.shape)
                if pixel is not None:
                    projected[index] = pixel
                    cv2.drawMarker(result, pixel, REPROJECTED, cv2.MARKER_TILTED_CROSS,
                                   size * 2, thickness, cv2.LINE_AA)
                    if index not in detected_ids:
                        _text(result, config.landmark_labels[index],
                              (pixel[0] + size, pixel[1] + size * 2),
                              REPROJECTED, 0.45 * scale, thickness)

    if detection is not None:
        for point in detection.keypoints:
            if point is None:
                continue
            pixel = _pixel(point.image_xy, frame.shape)
            if pixel is None:
                continue
            color = DETECTED if point.confidence >= config.keypoint_confidence else MUTED
            if point.landmark_id in projected:
                cv2.line(result, pixel, projected[point.landmark_id], MUTED,
                         max(1, thickness // 2), cv2.LINE_AA)
            cv2.circle(result, pixel, size, color, thickness, cv2.LINE_AA)
            _text(result, point.label, (pixel[0] + size, pixel[1] - size),
                  color, 0.45 * scale, thickness)

    for observation in observations:
        if not observation.visible:
            continue
        pixel = _pixel(observation.footpoint_xy, frame.shape)
        if pixel is None:
            continue
        color = TRACK_COLORS[observation.track_id % len(TRACK_COLORS)]
        cv2.circle(result, pixel, size + thickness, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(result, pixel, size, color, -1, cv2.LINE_AA)
        cv2.drawMarker(result, pixel, TEXT, cv2.MARKER_CROSS, size, 1, cv2.LINE_AA)
        _text(result, f"ID {observation.track_id}",
              (pixel[0] + size * 2, pixel[1] + size), color, 0.6 * scale, thickness)
    return result


class CourtDebugRenderer:
    """A source view beside a fixed-scale NBA court; no temporal dot caching."""

    PANEL_WIDTH = 800
    MIN_HEIGHT = 720
    COURT_TOP = 178
    MARGIN_CM = 100.0

    def __init__(self, configuration: CourtDetectorConfiguration | None = None):
        self.configuration = configuration or CourtDetectorConfiguration()
        points = np.asarray(self.configuration.landmark_points_cm)
        self.minimum = points.min(axis=0)
        self.maximum = points.max(axis=0)
        self.scale = (self.PANEL_WIDTH - 64) / (
            self.maximum[0] - self.minimum[0] + 2 * self.MARGIN_CM
        )

    def court_to_panel(self, point) -> tuple[int, int]:
        """Map court centimeters to panel pixels without changing aspect ratio."""
        xy = (np.asarray(point) - self.minimum + self.MARGIN_CM) * self.scale
        return round(float(xy[0]) + 32), round(float(xy[1]) + self.COURT_TOP)

    def _court(self, panel):
        line_color = (154, 182, 187)
        top_left = self.court_to_panel(self.minimum)
        bottom_right = self.court_to_panel(self.maximum)
        cv2.rectangle(panel, top_left, bottom_right, (57, 76, 82), -1)

        def path(points, closed=False):
            pixels = np.asarray([self.court_to_panel(p) for p in points], np.int32)
            cv2.polylines(panel, [pixels], closed, line_color, 2, cv2.LINE_AA)

        def arc(cx, cy, radius, start, end):
            angles = np.linspace(start, end, 100)
            path([(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles])

        # Court markings use the same centimeter template as the detector.
        path([(0, 0), (2865, 0), (2865, 1524), (0, 1524)], True)
        path([(1432, 0), (1432, 1524)])
        arc(1432, 762, 183, 0, 2 * math.pi)
        radius = math.hypot(424 - 160, 762 - 91)
        angle = math.atan2(762 - 91, 424 - 160)
        for right in (False, True):
            def mirrored(points):
                return [(2865 - x if right else x, y) for x, y in points]

            path(mirrored([(0, 518), (579, 518), (579, 1006), (0, 1006)]))
            path(mirrored([(0, 91), (424, 91)]))
            path(mirrored([(0, 1433), (424, 1433)]))
            path(mirrored([(120, 671), (120, 853)]))
            path(mirrored([(160 + radius * np.cos(a), 762 + radius * np.sin(a))
                           for a in np.linspace(-angle, angle, 100)]))
            arc(2865 - 579 if right else 579, 762, 183, 0, 2 * math.pi)
            hoop = self.court_to_panel((2705 if right else 160, 762))
            cv2.circle(panel, hoop, 5, DETECTED, 2, cv2.LINE_AA)
        _text(panel, "0, 0", (top_left[0], top_left[1] - 12), MUTED, 0.4)
        _text(panel, "2865 cm  /  X", (bottom_right[0] - 112, top_left[1] - 12), MUTED, 0.4)
        _text(panel, "1524 cm  /  Y", (top_left[0], bottom_right[1] + 22), MUTED, 0.4)

    def draw_birdseye(
        self,
        positions: tuple[PlayerCourtPosition, ...],
        calibration: CourtCalibration,
        *,
        fps: float,
        height: int = MIN_HEIGHT,
    ) -> np.ndarray:
        if height < self.MIN_HEIGHT:
            raise ValueError("bird's-eye panel height must be at least 720")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if any(p.frame_idx != calibration.frame_idx or p.segment_id != calibration.segment_id
               for p in positions):
            raise ValueError("positions must match calibration segment/frame")
        panel = np.full((height, self.PANEL_WIDTH, 3), BACKGROUND, np.uint8)
        _text(panel, "COURT / TOP DOWN", (32, 40), TEXT, 0.85, 2)
        _text(panel, f"FRAME {calibration.frame_idx:05d}   |   {calibration.frame_idx / fps:.3f} s",
              (32, 69), MUTED)
        status_color = INVALID if not calibration.valid else (
            DETECTED if calibration.source.value == "held" else (135, 219, 155)
        )
        _text(panel, calibration.source.value.upper(), (32, 109), status_color, 0.7, 2)
        source = calibration.source_frame_idx
        age = calibration.age_frames
        _text(panel, f"Calibration frame: {source if source is not None else '--'}"
              f"    Age: {age if age is not None else '--'} frames", (205, 108), MUTED)
        error = '--' if calibration.median_error_px is None else f'{calibration.median_error_px:.2f}'
        _text(panel, f"Inliers {calibration.inlier_count}/{calibration.keypoint_count}"
              f"   |   Fit error {error} px   |   Coverage {calibration.court_coverage_ratio:.0%}",
              (32, 143), MUTED, 0.5)
        self._court(panel)
        plotted = 0
        if calibration.valid:
            for position in positions:
                xy = position.raw_court_xy
                if xy is None or not np.isfinite(xy).all():
                    continue
                if (np.any(np.asarray(xy) < self.minimum - self.MARGIN_CM)
                        or np.any(np.asarray(xy) > self.maximum + self.MARGIN_CM)):
                    continue
                pixel = self.court_to_panel(xy)
                color = TRACK_COLORS[position.track_id % len(TRACK_COLORS)]
                cv2.circle(panel, pixel, 11, BACKGROUND, -1, cv2.LINE_AA)
                cv2.circle(panel, pixel, 8, color, -1, cv2.LINE_AA)
                _text(panel, str(position.track_id), (pixel[0] + 13, pixel[1] + 5), TEXT, 0.6, 2)
                plotted += 1
        if not calibration.valid or plotted == 0:
            message = "NO VALID CALIBRATION" if not calibration.valid else "NO PROJECTABLE PLAYERS"
            cv2.rectangle(panel, (185, 360), (615, 405), BACKGROUND, -1)
            _text(panel, message, (206, 390), status_color, 0.65, 2)
        _text(panel, f"{plotted} projected / {len(positions)} tracked   |   Raw positions, centimeters",
              (32, 622), TEXT, 0.52)
        flags = tuple(dict.fromkeys(calibration.quality_flags + tuple(
            flag for p in positions for flag in p.quality_flags
        )))
        reason = 'Flags: ' + (', '.join(flags) if flags else 'none')
        for index, line in enumerate(textwrap.wrap(reason, width=87)[:3]):
            _text(panel, line, (32, 650 + 20 * index), MUTED, 0.43)
        return panel

    def render(self, frame: np.ndarray, result: SceneFrame, *, fps: float) -> np.ndarray:
        """Return one even-sized composite at the original frame's timestamp."""
        source = draw_source_overlay(
            frame, result.masks, result.observations, result.court_detection,
            result.calibration, configuration=self.configuration,
        )
        factor = min(1.0, 1280 / frame.shape[1], 800 / frame.shape[0])
        width = max(2, round(frame.shape[1] * factor))
        source_height = max(2, round(frame.shape[0] * factor))
        source = cv2.resize(source, (width, source_height), interpolation=cv2.INTER_AREA)
        width += width % 2
        height = max(self.MIN_HEIGHT, source_height + 106)
        height += height % 2
        left = np.full((height, width, 3), BACKGROUND, np.uint8)
        _text(left, "SOURCE / PLAYER & COURT OVERLAY", (16, 29), TEXT, 0.6, 1)
        left[48:48 + source_height, :source.shape[1]] = source
        # Two rows remain readable even for a narrow source video.
        _text(left, "Yellow O: detected keypoint   Gray O: low confidence", (16, height - 35), DETECTED, 0.43)
        _text(left, "Cyan X: reprojected landmark   Colored +: player footpoint", (16, height - 15), REPROJECTED, 0.43)
        if result.court_detection is None:
            _text(left, "No raw keypoints for this frame", (16, 49 + source_height + 18), MUTED, 0.43)
        right = self.draw_birdseye(result.positions, result.calibration, fps=fps, height=height)
        return np.concatenate((left, right), axis=1)
