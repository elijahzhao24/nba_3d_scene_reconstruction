"""Robust court homography estimation and per-segment calibration state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import cv2
import numpy as np

from .configuration import (
    COURT_LANDMARK_POINTS_CM,
    CourtCalibrationConfiguration,
    CourtDetectorConfiguration,
)
from .schemas import (
    CalibrationSource,
    CourtCalibration,
    CourtDetection,
    Matrix3x3,
    Point,
)

PointArray: TypeAlias = np.ndarray


@dataclass(frozen=True)
class HomographyEstimate:
    """Result of one stateless RANSAC estimation attempt."""

    valid: bool
    image_to_court: Matrix3x3 | None
    court_to_image: Matrix3x3 | None
    keypoint_count: int
    inlier_count: int
    inlier_ratio: float
    court_coverage_ratio: float
    median_error_px: float | None
    inlier_landmark_ids: tuple[int, ...] = ()
    quality_flags: tuple[str, ...] = ()


@dataclass
class _SegmentCalibrationState:
    last_frame_idx: int = -1
    last_valid: CourtCalibration | None = None
    smoothed_image_points: dict[int, np.ndarray] = field(default_factory=dict)


def build_correspondences(
    detection: CourtDetection,
    *,
    landmark_points_cm: tuple[Point, ...] = COURT_LANDMARK_POINTS_CM,
    confidence_threshold: float,
    image_points_by_landmark: dict[int, Point] | None = None,
) -> tuple[PointArray, PointArray, tuple[int, ...]]:
    """Build aligned canonical/image arrays with one shared confidence mask."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if len(detection.keypoints) != len(landmark_points_cm):
        raise ValueError(
            "detection keypoints and canonical landmarks must have equal length"
        )

    court_points: list[Point] = []
    image_points: list[Point] = []
    landmark_ids: list[int] = []

    for dense_index, keypoint in enumerate(detection.keypoints):
        if keypoint is None or keypoint.confidence < confidence_threshold:
            continue
        if keypoint.landmark_id != dense_index:
            raise ValueError(
                "court detection keypoints must remain in dense landmark order"
            )

        image_xy = keypoint.image_xy
        if image_points_by_landmark is not None:
            image_xy = image_points_by_landmark.get(keypoint.landmark_id, image_xy)
        court_points.append(landmark_points_cm[keypoint.landmark_id])
        image_points.append(image_xy)
        landmark_ids.append(keypoint.landmark_id)

    return (
        np.asarray(court_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        tuple(landmark_ids),
    )


def project_points(points: PointArray, homography: PointArray) -> PointArray:
    """Apply a 3x3 homography to an N-by-2 point array."""
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(homography, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float64)
    transformed = cv2.perspectiveTransform(
        values.reshape(-1, 1, 2),
        matrix,
    )
    return transformed.reshape(-1, 2)


class CourtHomographyEstimator:
    """Estimate and validate a court-to-image homography from one detection."""

    def __init__(
        self,
        detector_configuration: CourtDetectorConfiguration | None = None,
        calibration_configuration: CourtCalibrationConfiguration | None = None,
    ) -> None:
        self.detector_configuration = (
            detector_configuration or CourtDetectorConfiguration.from_environment()
        )
        self.calibration_configuration = (
            calibration_configuration or CourtCalibrationConfiguration()
        )

    def estimate(
        self,
        detection: CourtDetection,
        *,
        image_points_by_landmark: dict[int, Point] | None = None,
    ) -> HomographyEstimate:
        """Fit court-to-image first so RANSAC residuals are measured in pixels."""
        court_points, image_points, landmark_ids = build_correspondences(
            detection,
            landmark_points_cm=self.detector_configuration.landmark_points_cm,
            confidence_threshold=self.detector_configuration.keypoint_confidence,
            image_points_by_landmark=image_points_by_landmark,
        )
        keypoint_count = len(landmark_ids)
        config = self.calibration_configuration

        if keypoint_count < config.minimum_correspondences:
            return _invalid_estimate(
                keypoint_count,
                ("insufficient_correspondences",),
            )
        if not _points_are_non_collinear(court_points) or not _points_are_non_collinear(
            image_points
        ):
            return _invalid_estimate(keypoint_count, ("collinear_correspondences",))

        try:
            court_to_image, raw_inlier_mask = cv2.findHomography(
                court_points,
                image_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=config.ransac_reprojection_threshold_px,
                maxIters=config.ransac_max_iterations,
                confidence=config.ransac_confidence,
            )
        except cv2.error:
            return _invalid_estimate(keypoint_count, ("estimation_failed",))

        if court_to_image is None or raw_inlier_mask is None:
            return _invalid_estimate(keypoint_count, ("estimation_failed",))

        court_to_image = _normalize_homography(court_to_image)
        if court_to_image is None:
            return _invalid_estimate(keypoint_count, ("invalid_homography",))

        inlier_mask = np.asarray(raw_inlier_mask, dtype=bool).reshape(-1)
        if len(inlier_mask) != keypoint_count:
            return _invalid_estimate(keypoint_count, ("invalid_inlier_mask",))
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / keypoint_count
        inlier_ids = tuple(
            landmark_id
            for landmark_id, is_inlier in zip(landmark_ids, inlier_mask, strict=True)
            if is_inlier
        )
        coverage_ratio = _court_coverage_ratio(
            court_points[inlier_mask],
            np.asarray(
                self.detector_configuration.landmark_points_cm,
                dtype=np.float64,
            ),
        )

        median_error_px: float | None = None
        if inlier_count:
            projected = project_points(court_points[inlier_mask], court_to_image)
            errors = np.linalg.norm(
                projected - image_points[inlier_mask],
                axis=1,
            )
            median_error_px = float(np.median(errors))

        quality_flags: list[str] = []
        blocking_flags: list[str] = []
        if inlier_count < keypoint_count:
            quality_flags.append("ransac_outliers")
        if inlier_count < config.minimum_inliers:
            blocking_flags.append("insufficient_inliers")
        if inlier_ratio < config.minimum_inlier_ratio:
            blocking_flags.append("low_inlier_ratio")
        if coverage_ratio < config.minimum_court_coverage_ratio:
            blocking_flags.append("insufficient_court_coverage")
        if (
            median_error_px is None
            or not np.isfinite(median_error_px)
            or median_error_px > config.maximum_median_reprojection_error_px
        ):
            blocking_flags.append("excessive_reprojection_error")

        image_to_court: np.ndarray | None = None
        if not blocking_flags:
            try:
                image_to_court = _normalize_homography(np.linalg.inv(court_to_image))
            except np.linalg.LinAlgError:
                image_to_court = None
            if image_to_court is None or not _has_valid_court_projection(
                court_to_image,
                self.detector_configuration.landmark_points_cm,
            ):
                blocking_flags.append("invalid_homography_geometry")

        all_flags = tuple(quality_flags + blocking_flags)
        if blocking_flags or image_to_court is None:
            return HomographyEstimate(
                valid=False,
                image_to_court=None,
                court_to_image=None,
                keypoint_count=keypoint_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                court_coverage_ratio=coverage_ratio,
                median_error_px=median_error_px,
                inlier_landmark_ids=inlier_ids,
                quality_flags=all_flags,
            )

        return HomographyEstimate(
            valid=True,
            image_to_court=_matrix_tuple(image_to_court),
            court_to_image=_matrix_tuple(court_to_image),
            keypoint_count=keypoint_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            court_coverage_ratio=coverage_ratio,
            median_error_px=median_error_px,
            inlier_landmark_ids=inlier_ids,
            quality_flags=all_flags,
        )


class CourtCalibrator:
    """Own stabilized calibration state independently for each video segment."""

    def __init__(
        self,
        estimator: CourtHomographyEstimator | None = None,
    ) -> None:
        self.estimator = estimator or CourtHomographyEstimator()
        self._states: dict[str, _SegmentCalibrationState] = {}

    def update(
        self,
        detection: CourtDetection | None,
        *,
        segment_id: str,
        frame_idx: int,
    ) -> CourtCalibration:
        """Process an attempted checkpoint detection or reuse recent valid state."""
        return self._advance(
            detection,
            segment_id=segment_id,
            frame_idx=frame_idx,
            missing_detection_flags=("court_detection_missing",),
        )

    def _advance(
        self,
        detection: CourtDetection | None,
        *,
        segment_id: str,
        frame_idx: int,
        missing_detection_flags: tuple[str, ...],
    ) -> CourtCalibration:
        if not segment_id:
            raise ValueError("segment_id must not be empty")
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        if detection is not None and detection.frame_idx != frame_idx:
            raise ValueError("detection frame_idx must match calibration frame_idx")

        state = self._states.setdefault(segment_id, _SegmentCalibrationState())
        if frame_idx <= state.last_frame_idx:
            raise ValueError(
                f"frame_idx must increase for segment {segment_id!r}; "
                f"last frame was {state.last_frame_idx}"
            )

        failure_flags = missing_detection_flags
        failed_estimate: HomographyEstimate | None = None
        if detection is not None:
            candidates = self._smoothed_candidates(state, detection)
            estimate = self.estimator.estimate(
                detection,
                image_points_by_landmark={
                    landmark_id: (float(point[0]), float(point[1]))
                    for landmark_id, point in candidates.items()
                },
            )
            failure_flags = estimate.quality_flags
            if estimate.valid:
                calibration = _detected_calibration(
                    segment_id,
                    frame_idx,
                    estimate,
                )
                state.last_valid = calibration
                state.smoothed_image_points.update(
                    {
                        landmark_id: candidates[landmark_id]
                        for landmark_id in estimate.inlier_landmark_ids
                    }
                )
                state.last_frame_idx = frame_idx
                return calibration
            failed_estimate = estimate

        calibration = self._fallback_calibration(
            state,
            segment_id=segment_id,
            frame_idx=frame_idx,
            failure_flags=failure_flags,
            failed_estimate=failed_estimate,
        )
        state.last_frame_idx = frame_idx
        return calibration

    def current(self, *, segment_id: str, frame_idx: int) -> CourtCalibration:
        """Return held/invalid state on a frame without court inference."""
        return self._advance(
            None,
            segment_id=segment_id,
            frame_idx=frame_idx,
            missing_detection_flags=(),
        )

    def reset(self, segment_id: str | None = None) -> None:
        """Forget temporal state after a cut, for one segment or all segments."""
        if segment_id is None:
            self._states.clear()
        else:
            self._states.pop(segment_id, None)

    def _smoothed_candidates(
        self,
        state: _SegmentCalibrationState,
        detection: CourtDetection,
    ) -> dict[int, np.ndarray]:
        alpha = self.estimator.calibration_configuration.landmark_smoothing_alpha
        threshold = self.estimator.detector_configuration.keypoint_confidence
        candidates: dict[int, np.ndarray] = {}
        for keypoint in detection.confident_keypoints(threshold):
            observed = np.asarray(keypoint.image_xy, dtype=np.float64)
            previous = state.smoothed_image_points.get(keypoint.landmark_id)
            candidates[keypoint.landmark_id] = (
                observed
                if previous is None
                else alpha * observed + (1.0 - alpha) * previous
            )
        return candidates

    def _fallback_calibration(
        self,
        state: _SegmentCalibrationState,
        *,
        segment_id: str,
        frame_idx: int,
        failure_flags: tuple[str, ...],
        failed_estimate: HomographyEstimate | None,
    ) -> CourtCalibration:
        last_valid = state.last_valid
        if last_valid is not None:
            assert last_valid.source_frame_idx is not None
            age_frames = frame_idx - last_valid.source_frame_idx
            maximum_age = (
                self.estimator.calibration_configuration.maximum_calibration_age_frames
            )
            if age_frames <= maximum_age:
                return CourtCalibration(
                    segment_id=segment_id,
                    frame_idx=frame_idx,
                    source_frame_idx=last_valid.source_frame_idx,
                    valid=True,
                    source=CalibrationSource.HELD,
                    image_to_court=last_valid.image_to_court,
                    court_to_image=last_valid.court_to_image,
                    keypoint_count=last_valid.keypoint_count,
                    inlier_count=last_valid.inlier_count,
                    inlier_ratio=last_valid.inlier_ratio,
                    court_coverage_ratio=last_valid.court_coverage_ratio,
                    median_error_px=last_valid.median_error_px,
                    age_frames=age_frames,
                    quality_flags=_merge_flags(last_valid.quality_flags, failure_flags),
                )

        invalid_flags = failure_flags
        if last_valid is not None:
            invalid_flags = _merge_flags(failure_flags, ("calibration_expired",))
        return CourtCalibration(
            segment_id=segment_id,
            frame_idx=frame_idx,
            source_frame_idx=None,
            valid=False,
            source=CalibrationSource.INVALID,
            image_to_court=None,
            court_to_image=None,
            keypoint_count=(
                failed_estimate.keypoint_count if failed_estimate is not None else 0
            ),
            inlier_count=(
                failed_estimate.inlier_count if failed_estimate is not None else 0
            ),
            inlier_ratio=(
                failed_estimate.inlier_ratio if failed_estimate is not None else 0.0
            ),
            court_coverage_ratio=(
                failed_estimate.court_coverage_ratio
                if failed_estimate is not None
                else 0.0
            ),
            median_error_px=(
                failed_estimate.median_error_px if failed_estimate is not None else None
            ),
            age_frames=None,
            quality_flags=invalid_flags,
        )


def _invalid_estimate(
    keypoint_count: int,
    flags: tuple[str, ...],
) -> HomographyEstimate:
    return HomographyEstimate(
        valid=False,
        image_to_court=None,
        court_to_image=None,
        keypoint_count=keypoint_count,
        inlier_count=0,
        inlier_ratio=0.0,
        court_coverage_ratio=0.0,
        median_error_px=None,
        quality_flags=flags,
    )


def _detected_calibration(
    segment_id: str,
    frame_idx: int,
    estimate: HomographyEstimate,
) -> CourtCalibration:
    return CourtCalibration(
        segment_id=segment_id,
        frame_idx=frame_idx,
        source_frame_idx=frame_idx,
        valid=True,
        source=CalibrationSource.DETECTED,
        image_to_court=estimate.image_to_court,
        court_to_image=estimate.court_to_image,
        keypoint_count=estimate.keypoint_count,
        inlier_count=estimate.inlier_count,
        inlier_ratio=estimate.inlier_ratio,
        court_coverage_ratio=estimate.court_coverage_ratio,
        median_error_px=estimate.median_error_px,
        age_frames=0,
        quality_flags=estimate.quality_flags,
    )


def _points_are_non_collinear(points: PointArray) -> bool:
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return False
    centered = points - np.mean(points, axis=0)
    return bool(np.linalg.matrix_rank(centered, tol=1e-8) >= 2)


def _normalize_homography(matrix: PointArray) -> PointArray | None:
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        return None
    scale = result[2, 2]
    if abs(scale) < 1e-12:
        scale = np.linalg.norm(result)
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        return None
    result = result / scale
    determinant = float(np.linalg.det(result))
    condition_number = float(np.linalg.cond(result))
    if (
        not np.isfinite(determinant)
        or abs(determinant) < 1e-12
        or not np.isfinite(condition_number)
        or condition_number > 1e12
    ):
        return None
    return result


def _court_coverage_ratio(
    inlier_court_points: PointArray,
    all_court_points: PointArray,
) -> float:
    if len(inlier_court_points) < 3:
        return 0.0
    full_hull = cv2.convexHull(all_court_points.astype(np.float32))
    inlier_hull = cv2.convexHull(inlier_court_points.astype(np.float32))
    full_area = float(cv2.contourArea(full_hull))
    if full_area <= 0.0:
        return 0.0
    return min(1.0, float(cv2.contourArea(inlier_hull)) / full_area)


def _has_valid_court_projection(
    court_to_image: PointArray,
    landmark_points_cm: tuple[Point, ...],
) -> bool:
    points = np.asarray(landmark_points_cm, dtype=np.float64)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    corners = np.asarray(
        [
            minimum,
            (maximum[0], minimum[1]),
            maximum,
            (minimum[0], maximum[1]),
        ],
        dtype=np.float64,
    )
    try:
        projected = project_points(corners, court_to_image)
    except cv2.error:
        return False
    if not np.all(np.isfinite(projected)):
        return False
    contour = projected.astype(np.float32).reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(contour) and cv2.contourArea(contour) > 1.0)


def _matrix_tuple(matrix: PointArray) -> Matrix3x3:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


def _merge_flags(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flag for group in groups for flag in group))
