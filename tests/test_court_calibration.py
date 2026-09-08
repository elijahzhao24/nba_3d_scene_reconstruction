from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from nba_3d_scene_reconstruction.court.calibration import (
    CourtCalibrator,
    CourtHomographyEstimator,
    build_correspondences,
    project_points,
)
from nba_3d_scene_reconstruction.court.configuration import (
    COURT_LANDMARK_LABELS,
    COURT_LANDMARK_POINTS_CM,
    CourtCalibrationConfiguration,
)
from nba_3d_scene_reconstruction.court.schemas import (
    CalibrationSource,
    CourtDetection,
    CourtKeypoint,
)

KNOWN_COURT_TO_IMAGE = np.asarray(
    [
        [0.42, 0.03, 140.0],
        [0.01, 0.31, 90.0],
        [0.00002, 0.00004, 1.0],
    ],
    dtype=np.float64,
)


def synthetic_detection(
    frame_idx: int = 0,
    *,
    confidence: float = 0.99,
    included_ids: tuple[int, ...] | None = None,
    outlier_id: int | None = None,
) -> CourtDetection:
    court_points = np.asarray(COURT_LANDMARK_POINTS_CM, dtype=np.float64)
    image_points = project_points(court_points, KNOWN_COURT_TO_IMAGE)
    included = set(
        included_ids if included_ids is not None else range(len(court_points))
    )
    keypoints: list[CourtKeypoint | None] = []
    for landmark_id, (image_x, image_y) in enumerate(image_points):
        if landmark_id not in included:
            keypoints.append(None)
            continue
        if landmark_id == outlier_id:
            image_x += 500.0
            image_y -= 400.0
        keypoints.append(
            CourtKeypoint(
                landmark_id=landmark_id,
                label=COURT_LANDMARK_LABELS[landmark_id],
                image_xy=(float(image_x), float(image_y)),
                confidence=confidence,
            )
        )
    return CourtDetection(
        frame_idx=frame_idx,
        confidence=0.95,
        bbox_xyxy=(0.0, 0.0, 1920.0, 1080.0),
        keypoints=tuple(keypoints),
        image_width=1920,
        image_height=1080,
    )


class CorrespondenceHelpersTest(unittest.TestCase):
    def test_builds_aligned_arrays_with_one_confidence_mask(self) -> None:
        detection = synthetic_detection(included_ids=(0, 1, 2, 27))
        low_confidence = list(detection.keypoints)
        assert low_confidence[1] is not None
        low_confidence[1] = CourtKeypoint(
            landmark_id=1,
            label="02",
            image_xy=low_confidence[1].image_xy,
            confidence=0.2,
        )
        detection = CourtDetection(
            frame_idx=0,
            confidence=detection.confidence,
            bbox_xyxy=detection.bbox_xyxy,
            keypoints=tuple(low_confidence),
        )

        court, image, landmark_ids = build_correspondences(
            detection,
            confidence_threshold=0.5,
        )

        self.assertEqual(landmark_ids, (0, 2, 27))
        np.testing.assert_array_equal(
            court,
            np.asarray([COURT_LANDMARK_POINTS_CM[index] for index in landmark_ids]),
        )
        self.assertEqual(image.shape, (3, 2))

    def test_projects_points_and_validates_shapes(self) -> None:
        points = np.asarray([[0.0, 0.0], [100.0, 200.0]])

        projected = project_points(points, KNOWN_COURT_TO_IMAGE)

        self.assertEqual(projected.shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "shape"):
            project_points(np.asarray([1.0, 2.0]), KNOWN_COURT_TO_IMAGE)


class CourtHomographyEstimatorTest(unittest.TestCase):
    def test_recovers_known_homography_and_inverse(self) -> None:
        estimate = CourtHomographyEstimator().estimate(synthetic_detection())

        self.assertTrue(estimate.valid)
        self.assertEqual(estimate.inlier_count, 33)
        self.assertEqual(estimate.inlier_ratio, 1.0)
        self.assertGreater(estimate.court_coverage_ratio, 0.99)
        self.assertIsNotNone(estimate.court_to_image)
        self.assertIsNotNone(estimate.image_to_court)
        assert estimate.court_to_image is not None
        assert estimate.image_to_court is not None
        np.testing.assert_allclose(
            np.asarray(estimate.court_to_image),
            KNOWN_COURT_TO_IMAGE,
            rtol=1e-5,
            atol=1e-5,
        )
        round_trip = project_points(
            project_points(
                np.asarray(COURT_LANDMARK_POINTS_CM),
                np.asarray(estimate.court_to_image),
            ),
            np.asarray(estimate.image_to_court),
        )
        np.testing.assert_allclose(
            round_trip,
            np.asarray(COURT_LANDMARK_POINTS_CM),
            atol=1e-5,
        )

    def test_ransac_rejects_an_outlier(self) -> None:
        estimate = CourtHomographyEstimator().estimate(
            synthetic_detection(outlier_id=16)
        )

        self.assertTrue(estimate.valid)
        self.assertEqual(estimate.inlier_count, 32)
        self.assertNotIn(16, estimate.inlier_landmark_ids)
        self.assertIn("ransac_outliers", estimate.quality_flags)

    def test_rejects_insufficient_and_collinear_correspondences(self) -> None:
        insufficient = CourtHomographyEstimator().estimate(
            synthetic_detection(included_ids=(0, 1, 2))
        )
        collinear = CourtHomographyEstimator().estimate(
            synthetic_detection(included_ids=(0, 1, 2, 3, 4, 5))
        )

        self.assertFalse(insufficient.valid)
        self.assertEqual(
            insufficient.quality_flags,
            ("insufficient_correspondences",),
        )
        self.assertFalse(collinear.valid)
        self.assertEqual(collinear.quality_flags, ("collinear_correspondences",))

    def test_rejects_landmarks_with_too_little_court_coverage(self) -> None:
        configuration = CourtCalibrationConfiguration(minimum_court_coverage_ratio=0.9)
        estimator = CourtHomographyEstimator(calibration_configuration=configuration)

        estimate = estimator.estimate(
            synthetic_detection(included_ids=(0, 5, 8, 12, 14, 17, 27))
        )

        self.assertFalse(estimate.valid)
        self.assertIn("insufficient_court_coverage", estimate.quality_flags)

    def test_rejects_low_inlier_count_and_ratio(self) -> None:
        inliers = np.zeros((33, 1), dtype=np.uint8)
        inliers[:3] = 1
        with patch(
            "nba_3d_scene_reconstruction.court.calibration.cv2.findHomography",
            return_value=(KNOWN_COURT_TO_IMAGE.copy(), inliers),
        ):
            estimate = CourtHomographyEstimator().estimate(synthetic_detection())

        self.assertFalse(estimate.valid)
        self.assertIn("insufficient_inliers", estimate.quality_flags)
        self.assertIn("low_inlier_ratio", estimate.quality_flags)

    def test_accepts_six_well_spread_inliers_out_of_twelve(self) -> None:
        included_ids = (0, 5, 9, 17, 27, 32, 1, 4, 12, 20, 24, 30)
        inliers = np.zeros((12, 1), dtype=np.uint8)
        inliers[:6] = 1
        with patch(
            "nba_3d_scene_reconstruction.court.calibration.cv2.findHomography",
            return_value=(KNOWN_COURT_TO_IMAGE.copy(), inliers),
        ):
            estimate = CourtHomographyEstimator().estimate(
                synthetic_detection(included_ids=included_ids)
            )

        self.assertTrue(estimate.valid)
        self.assertEqual(estimate.inlier_ratio, 0.5)
        self.assertIn("ransac_outliers", estimate.quality_flags)

    def test_rejects_singular_homography(self) -> None:
        with patch(
            "nba_3d_scene_reconstruction.court.calibration.cv2.findHomography",
            return_value=(np.zeros((3, 3)), np.ones((33, 1), dtype=np.uint8)),
        ):
            estimate = CourtHomographyEstimator().estimate(synthetic_detection())

        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.quality_flags, ("invalid_homography",))

    def test_rejects_excessive_reprojection_error(self) -> None:
        detection = synthetic_detection()
        shifted_keypoints = tuple(
            None
            if point is None
            else CourtKeypoint(
                landmark_id=point.landmark_id,
                label=point.label,
                image_xy=(point.image_xy[0] + 100.0, point.image_xy[1]),
                confidence=point.confidence,
            )
            for point in detection.keypoints
        )
        shifted = CourtDetection(
            frame_idx=0,
            confidence=detection.confidence,
            bbox_xyxy=detection.bbox_xyxy,
            keypoints=shifted_keypoints,
        )
        with patch(
            "nba_3d_scene_reconstruction.court.calibration.cv2.findHomography",
            return_value=(
                KNOWN_COURT_TO_IMAGE.copy(),
                np.ones((33, 1), dtype=np.uint8),
            ),
        ):
            estimate = CourtHomographyEstimator().estimate(shifted)

        self.assertFalse(estimate.valid)
        self.assertIn("excessive_reprojection_error", estimate.quality_flags)


class CourtCalibratorTest(unittest.TestCase):
    def setUp(self) -> None:
        configuration = CourtCalibrationConfiguration(
            maximum_calibration_age_frames=2,
            landmark_smoothing_alpha=1.0,
        )
        self.calibrator = CourtCalibrator(
            CourtHomographyEstimator(
                calibration_configuration=configuration,
            )
        )

    def test_detects_holds_expires_and_resets_calibration(self) -> None:
        detected = self.calibrator.update(
            synthetic_detection(frame_idx=0),
            segment_id="segment-1",
            frame_idx=0,
        )
        held = self.calibrator.current(segment_id="segment-1", frame_idx=1)
        held_after_failure = self.calibrator.update(
            synthetic_detection(frame_idx=2, included_ids=(0, 1, 2)),
            segment_id="segment-1",
            frame_idx=2,
        )
        expired = self.calibrator.current(segment_id="segment-1", frame_idx=3)

        self.assertEqual(detected.source, CalibrationSource.DETECTED)
        self.assertEqual(detected.source_frame_idx, 0)
        self.assertEqual(detected.age_frames, 0)
        self.assertEqual(held.source, CalibrationSource.HELD)
        self.assertEqual(held.age_frames, 1)
        self.assertEqual(held.image_to_court, detected.image_to_court)
        self.assertNotIn("court_detection_missing", held.quality_flags)
        self.assertEqual(held_after_failure.source, CalibrationSource.HELD)
        self.assertIn(
            "insufficient_correspondences",
            held_after_failure.quality_flags,
        )
        self.assertEqual(expired.source, CalibrationSource.INVALID)
        self.assertFalse(expired.valid)
        self.assertIn("calibration_expired", expired.quality_flags)

        self.calibrator.reset("segment-1")
        after_reset = self.calibrator.current(
            segment_id="segment-1",
            frame_idx=4,
        )
        self.assertEqual(after_reset.source, CalibrationSource.INVALID)
        self.assertNotIn("calibration_expired", after_reset.quality_flags)

    def test_keeps_segment_state_independent(self) -> None:
        self.calibrator.update(
            synthetic_detection(frame_idx=0),
            segment_id="segment-1",
            frame_idx=0,
        )

        other = self.calibrator.current(segment_id="segment-2", frame_idx=0)

        self.assertEqual(other.source, CalibrationSource.INVALID)
        self.assertEqual(other.quality_flags, ())

    def test_checkpoint_without_detection_is_distinct_from_unscheduled_frame(self) -> None:
        self.calibrator.update(
            synthetic_detection(frame_idx=0),
            segment_id="segment-1",
            frame_idx=0,
        )
        missing_checkpoint = self.calibrator.update(
            None,
            segment_id="segment-1",
            frame_idx=1,
        )

        self.assertIn("court_detection_missing", missing_checkpoint.quality_flags)

    def test_smooths_landmarks_before_the_next_fit(self) -> None:
        configuration = CourtCalibrationConfiguration(
            landmark_smoothing_alpha=0.5,
        )
        calibrator = CourtCalibrator(
            CourtHomographyEstimator(calibration_configuration=configuration)
        )
        first = calibrator.update(
            synthetic_detection(frame_idx=0),
            segment_id="segment-1",
            frame_idx=0,
        )
        moved = synthetic_detection(frame_idx=1)
        moved_keypoints = tuple(
            None
            if point is None
            else CourtKeypoint(
                landmark_id=point.landmark_id,
                label=point.label,
                image_xy=(point.image_xy[0] + 20.0, point.image_xy[1]),
                confidence=point.confidence,
            )
            for point in moved.keypoints
        )
        second = calibrator.update(
            CourtDetection(
                frame_idx=1,
                confidence=moved.confidence,
                bbox_xyxy=moved.bbox_xyxy,
                keypoints=moved_keypoints,
            ),
            segment_id="segment-1",
            frame_idx=1,
        )

        assert first.court_to_image is not None
        assert second.court_to_image is not None
        first_origin = project_points(
            np.asarray([[0.0, 0.0]]),
            np.asarray(first.court_to_image),
        )[0]
        second_origin = project_points(
            np.asarray([[0.0, 0.0]]),
            np.asarray(second.court_to_image),
        )[0]
        np.testing.assert_allclose(
            second_origin - first_origin,
            (10.0, 0.0),
            atol=1e-6,
        )

    def test_requires_matching_monotonic_frame_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            self.calibrator.update(
                synthetic_detection(frame_idx=1),
                segment_id="segment-1",
                frame_idx=0,
            )

        self.calibrator.current(segment_id="segment-1", frame_idx=0)
        with self.assertRaisesRegex(ValueError, "must increase"):
            self.calibrator.current(segment_id="segment-1", frame_idx=0)

    def test_invalid_attempt_preserves_its_quality_counts(self) -> None:
        invalid = self.calibrator.update(
            synthetic_detection(frame_idx=0, included_ids=(0, 1, 2)),
            segment_id="segment-1",
            frame_idx=0,
        )

        self.assertEqual(invalid.source, CalibrationSource.INVALID)
        self.assertEqual(invalid.keypoint_count, 3)


if __name__ == "__main__":
    unittest.main()
