from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from nba_3d_scene_reconstruction.court.configuration import (
    COURT_LANDMARK_LABELS,
    COURT_LANDMARK_POINTS_CM,
    CourtCalibrationConfiguration,
    CourtDetectorConfiguration,
)


class CourtDetectorConfigurationTest(unittest.TestCase):
    def test_default_skeleton_has_expected_33_landmarks(self) -> None:
        configuration = CourtDetectorConfiguration()

        self.assertEqual(configuration.model_id, "basketball-court-detection-2/22")
        self.assertEqual(len(COURT_LANDMARK_LABELS), 33)
        self.assertEqual(len(COURT_LANDMARK_POINTS_CM), 33)
        self.assertEqual(COURT_LANDMARK_LABELS[:3], ("01", "02", "04"))
        self.assertEqual(COURT_LANDMARK_LABELS[-1], "41")
        self.assertEqual(COURT_LANDMARK_POINTS_CM[0], (0.0, 0.0))
        self.assertEqual(COURT_LANDMARK_POINTS_CM[16], (1432.0, 762.0))
        self.assertEqual(COURT_LANDMARK_POINTS_CM[-1], (2865.0, 1524.0))

    def test_reads_non_secret_configuration_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "COURT_MODEL_ID": "workspace/court/1",
                "COURT_API_URL": "https://example.test",
                "COURT_DETECTION_CONFIDENCE": "0.4",
                "COURT_KEYPOINT_CONFIDENCE": "0.6",
            },
            clear=True,
        ):
            configuration = CourtDetectorConfiguration.from_environment()

        self.assertEqual(configuration.model_id, "workspace/court/1")
        self.assertEqual(configuration.api_url, "https://example.test")
        self.assertEqual(configuration.detection_confidence, 0.4)
        self.assertEqual(configuration.keypoint_confidence, 0.6)

    def test_rejects_invalid_thresholds_and_skeletons(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            CourtDetectorConfiguration(detection_confidence=1.1)

        with self.assertRaisesRegex(ValueError, "equal length"):
            CourtDetectorConfiguration(
                landmark_labels=("01",),
                landmark_points_cm=((0.0, 0.0), (1.0, 1.0)),
            )

        with self.assertRaisesRegex(ValueError, "unique"):
            CourtDetectorConfiguration(
                landmark_labels=("01", "01"),
                landmark_points_cm=((0.0, 0.0), (1.0, 1.0)),
            )


class CourtCalibrationConfigurationTest(unittest.TestCase):
    def test_has_documented_production_defaults(self) -> None:
        configuration = CourtCalibrationConfiguration()

        self.assertEqual(configuration.ransac_reprojection_threshold_px, 8.0)
        self.assertEqual(configuration.minimum_correspondences, 4)
        self.assertEqual(configuration.minimum_inliers, 5)
        self.assertEqual(configuration.minimum_inlier_ratio, 0.45)
        self.assertEqual(configuration.minimum_court_coverage_ratio, 0.03)
        self.assertEqual(configuration.maximum_calibration_age_frames, 15)
        self.assertEqual(configuration.landmark_smoothing_alpha, 0.25)

    def test_rejects_invalid_numerical_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            CourtCalibrationConfiguration(ransac_reprojection_threshold_px=0.0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            CourtCalibrationConfiguration(
                maximum_median_reprojection_error_px=float("nan")
            )
        with self.assertRaisesRegex(ValueError, "at least 4"):
            CourtCalibrationConfiguration(minimum_correspondences=3)
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            CourtCalibrationConfiguration(minimum_inlier_ratio=0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CourtCalibrationConfiguration(maximum_calibration_age_frames=-1)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            CourtCalibrationConfiguration(ransac_confidence=1.0)


if __name__ == "__main__":
    unittest.main()
