from __future__ import annotations

import os
import unittest
from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

from nba_3d_scene_reconstruction.court.configuration import (
    CourtDetectorConfiguration,
)
from nba_3d_scene_reconstruction.court.detector import (
    CourtSchemaMismatchError,
    RoboflowCourtDetector,
)


class FakeClient:
    def __init__(self, result: Mapping[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[object, str]] = []

    def infer(
        self,
        inference_input: object,
        *,
        model_id: str,
    ) -> Mapping[str, Any]:
        self.calls.append((inference_input, model_id))
        return self.result


def court_prediction(
    *,
    confidence: float = 0.9,
    keypoints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "x": 50,
        "y": 40,
        "width": 80,
        "height": 60,
        "confidence": confidence,
        "class": "court",
        "class_id": 0,
        "keypoints": keypoints or [],
    }


class RoboflowCourtDetectorTest(unittest.TestCase):
    def test_runs_model_and_densifies_shuffled_sparse_keypoints(self) -> None:
        result = {
            "predictions": [
                {
                    **court_prediction(confidence=0.4),
                    "class": "not-court",
                },
                court_prediction(
                    keypoints=[
                        {
                            "x": 30,
                            "y": 40,
                            "confidence": 0.4,
                            "class_id": 2,
                            "class": "04",
                        },
                        {
                            "x": 10,
                            "y": 20,
                            "confidence": 0.9,
                            "class_id": 0,
                            "class": "01",
                        },
                    ]
                ),
            ],
            "image": {"width": 100, "height": 80},
        }
        client = FakeClient(result)
        detector = RoboflowCourtDetector(client=client)

        detection = detector.detect("frame.jpg", frame_idx=7)

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(
            client.calls,
            [("frame.jpg", "basketball-court-detection-2/22")],
        )
        self.assertEqual(detection.frame_idx, 7)
        self.assertEqual(detection.bbox_xyxy, (10.0, 10.0, 90.0, 70.0))
        self.assertEqual(detection.image_width, 100)
        self.assertEqual(detection.image_height, 80)
        self.assertEqual(len(detection.keypoints), 33)
        self.assertEqual(detection.keypoints[0].label, "01")
        self.assertIsNone(detection.keypoints[1])
        self.assertEqual(detection.keypoints[2].label, "04")
        self.assertEqual(
            [point.label for point in detection.confident_keypoints(0.5)],
            ["01"],
        )

    def test_selects_highest_confidence_court_above_threshold(self) -> None:
        client = FakeClient(
            {
                "predictions": [
                    court_prediction(confidence=0.2),
                    court_prediction(confidence=0.6),
                    court_prediction(confidence=0.8),
                ]
            }
        )
        detector = RoboflowCourtDetector(
            CourtDetectorConfiguration(detection_confidence=0.5),
            client=client,
        )

        detection = detector.detect("frame.jpg", frame_idx=0)

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(detection.confidence, 0.8)

    def test_returns_none_without_usable_court_prediction(self) -> None:
        detector = RoboflowCourtDetector(
            CourtDetectorConfiguration(detection_confidence=0.5),
            client=FakeClient(
                {
                    "predictions": [
                        court_prediction(confidence=0.2),
                        {
                            **court_prediction(confidence=0.9),
                            "class": "player",
                        },
                    ]
                }
            ),
        )

        self.assertIsNone(detector.detect("frame.jpg", frame_idx=0))

    def test_rejects_landmark_label_mismatch_and_duplicate_id(self) -> None:
        mismatch = RoboflowCourtDetector(
            client=FakeClient(
                {
                    "predictions": [
                        court_prediction(
                            keypoints=[
                                {
                                    "x": 1,
                                    "y": 2,
                                    "confidence": 0.9,
                                    "class_id": 2,
                                    "class": "03",
                                }
                            ]
                        )
                    ]
                }
            )
        )
        with self.assertRaisesRegex(CourtSchemaMismatchError, "expected label 04"):
            mismatch.detect("frame.jpg", frame_idx=0)

        duplicate = RoboflowCourtDetector(
            client=FakeClient(
                {
                    "predictions": [
                        court_prediction(
                            keypoints=[
                                {
                                    "x": 1,
                                    "y": 2,
                                    "confidence": 0.9,
                                    "class_id": 0,
                                    "class": "01",
                                },
                                {
                                    "x": 3,
                                    "y": 4,
                                    "confidence": 0.8,
                                    "class_id": 0,
                                    "class": "01",
                                },
                            ]
                        )
                    ]
                }
            )
        )
        with self.assertRaisesRegex(CourtSchemaMismatchError, "duplicate"):
            duplicate.detect("frame.jpg", frame_idx=0)

    def test_rejects_parent_class_id_mismatch(self) -> None:
        detector = RoboflowCourtDetector(
            client=FakeClient(
                {
                    "predictions": [
                        {
                            **court_prediction(),
                            "class_id": 7,
                        }
                    ]
                }
            )
        )

        with self.assertRaisesRegex(CourtSchemaMismatchError, "expected id 0"):
            detector.detect("frame.jpg", frame_idx=0)

    def test_requires_api_key_only_when_constructing_real_client(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "ROBOFLOW_API_KEY"),
        ):
            RoboflowCourtDetector()

        client = FakeClient({"predictions": []})
        with patch.dict(
            os.environ,
            {"ROBOFLOW_API_KEY": "secret"},
            clear=True,
        ), patch(
            "nba_3d_scene_reconstruction.court.detector._load_client",
            return_value=client,
        ) as load_client:
            detector = RoboflowCourtDetector()

        load_client.assert_called_once_with(
            "https://serverless.roboflow.com",
            "secret",
        )
        self.assertIs(detector.client, client)

    def test_rejects_negative_frame_index(self) -> None:
        detector = RoboflowCourtDetector(client=FakeClient({"predictions": []}))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            detector.detect("frame.jpg", frame_idx=-1)


if __name__ == "__main__":
    unittest.main()
