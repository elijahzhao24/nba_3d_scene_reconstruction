from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from nba_3d_scene_reconstruction.tracking.player_detector import (
    RoboflowPlayerDetector,
)


@dataclass(frozen=True)
class FakePrediction:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class FakeImageMetadata:
    width: int
    height: int


@dataclass(frozen=True)
class FakeResponse:
    predictions: tuple[FakePrediction, ...]
    image: FakeImageMetadata


class FakeModel:
    def __init__(self, responses: tuple[FakeResponse, ...]) -> None:
        self.responses = responses
        self.calls: list[tuple[object, float]] = []

    def infer(
        self,
        image: object,
        *,
        confidence: float,
    ) -> tuple[FakeResponse, ...]:
        self.calls.append((image, confidence))
        return self.responses


class RoboflowPlayerDetectorTest(unittest.TestCase):
    def test_filters_and_clips_predictions(self) -> None:
        model = FakeModel(
            (
                FakeResponse(
                    predictions=(
                        FakePrediction(10, 10, 30, 30, 0.9, 1, "player"),
                        FakePrediction(50, 50, 20, 20, 0.9, 2, "referee"),
                        FakePrediction(50, 50, 20, 20, 0.3, 1, "player"),
                        FakePrediction(150, 50, 20, 20, 0.9, 1, "player"),
                        FakePrediction(
                            70,
                            80,
                            20,
                            30,
                            0.8,
                            3,
                            "player-in-possession",
                        ),
                    ),
                    image=FakeImageMetadata(width=100, height=100),
                ),
            )
        )
        detector = RoboflowPlayerDetector(
            model_id="workspace/model",
            confidence_threshold=0.5,
            model=model,
        )

        detections = detector.detect("frame.jpg", frame_idx=12)

        self.assertEqual(model.calls, [("frame.jpg", 0.5)])
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].frame_idx, 12)
        self.assertEqual(detections[0].bbox_xyxy, (0.0, 0.0, 25.0, 25.0))
        self.assertEqual(detections[0].confidence, 0.9)
        self.assertEqual(detections[0].class_id, 1)
        self.assertEqual(detections[1].bbox_xyxy, (60.0, 65.0, 80.0, 95.0))

    def test_reads_confidence_threshold_from_environment(self) -> None:
        model = FakeModel(())
        with patch.dict(
            os.environ,
            {"RFDETR_CONFIDENCE_THRESHOLD": "0.65"},
            clear=True,
        ):
            detector = RoboflowPlayerDetector(
                model_id="workspace/model",
                model=model,
            )

        self.assertEqual(detector.confidence_threshold, 0.65)
        self.assertEqual(detector.detect("frame.jpg", 0), ())

    def test_rejects_invalid_configuration_and_frame_index(self) -> None:
        model = FakeModel(())

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            RoboflowPlayerDetector(
                model_id="workspace/model",
                confidence_threshold=1.1,
                model=model,
            )

        detector = RoboflowPlayerDetector(
            model_id="workspace/model",
            model=model,
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            detector.detect("frame.jpg", -1)

    def test_requires_model_configuration_before_loading(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "ROBOFLOW_MODEL_ID"),
        ):
            RoboflowPlayerDetector()


if __name__ == "__main__":
    unittest.main()
