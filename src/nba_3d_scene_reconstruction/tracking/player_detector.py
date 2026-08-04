"""Roboflow-backed player detection adapter."""

from __future__ import annotations

import os

from inference_sdk import InferenceHTTPClient

from .schemas import PlayerDetection


PLAYER_CLASSES = {
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
}


class RoboflowPlayerDetector:
    def __init__(self, model_id: str | None = None) -> None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("ROBOFLOW_API_KEY is not set")

        self.model_id = model_id or os.environ.get(
            "ROBOFLOW_MODEL_ID",
            "basketball-player-detection-3-ycjdo/4",
        )
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )

    def detect(
        self,
        image_path: str,
        frame_idx: int,
    ) -> tuple[PlayerDetection, ...]:
        result = self.client.infer(image_path, model_id=self.model_id)
        detections = []

        for prediction in result["predictions"]:
            if prediction["class"] not in PLAYER_CLASSES:
                continue

            center_x = prediction["x"]
            center_y = prediction["y"]
            width = prediction["width"]
            height = prediction["height"]

            detections.append(
                PlayerDetection(
                    frame_idx=frame_idx,
                    bbox_xyxy=(
                        center_x - width / 2,
                        center_y - height / 2,
                        center_x + width / 2,
                        center_y + height / 2,
                    ),
                    confidence=prediction["confidence"],
                    class_id=prediction["class_id"],
                )
            )

        return tuple(detections)
