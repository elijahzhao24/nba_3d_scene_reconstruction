"""Roboflow Serverless adapter for basketball court keypoint detection."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

from .configuration import CourtDetectorConfiguration
from .schemas import CourtDetection, CourtKeypoint

if TYPE_CHECKING:
    import numpy as np

    ImageInput: TypeAlias = str | os.PathLike[str] | np.ndarray
else:
    ImageInput = object


class CourtSchemaMismatchError(ValueError):
    """The deployed model's landmark skeleton differs from configuration."""


class _InferenceClient(Protocol):
    def infer(
        self,
        inference_input: ImageInput,
        *,
        model_id: str,
    ) -> Mapping[str, Any]: ...


def _load_client(api_url: str, api_key: str) -> _InferenceClient:
    # Keep the optional network SDK out of module import time so configuration,
    # parsing, and unit tests do not create clients or perform network I/O.
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key).configure(
        InferenceConfiguration(api_key_transport="header")
    )
    return cast(_InferenceClient, client)


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array")
    return cast(Sequence[object], value)


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _confidence(value: object, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


class RoboflowCourtDetector:
    """Run the hosted model and normalize its sparse nested keypoints."""

    def __init__(
        self,
        configuration: CourtDetectorConfiguration | None = None,
        *,
        client: _InferenceClient | None = None,
    ) -> None:
        self.configuration = (
            configuration or CourtDetectorConfiguration.from_environment()
        )

        if client is None:
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if not api_key:
                raise RuntimeError("ROBOFLOW_API_KEY is not set")
            client = _load_client(self.configuration.api_url, api_key)
        self.client = client

    def detect(
        self,
        image: ImageInput,
        frame_idx: int,
    ) -> CourtDetection | None:
        """Return the best valid court prediction for one original frame."""
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")

        result = self.client.infer(
            image,
            model_id=self.configuration.model_id,
        )
        return self.parse_result(result, frame_idx=frame_idx)

    def parse_result(
        self,
        result: Mapping[str, Any],
        *,
        frame_idx: int,
    ) -> CourtDetection | None:
        """Parse hosted JSON without depending on SDK response classes."""
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")

        predictions = _sequence(result.get("predictions", ()), "predictions")
        court_predictions: list[tuple[float, Mapping[str, Any]]] = []

        for index, value in enumerate(predictions):
            prediction = _mapping(value, f"predictions[{index}]")
            class_name = prediction.get("class", prediction.get("class_name"))
            if class_name != self.configuration.parent_class_name:
                continue
            class_id = prediction.get("class_id")
            if class_id != self.configuration.parent_class_id:
                raise CourtSchemaMismatchError(
                    f"parent class {class_name!r} expected id "
                    f"{self.configuration.parent_class_id}, received {class_id!r}"
                )
            confidence = _confidence(
                prediction.get("confidence"),
                f"predictions[{index}].confidence",
            )
            if confidence >= self.configuration.detection_confidence:
                court_predictions.append((confidence, prediction))

        if not court_predictions:
            return None

        confidence, prediction = max(court_predictions, key=lambda item: item[0])
        keypoints = self._parse_keypoints(prediction)
        bbox_xyxy = self._parse_bbox(prediction)

        image_width: int | None = None
        image_height: int | None = None
        image_metadata = result.get("image")
        if image_metadata is not None:
            image = _mapping(image_metadata, "image")
            image_width = int(_finite_float(image.get("width"), "image.width"))
            image_height = int(
                _finite_float(image.get("height"), "image.height")
            )
            if image_width <= 0 or image_height <= 0:
                raise ValueError("image dimensions must be positive")

        return CourtDetection(
            frame_idx=frame_idx,
            confidence=confidence,
            bbox_xyxy=bbox_xyxy,
            keypoints=keypoints,
            image_width=image_width,
            image_height=image_height,
        )

    def _parse_keypoints(
        self,
        prediction: Mapping[str, Any],
    ) -> tuple[CourtKeypoint | None, ...]:
        raw_keypoints = _sequence(prediction.get("keypoints", ()), "keypoints")
        expected_labels = self.configuration.landmark_labels
        dense: list[CourtKeypoint | None] = [None] * len(expected_labels)

        for index, value in enumerate(raw_keypoints):
            keypoint = _mapping(value, f"keypoints[{index}]")
            raw_id = keypoint.get("class_id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise CourtSchemaMismatchError(
                    f"keypoints[{index}].class_id must be an integer"
                )
            if not 0 <= raw_id < len(expected_labels):
                raise CourtSchemaMismatchError(
                    f"landmark id {raw_id} is outside the configured skeleton"
                )
            if dense[raw_id] is not None:
                raise CourtSchemaMismatchError(
                    f"duplicate landmark id {raw_id} in court response"
                )

            label = keypoint.get("class", keypoint.get("class_name"))
            expected_label = expected_labels[raw_id]
            if label != expected_label:
                raise CourtSchemaMismatchError(
                    f"landmark id {raw_id} expected label {expected_label}, "
                    f"received {label!r}"
                )

            dense[raw_id] = CourtKeypoint(
                landmark_id=raw_id,
                label=expected_label,
                image_xy=(
                    _finite_float(keypoint.get("x"), f"keypoints[{index}].x"),
                    _finite_float(keypoint.get("y"), f"keypoints[{index}].y"),
                ),
                confidence=_confidence(
                    keypoint.get("confidence"),
                    f"keypoints[{index}].confidence",
                ),
            )

        return tuple(dense)

    @staticmethod
    def _parse_bbox(
        prediction: Mapping[str, Any],
    ) -> tuple[float, float, float, float]:
        center_x = _finite_float(prediction.get("x"), "prediction.x")
        center_y = _finite_float(prediction.get("y"), "prediction.y")
        width = _finite_float(prediction.get("width"), "prediction.width")
        height = _finite_float(prediction.get("height"), "prediction.height")
        if width < 0 or height < 0:
            raise ValueError("court bounding-box dimensions must be non-negative")
        return (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
