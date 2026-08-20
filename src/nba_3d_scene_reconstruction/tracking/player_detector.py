"""Native GPU RF-DETR adapter for on-court player detection."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from .schemas import PlayerDetection

if TYPE_CHECKING:
    import numpy as np

    ImageInput: TypeAlias = str | os.PathLike[str] | np.ndarray
else:
    ImageInput = object


DEFAULT_CONFIDENCE_THRESHOLD = 0.4
# Discard non player classes in our fine-tuned model
PLAYER_CLASSES = frozenset(
    {
        "player",
        "player-in-possession",
        "player-jump-shot",
        "player-layup-dunk",
        "player-shot-block",
    }
)

# expected shape from roboflow
class _Prediction(Protocol):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str


class _ImageMetadata(Protocol):
    width: int
    height: int


class _InferenceResponse(Protocol):
    predictions: Sequence[_Prediction]
    image: _ImageMetadata


class _InferenceModel(Protocol):
    def infer(
        self,
        image: ImageInput,
        *,
        confidence: float,
    ) -> Sequence[_InferenceResponse]: ...


def _load_model(model_id: str, api_key: str) -> _InferenceModel:
    # Keep the large optional ML dependency out of module import time. This also
    # lets schema and adapter tests run without loading CUDA or model weights.
    from inference import get_model

    return cast(
        _InferenceModel,
        get_model(model_id=model_id, api_key=api_key),
    )


class RoboflowPlayerDetector:
    """Convert native RF-DETR predictions into tracking data contracts.

    One detector instance owns one loaded model. Create it once per processing
    worker and reuse it at each RF-DETR checkpoint; constructing it per frame
    would repeatedly allocate model resources.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        confidence_threshold: float | None = None,
        player_classes: Iterable[str] = PLAYER_CLASSES,
        model: _InferenceModel | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("ROBOFLOW_MODEL_ID")
        if not self.model_id:
            raise RuntimeError("ROBOFLOW_MODEL_ID is not set")

        if confidence_threshold is None:
            configured_threshold = os.environ.get("RFDETR_CONFIDENCE_THRESHOLD")
            confidence_threshold = (
                float(configured_threshold)
                if configured_threshold is not None
                else DEFAULT_CONFIDENCE_THRESHOLD
            )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.confidence_threshold = confidence_threshold
        self.player_classes = frozenset(player_classes)
        if not self.player_classes:
            raise ValueError("player_classes must contain at least one class")

        if model is None:
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if not api_key:
                raise RuntimeError("ROBOFLOW_API_KEY is not set")
            model = _load_model(self.model_id, api_key)
        self.model = model

    def detect(
        self,
        image: ImageInput,
        frame_idx: int,
    ) -> tuple[PlayerDetection, ...]:
        """Detect players in one original-resolution frame.

        ``image`` may be a filesystem path or an OpenCV/NumPy image. Passing an
        in-memory frame avoids JPEG encoding and disk I/O in a video worker.
        """
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")

        responses = self.model.infer(
            image,
            confidence=self.confidence_threshold,
        )
        if not responses:
            return ()

        response = responses[0]
        image_width = float(response.image.width)
        image_height = float(response.image.height)
        if image_width <= 0 or image_height <= 0:
            raise ValueError("RF-DETR returned invalid image dimensions")

        detections = []
        for prediction in response.predictions:
            if prediction.class_name not in self.player_classes:
                continue
            if prediction.confidence < self.confidence_threshold:
                continue

            half_width = prediction.width / 2.0
            half_height = prediction.height / 2.0
            x1 = min(max(prediction.x - half_width, 0.0), image_width)
            y1 = min(max(prediction.y - half_height, 0.0), image_height)
            x2 = min(max(prediction.x + half_width, 0.0), image_width)
            y2 = min(max(prediction.y + half_height, 0.0), image_height)

            # Ignore boxes that are empty after clipping instead of passing
            # invalid geometry into association or SAM 2 prompting.
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                PlayerDetection(
                    frame_idx=frame_idx,
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(prediction.confidence),
                    class_id=int(prediction.class_id),
                )
            )

        return tuple(detections)