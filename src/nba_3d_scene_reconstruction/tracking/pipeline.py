"""Orchestrate player detection, mask tracking, and ID lifecycle."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .association_engine import PlayerAssociationEngine
from .observations import build_player_observations
from .player_detector import RoboflowPlayerDetector
from .sam2_tracker import Sam2PlayerTracker
from .schemas import PlayerDetection, PlayerObservation, SamMaskPrediction
from .track_manager import PlayerTrackManager

DEFAULT_MAXIMUM_ACTIVE_PLAYER_TRACKS = 10
DEFAULT_NEW_TRACK_CONFIRMATION_CHECKPOINTS = 2
DEFAULT_REPROMPT_SCORE_THRESHOLD = 0.75
PENDING_DETECTION_IOU_THRESHOLD = 0.30


@dataclass
class _PendingDetection:
    detection: PlayerDetection
    checkpoint_hits: int
    last_frame_idx: int


class PlayerTrackingPipeline:
    """Coordinate tracking for one continuous video segment."""

    def __init__(
        self,
        *,
        detector: RoboflowPlayerDetector,
        sam_tracker: Sam2PlayerTracker,
        association_engine: PlayerAssociationEngine,
        track_manager: PlayerTrackManager,
        detector_interval: int = 5,
        fps: float = 30.0,
        maximum_active_tracks: int | None = DEFAULT_MAXIMUM_ACTIVE_PLAYER_TRACKS,
        new_track_confirmation_checkpoints: int = (
            DEFAULT_NEW_TRACK_CONFIRMATION_CHECKPOINTS
        ),
        reprompt_score_threshold: float = DEFAULT_REPROMPT_SCORE_THRESHOLD,
    ) -> None:
        if not isinstance(detector_interval, int) or detector_interval <= 0:
            raise ValueError("detector_interval must be a positive integer")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if maximum_active_tracks is not None and (
            not isinstance(maximum_active_tracks, int) or maximum_active_tracks <= 0
        ):
            raise ValueError("maximum_active_tracks must be positive or None")
        if (
            not isinstance(new_track_confirmation_checkpoints, int)
            or new_track_confirmation_checkpoints <= 0
        ):
            raise ValueError(
                "new_track_confirmation_checkpoints must be a positive integer"
            )
        if not 0.0 <= reprompt_score_threshold <= 1.0:
            raise ValueError("reprompt_score_threshold must be between 0 and 1")
        self.detector = detector
        self.sam_tracker = sam_tracker
        self.association_engine = association_engine
        self.track_manager = track_manager
        self.detector_interval = detector_interval
        self.fps = fps
        self.maximum_active_tracks = maximum_active_tracks
        self.new_track_confirmation_checkpoints = new_track_confirmation_checkpoints
        self.reprompt_score_threshold = reprompt_score_threshold
        self.observations: tuple[PlayerObservation, ...] = ()
        self._frame_detection_confidences: dict[int, float] = {}
        self._pending_detections: list[_PendingDetection] = []

        self._started = False
        self._last_frame_idx = -1

    def start_segment(self, frames_dir: str) -> None:
        """Initialize SAM 2 before processing the segment's first frame."""
        if self._started:
            raise RuntimeError("this pipeline has already started a segment")
        self.sam_tracker.start_segment(frames_dir)
        self._started = True

    def process_frame(
        self,
        frame: Any,
        frame_idx: int,
        *,
        detection_filter: Callable[[PlayerDetection], bool] | None = None,
        mask_filter: Callable[[SamMaskPrediction], bool] | None = None,
    ) -> tuple[SamMaskPrediction, ...]:
        """Process one frame; frames must arrive sequentially starting at zero."""
        self._validate_next_frame(frame_idx)
        self._frame_detection_confidences = {}

        if frame_idx == 0:
            self._initialize_players(frame, detection_filter)
            masks = self._filter_masks(
                self._known_masks(self.sam_tracker.propagate_frame(frame_idx)),
                mask_filter,
            )
        else:
            masks = self._filter_masks(
                self._known_masks(self.sam_tracker.propagate_frame(frame_idx)),
                mask_filter,
            )
            self._update_track_visibility(masks, frame_idx)

            if frame_idx % self.detector_interval == 0:
                self._run_detector_checkpoint(
                    frame,
                    frame_idx,
                    masks,
                    detection_filter,
                )

        self.observations = build_player_observations(
            masks,
            segment_id=self.track_manager.segment_id,
            frame_idx=frame_idx,
            fps=self.fps,
            track_ids=self.track_manager.tracks,
            detection_confidences=self._frame_detection_confidences,
            fallback_bboxes={
                track_id: track.latest_bbox_xyxy
                for track_id, track in self.track_manager.tracks.items()
            },
        )
        self._last_frame_idx = frame_idx
        return masks

    def _initialize_players(
        self,
        frame: Any,
        detection_filter: Callable[[PlayerDetection], bool] | None,
    ) -> None:
        detections = self._filter_detections(
            self.detector.detect(frame, frame_idx=0),
            detection_filter,
        )
        detections = tuple(
            sorted(detections, key=lambda item: item.confidence, reverse=True)
        )
        if self.maximum_active_tracks is not None:
            detections = detections[: self.maximum_active_tracks]
        for detection in detections:
            self._create_and_prompt_track(detection)

    def _run_detector_checkpoint(
        self,
        frame: Any,
        frame_idx: int,
        masks: tuple[SamMaskPrediction, ...],
        detection_filter: Callable[[PlayerDetection], bool] | None,
    ) -> None:
        detections = self._filter_detections(
            self.detector.detect(frame, frame_idx),
            detection_filter,
        )
        result = self.association_engine.associate(
            detections=detections,
            sam_masks=masks,
        )

        for match in result.matches:
            detection = detections[match.detection_index]
            self._frame_detection_confidences[match.track_id] = detection.confidence
            self.track_manager.mark_visible(
                match.track_id,
                frame_idx,
                detection.bbox_xyxy,
            )
            # A high-overlap SAM mask is already stable. Re-prompt only when
            # RF-DETR materially disagrees; unconditional checkpoint prompts
            # cause visible mask/footpoint snaps every detector interval.
            if match.score < self.reprompt_score_threshold:
                self.sam_tracker.prompt_player(
                    frame_idx,
                    match.track_id,
                    detection.bbox_xyxy,
                )

        self._consider_new_tracks(
            tuple(detections[index] for index in result.unmatched_detection_indices),
            frame_idx=frame_idx,
        )

        # An unmatched RF-DETR box does not make a valid SAM mask missing.
        # Visibility was already handled from the mask results above.

    def _create_and_prompt_track(self, detection: PlayerDetection) -> None:
        track = self.track_manager.create_track(detection)
        self._frame_detection_confidences[track.track_id] = detection.confidence
        self.sam_tracker.prompt_player(
            detection.frame_idx,
            track.track_id,
            detection.bbox_xyxy,
        )

    def _consider_new_tracks(
        self,
        detections: tuple[PlayerDetection, ...],
        *,
        frame_idx: int,
    ) -> None:
        """Require repeat checkpoint evidence before adding a new SAM object."""
        if not self._has_track_capacity():
            self._pending_detections.clear()
            return

        pending = [
            candidate
            for candidate in self._pending_detections
            if frame_idx - candidate.last_frame_idx <= self.detector_interval
        ]
        used_pending_indices: set[int] = set()
        next_pending: list[_PendingDetection] = []

        for detection in sorted(
            detections, key=lambda item: item.confidence, reverse=True
        ):
            best_index = None
            best_iou = PENDING_DETECTION_IOU_THRESHOLD
            for index, candidate in enumerate(pending):
                if index in used_pending_indices:
                    continue
                iou = self.association_engine._calculate_iou(
                    detection.bbox_xyxy,
                    candidate.detection.bbox_xyxy,
                )
                if iou >= best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is None:
                candidate = _PendingDetection(detection, 1, frame_idx)
            else:
                used_pending_indices.add(best_index)
                previous = pending[best_index]
                candidate = _PendingDetection(
                    detection,
                    previous.checkpoint_hits + 1,
                    frame_idx,
                )

            if (
                candidate.checkpoint_hits >= self.new_track_confirmation_checkpoints
                and self._has_track_capacity()
            ):
                self._create_and_prompt_track(detection)
            else:
                next_pending.append(candidate)

        self._pending_detections = next_pending

    def _has_track_capacity(self) -> bool:
        return (
            self.maximum_active_tracks is None
            or len(self.track_manager.tracks) < self.maximum_active_tracks
        )

    @staticmethod
    def _filter_detections(
        detections: tuple[PlayerDetection, ...],
        detection_filter: Callable[[PlayerDetection], bool] | None,
    ) -> tuple[PlayerDetection, ...]:
        if detection_filter is None:
            return detections
        return tuple(
            detection for detection in detections if detection_filter(detection)
        )

    def _known_masks(
        self,
        masks: tuple[SamMaskPrediction, ...],
    ) -> tuple[SamMaskPrediction, ...]:
        """Ignore stale SAM objects whose track has already been retired."""
        return tuple(
            mask for mask in masks if mask.track_id in self.track_manager.tracks
        )

    @staticmethod
    def _filter_masks(
        masks: tuple[SamMaskPrediction, ...],
        mask_filter: Callable[[SamMaskPrediction], bool] | None,
    ) -> tuple[SamMaskPrediction, ...]:
        if mask_filter is None:
            return masks
        return tuple(mask for mask in masks if mask_filter(mask))

    def _update_track_visibility(
        self,
        masks: tuple[SamMaskPrediction, ...],
        frame_idx: int,
    ) -> None:
        visible_track_ids = {mask.track_id for mask in masks if mask.mask.any()}

        # Copy the IDs because mark_missing may retire and remove a track.
        for track_id in tuple(self.track_manager.tracks):
            if track_id in visible_track_ids:
                self.track_manager.mark_visible(track_id, frame_idx)
            else:
                retired = self.track_manager.mark_missing(track_id, frame_idx)
                if retired is not None:
                    self.sam_tracker.remove_player(track_id)

    def _validate_next_frame(self, frame_idx: int) -> None:
        if not self._started:
            raise RuntimeError("start_segment must be called before process_frame")
        expected_frame_idx = self._last_frame_idx + 1
        if frame_idx != expected_frame_idx:
            raise ValueError(
                f"expected frame {expected_frame_idx}, received frame {frame_idx}"
            )

    def validate_next_frame(self, frame_idx: int) -> None:
        """Validate ordering before another subsystem performs frame work."""
        self._validate_next_frame(frame_idx)
