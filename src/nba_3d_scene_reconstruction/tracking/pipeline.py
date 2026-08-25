"""Orchestrate player detection, mask tracking, and ID lifecycle."""

from __future__ import annotations

from typing import Any

from .association_engine import PlayerAssociationEngine
from .player_detector import RoboflowPlayerDetector
from .sam2_tracker import Sam2PlayerTracker
from .schemas import PlayerDetection, SamMaskPrediction
from .track_manager import PlayerTrackManager


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
    ) -> None:
        self.detector = detector
        self.sam_tracker = sam_tracker
        self.association_engine = association_engine
        self.track_manager = track_manager
        self.detector_interval = detector_interval

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
    ) -> tuple[SamMaskPrediction, ...]:
        """Process one frame; frames must arrive sequentially starting at zero."""
        self._validate_next_frame(frame_idx)

        if frame_idx == 0:
            self._initialize_players(frame)
            masks = self._known_masks(self.sam_tracker.propagate_frame(frame_idx))
        else:
            masks = self._known_masks(self.sam_tracker.propagate_frame(frame_idx))
            self._update_track_visibility(masks, frame_idx)

            if frame_idx % self.detector_interval == 0:
                self._run_detector_checkpoint(frame, frame_idx, masks)

        self._last_frame_idx = frame_idx
        return masks

    def _initialize_players(self, frame: Any) -> None:
        detections = self.detector.detect(frame, frame_idx=0)
        for detection in detections:
            self._create_and_prompt_track(detection)

    def _run_detector_checkpoint(
        self,
        frame: Any,
        frame_idx: int,
        masks: tuple[SamMaskPrediction, ...],
    ) -> None:
        detections = self.detector.detect(frame, frame_idx)
        result = self.association_engine.associate(
            detections=detections,
            sam_masks=masks,
        )

        for match in result.matches:
            detection = detections[match.detection_index]
            self.track_manager.mark_visible(
                match.track_id,
                frame_idx,
                detection.bbox_xyxy,
            )
            self.sam_tracker.prompt_player(
                frame_idx,
                match.track_id,
                detection.bbox_xyxy,
            )

        for detection_index in result.unmatched_detection_indices:
            self._create_and_prompt_track(detections[detection_index])

        # An unmatched RF-DETR box does not make a valid SAM mask missing.
        # Visibility was already handled from the mask results above.

    def _create_and_prompt_track(self, detection: PlayerDetection) -> None:
        track = self.track_manager.create_track(detection)
        self.sam_tracker.prompt_player(
            detection.frame_idx,
            track.track_id,
            detection.bbox_xyxy,
        )

    def _known_masks(
        self,
        masks: tuple[SamMaskPrediction, ...],
    ) -> tuple[SamMaskPrediction, ...]:
        """Ignore stale SAM objects whose track has already been retired."""
        return tuple(
            mask
            for mask in masks
            if mask.track_id in self.track_manager.tracks
        )

    def _update_track_visibility(
        self,
        masks: tuple[SamMaskPrediction, ...],
        frame_idx: int,
    ) -> None:
        visible_track_ids = {
            mask.track_id
            for mask in masks
            if mask.mask.any()
        }

        # Copy the IDs because mark_missing may retire and remove a track.
        for track_id in tuple(self.track_manager.tracks):
            if track_id in visible_track_ids:
                self.track_manager.mark_visible(track_id, frame_idx)
            else:
                self.track_manager.mark_missing(track_id, frame_idx)

    def _validate_next_frame(self, frame_idx: int) -> None:
        if not self._started:
            raise RuntimeError("start_segment must be called before process_frame")
        expected_frame_idx = self._last_frame_idx + 1
        if frame_idx != expected_frame_idx:
            raise ValueError(
                f"expected frame {expected_frame_idx}, received frame {frame_idx}"
            )
