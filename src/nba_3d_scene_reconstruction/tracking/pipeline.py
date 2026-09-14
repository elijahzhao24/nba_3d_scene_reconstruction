"""Orchestrate player detection, mask tracking, validation, and recovery."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .association_engine import PlayerAssociationEngine
from .observations import build_player_observations
from .player_detector import RoboflowPlayerDetector
from .sam2_tracker import Sam2PlayerTracker
from .schemas import (
    ObservationSource,
    PlayerDetection,
    PlayerObservation,
    SamMaskPrediction,
    TrackStatus,
)
from .track_manager import PlayerTrackManager

DEFAULT_MAXIMUM_ACTIVE_PLAYER_TRACKS = 10
DEFAULT_MAXIMUM_TENTATIVE_PLAYER_TRACKS = 2
DEFAULT_NEW_TRACK_CONFIRMATION_CHECKPOINTS = 2
DEFAULT_REPROMPT_SCORE_THRESHOLD = 0.75
INITIAL_PLAYER_CONFIDENCE_THRESHOLD = 0.75
TENTATIVE_MEAN_CONFIDENCE_THRESHOLD = 0.72
PENDING_DETECTION_IOU_THRESHOLD = 0.20
PENDING_DETECTION_DISTANCE_HEIGHT_RATIO = 0.75
DUPLICATE_MASK_IOU_THRESHOLD = 0.80
DUPLICATE_MASK_PATIENCE_FRAMES = 3
DETECTOR_HISTORY_CHECKPOINTS = 3
TENTATIVE_TIMEOUT_CHECKPOINTS = 3
MAX_BACKWARD_RECOVERY_FRAMES = 30
OFF_COURT_PATIENCE_SECONDS = 0.25
WEAK_TRACK_CONFIDENCE_THRESHOLD = 0.72
REPLACEMENT_CONFIDENCE_MARGIN = 0.15


@dataclass
class _PendingDetection:
    """Repeated unmatched RF-DETR evidence for one potential player."""

    detections: list[PlayerDetection]

    @property
    def latest(self) -> PlayerDetection:
        return self.detections[-1]

    @property
    def start_frame_idx(self) -> int:
        return self.detections[0].frame_idx

    def append(self, detection: PlayerDetection) -> None:
        self.detections.append(detection)

    def recent(self, maximum_hits: int) -> list[PlayerDetection]:
        return self.detections[-maximum_hits:]


@dataclass(frozen=True)
class _TentativeTrack:
    candidate_start_frame: int
    created_frame: int


@dataclass(frozen=True)
class _RecoveryRequest:
    track_id: int
    confirmation_frame: int
    candidate_start_frame: int


class PlayerTrackingPipeline:
    """Coordinate one offline-capable player-tracking segment.

    ``process_frame`` remains a forward-only, sequential interface for callers
    that need progress feedback. Its observations deliberately exclude
    tentative objects. ``finalize_segment`` is the authoritative batch phase:
    it suppresses proven duplicates, runs bounded backward SAM2 recovery, and
    writes final artifacts when an artifact directory has been configured.
    """

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
        maximum_tentative_tracks: int = DEFAULT_MAXIMUM_TENTATIVE_PLAYER_TRACKS,
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
            not isinstance(maximum_tentative_tracks, int)
            or maximum_tentative_tracks < 0
        ):
            raise ValueError("maximum_tentative_tracks must be non-negative")
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
        self.maximum_tentative_tracks = maximum_tentative_tracks
        self.new_track_confirmation_checkpoints = new_track_confirmation_checkpoints
        self.reprompt_score_threshold = reprompt_score_threshold

        self.observations: tuple[PlayerObservation, ...] = ()
        self.finalized_observations: dict[int, tuple[PlayerObservation, ...]] = {}
        self._frame_detection_confidences: dict[int, float] = {}
        self._pending_detections: list[_PendingDetection] = []
        self._tentative_tracks: dict[int, _TentativeTrack] = {}
        self._detector_history: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._rejected_mask_streaks: dict[int, int] = {}
        self._duplicate_streaks: dict[tuple[int, int], tuple[int, int]] = {}
        self._suppressed_from_frame: dict[int, int] = {}
        self._recovery_requests: list[_RecoveryRequest] = []
        self._recovered_masks: dict[int, dict[int, SamMaskPrediction]] = defaultdict(
            dict
        )
        self._observation_history: dict[int, tuple[PlayerObservation, ...]] = {}
        self._recovery_mask_filters: dict[
            int, Callable[[SamMaskPrediction], bool] | None
        ] = {}
        self._artifact_dir: Path | None = None
        self._started = False
        self._finalized = False
        self._last_frame_idx = -1

    def configure_artifacts(self, artifact_dir: str | Path) -> None:
        """Configure durable raw and finalized tracking artifacts."""
        if self._started:
            raise RuntimeError("configure artifacts before start_segment")
        self._artifact_dir = Path(artifact_dir)

    def start_segment(self, frames_dir: str) -> None:
        """Initialize SAM 2 and artifact streams for one video segment."""
        if self._started:
            raise RuntimeError("this pipeline has already started a segment")
        self.sam_tracker.start_segment(frames_dir)
        self._started = True
        if self._artifact_dir is not None:
            (self._artifact_dir / "masks" / "raw").mkdir(parents=True, exist_ok=True)
            (self._artifact_dir / "masks").mkdir(parents=True, exist_ok=True)
            for name in (
                "detections.jsonl",
                "tracking_events.jsonl",
                "raw_observations.jsonl",
                "observations.jsonl",
            ):
                (self._artifact_dir / name).write_text("", encoding="utf-8")

    def process_frame(
        self,
        frame: Any,
        frame_idx: int,
        *,
        detection_filter: Callable[[PlayerDetection], bool] | None = None,
        mask_filter: Callable[[SamMaskPrediction], bool] | None = None,
    ) -> tuple[SamMaskPrediction, ...]:
        """Process one sequential frame and expose only confirmed player masks."""
        self._validate_next_frame(frame_idx)
        self._frame_detection_confidences = {}
        self._recovery_mask_filters[frame_idx] = mask_filter

        if frame_idx == 0:
            self._initialize_players(frame, detection_filter)
            all_masks = self._filter_masks(
                self._known_masks(self.sam_tracker.propagate_frame(frame_idx)),
                mask_filter,
            )
        else:
            raw_masks = self._known_masks(self.sam_tracker.propagate_frame(frame_idx))
            all_masks = self._filter_masks(raw_masks, mask_filter)
            accepted_ids = {mask.track_id for mask in all_masks}
            rejected_ids = {
                mask.track_id for mask in raw_masks
                if mask.mask.any() and mask.track_id not in accepted_ids
            }
            self._update_track_visibility(all_masks, frame_idx, rejected_ids)
            if frame_idx % self.detector_interval == 0:
                self._run_detector_checkpoint(
                    frame,
                    frame_idx,
                    all_masks,
                    detection_filter,
                )

        self._update_duplicate_streaks(all_masks, frame_idx)
        confirmed_track_ids = self._confirmed_track_ids()
        masks = tuple(mask for mask in all_masks if mask.track_id in confirmed_track_ids)
        self.observations = build_player_observations(
            masks,
            segment_id=self.track_manager.segment_id,
            frame_idx=frame_idx,
            fps=self.fps,
            track_ids=confirmed_track_ids,
            detection_confidences=self._frame_detection_confidences,
            fallback_bboxes={
                track_id: self.track_manager.tracks[track_id].latest_bbox_xyxy
                for track_id in confirmed_track_ids
            },
        )
        self._observation_history[frame_idx] = self.observations
        self._record_masks(frame_idx, all_masks)
        self._append_jsonl(
            "raw_observations.jsonl", *(asdict(item) for item in self.observations)
        )
        self._last_frame_idx = frame_idx
        return masks

    def finalize_segment(self) -> dict[int, tuple[PlayerObservation, ...]]:
        """Produce authoritative observations and masks after the final frame."""
        if not self._started:
            raise RuntimeError("start_segment must be called before finalize_segment")
        if self._finalized:
            return self.finalized_observations

        for request in self._recovery_requests:
            if request.track_id not in self.track_manager.tracks:
                continue
            start_frame = max(
                0,
                request.candidate_start_frame - MAX_BACKWARD_RECOVERY_FRAMES,
            )
            recovered = self.sam_tracker.propagate_range(
                request.confirmation_frame,
                start_frame,
                reverse=True,
                track_ids=(request.track_id,),
            )
            invalid_streak = 0
            for frame_idx in sorted(recovered, reverse=True):
                prediction = next(
                    (
                        value
                        for value in recovered[frame_idx]
                        if value.track_id == request.track_id
                    ),
                    None,
                )
                if prediction is None or not prediction.mask.any():
                    invalid_streak += 1
                elif (
                    self._recovery_mask_filters.get(frame_idx) is not None
                    and not self._recovery_mask_filters[frame_idx](prediction)
                ):
                    invalid_streak += 1
                elif self._recovery_collides(prediction, frame_idx):
                    invalid_streak += 1
                else:
                    invalid_streak = 0
                    self._recovered_masks[frame_idx][request.track_id] = prediction
                    self._record_event(
                        "backward_recovery",
                        frame_idx=frame_idx,
                        track_id=request.track_id,
                        confirmation_frame=request.confirmation_frame,
                    )
                if invalid_streak >= 2:
                    break

        finalized: dict[int, tuple[PlayerObservation, ...]] = {}
        for frame_idx, observations in self._observation_history.items():
            visible = [
                observation
                for observation in observations
                if frame_idx
                < self._suppressed_from_frame.get(observation.track_id, math.inf)
            ]
            by_track = {observation.track_id: observation for observation in visible}
            for track_id, prediction in self._recovered_masks.get(frame_idx, {}).items():
                recovered_observation = build_player_observations(
                    (prediction,),
                    segment_id=self.track_manager.segment_id,
                    frame_idx=frame_idx,
                    fps=self.fps,
                    track_ids=(track_id,),
                )[0]
                by_track[track_id] = replace(
                    recovered_observation,
                    source=ObservationSource.SAM2_BACKWARD_RECOVERY,
                )
            finalized[frame_idx] = tuple(
                sorted(by_track.values(), key=lambda item: item.track_id)
            )

        self.finalized_observations = self._persist_finalized_observations(finalized)
        self._finalized = True
        return self.finalized_observations

    def finalized_masks_for_frame(self, frame_idx: int) -> tuple[SamMaskPrediction, ...]:
        """Load finalized masks by frame for a second-pass debug renderer."""
        if not self._finalized:
            raise RuntimeError("finalize_segment must be called before reading final masks")
        if self._artifact_dir is None:
            return tuple(self._recovered_masks.get(frame_idx, {}).values())
        directory = self._artifact_dir / "masks" / f"{frame_idx:06d}"
        masks: list[SamMaskPrediction] = []
        if directory.exists():
            for path in sorted(directory.glob("*.png"), key=lambda item: int(item.stem)):
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks.append(
                        SamMaskPrediction(
                            frame_idx=frame_idx,
                            track_id=int(path.stem),
                            mask=mask.astype(bool),
                        )
                    )
        return tuple(masks)

    def _initialize_players(
        self,
        frame: Any,
        detection_filter: Callable[[PlayerDetection], bool] | None,
    ) -> None:
        detections = tuple(
            sorted(
                self._filter_detections(
                    self.detector.detect(frame, frame_idx=0), detection_filter
                ),
                key=lambda item: item.confidence,
                reverse=True,
            )
        )
        self._record_detections(detections)
        for detection in detections:
            if (
                detection.confidence >= INITIAL_PLAYER_CONFIDENCE_THRESHOLD
                and self._has_confirmed_capacity()
            ):
                self._create_and_prompt_track(detection, tentative=False)
            else:
                self._add_pending_detection(detection)

    def _run_detector_checkpoint(
        self,
        frame: Any,
        frame_idx: int,
        masks: tuple[SamMaskPrediction, ...],
        detection_filter: Callable[[PlayerDetection], bool] | None,
    ) -> None:
        detections = self._filter_detections(
            self.detector.detect(frame, frame_idx), detection_filter
        )
        self._record_detections(detections)
        result = self.association_engine.associate(
            detections=detections,
            sam_masks=masks,
            track_bboxes={
                track_id: track.latest_bbox_xyxy
                for track_id, track in self.track_manager.tracks.items()
            },
        )

        matches_by_track = {match.track_id: match for match in result.matches}
        for match in result.matches:
            detection = detections[match.detection_index]
            self._frame_detection_confidences[match.track_id] = detection.confidence
            self._record_detector_support(match.track_id, frame_idx, detection.confidence)
            self.track_manager.mark_visible(
                match.track_id,
                frame_idx,
                detection.bbox_xyxy,
            )
            if match.score < self.reprompt_score_threshold:
                self.sam_tracker.prompt_player(
                    frame_idx,
                    match.track_id,
                    detection.bbox_xyxy,
                )

        self._arbitrate_duplicates(masks, matches_by_track, frame_idx)
        self._promote_valid_tentatives(masks, matches_by_track, frame_idx)
        self._consider_new_tracks(
            tuple(detections[index] for index in result.unmatched_detection_indices),
            frame_idx=frame_idx,
        )
        self._retire_stale_tentatives(frame_idx)

    def _create_and_prompt_track(
        self,
        detection: PlayerDetection,
        *,
        tentative: bool,
        candidate_start_frame: int | None = None,
    ) -> int:
        track = self.track_manager.create_track(detection, tentative=tentative)
        self._frame_detection_confidences[track.track_id] = detection.confidence
        self._record_detector_support(track.track_id, detection.frame_idx, detection.confidence)
        self.sam_tracker.prompt_player(
            detection.frame_idx,
            track.track_id,
            detection.bbox_xyxy,
        )
        if tentative:
            self._tentative_tracks[track.track_id] = _TentativeTrack(
                candidate_start_frame=(candidate_start_frame or detection.frame_idx),
                created_frame=detection.frame_idx,
            )
            self._record_event(
                "tentative_created",
                frame_idx=detection.frame_idx,
                track_id=track.track_id,
                candidate_start_frame=candidate_start_frame or detection.frame_idx,
            )
        return track.track_id

    def _consider_new_tracks(
        self,
        detections: tuple[PlayerDetection, ...],
        *,
        frame_idx: int,
    ) -> None:
        """Retain evidence while full; only SAM validation consumes overflow."""
        still_recent = [
            candidate
            for candidate in self._pending_detections
            if frame_idx - candidate.latest.frame_idx <= self.detector_interval * 3
        ]
        used_indices: set[int] = set()
        next_pending: list[_PendingDetection] = []

        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            best_index: int | None = None
            best_score = -math.inf
            for index, candidate in enumerate(still_recent):
                if index in used_indices:
                    continue
                score = self._pending_similarity(detection, candidate.latest)
                if score > best_score and self._same_pending_player(
                    detection, candidate.latest
                ):
                    best_index = index
                    best_score = score
            if best_index is None:
                candidate = _PendingDetection([detection])
            else:
                used_indices.add(best_index)
                candidate = still_recent[best_index]
                candidate.append(detection)

            if self._pending_is_confirmed(candidate) and self._has_tentative_capacity():
                self._create_and_prompt_track(
                    detection,
                    tentative=True,
                    candidate_start_frame=candidate.start_frame_idx,
                )
            else:
                next_pending.append(candidate)

        next_pending.extend(
            candidate
            for index, candidate in enumerate(still_recent)
            if index not in used_indices
        )
        self._pending_detections = next_pending

    def _add_pending_detection(self, detection: PlayerDetection) -> None:
        self._pending_detections.append(_PendingDetection([detection]))

    def _pending_is_confirmed(self, candidate: _PendingDetection) -> bool:
        recent_three = candidate.recent(3)
        if (
            len(recent_three) >= self.new_track_confirmation_checkpoints
            and len(recent_three) >= 2
            and statistics_mean(item.confidence for item in recent_three)
            >= TENTATIVE_MEAN_CONFIDENCE_THRESHOLD
        ):
            return True
        recent_four = candidate.recent(4)
        return len(recent_four) >= 3 and all(
            item.confidence >= 0.66 for item in recent_four
        )

    def _promote_valid_tentatives(
        self,
        masks: tuple[SamMaskPrediction, ...],
        matches_by_track: dict[int, Any],
        frame_idx: int,
    ) -> None:
        visible_track_ids = {mask.track_id for mask in masks if mask.mask.any()}
        for track_id, tentative in tuple(self._tentative_tracks.items()):
            if track_id not in matches_by_track or track_id not in visible_track_ids:
                continue
            victim = None
            if not self._has_confirmed_capacity():
                candidate_hits = self._recent_detector_hits(track_id, frame_idx)
                victim = self._find_replaceable_track(
                    frame_idx,
                    candidate_confidence=statistics_mean(
                        confidence for _, confidence in candidate_hits
                    ),
                )
                if victim is None:
                    continue
                self._retire_track(victim, frame_idx, "capacity_replacement")
            self.track_manager.promote_track(track_id, frame_idx)
            del self._tentative_tracks[track_id]
            self._recovery_requests.append(
                _RecoveryRequest(
                    track_id=track_id,
                    confirmation_frame=frame_idx,
                    candidate_start_frame=tentative.candidate_start_frame,
                )
            )
            self._record_event(
                "tentative_promoted",
                frame_idx=frame_idx,
                track_id=track_id,
                replaced_track_id=victim,
            )

    def _find_replaceable_track(
        self, frame_idx: int, *, candidate_confidence: float = 0.0,
    ) -> int | None:
        missing = [
            track_id
            for track_id, track in self.track_manager.tracks.items()
            if track.status is TrackStatus.MISSING
        ]
        if missing:
            return min(missing)
        unsupported = [
            track_id
            for track_id in self._confirmed_track_ids()
            if not self._recent_detector_hits(track_id, frame_idx)
        ]
        if unsupported:
            return min(unsupported)
        # Require repeated weak evidence and a clear improvement; one poor
        # detection during an occlusion must not evict an established player.
        weak: list[tuple[float, int]] = []
        for track_id in self._confirmed_track_ids():
            hits = self._recent_detector_hits(track_id, frame_idx)[-2:]
            if len(hits) < 2 or any(
                confidence >= WEAK_TRACK_CONFIDENCE_THRESHOLD
                for _, confidence in hits
            ):
                continue
            confidence = statistics_mean(value for _, value in hits)
            if candidate_confidence >= confidence + REPLACEMENT_CONFIDENCE_MARGIN:
                weak.append((confidence, track_id))
        return min(weak)[1] if weak else None

    def _retire_stale_tentatives(self, frame_idx: int) -> None:
        timeout = self.detector_interval * TENTATIVE_TIMEOUT_CHECKPOINTS
        for track_id, tentative in tuple(self._tentative_tracks.items()):
            if frame_idx - tentative.created_frame >= timeout:
                self._retire_track(track_id, frame_idx, "tentative_timeout")

    def _update_duplicate_streaks(
        self,
        masks: tuple[SamMaskPrediction, ...],
        frame_idx: int,
    ) -> None:
        visible = [mask for mask in masks if mask.mask.any()]
        active_pairs: set[tuple[int, int]] = set()
        for index, first in enumerate(visible):
            for second in visible[index + 1 :]:
                pair = tuple(sorted((first.track_id, second.track_id)))
                if self._mask_iou(first.mask, second.mask) < DUPLICATE_MASK_IOU_THRESHOLD:
                    continue
                active_pairs.add(pair)
                count, start_frame = self._duplicate_streaks.get(pair, (0, frame_idx))
                self._duplicate_streaks[pair] = (count + 1, start_frame)
        for pair in tuple(self._duplicate_streaks):
            if pair not in active_pairs:
                del self._duplicate_streaks[pair]

    def _arbitrate_duplicates(
        self,
        masks: tuple[SamMaskPrediction, ...],
        matches_by_track: dict[int, Any],
        frame_idx: int,
    ) -> None:
        mask_ids = {mask.track_id for mask in masks if mask.mask.any()}
        for pair, (streak, start_frame) in tuple(self._duplicate_streaks.items()):
            if (
                streak < DUPLICATE_MASK_PATIENCE_FRAMES
                or pair[0] not in mask_ids
                or pair[1] not in mask_ids
            ):
                continue
            independent_support = sum(track_id in matches_by_track for track_id in pair)
            if independent_support >= 2:
                continue
            winner = max(
                pair,
                key=lambda track_id: self._duplicate_survivor_key(track_id, frame_idx),
            )
            loser = pair[1] if winner == pair[0] else pair[0]
            self._suppressed_from_frame[loser] = min(
                self._suppressed_from_frame.get(loser, start_frame), start_frame
            )
            self._retire_track(loser, frame_idx, "duplicate_mask")
            self._record_event(
                "duplicate_suppressed",
                frame_idx=frame_idx,
                track_id=loser,
                survivor_track_id=winner,
                duplicate_start_frame=start_frame,
                detector_support=independent_support,
            )
            self._duplicate_streaks.pop(pair, None)

    def _duplicate_survivor_key(self, track_id: int, frame_idx: int) -> tuple[int, float, int]:
        hits = self._recent_detector_hits(track_id, frame_idx)
        return (
            len(hits),
            statistics_mean(confidence for _, confidence in hits),
            -track_id,
        )

    def _retire_track(self, track_id: int, frame_idx: int, reason: str) -> None:
        retired = self.track_manager.remove_track(track_id, frame_idx)
        if retired is None:
            return
        self.sam_tracker.remove_player(track_id)
        self._tentative_tracks.pop(track_id, None)
        self._rejected_mask_streaks.pop(track_id, None)
        self._record_event(reason, frame_idx=frame_idx, track_id=track_id)

    def _update_track_visibility(
        self,
        masks: tuple[SamMaskPrediction, ...],
        frame_idx: int,
        rejected_track_ids: set[int] | None = None,
    ) -> None:
        visible_track_ids = {mask.track_id for mask in masks if mask.mask.any()}
        for track_id in tuple(self.track_manager.tracks):
            if rejected_track_ids and track_id in rejected_track_ids:
                streak = self._rejected_mask_streaks.get(track_id, 0) + 1
                self._rejected_mask_streaks[track_id] = streak
                if streak >= max(2, math.ceil(self.fps * OFF_COURT_PATIENCE_SECONDS)):
                    self._retire_track(track_id, frame_idx, "off_court_retired")
                    continue
            else:
                self._rejected_mask_streaks.pop(track_id, None)
            if track_id in visible_track_ids:
                self.track_manager.mark_visible(track_id, frame_idx)
            else:
                retired = self.track_manager.mark_missing(track_id, frame_idx)
                if retired is not None:
                    self.sam_tracker.remove_player(track_id)
                    self._tentative_tracks.pop(track_id, None)
                    self._record_event(
                        "mask_missing_retired", frame_idx=frame_idx, track_id=track_id
                    )

    def _confirmed_track_ids(self) -> tuple[int, ...]:
        return tuple(
            track_id
            for track_id, track in self.track_manager.tracks.items()
            if track.status is not TrackStatus.TENTATIVE
        )

    def _has_confirmed_capacity(self) -> bool:
        return (
            self.maximum_active_tracks is None
            or len(self._confirmed_track_ids()) < self.maximum_active_tracks
        )

    def _has_tentative_capacity(self) -> bool:
        return len(self._tentative_tracks) < self.maximum_tentative_tracks

    def _record_detector_support(
        self, track_id: int, frame_idx: int, confidence: float
    ) -> None:
        history = self._detector_history[track_id]
        history.append((frame_idx, confidence))
        minimum_frame = frame_idx - self.detector_interval * DETECTOR_HISTORY_CHECKPOINTS
        self._detector_history[track_id] = [
            value for value in history if value[0] >= minimum_frame
        ]

    def _recent_detector_hits(
        self, track_id: int, frame_idx: int
    ) -> list[tuple[int, float]]:
        minimum_frame = frame_idx - self.detector_interval * DETECTOR_HISTORY_CHECKPOINTS
        return [
            value
            for value in self._detector_history.get(track_id, ())
            if value[0] >= minimum_frame
        ]

    @staticmethod
    def _pending_similarity(
        current: PlayerDetection, previous: PlayerDetection
    ) -> float:
        iou = PlayerAssociationEngine._calculate_iou(
            current.bbox_xyxy, previous.bbox_xyxy
        )
        current_foot = (
            (current.bbox_xyxy[0] + current.bbox_xyxy[2]) / 2.0,
            current.bbox_xyxy[3],
        )
        previous_foot = (
            (previous.bbox_xyxy[0] + previous.bbox_xyxy[2]) / 2.0,
            previous.bbox_xyxy[3],
        )
        height = max(
            1.0,
            current.bbox_xyxy[3] - current.bbox_xyxy[1],
            previous.bbox_xyxy[3] - previous.bbox_xyxy[1],
        )
        return iou + max(0.0, 1.0 - math.dist(current_foot, previous_foot) / height)

    @staticmethod
    def _same_pending_player(
        current: PlayerDetection, previous: PlayerDetection
    ) -> bool:
        if (
            PlayerAssociationEngine._calculate_iou(
                current.bbox_xyxy, previous.bbox_xyxy
            )
            >= PENDING_DETECTION_IOU_THRESHOLD
        ):
            return True
        current_foot = (
            (current.bbox_xyxy[0] + current.bbox_xyxy[2]) / 2.0,
            current.bbox_xyxy[3],
        )
        previous_foot = (
            (previous.bbox_xyxy[0] + previous.bbox_xyxy[2]) / 2.0,
            previous.bbox_xyxy[3],
        )
        height = max(
            1.0,
            current.bbox_xyxy[3] - current.bbox_xyxy[1],
            previous.bbox_xyxy[3] - previous.bbox_xyxy[1],
        )
        return math.dist(current_foot, previous_foot) <= (
            PENDING_DETECTION_DISTANCE_HEIGHT_RATIO * height
        )

    @staticmethod
    def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
        intersection = int(np.logical_and(first, second).sum())
        union = int(np.logical_or(first, second).sum())
        return 0.0 if union == 0 else intersection / union

    def _recovery_collides(
        self, prediction: SamMaskPrediction, frame_idx: int
    ) -> bool:
        if self._artifact_dir is None:
            return False
        for observation in self._observation_history.get(frame_idx, ()):
            if observation.track_id == prediction.track_id:
                continue
            mask = cv2.imread(
                str(self._raw_mask_path(frame_idx, observation.track_id)),
                cv2.IMREAD_GRAYSCALE,
            )
            if mask is not None and self._mask_iou(
                prediction.mask, mask.astype(bool)
            ) >= DUPLICATE_MASK_IOU_THRESHOLD:
                return True
        return False

    def _record_detections(self, detections: tuple[PlayerDetection, ...]) -> None:
        self._append_jsonl("detections.jsonl", *(asdict(item) for item in detections))

    def _record_event(self, event: str, **values: object) -> None:
        self._append_jsonl("tracking_events.jsonl", {"event": event, **values})

    def _record_masks(
        self, frame_idx: int, masks: tuple[SamMaskPrediction, ...]
    ) -> None:
        if self._artifact_dir is None:
            return
        for prediction in masks:
            destination = self._raw_mask_path(frame_idx, prediction.track_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), prediction.mask.astype(np.uint8) * 255):
                raise OSError(f"could not write mask: {destination}")

    def _persist_finalized_observations(
        self, finalized: dict[int, tuple[PlayerObservation, ...]]
    ) -> dict[int, tuple[PlayerObservation, ...]]:
        persisted: dict[int, tuple[PlayerObservation, ...]] = {}
        for frame_idx, observations in finalized.items():
            frame_observations: list[PlayerObservation] = []
            for observation in observations:
                mask_ref = self._persist_final_mask(frame_idx, observation.track_id)
                if observation.track_id in self._recovered_masks.get(frame_idx, {}):
                    mask_ref = self._write_final_mask(
                        frame_idx, self._recovered_masks[frame_idx][observation.track_id]
                    )
                frame_observations.append(replace(observation, mask_ref=mask_ref))
            persisted[frame_idx] = tuple(frame_observations)
        self._append_jsonl(
            "observations.jsonl",
            *(asdict(observation) for frame in persisted.values() for observation in frame),
        )
        return persisted

    def _persist_final_mask(self, frame_idx: int, track_id: int) -> str | None:
        if self._artifact_dir is None:
            return None
        source = self._raw_mask_path(frame_idx, track_id)
        if not source.exists():
            return None
        destination = self._final_mask_path(frame_idx, track_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return str(destination.relative_to(self._artifact_dir))

    def _write_final_mask(
        self, frame_idx: int, prediction: SamMaskPrediction
    ) -> str | None:
        if self._artifact_dir is None:
            return None
        destination = self._final_mask_path(frame_idx, prediction.track_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), prediction.mask.astype(np.uint8) * 255):
            raise OSError(f"could not write mask: {destination}")
        return str(destination.relative_to(self._artifact_dir))

    def _append_jsonl(self, name: str, *rows: object) -> None:
        if self._artifact_dir is None or not rows:
            return
        with (self._artifact_dir / name).open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def _raw_mask_path(self, frame_idx: int, track_id: int) -> Path:
        assert self._artifact_dir is not None
        return (
            self._artifact_dir
            / "masks"
            / "raw"
            / f"{frame_idx:06d}"
            / f"{track_id}.png"
        )

    def _final_mask_path(self, frame_idx: int, track_id: int) -> Path:
        assert self._artifact_dir is not None
        return self._artifact_dir / "masks" / f"{frame_idx:06d}" / f"{track_id}.png"

    @staticmethod
    def _filter_detections(
        detections: tuple[PlayerDetection, ...],
        detection_filter: Callable[[PlayerDetection], bool] | None,
    ) -> tuple[PlayerDetection, ...]:
        if detection_filter is None:
            return detections
        return tuple(detection for detection in detections if detection_filter(detection))

    def _known_masks(
        self, masks: tuple[SamMaskPrediction, ...]
    ) -> tuple[SamMaskPrediction, ...]:
        return tuple(mask for mask in masks if mask.track_id in self.track_manager.tracks)

    @staticmethod
    def _filter_masks(
        masks: tuple[SamMaskPrediction, ...],
        mask_filter: Callable[[SamMaskPrediction], bool] | None,
    ) -> tuple[SamMaskPrediction, ...]:
        if mask_filter is None:
            return masks
        return tuple(mask for mask in masks if mask_filter(mask))

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


def statistics_mean(values: Any) -> float:
    """Mean with a useful zero default for empty detector-history windows."""
    items = tuple(values)
    return sum(items) / len(items) if items else 0.0
