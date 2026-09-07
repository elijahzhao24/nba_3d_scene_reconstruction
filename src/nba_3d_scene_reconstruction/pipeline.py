"""Coordinate tracking, court calibration, and projection for one segment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .court.calibration import CourtCalibrator, CourtHomographyEstimator
from .court.detector import RoboflowCourtDetector
from .court.projector import PlayerCourtProjector
from .court.schemas import CourtCalibration, CourtDetection, PlayerCourtPosition
from .tracking.pipeline import PlayerTrackingPipeline
from .tracking.schemas import PlayerObservation, SamMaskPrediction


@dataclass(frozen=True)
class SceneFrame:
    """One synchronized frame; callers can persist each subsystem separately."""

    masks: tuple[SamMaskPrediction, ...]
    observations: tuple[PlayerObservation, ...]
    court_detection: CourtDetection | None
    calibration: CourtCalibration
    positions: tuple[PlayerCourtPosition, ...]


class SceneReconstructionPipeline:
    """Detect the court at frame zero and every five frames by default."""

    def __init__(
        self,
        *,
        tracking_pipeline: PlayerTrackingPipeline,
        court_detector: RoboflowCourtDetector,
        calibrator: CourtCalibrator | None = None,
        projector: PlayerCourtProjector | None = None,
        court_detector_interval: int = 5,
    ) -> None:
        if not isinstance(court_detector_interval, int) or court_detector_interval <= 0:
            raise ValueError("court_detector_interval must be a positive integer")
        self.tracking_pipeline = tracking_pipeline
        self.court_detector = court_detector
        self.calibrator = calibrator or CourtCalibrator(
            CourtHomographyEstimator(detector_configuration=court_detector.configuration)
        )
        self.projector = projector or PlayerCourtProjector(
            landmark_points_cm=self.calibrator.estimator.detector_configuration.landmark_points_cm
        )
        self.court_detector_interval = court_detector_interval
        self.segment_id = tracking_pipeline.track_manager.segment_id

    def start_segment(self, frames_dir: str) -> None:
        """Start a fresh tracker and clear calibration for this segment."""
        self.tracking_pipeline.start_segment(frames_dir)
        self.calibrator.reset(self.segment_id)

    def process_frame(self, frame: Any, frame_idx: int) -> SceneFrame:
        """Process sequential original-resolution frames starting at zero."""
        masks = self.tracking_pipeline.process_frame(frame, frame_idx)
        detection = None
        if frame_idx % self.court_detector_interval == 0:
            detection = self.court_detector.detect(frame, frame_idx)
            calibration = self.calibrator.update(
                detection, segment_id=self.segment_id, frame_idx=frame_idx,
            )
        else:
            calibration = self.calibrator.current(
                segment_id=self.segment_id, frame_idx=frame_idx,
            )
        observations = self.tracking_pipeline.observations
        return SceneFrame(
            masks=masks,
            observations=observations,
            court_detection=detection,
            calibration=calibration,
            positions=self.projector.project(observations, calibration),
        )
