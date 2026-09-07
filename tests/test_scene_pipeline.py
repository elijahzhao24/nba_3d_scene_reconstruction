from __future__ import annotations

import unittest

import numpy as np

from nba_3d_scene_reconstruction.court.calibration import CourtCalibrator
from nba_3d_scene_reconstruction.court.configuration import CourtDetectorConfiguration
from nba_3d_scene_reconstruction.court.schemas import CalibrationSource
from nba_3d_scene_reconstruction.pipeline import SceneReconstructionPipeline
from nba_3d_scene_reconstruction.tracking.association_engine import PlayerAssociationEngine
from nba_3d_scene_reconstruction.tracking.pipeline import PlayerTrackingPipeline
from nba_3d_scene_reconstruction.tracking.schemas import SamMaskPrediction
from nba_3d_scene_reconstruction.tracking.track_manager import PlayerTrackManager
from test_court_calibration import synthetic_detection
from test_pipeline import FakeDetector, FakeSamTracker, detection


class FakeCourtDetector:
    configuration = CourtDetectorConfiguration()

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def detect(self, frame, frame_idx):
        self.calls.append(frame_idx)
        return self.outputs.get(frame_idx)


def make_scene(court_detector, calibrator=None):
    player_mask = np.zeros((480, 640), dtype=bool)
    player_mask[200:240, 300:320] = True
    tracking = PlayerTrackingPipeline(
        detector=FakeDetector({0: (detection(0, (300, 200, 320, 240)),)}),
        sam_tracker=FakeSamTracker({
            i: (SamMaskPrediction(i, 1, player_mask),) for i in range(16)
        }),
        association_engine=PlayerAssociationEngine(),
        track_manager=PlayerTrackManager("segment-1"), fps=25.0,
    )
    return SceneReconstructionPipeline(
        tracking_pipeline=tracking, court_detector=court_detector, calibrator=calibrator,
    )


class ScenePipelineTest(unittest.TestCase):
    def test_five_frame_checkpoints_hold_expire_and_recover(self):
        court = FakeCourtDetector({0: synthetic_detection(0), 15: synthetic_detection(15)})
        scene = make_scene(court)
        scene.start_segment("frames")
        results = [scene.process_frame(object(), i) for i in range(16)]
        self.assertEqual(court.calls, [0, 5, 10, 15])
        self.assertEqual(results[0].calibration.source, CalibrationSource.DETECTED)
        self.assertEqual(results[4].calibration.source, CalibrationSource.HELD)
        self.assertEqual(results[10].calibration.age_frames, 10)
        self.assertFalse(results[11].calibration.valid)
        self.assertIsNone(results[11].positions[0].raw_court_xy)
        self.assertEqual(results[15].calibration.source, CalibrationSource.DETECTED)
        self.assertEqual(results[4].observations[0].timestamp_seconds, 4 / 25)
        self.assertEqual(results[4].positions[0].frame_idx, 4)
        self.assertIsNotNone(results[4].positions[0].raw_court_xy)
        self.assertIsNotNone(results[15].positions[0].raw_court_xy)
        self.assertIsNone(results[4].court_detection)

    def test_start_resets_old_calibration_and_bad_frame_does_not_run_court(self):
        calibrator = CourtCalibrator()
        calibrator.update(synthetic_detection(), segment_id="segment-1", frame_idx=0)
        court = FakeCourtDetector({})
        scene = make_scene(court, calibrator)
        with self.assertRaisesRegex(RuntimeError, "start_segment"):
            scene.process_frame(object(), 0)
        scene.start_segment("frames")
        with self.assertRaisesRegex(ValueError, "expected frame 0"):
            scene.process_frame(object(), 1)
        self.assertEqual(court.calls, [])
        result = scene.process_frame(object(), 0)
        self.assertFalse(result.calibration.valid)
