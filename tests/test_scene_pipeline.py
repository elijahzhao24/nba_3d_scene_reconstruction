from __future__ import annotations

import unittest

import numpy as np
from test_court_calibration import KNOWN_COURT_TO_IMAGE, synthetic_detection
from test_pipeline import FakeDetector, FakeSamTracker, detection

from nba_3d_scene_reconstruction.court.calibration import (
    CourtCalibrator,
    CourtHomographyEstimator,
)
from nba_3d_scene_reconstruction.court.configuration import (
    CourtCalibrationConfiguration,
    CourtDetectorConfiguration,
)
from nba_3d_scene_reconstruction.court.schemas import CalibrationSource
from nba_3d_scene_reconstruction.pipeline import SceneReconstructionPipeline
from nba_3d_scene_reconstruction.tracking.association_engine import (
    PlayerAssociationEngine,
)
from nba_3d_scene_reconstruction.tracking.pipeline import PlayerTrackingPipeline
from nba_3d_scene_reconstruction.tracking.schemas import SamMaskPrediction
from nba_3d_scene_reconstruction.tracking.track_manager import PlayerTrackManager


class FakeCourtDetector:
    configuration = CourtDetectorConfiguration()

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def detect(self, frame, frame_idx):
        self.calls.append(frame_idx)
        return self.outputs.get(frame_idx)


def make_scene(court_detector, calibrator=None, court_detector_interval=1):
    player_mask = np.zeros((480, 640), dtype=bool)
    player_mask[200:240, 300:320] = True
    tracking = PlayerTrackingPipeline(
        detector=FakeDetector({0: (detection(0, (300, 200, 320, 240)),)}),
        sam_tracker=FakeSamTracker(
            {i: (SamMaskPrediction(i, 1, player_mask),) for i in range(16)}
        ),
        association_engine=PlayerAssociationEngine(),
        track_manager=PlayerTrackManager("segment-1"),
        fps=25.0,
    )
    return SceneReconstructionPipeline(
        tracking_pipeline=tracking,
        court_detector=court_detector,
        calibrator=calibrator,
        court_detector_interval=court_detector_interval,
    )


class ScenePipelineTest(unittest.TestCase):
    def test_five_frame_checkpoints_hold_expire_and_recover(self):
        court = FakeCourtDetector(
            {0: synthetic_detection(0), 15: synthetic_detection(15)}
        )
        calibrator = CourtCalibrator(
            CourtHomographyEstimator(
                calibration_configuration=CourtCalibrationConfiguration(
                    maximum_calibration_age_frames=10,
                ),
            )
        )
        scene = make_scene(court, calibrator, court_detector_interval=5)
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

    def test_default_runs_court_detection_every_frame(self):
        court = FakeCourtDetector(
            {index: synthetic_detection(index) for index in range(3)}
        )
        scene = make_scene(court)
        scene.start_segment("frames")

        results = [scene.process_frame(object(), index) for index in range(3)]

        self.assertEqual(court.calls, [0, 1, 2])
        self.assertTrue(
            all(
                result.calibration.source is CalibrationSource.DETECTED
                for result in results
            )
        )

    def test_filters_new_tracks_by_current_court_projection(self):
        from nba_3d_scene_reconstruction.court.calibration import project_points

        inside, outside = project_points(
            np.asarray([[1000.0, 700.0], [1000.0, -200.0]]),
            KNOWN_COURT_TO_IMAGE,
        )

        def player_box(anchor):
            x, y = anchor
            return (float(x - 10), float(y - 50), float(x + 10), float(y))

        tracking = PlayerTrackingPipeline(
            detector=FakeDetector(
                {
                    0: (
                        detection(0, player_box(inside)),
                        detection(0, player_box(outside)),
                    ),
                }
            ),
            sam_tracker=FakeSamTracker({}),
            association_engine=PlayerAssociationEngine(),
            track_manager=PlayerTrackManager("segment-1"),
        )
        scene = SceneReconstructionPipeline(
            tracking_pipeline=tracking,
            court_detector=FakeCourtDetector({0: synthetic_detection(0)}),
        )
        scene.start_segment("frames")

        scene.process_frame(object(), 0)

        self.assertEqual(len(tracking.track_manager.tracks), 1)
        only_track = next(iter(tracking.track_manager.tracks.values()))
        self.assertEqual(only_track.latest_bbox_xyxy, player_box(inside))

    def test_filters_propagated_masks_far_outside_the_court(self):
        from nba_3d_scene_reconstruction.court.calibration import project_points

        outside = project_points(
            np.asarray([[1000.0, -200.0]]),
            KNOWN_COURT_TO_IMAGE,
        )[0]
        x, y = np.rint(outside).astype(int)
        outside_mask = np.zeros((480, 640), dtype=bool)
        outside_mask[y - 10 : y + 1, x - 5 : x + 6] = True
        tracking = PlayerTrackingPipeline(
            detector=FakeDetector({0: (detection(0, (300, 200, 320, 240)),)}),
            sam_tracker=FakeSamTracker(
                {
                    0: (SamMaskPrediction(0, 1, outside_mask),),
                }
            ),
            association_engine=PlayerAssociationEngine(),
            track_manager=PlayerTrackManager("segment-1"),
        )
        scene = SceneReconstructionPipeline(
            tracking_pipeline=tracking,
            court_detector=FakeCourtDetector({0: synthetic_detection(0)}),
        )
        scene.start_segment("frames")

        result = scene.process_frame(object(), 0)

        self.assertEqual(result.masks, ())
        self.assertFalse(result.observations[0].visible)

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
