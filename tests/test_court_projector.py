from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from nba_3d_scene_reconstruction.court.calibration import CourtCalibrator, project_points
from nba_3d_scene_reconstruction.court.projector import PlayerCourtProjector
from nba_3d_scene_reconstruction.tracking.schemas import ObservationSource, PlayerObservation
from test_court_calibration import KNOWN_COURT_TO_IMAGE, synthetic_detection


class PlayerCourtProjectorTest(unittest.TestCase):
    def setUp(self):
        self.calibrator = CourtCalibrator()
        self.calibration = self.calibrator.update(
            synthetic_detection(), segment_id="segment-1", frame_idx=0,
        )
        image_xy = project_points(np.asarray([[1000.0, 700.0]]), KNOWN_COURT_TO_IMAGE)[0]
        self.observation = PlayerObservation(
            segment_id="segment-1", frame_idx=0, timestamp_seconds=0.0,
            track_id=7, visible=True, source=ObservationSource.SAM2_PROPAGATION,
            footpoint_xy=tuple(image_xy),
        )
        self.projector = PlayerCourtProjector()

    def project(self, observation=None, calibration=None):
        return self.projector.project(
            (observation or self.observation,), calibration or self.calibration,
        )[0]

    def test_recovers_court_centimeters_and_held_provenance(self):
        position = self.project()
        np.testing.assert_allclose(position.raw_court_xy, (1000, 700), atol=1e-3)
        self.assertEqual(position.track_id, 7)
        held = self.calibrator.current(segment_id="segment-1", frame_idx=3)
        position = self.project(replace(self.observation, frame_idx=3), held)
        self.assertEqual(position.calibration_source_frame_idx, 0)
        self.assertEqual(position.calibration_age_frames, 3)

    def test_missing_player_footpoint_and_expired_calibration(self):
        missing_but_projected = self.project(replace(self.observation, visible=False))
        self.assertIsNotNone(missing_but_projected.raw_court_xy)
        self.assertIn("player_missing", missing_but_projected.quality_flags)
        for observation, flag in (
            (replace(self.observation, visible=False, footpoint_xy=None), "player_missing"),
            (replace(self.observation, footpoint_xy=None), "footpoint_missing"),
            (replace(self.observation, footpoint_xy=(float("nan"), 1)), "invalid_footpoint"),
        ):
            position = self.project(observation)
            self.assertIsNone(position.raw_court_xy)
            self.assertIn(flag, position.quality_flags)
        expired = self.calibrator.current(segment_id="segment-1", frame_idx=31)
        position = self.project(replace(self.observation, frame_idx=31), expired)
        self.assertIsNone(position.raw_court_xy)
        self.assertIn("calibration_expired", position.quality_flags)

    def test_horizon_and_outside_court_are_rejected_without_clamping(self):
        horizon = replace(self.calibration, image_to_court=((1, 0, 0), (0, 1, 0), (0, 1, -1)))
        position = self.project(replace(self.observation, footpoint_xy=(3, 1)), horizon)
        self.assertIsNone(position.raw_court_xy)
        self.assertIn("projection_at_infinity", position.quality_flags)
        identity = replace(self.calibration, image_to_court=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        self.assertEqual(self.project(replace(self.observation, footpoint_xy=(-50, 0)), identity).raw_court_xy, (-50, 0))
        position = self.project(replace(self.observation, footpoint_xy=(-101, 0)), identity)
        self.assertIsNone(position.raw_court_xy)
        self.assertIn("outside_court", position.quality_flags)

    def test_checks_image_point_against_calibrated_court(self):
        inside = project_points(
            np.asarray([[1000.0, 700.0]]), KNOWN_COURT_TO_IMAGE,
        )[0]
        outside = project_points(
            np.asarray([[1000.0, -200.0]]), KNOWN_COURT_TO_IMAGE,
        )[0]

        self.assertTrue(
            self.projector.contains_image_point(tuple(inside), self.calibration)
        )
        self.assertFalse(
            self.projector.contains_image_point(tuple(outside), self.calibration)
        )

    def test_rejects_cross_frame_or_segment_join(self):
        for observation in (replace(self.observation, frame_idx=1),
                            replace(self.observation, segment_id="other")):
            with self.assertRaisesRegex(ValueError, "must match"):
                self.project(observation)
