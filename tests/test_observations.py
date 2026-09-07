from __future__ import annotations

import unittest

import numpy as np

from nba_3d_scene_reconstruction.tracking.observations import build_player_observations
from nba_3d_scene_reconstruction.tracking.schemas import ObservationSource, SamMaskPrediction


class PlayerObservationsTest(unittest.TestCase):
    def build(self, masks=(), **kwargs):
        return build_player_observations(
            masks, segment_id="segment-1", frame_idx=5,
            fps=kwargs.pop("fps", 25.0), track_ids=(1, 2), **kwargs,
        )

    def test_cleans_noise_and_uses_feet_instead_of_full_bbox_center(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:50, 20:40] = True
        mask[50:60, 30:40] = True
        mask[95, 95] = True
        observation, missing = self.build(
            (SamMaskPrediction(5, 1, mask),), detection_confidences={1: 0.9},
        )
        self.assertTrue(observation.visible)
        self.assertEqual(observation.bbox_xyxy, (20, 10, 40, 60))
        self.assertEqual(observation.footpoint_xy, (34.5, 59))
        self.assertEqual(observation.mask_area, 900)
        np.testing.assert_allclose(observation.centroid_xy, (30.05555556, 32.27777778))
        self.assertEqual(observation.timestamp_seconds, 0.2)
        self.assertEqual(observation.detection_confidence, 0.9)
        self.assertIn("disconnected_mask_regions_removed", observation.quality_flags)
        self.assertIsNone(observation.mask_ref)
        self.assertEqual(missing.source, ObservationSource.MISSING)
        self.assertIsNone(missing.footpoint_xy)

    def test_empty_mask_has_no_geometry_or_fabricated_confidence(self):
        observation, _ = self.build((SamMaskPrediction(5, 1, np.zeros((10, 10), bool)),))
        self.assertFalse(observation.visible)
        self.assertIsNone(observation.bbox_xyxy)
        self.assertIsNone(observation.centroid_xy)
        self.assertIsNone(observation.footpoint_xy)
        self.assertIsNone(observation.detection_confidence)
        self.assertIn("empty_mask", observation.quality_flags)

    def test_rejects_wrong_frame_shape_and_fps(self):
        with self.assertRaisesRegex(ValueError, "frame_idx"):
            self.build((SamMaskPrediction(4, 1, np.ones((2, 2), bool)),))
        with self.assertRaisesRegex(ValueError, "2D boolean"):
            self.build((SamMaskPrediction(5, 1, np.ones((1, 2, 2), bool)),))
        for fps in (0, -1, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, "fps"):
                self.build(fps=fps)
