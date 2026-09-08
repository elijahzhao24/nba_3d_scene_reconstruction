from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from nba_3d_scene_reconstruction.court.debug import CourtDebugRenderer, draw_source_overlay
from nba_3d_scene_reconstruction.court.schemas import CalibrationSource
from nba_3d_scene_reconstruction.pipeline import SceneFrame
from test_court_calibration import synthetic_detection
from test_scene_pipeline import FakeCourtDetector, make_scene


class CourtDebugRendererTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        scene = make_scene(FakeCourtDetector({0: synthetic_detection()}))
        scene.start_segment("frames")
        self.result = scene.process_frame(self.frame, 0)
        self.renderer = CourtDebugRenderer()

    def test_source_overlay_draws_masks_landmarks_and_footpoints(self) -> None:
        overlay = draw_source_overlay(
            self.frame,
            self.result.masks,
            self.result.observations,
            self.result.court_detection,
            self.result.calibration,
        )

        self.assertEqual(overlay.shape, self.frame.shape)
        self.assertGreater(int(np.count_nonzero(overlay)), 500)
        footpoint = self.result.observations[0].footpoint_xy
        assert footpoint is not None
        x, y = map(round, footpoint)
        self.assertGreater(int(overlay[y, x].max()), 0)

    def test_renders_synchronized_composite_with_player_on_court(self) -> None:
        rendered = self.renderer.render(self.frame, self.result, fps=25.0)

        self.assertEqual(rendered.dtype, np.uint8)
        self.assertEqual(rendered.shape[0] % 2, 0)
        self.assertEqual(rendered.shape[1] % 2, 0)
        self.assertGreater(rendered.shape[1], self.frame.shape[1])
        position = self.result.positions[0].raw_court_xy
        assert position is not None
        panel_x, panel_y = self.renderer.court_to_panel(position)
        right_start = rendered.shape[1] - self.renderer.PANEL_WIDTH
        self.assertGreater(int(rendered[panel_y, right_start + panel_x].max()), 0)

    def test_invalid_calibration_renders_without_player_dots(self) -> None:
        invalid = replace(
            self.result.calibration,
            valid=False,
            source=CalibrationSource.INVALID,
            image_to_court=None,
            court_to_image=None,
            source_frame_idx=None,
            age_frames=None,
            quality_flags=("low_inlier_ratio",),
        )
        positions = tuple(replace(position, raw_court_xy=None) for position in self.result.positions)
        result = SceneFrame(
            masks=self.result.masks,
            observations=self.result.observations,
            court_detection=self.result.court_detection,
            calibration=invalid,
            positions=positions,
        )

        rendered = self.renderer.render(self.frame, result, fps=25.0)

        self.assertGreater(int(np.count_nonzero(rendered)), 500)

    def test_rejects_unsynchronized_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "match calibration"):
            draw_source_overlay(
                self.frame,
                self.result.masks,
                self.result.observations,
                self.result.court_detection,
                replace(self.result.calibration, frame_idx=1),
            )
        with self.assertRaisesRegex(ValueError, "fps"):
            self.renderer.draw_birdseye(
                self.result.positions, self.result.calibration, fps=0,
            )
