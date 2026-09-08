from __future__ import annotations

import unittest

import numpy as np

from nba_3d_scene_reconstruction.court.schemas import PlayerCourtPosition
from nba_3d_scene_reconstruction.court.smoothing import (
    DEFAULT_MINIMUM_JUMP_DISTANCE_CM,
    TrajectorySmoothingConfiguration,
    clean_paths,
    clean_player_position_frames,
)


def position(
    frame_idx: int,
    xy: tuple[float, float] | None,
    *,
    track_id: int = 1,
) -> PlayerCourtPosition:
    return PlayerCourtPosition(
        segment_id="segment-1",
        frame_idx=frame_idx,
        timestamp_seconds=frame_idx / 30,
        track_id=track_id,
        footpoint_image_xy=None,
        raw_court_xy=xy,
        calibration_source_frame_idx=frame_idx,
        calibration_age_frames=0,
    )


class CleanPathsTest(unittest.TestCase):
    def test_ports_reference_notebook_units_and_defaults(self) -> None:
        configuration = TrajectorySmoothingConfiguration()

        self.assertAlmostEqual(DEFAULT_MINIMUM_JUMP_DISTANCE_CM, 18.288)
        self.assertEqual(configuration.jump_sigma, 3.5)
        self.assertEqual(configuration.maximum_jump_run, 18)
        self.assertEqual(configuration.jump_padding_frames, 2)
        self.assertEqual(configuration.smoothing_window, 9)
        self.assertEqual(configuration.smoothing_polynomial, 2)

    def test_removes_a_teleport_and_smooths_the_path(self) -> None:
        x = np.arange(15, dtype=float) * 5
        raw = np.stack((x, np.zeros_like(x)), axis=1)[:, None, :]
        raw[7, 0] += (400, 300)

        cleaned, edited = clean_paths(raw)

        self.assertTrue(edited[7, 0])
        self.assertLess(
            np.linalg.norm(np.diff(cleaned[:, 0], axis=0), axis=1).max(),
            10.0,
        )
        np.testing.assert_allclose(cleaned[:, 0, 0], x, atol=1e-6)

    def test_rejects_invalid_shapes_and_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            clean_paths(np.zeros((3, 2)))
        with self.assertRaisesRegex(ValueError, "finite"):
            clean_paths(np.full((3, 1, 2), np.nan))
        with self.assertRaisesRegex(ValueError, "odd"):
            TrajectorySmoothingConfiguration(smoothing_window=8)


class PlayerPositionFrameCleaningTest(unittest.TestCase):
    def test_preserves_raw_values_and_interpolates_a_short_gap(self) -> None:
        frames = tuple(
            (position(frame_idx, None if frame_idx == 5 else (frame_idx * 5, 0)),)
            for frame_idx in range(11)
        )

        cleaned = clean_player_position_frames(frames)

        gap = cleaned[5][0]
        self.assertIsNone(gap.raw_court_xy)
        np.testing.assert_allclose(gap.clean_court_xy, (25, 0), atol=1e-6)
        self.assertTrue(gap.position_was_edited)
        self.assertIn("trajectory_gap_interpolated", gap.quality_flags)

    def test_does_not_bridge_a_long_missing_gap(self) -> None:
        config = TrajectorySmoothingConfiguration(maximum_interpolation_gap=2)
        frames = tuple(
            (
                position(
                    frame_idx,
                    (float(frame_idx), 0.0) if frame_idx in (0, 1, 5, 6) else None,
                ),
            )
            for frame_idx in range(7)
        )

        cleaned = clean_player_position_frames(frames, configuration=config)

        self.assertIsNone(cleaned[2][0].clean_court_xy)
        self.assertIsNone(cleaned[3][0].clean_court_xy)
        self.assertIsNone(cleaned[4][0].clean_court_xy)
        self.assertIsNotNone(cleaned[1][0].clean_court_xy)
        self.assertIsNotNone(cleaned[5][0].clean_court_xy)


if __name__ == "__main__":
    unittest.main()
