from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nba_3d_scene_reconstruction.court.pose_camera import estimate_pose_camera
from nba_3d_scene_reconstruction.scene_export import build_scene

_IMAGE_WIDTH = 1280
_IMAGE_HEIGHT = 720


def court_homography(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    right = np.array([math.cos(yaw), 0, math.sin(yaw)])
    forward = np.array(
        [math.sin(yaw) * math.cos(pitch), -math.sin(pitch), -math.cos(yaw) * math.cos(pitch)]
    )
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    # The source camera is 35 m from center court and aims at center court.
    position = np.array([1432.5, 0, 762]) - 3500 * forward
    intrinsics = np.array(
        [[1800, 0, _IMAGE_WIDTH / 2], [0, 1800, _IMAGE_HEIGHT / 2], [0, 0, 1]]
    )
    return intrinsics @ np.column_stack([rotation[:, 0], rotation[:, 2], -rotation @ position])


class PoseCameraTest(unittest.TestCase):
    def test_recovers_camera_angles_and_resolves_homography_sign(self):
        for yaw, pitch in ((-37.5, 15.0), (0.0, 21.0), (145.0, 34.0)):
            with self.subTest(yaw=yaw, pitch=pitch):
                camera = estimate_pose_camera(
                    [-2 * court_homography(yaw, pitch)], _IMAGE_WIDTH, _IMAGE_HEIGHT
                )
                self.assertIsNotNone(camera)
                self.assertAlmostEqual(camera.yaw_deg, yaw)
                self.assertAlmostEqual(camera.pitch_deg, pitch)

    def test_rejects_unphysical_or_unreliable_views(self):
        self.assertIsNone(
            estimate_pose_camera(
                [np.eye(3), court_homography(30.0, -15.0), np.full((3, 3), np.nan)],
                _IMAGE_WIDTH,
                _IMAGE_HEIGHT,
            )
        )
        self.assertIsNone(
            estimate_pose_camera(
                [court_homography(yaw, 20.0) for yaw in (20.0, -160.0)],
                _IMAGE_WIDTH,
                _IMAGE_HEIGHT,
            )
        )

    def test_uses_consistent_directions_across_angle_wrap(self):
        camera = estimate_pose_camera(
            [court_homography(yaw, 20.0) for yaw in (178.0, 179.0, -179.0, -178.0, 15.0)],
            _IMAGE_WIDTH,
            _IMAGE_HEIGHT,
        )
        self.assertIsNotNone(camera)
        self.assertAlmostEqual(abs(camera.yaw_deg), 180.0)
        self.assertAlmostEqual(camera.pitch_deg, 20.0, places=2)

    def test_scene_exports_camera_only_from_usable_detected_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "clip_id": "game", "segment_id": "seg", "fps": 30.0,
                "width": _IMAGE_WIDTH, "height": _IMAGE_HEIGHT, "frame_count": 1,
                "source_path": "not-needed.webm", "frames_dir": "not-needed",
            }
            track = {"segment_id": "seg", "frame_idx": 0, "timestamp_seconds": 0.0, "track_id": 1}
            files = {
                "video_manifest.json": manifest,
                "observations.jsonl": {**track, "visible": True, "source": "sam2_propagation"},
                "player_court_positions_clean.jsonl": {
                    **track, "footpoint_image_xy": [40.0, 80.0], "raw_court_xy": [40.0, 80.0],
                    "calibration_source_frame_idx": 0, "calibration_age_frames": 0,
                    "clean_court_xy": [40.0, 80.0],
                },
                "poses_3d_local.jsonl": {
                    **track, "clip_id": "game", "sequence_id": "a", "layout": "h36m17",
                    "joints": [
                        {"name": "pelvis", "xyz": [0.0, 0.0, 0.0],
                         "input_score": 1.0, "input_observed": True},
                    ],
                },
            }
            for name, row in files.items():
                (root / name).write_text(f"{json.dumps(row)}\n", encoding="utf-8")

            calibration = {
                **track, "valid": True, "source": "detected",
                "court_to_image": court_homography(25.0, 18.0).tolist(),
            }
            for source in ("detected", "held"):
                with self.subTest(source=source):
                    calibration["source"] = source
                    (root / "calibrations.jsonl").write_text(
                        f"{json.dumps(calibration)}\n", encoding="utf-8",
                    )
                    scene = build_scene(root)
                    pose = scene["pose"]
                    self.assertEqual(scene["frames"][0]["players"][0]["pose_sequence"], "a")
                    if source == "detected":
                        self.assertAlmostEqual(pose["camera"]["yaw_deg"], 25.0)
                        self.assertAlmostEqual(pose["camera"]["pitch_deg"], 18.0)
                    else:
                        self.assertNotIn("camera", pose)
