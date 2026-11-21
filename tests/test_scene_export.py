from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from nba_3d_scene_reconstruction.court.schemas import (
    CalibrationSource,
    CourtCalibration,
    PlayerCourtPosition,
)
from nba_3d_scene_reconstruction.pose.io import write_json, write_jsonl
from nba_3d_scene_reconstruction.pose.schemas import Pose3D, PoseJoint3D
from nba_3d_scene_reconstruction.pose.skeletons import H36M_17
from nba_3d_scene_reconstruction.scene_export import build_scene, main
from nba_3d_scene_reconstruction.tracking.schemas import (
    ObservationSource,
    PlayerObservation,
)


class SceneExportTest(unittest.TestCase):
    fps = 29.75
    segment_id = "segment_004"

    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.root = Path(temporary_dir.name)
        self.segment_dir = self.root / "game" / self.segment_id
        self.segment_dir.mkdir(parents=True)

    def write_manifest(self):
        write_json(
            self.segment_dir / "video_manifest.json",
            {
                "clip_id": "game",
                "segment_id": self.segment_id,
                "source_path": "deleted-source.webm",
                "frames_dir": "deleted-frames",
                "fps": self.fps,
                "width": 1280,
                "height": 720,
                "frame_count": 100,
            },
        )

    def observation(self, frame, track, *, visible=True):
        source = (
            ObservationSource.SAM2_PROPAGATION if visible else ObservationSource.MISSING
        )
        return PlayerObservation(
            self.segment_id,
            frame,
            frame / self.fps,
            track,
            visible,
            source,
        )

    def position(self, frame, track, clean_xy):
        return PlayerCourtPosition(
            segment_id=self.segment_id,
            frame_idx=frame,
            timestamp_seconds=frame / self.fps,
            track_id=track,
            footpoint_image_xy=(25.0, 50.0),
            raw_court_xy=(10.0, 20.0),
            calibration_source_frame_idx=frame,
            calibration_age_frames=0,
            clean_court_xy=clean_xy,
        )

    def write_tracking(self, observations, positions):
        write_jsonl(self.segment_dir / "observations.jsonl", observations)
        write_jsonl(
            self.segment_dir / "player_court_positions_clean.jsonl", positions
        )

    def test_cli_exports_visible_clean_positions_on_processed_timeline(self):
        self.write_manifest()
        self.write_tracking(
            (
                self.observation(0, 12),
                self.observation(0, 17, visible=False),
                self.observation(1, 12),
                self.observation(2, 12),
            ),
            (
                self.position(0, 12, (0.0, 0.0)),
                self.position(0, 17, (1432.5, 762.0)),
                self.position(1, 12, None),
                self.position(2, 12, (2865.0, 1524.0)),
            ),
        )
        write_jsonl(
            self.segment_dir / "calibrations.jsonl",
            tuple(
                CourtCalibration(
                    self.segment_id,
                    frame,
                    None,
                    False,
                    CalibrationSource.INVALID,
                    None,
                    None,
                    0,
                    0,
                    0.0,
                    0.0,
                    None,
                    None,
                )
                for frame in range(5)
            ),
        )
        output = self.root / "scene.json"
        with redirect_stdout(io.StringIO()):
            main([str(self.segment_dir), "--output", str(output)])
        scene = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(scene["version"], 1)
        self.assertEqual(scene["clip_id"], "game")
        self.assertEqual(scene["segment_id"], self.segment_id)
        self.assertEqual(scene["units"], "meters")
        self.assertEqual(scene["fps"], self.fps)
        self.assertAlmostEqual(scene["court"]["length_m"], 28.65)
        self.assertAlmostEqual(scene["court"]["width_m"], 15.24)
        frames = scene["frames"]
        self.assertEqual([frame["frame_idx"] for frame in frames], list(range(5)))
        self.assertEqual(
            [frame["t"] for frame in frames], [i / self.fps for i in range(5)]
        )
        for frame, expected in ((0, (-14.325, 0.0, -7.62)), (2, (14.325, 0.0, 7.62))):
            player, = frames[frame]["players"]
            self.assertEqual(player["id"], 12)
            for actual, coordinate in zip(player["p"], expected, strict=True):
                self.assertAlmostEqual(actual, coordinate)
        for frame in (1, 3, 4):
            self.assertEqual(frames[frame]["players"], [])

    def test_pose_metadata_keeps_model_joints_distinct_from_court_positions(self):
        self.write_manifest()
        self.write_tracking(
            (self.observation(0, 4), self.observation(0, 5)),
            (
                self.position(0, 4, (1432.5, 762.0)),
                self.position(0, 5, (1200.0, 800.0)),
            ),
        )
        joints = tuple(
            PoseJoint3D(name, (index / 100, -index / 100, 0.0), 0.9, True)
            for index, name in enumerate(H36M_17.joints)
        )
        write_jsonl(
            self.segment_dir / "poses_3d_local.jsonl",
            (
                Pose3D(
                    "game", self.segment_id, 0, 0.0, 4, "sequence", H36M_17.id, joints
                ),
            ),
        )

        scene = build_scene(self.segment_dir)

        self.assertEqual(
            scene["pose"],
            {
                "layout": H36M_17.id,
                "coordinate_space": "pelvis_relative",
                "units": "model_relative",
                "axes": "motionbert_camera_axes",
                "joints": list(H36M_17.joints),
            },
        )
        players = {player["id"]: player for player in scene["frames"][0]["players"]}
        self.assertEqual(list(players[4]["p"]), [0.0, 0.0, 0.0])
        exported_joints = [tuple(xyz) for xyz in players[4]["joints"]]
        self.assertEqual(
            exported_joints, [joint.xyz for joint in joints]
        )
        self.assertNotIn("joints", players[5])

    def test_legacy_source_supplies_fps_and_sparse_records_define_extent(self):
        self.write_tracking(
            (self.observation(3, 4),),
            (self.position(3, 4, (1432.5, 762.0)),),
        )
        source = self.root / "game.webm"
        with patch(
            "nba_3d_scene_reconstruction.scene_export._read_legacy_video_metadata",
            return_value=(self.fps, 1280, 720),
        ) as read_metadata:
            scene = build_scene(self.segment_dir, source_video=source)

        read_metadata.assert_called_once_with(source)
        self.assertEqual(scene["clip_id"], "game")
        self.assertEqual(scene["segment_id"], self.segment_id)
        self.assertEqual(scene["fps"], self.fps)
        self.assertEqual(
            [frame["frame_idx"] for frame in scene["frames"]], [0, 1, 2, 3]
        )
        self.assertEqual(scene["frames"][0]["players"], [])
        self.assertEqual(scene["frames"][3]["t"], 3 / self.fps)

    def test_incompatible_artifact_timestamps_are_rejected(self):
        self.write_manifest()
        position = self.position(0, 4, (1432.5, 762.0))
        self.write_tracking(
            (self.observation(0, 4),),
            (replace(position, timestamp_seconds=0.1),),
        )

        with self.assertRaisesRegex(ValueError, "timestamp mismatch at frame 0"):
            build_scene(self.segment_dir)
