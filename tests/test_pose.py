from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from nba_3d_scene_reconstruction.pose.debug import render_pose_3d_videos
from nba_3d_scene_reconstruction.pose.estimator import MMPosePose2DEstimator
from nba_3d_scene_reconstruction.pose.io import read_poses_2d, write_jsonl
from nba_3d_scene_reconstruction.pose.lifter import (
    make_root_relative,
    normalize_sequence,
    temporal_windows,
)
from nba_3d_scene_reconstruction.pose.pipeline import (
    estimate_tracked_poses,
    lift_prepared_sequences,
)
from nba_3d_scene_reconstruction.pose.preparation import prepare_pose_sequences
from nba_3d_scene_reconstruction.pose.schemas import (
    Pose2D,
    Pose3D,
    PoseJoint2D,
    PoseJoint3D,
)
from nba_3d_scene_reconstruction.pose.skeletons import H36M_17, HALPE_26
from nba_3d_scene_reconstruction.tracking.schemas import (
    ObservationSource,
    PlayerObservation,
    VideoManifest,
)


def pose(frame: int, track: int = 4, *, low_joint: str | None = None) -> Pose2D:
    joints = tuple(
        PoseJoint2D(
            name,
            (float(index * 10 + frame), float(index * 10 + frame + 1)),
            0.1 if name == low_joint else 0.9,
            True,
        )
        for index, name in enumerate(HALPE_26.joints)
    )
    return Pose2D("clip", "segment", frame, frame / 30, track, HALPE_26.id, joints)


class FakeEstimator:
    def estimate(self, frame, observations, *, clip_id):
        return tuple(pose(item.frame_idx, item.track_id) for item in observations)


class FakeLifter:
    def lift(self, sequence):
        result = []
        for item in sequence:
            joints = tuple(
                PoseJoint3D(
                    name,
                    (float(i), float(i + 1), float(i + 2)),
                    joint.score,
                    joint.observed,
                )
                for i, (name, joint) in enumerate(zip(H36M_17.joints, item.joints))
            )
            result.append(
                Pose3D(
                    item.clip_id,
                    item.segment_id,
                    item.frame_idx,
                    item.timestamp_seconds,
                    item.track_id,
                    item.sequence_id,
                    H36M_17.id,
                    joints,
                )
            )
        return tuple(result)


class PosePipelineTest(unittest.TestCase):
    def test_halpe_conversion_has_exact_order_sides_and_derived_spine(self):
        prepared = prepare_pose_sequences((pose(0),), fps=30)
        self.assertEqual(tuple(j.name for j in prepared[0].joints), H36M_17.joints)
        source = {joint.name: joint for joint in pose(0).joints}
        output = {joint.name: joint for joint in prepared[0].joints}
        self.assertEqual(output["left_wrist"].image_xy, source["left_wrist"].image_xy)
        self.assertEqual(output["right_wrist"].image_xy, source["right_wrist"].image_xy)
        self.assertEqual(output["spine"].provenance, "derived")
        self.assertEqual(H36M_17.semantic_endpoints["left_hand"], "left_wrist")

    def test_short_bounded_low_confidence_gap_is_interpolated(self):
        poses = (pose(0), pose(1, low_joint="left_wrist"), pose(2))
        prepared = prepare_pose_sequences(poses, fps=30, maximum_gap_seconds=0.1)
        wrist = {joint.name: joint for joint in prepared[1].joints}["left_wrist"]
        self.assertEqual(wrist.provenance, "interpolated")
        self.assertEqual(wrist.score, 0.0)
        self.assertFalse(wrist.observed)
        self.assertEqual(wrist.image_xy, (91.0, 92.0))

    def test_gap_beyond_limit_stays_finite_and_missing(self):
        poses = tuple(
            pose(i, low_joint="left_wrist") if 1 <= i <= 4 else pose(i)
            for i in range(6)
        )
        prepared = prepare_pose_sequences(poses, fps=30, maximum_gap_seconds=0.1)
        wrist = {joint.name: joint for joint in prepared[2].joints}["left_wrist"]
        self.assertEqual(wrist.image_xy, (0.0, 0.0))
        self.assertEqual(wrist.score, 0.0)

    def test_windows_cover_tail_and_overlap(self):
        windows = temporal_windows(500)
        self.assertEqual((windows[0].start, windows[0].stop), (0, 243))
        self.assertEqual(windows[-1].stop, 500)
        self.assertGreater(sum(100 in window for window in windows), 1)
        self.assertEqual(temporal_windows(12), (range(12),))

    def test_normalization_uses_valid_joints_and_is_finite(self):
        values = np.zeros((2, 17, 3), dtype=np.float32)
        values[:, :3] = [[[10, 20, 1], [30, 40, 1], [50, 20, 1]]] * 2
        result = normalize_sequence(values)
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(abs(result[..., :2]).max(), 1)

    def test_root_relative_output_places_pelvis_at_origin(self):
        coordinates = np.arange(2 * 17 * 3, dtype=float).reshape(2, 17, 3)
        rooted = make_root_relative(coordinates)
        np.testing.assert_array_equal(rooted[:, 0], np.zeros((2, 3)))
        np.testing.assert_array_equal(rooted[:, 1], np.full((2, 3), 3.0))

    def test_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.jsonl"
            write_jsonl(path, (pose(0),))
            restored = read_poses_2d(path)
            self.assertEqual(restored, (pose(0),))

    def test_model_independent_end_to_end_preserves_tracks_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory) / "frames"
            frames.mkdir()
            for index in range(2):
                cv2.imwrite(
                    str(frames / f"{index:06d}.jpg"), np.zeros((20, 20, 3), np.uint8)
                )
            manifest = VideoManifest(
                "clip", "segment", "source", str(frames), 30, 20, 20, 2
            )
            observations = tuple(
                PlayerObservation(
                    "segment",
                    i,
                    i / 30,
                    track,
                    True,
                    ObservationSource.SAM2_PROPAGATION,
                    (0, 0, 10, 19),
                )
                for i in range(2)
                for track in (3, 7)
            )
            raw = estimate_tracked_poses(manifest, observations, FakeEstimator())
            prepared = prepare_pose_sequences(raw, fps=manifest.fps)
            lifted = lift_prepared_sequences(prepared, FakeLifter())
            self.assertEqual(
                [(p.frame_idx, p.track_id) for p in lifted],
                [(0, 3), (0, 7), (1, 3), (1, 7)],
            )

    def test_mmpose_adapter_passes_boxes_and_preserves_image_coordinates(self):
        captured = {}

        def inference_topdown(model, frame, *, bboxes):
            captured["boxes"] = bboxes.copy()
            points = np.arange(52, dtype=float).reshape(1, 26, 2)
            scores = np.full((1, 26), 0.8)
            instances = types.SimpleNamespace(keypoints=points, keypoint_scores=scores)
            return [types.SimpleNamespace(pred_instances=instances)]

        package = types.ModuleType("mmpose")
        api = types.ModuleType("mmpose.apis")
        utilities = types.ModuleType("mmpose.utils")
        api.inference_topdown = inference_topdown
        utilities.register_all_modules = lambda: None
        estimator = MMPosePose2DEstimator.__new__(MMPosePose2DEstimator)
        estimator.model = object()
        visible = PlayerObservation(
            "segment",
            2,
            2 / 30,
            8,
            True,
            ObservationSource.SAM2_PROPAGATION,
            (4.0, 5.0, 14.0, 25.0),
        )
        missing = PlayerObservation(
            "segment",
            2,
            2 / 30,
            9,
            False,
            ObservationSource.MISSING,
        )
        with patch.dict(
            "sys.modules",
            {"mmpose": package, "mmpose.apis": api, "mmpose.utils": utilities},
        ):
            result = estimator.estimate(
                np.zeros((30, 30, 3)), (visible, missing), clip_id="clip"
            )
        np.testing.assert_array_equal(captured["boxes"], [[4.0, 5.0, 14.0, 25.0]])
        self.assertEqual(result[0].joints[3].image_xy, (6.0, 7.0))
        self.assertEqual(result[0].track_id, 8)
        self.assertFalse(result[1].visible)
        self.assertIn("pose_input_missing", result[1].quality_flags)

    def test_3d_debug_video_preserves_missing_frame_timing(self):
        prepared = prepare_pose_sequences((pose(0), pose(2)), fps=30)
        lifted = lift_prepared_sequences(prepared, FakeLifter())
        with tempfile.TemporaryDirectory() as directory:
            outputs = render_pose_3d_videos(lifted, directory, fps=30, frame_count=3)
            capture = cv2.VideoCapture(str(outputs[0]))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            capture.release()


if __name__ == "__main__":
    unittest.main()
