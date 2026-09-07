from __future__ import annotations

import os
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from nba_3d_scene_reconstruction.ingest_video import processNewVideo
from nba_3d_scene_reconstruction.tracking.schemas import SamMaskPrediction, VideoManifest
from nba_3d_scene_reconstruction.tracking_demo import render_tracking_video
from test_court_calibration import synthetic_detection
from test_scene_pipeline import FakeCourtDetector, make_scene


class FakeTrackingPipeline:
    def __init__(self) -> None:
        self.frames_dir: str | None = None
        self.processed_frames: list[int] = []

    def start_segment(self, frames_dir: str) -> None:
        self.frames_dir = frames_dir

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> tuple[SamMaskPrediction, ...]:
        self.processed_frames.append(frame_idx)
        mask = np.zeros(frame.shape[:2], dtype=np.bool_)
        mask[10:35, 15 + frame_idx : 30 + frame_idx] = True
        return (SamMaskPrediction(frame_idx, 7, mask),)


class TrackingDemoTest(unittest.TestCase):
    def test_scene_demo_persists_synchronized_raw_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = root / "frames"
            frames.mkdir()
            for frame_idx in range(3):
                cv2.imwrite(str(frames / f"{frame_idx:06d}.jpg"),
                            np.zeros((480, 640, 3), dtype=np.uint8))
            scene = make_scene(FakeCourtDetector({0: synthetic_detection()}))
            manifest = VideoManifest("clip", "segment-1", "source.mp4", str(frames),
                                     25.0, 640, 480, 3)
            render_tracking_video(manifest, scene, root / "overlay.mp4",
                                  records_dir=root / "records", max_frames=2)
            records = {}
            for name in ("observations", "court_detections", "calibrations",
                         "player_court_positions_raw"):
                records[name] = [json.loads(line) for line in
                                 (root / "records" / f"{name}.jsonl").read_text().splitlines()]
            self.assertEqual(len(records["court_detections"]), 1)
            for name in ("observations", "calibrations", "player_court_positions_raw"):
                self.assertEqual([row["frame_idx"] for row in records[name]], [0, 1])
            position = records["player_court_positions_raw"][1]
            self.assertIsNotNone(position["raw_court_xy"])
            self.assertEqual(position["timestamp_seconds"], 1 / 25)
            self.assertEqual(position["calibration_age_frames"], 1)

    def test_ingests_video_and_renders_mask_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.avi"
            source_writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(source_writer.isOpened())
            for value in (20, 40, 60):
                source_writer.write(
                    np.full((48, 64, 3), value, dtype=np.uint8)
                )
            source_writer.release()

            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                manifest = processNewVideo(str(source))
            finally:
                os.chdir(previous_directory)
            manifest = replace(
                manifest,
                frames_dir=str(root / manifest.frames_dir),
            )

            pipeline = FakeTrackingPipeline()
            output = render_tracking_video(
                manifest,
                pipeline,
                root / "tracking_overlay.webm",
            )

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(pipeline.frames_dir, manifest.frames_dir)
            self.assertEqual(pipeline.processed_frames, [0, 1, 2])

            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            success, rendered_frame = capture.read()
            capture.release()
            self.assertTrue(success)
            self.assertGreater(int(rendered_frame[20, 20].max()), 20)


if __name__ == "__main__":
    unittest.main()
