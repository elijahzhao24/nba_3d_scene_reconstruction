from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from nba_3d_scene_reconstruction.ingest_video import processNewVideo
from nba_3d_scene_reconstruction.tracking.schemas import SamMaskPrediction
from nba_3d_scene_reconstruction.tracking_demo import render_tracking_video


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
