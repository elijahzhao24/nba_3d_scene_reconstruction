from __future__ import annotations

import unittest

import numpy as np

from nba_3d_scene_reconstruction.tracking.association_engine import (
    PlayerAssociationEngine,
)
from nba_3d_scene_reconstruction.tracking.pipeline import PlayerTrackingPipeline
from nba_3d_scene_reconstruction.tracking.schemas import (
    PlayerDetection,
    SamMaskPrediction,
    TrackStatus,
)
from nba_3d_scene_reconstruction.tracking.track_manager import PlayerTrackManager


def detection(frame_idx: int, box: tuple[float, float, float, float]):
    return PlayerDetection(frame_idx, box, 0.9, 1)


def mask(frame_idx: int, track_id: int, box: tuple[int, int, int, int]):
    value = np.zeros((100, 100), dtype=np.bool_)
    x1, y1, x2, y2 = box
    value[y1:y2, x1:x2] = True
    return SamMaskPrediction(frame_idx, track_id, value)


class FakeDetector:
    def __init__(self, outputs: dict[int, tuple[PlayerDetection, ...]]) -> None:
        self.outputs = outputs
        self.calls: list[int] = []

    def detect(self, frame: object, frame_idx: int):
        self.calls.append(frame_idx)
        return self.outputs.get(frame_idx, ())


class FakeSamTracker:
    def __init__(self, outputs: dict[int, tuple[SamMaskPrediction, ...]]) -> None:
        self.outputs = outputs
        self.started_with: str | None = None
        self.prompts: list[tuple[int, int, tuple[float, float, float, float]]] = []
        self.removed_ids: list[int] = []

    def start_segment(self, frames_dir: str) -> None:
        self.started_with = frames_dir

    def prompt_player(self, frame_idx: int, track_id: int, bbox_xyxy) -> None:
        self.prompts.append((frame_idx, track_id, bbox_xyxy))

    def propagate_frame(self, frame_idx: int):
        return self.outputs.get(frame_idx, ())

    def remove_player(self, track_id: int) -> None:
        self.removed_ids.append(track_id)


class PlayerTrackingPipelineTest(unittest.TestCase):
    def test_initializes_ids_then_confirms_new_player_at_two_checkpoints(self) -> None:
        detector = FakeDetector(
            {
                0: (detection(0, (10, 20, 30, 60)),),
                2: (
                    detection(2, (10, 20, 30, 60)),
                    detection(2, (10.2, 20.2, 30.2, 60.2)),
                    detection(2, (60, 20, 80, 60)),
                ),
                4: (
                    detection(4, (10, 20, 30, 60)),
                    detection(4, (60, 20, 80, 60)),
                ),
            }
        )
        sam = FakeSamTracker(
            {
                0: (mask(0, 1, (10, 20, 30, 60)),),
                1: (mask(1, 1, (10, 20, 30, 60)),),
                2: (mask(2, 1, (10, 20, 30, 60)),),
                3: (mask(3, 1, (10, 20, 30, 60)),),
                4: (mask(4, 1, (10, 20, 30, 60)),),
            }
        )
        manager = PlayerTrackManager("segment-1")
        pipeline = PlayerTrackingPipeline(
            detector=detector,
            sam_tracker=sam,
            association_engine=PlayerAssociationEngine(),
            track_manager=manager,
            detector_interval=2,
        )

        pipeline.start_segment("frames")
        pipeline.process_frame(object(), 0)
        pipeline.process_frame(object(), 1)
        pipeline.process_frame(object(), 2)
        self.assertEqual(set(manager.tracks), {1})
        pipeline.process_frame(object(), 3)
        pipeline.process_frame(object(), 4)

        observations = {item.track_id: item for item in pipeline.observations}
        self.assertEqual(observations[1].footpoint_xy, (19.5, 59))
        self.assertEqual(observations[1].detection_confidence, 0.9)
        # A newly prompted track uses its detector box until SAM returns a mask.
        self.assertFalse(observations[2].visible)
        self.assertEqual(observations[2].footpoint_xy, (70.0, 60.0))
        self.assertIn("bbox_footpoint_fallback", observations[2].quality_flags)
        self.assertEqual(observations[2].detection_confidence, 0.9)

        self.assertEqual(detector.calls, [0, 2, 4])
        self.assertEqual(set(manager.tracks), {1, 2})
        self.assertEqual(
            sam.prompts,
            [
                (0, 1, (10, 20, 30, 60)),
                (4, 2, (60, 20, 80, 60)),
            ],
        )

    def test_caps_initial_tracks_at_ten_highest_confidence_players(self) -> None:
        detections = tuple(
            PlayerDetection(
                frame_idx=0,
                bbox_xyxy=(float(index), 0.0, float(index + 1), 10.0),
                confidence=index / 20,
                class_id=1,
            )
            for index in range(12)
        )
        detector = FakeDetector({0: detections})
        sam = FakeSamTracker({})
        pipeline = PlayerTrackingPipeline(
            detector=detector,
            sam_tracker=sam,
            association_engine=PlayerAssociationEngine(),
            track_manager=PlayerTrackManager("segment-1"),
        )

        pipeline.start_segment("frames")
        pipeline.process_frame(object(), 0)

        self.assertEqual(len(pipeline.track_manager.tracks), 10)
        prompted_boxes = {prompt[2] for prompt in sam.prompts}
        self.assertNotIn((0.0, 0.0, 1.0, 10.0), prompted_boxes)
        self.assertNotIn((1.0, 0.0, 2.0, 10.0), prompted_boxes)

    def test_marks_missing_tracks_and_retires_them(self) -> None:
        detector = FakeDetector({0: (detection(0, (10, 20, 30, 60)),)})
        sam = FakeSamTracker({})
        manager = PlayerTrackManager("segment-1")
        manager.MAX_MISSING_FRAMES = 1
        pipeline = PlayerTrackingPipeline(
            detector=detector,
            sam_tracker=sam,
            association_engine=PlayerAssociationEngine(),
            track_manager=manager,
        )

        pipeline.start_segment("frames")
        pipeline.process_frame(object(), 0)
        pipeline.process_frame(object(), 1)
        self.assertEqual(manager.tracks[1].status, TrackStatus.MISSING)
        self.assertEqual(sam.removed_ids, [])

        pipeline.process_frame(object(), 2)
        self.assertEqual(manager.tracks, {})
        self.assertEqual(pipeline.observations, ())
        self.assertEqual(sam.removed_ids, [1])

    def test_requires_start_and_sequential_frames(self) -> None:
        pipeline = PlayerTrackingPipeline(
            detector=FakeDetector({}),
            sam_tracker=FakeSamTracker({}),
            association_engine=PlayerAssociationEngine(),
            track_manager=PlayerTrackManager("segment-1"),
        )

        with self.assertRaisesRegex(RuntimeError, "start_segment"):
            pipeline.process_frame(object(), 0)

        pipeline.start_segment("frames")
        with self.assertRaisesRegex(ValueError, "expected frame 0"):
            pipeline.process_frame(object(), 1)


if __name__ == "__main__":
    unittest.main()
