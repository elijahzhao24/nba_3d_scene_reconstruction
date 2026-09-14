from __future__ import annotations

import unittest

import numpy as np

from nba_3d_scene_reconstruction.tracking.association_engine import (
    PlayerAssociationEngine,
)
from nba_3d_scene_reconstruction.tracking.pipeline import PlayerTrackingPipeline
from nba_3d_scene_reconstruction.tracking.schemas import (
    ObservationSource,
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

    def propagate_range(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        *,
        reverse: bool = False,
        track_ids=None,
    ):
        if reverse:
            indices = range(start_frame_idx, end_frame_idx - 1, -1)
        else:
            indices = range(start_frame_idx, end_frame_idx + 1)
        requested = None if track_ids is None else set(track_ids)
        return {
            frame_idx: tuple(
                prediction
                for prediction in self.outputs.get(frame_idx, ())
                if requested is None or prediction.track_id in requested
            )
            for frame_idx in indices
        }


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
                6: (
                    detection(6, (10, 20, 30, 60)),
                    detection(6, (60, 20, 80, 60)),
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
                5: (
                    mask(5, 1, (10, 20, 30, 60)),
                    mask(5, 2, (60, 20, 80, 60)),
                ),
                6: (
                    mask(6, 1, (10, 20, 30, 60)),
                    mask(6, 2, (60, 20, 80, 60)),
                ),
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
        # The repeated entrant is SAM-prompted but remains hidden until its
        # next detector checkpoint independently validates the mask.
        self.assertNotIn(2, observations)
        self.assertEqual(pipeline.track_manager.tracks[2].status, TrackStatus.TENTATIVE)

        pipeline.process_frame(object(), 5)
        pipeline.process_frame(object(), 6)
        observations = {item.track_id: item for item in pipeline.observations}
        self.assertTrue(observations[2].visible)
        self.assertEqual(observations[2].detection_confidence, 0.9)

        self.assertEqual(detector.calls, [0, 2, 4, 6])
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
                confidence=0.75 + index / 100,
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

    def test_suppresses_duplicate_masks_after_one_detector_support(self) -> None:
        detector = FakeDetector(
            {
                0: (
                    detection(0, (10, 20, 30, 60)),
                    detection(0, (11, 20, 31, 60)),
                ),
                5: (detection(5, (10, 20, 30, 60)),),
            }
        )
        overlapping_masks = {
            index: (
                mask(index, 1, (10, 20, 30, 60)),
                mask(index, 2, (10, 20, 30, 60)),
            )
            for index in range(6)
        }
        pipeline = PlayerTrackingPipeline(
            detector=detector,
            sam_tracker=FakeSamTracker(overlapping_masks),
            association_engine=PlayerAssociationEngine(),
            track_manager=PlayerTrackManager("segment-1"),
        )

        pipeline.start_segment("frames")
        for frame_idx in range(6):
            pipeline.process_frame(object(), frame_idx)
        finalized = pipeline.finalize_segment()

        self.assertEqual(set(pipeline.track_manager.tracks), {1})
        self.assertEqual([item.track_id for item in finalized[0]], [1])
        self.assertEqual([item.track_id for item in finalized[5]], [1])

    def test_full_capacity_retains_candidate_until_unsupported_track_can_be_replaced(
        self,
    ) -> None:
        detector = FakeDetector(
            {
                0: (detection(0, (10, 20, 30, 60)),),
                1: (detection(1, (60, 20, 80, 60)),),
                2: (detection(2, (60, 20, 80, 60)),),
                3: (detection(3, (60, 20, 80, 60)),),
                4: (detection(4, (60, 20, 80, 60)),),
            }
        )
        sam = FakeSamTracker(
            {
                0: (mask(0, 1, (10, 20, 30, 60)),),
                1: (mask(1, 1, (10, 20, 30, 60)),),
                2: (mask(2, 1, (10, 20, 30, 60)),),
                3: (
                    mask(3, 1, (10, 20, 30, 60)),
                    mask(3, 2, (60, 20, 80, 60)),
                ),
                4: (
                    mask(4, 1, (10, 20, 30, 60)),
                    mask(4, 2, (60, 20, 80, 60)),
                ),
            }
        )
        pipeline = PlayerTrackingPipeline(
            detector=detector,
            sam_tracker=sam,
            association_engine=PlayerAssociationEngine(),
            track_manager=PlayerTrackManager("segment-1"),
            detector_interval=1,
            maximum_active_tracks=1,
        )

        pipeline.start_segment("frames")
        for frame_idx in range(5):
            pipeline.process_frame(object(), frame_idx)

        self.assertEqual(set(pipeline.track_manager.tracks), {2})
        self.assertEqual(pipeline.track_manager.tracks[2].status, TrackStatus.ACTIVE)
        self.assertIn(1, sam.removed_ids)

    def test_backward_recovery_publishes_masks_for_late_low_confidence_player(self) -> None:
        low = lambda frame_idx: PlayerDetection(
            frame_idx, (10, 20, 30, 60), 0.7, 1
        )
        detector = FakeDetector({index: (low(index),) for index in range(4)})
        sam = FakeSamTracker(
            {
                3: (mask(3, 1, (10, 20, 30, 60)),),
                2: (mask(2, 1, (10, 20, 30, 60)),),
                1: (mask(1, 1, (10, 20, 30, 60)),),
                0: (mask(0, 1, (10, 20, 30, 60)),),
            }
        )
        pipeline = PlayerTrackingPipeline(
            detector=detector,
            sam_tracker=sam,
            association_engine=PlayerAssociationEngine(),
            track_manager=PlayerTrackManager("segment-1"),
            detector_interval=1,
        )

        pipeline.start_segment("frames")
        for frame_idx in range(4):
            pipeline.process_frame(object(), frame_idx)
        finalized = pipeline.finalize_segment()

        self.assertEqual(finalized[0][0].source, ObservationSource.SAM2_BACKWARD_RECOVERY)
        self.assertTrue(finalized[0][0].visible)
        self.assertEqual(finalized[0][0].track_id, 1)

    def test_stronger_candidate_replaces_repeatedly_weak_track(self) -> None:
        for scores, expected in (((0.70, 0.70), {2}), ((0.90, 0.70), {1, 2})):
            with self.subTest(scores=scores):
                outputs = {0: (detection(0, (10, 20, 30, 60)),)}
                for index in range(1, 4):
                    outputs[index] = (
                        PlayerDetection(index, (10, 20, 30, 60),
                                        scores[0] if index < 3 else scores[1], 1),
                        detection(index, (60, 20, 80, 60)),
                    )
                sam = FakeSamTracker({
                    index: (mask(index, 1, (10, 20, 30, 60)),) + (
                        (mask(index, 2, (60, 20, 80, 60)),) if index == 3 else ()
                    )
                    for index in range(4)
                })
                pipeline = PlayerTrackingPipeline(
                    detector=FakeDetector(outputs), sam_tracker=sam,
                    association_engine=PlayerAssociationEngine(),
                    track_manager=PlayerTrackManager("segment-1"),
                    detector_interval=1, maximum_active_tracks=1,
                )
                pipeline.start_segment("frames")
                for index in range(4):
                    pipeline.process_frame(object(), index)
                self.assertEqual(set(pipeline.track_manager.tracks), expected)
                self.assertEqual(pipeline.track_manager.tracks[2].status,
                                 TrackStatus.ACTIVE if expected == {2} else TrackStatus.TENTATIVE)

    def test_off_court_retirement_requires_consecutive_rejections(self) -> None:
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                sam = FakeSamTracker({
                    index: (mask(index, 1, (10, 20, 30, 60)),)
                    for index in range(9)
                })
                pipeline = PlayerTrackingPipeline(
                    detector=FakeDetector({0: (detection(0, (10, 20, 30, 60)),)}),
                    sam_tracker=sam, association_engine=PlayerAssociationEngine(),
                    track_manager=PlayerTrackManager("segment-1"), fps=30,
                )
                pipeline.start_segment("frames")
                pipeline.process_frame(object(), 0)
                for index in range(1, 9):
                    pipeline.process_frame(
                        object(), index,
                        mask_filter=lambda prediction: interrupted and prediction.frame_idx == 4,
                    )
                self.assertEqual(sam.removed_ids, [] if interrupted else [1])

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
