from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from nba_3d_scene_reconstruction.tracking.sam2_tracker import Sam2PlayerTracker


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeSam2Predictor:
    def __init__(self) -> None:
        self.state = object()
        self.init_calls: list[str] = []
        self.prompt_calls: list[dict[str, object]] = []
        self.propagate_calls: list[dict[str, object]] = []
        self.reset_calls: list[object] = []
        self.output_object_ids: tuple[int, ...] = ()
        self.output_mask_logits = np.empty((0, 1, 2, 2), dtype=np.float32)

    def init_state(self, video_path: str) -> object:
        self.init_calls.append(video_path)
        return self.state

    def add_new_points_or_box(
        self,
        inference_state: object,
        **kwargs: object,
    ) -> tuple[int, tuple[int, ...], FakeTensor]:
        self.prompt_calls.append({"state": inference_state, **kwargs})
        return (
            int(kwargs["frame_idx"]),
            self.output_object_ids,
            FakeTensor(self.output_mask_logits),
        )

    def propagate_in_video(
        self,
        inference_state: object,
        **kwargs: object,
    ):
        self.propagate_calls.append({"state": inference_state, **kwargs})
        yield (
            int(kwargs["start_frame_idx"]),
            self.output_object_ids,
            FakeTensor(self.output_mask_logits),
        )

    def reset_state(self, inference_state: object) -> None:
        self.reset_calls.append(inference_state)


class Sam2PlayerTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = FakeSam2Predictor()
        self.tracker = Sam2PlayerTracker(predictor=self.predictor)

    def test_starts_segment_and_prompts_player(self) -> None:
        self.tracker.start_segment(Path("frames"))
        self.tracker.prompt_player(0, 7, (10.0, 20.0, 30.0, 60.0))

        self.assertEqual(self.predictor.init_calls, ["frames"])
        prompt = self.predictor.prompt_calls[0]
        self.assertIs(prompt["state"], self.predictor.state)
        self.assertEqual(prompt["frame_idx"], 0)
        self.assertEqual(prompt["obj_id"], 7)
        np.testing.assert_array_equal(
            prompt["box"],
            np.asarray([10, 20, 30, 60], dtype=np.float32),
        )
        self.assertEqual(self.tracker.track_ids, frozenset({7}))

    def test_propagates_one_frame_and_converts_masks(self) -> None:
        self.predictor.output_object_ids = (7, 11)
        self.predictor.output_mask_logits = np.asarray(
            [
                [[[-1.0, 0.1], [2.0, -0.5]]],
                [[[0.2, -0.1], [-3.0, 4.0]]],
            ],
            dtype=np.float32,
        )
        self.tracker.start_segment("frames")
        self.tracker.prompt_player(0, 7, (0, 0, 10, 20))
        self.tracker.prompt_player(0, 11, (20, 0, 30, 20))

        predictions = self.tracker.propagate_frame(5)

        self.assertEqual(
            self.predictor.propagate_calls,
            [
                {
                    "state": self.predictor.state,
                    "start_frame_idx": 5,
                    "max_frame_num_to_track": 0,
                }
            ],
        )
        self.assertEqual([item.track_id for item in predictions], [7, 11])
        np.testing.assert_array_equal(
            predictions[0].mask,
            np.asarray([[False, True], [True, False]]),
        )

    def test_same_prompt_method_adds_and_corrects_players(self) -> None:
        self.tracker.start_segment("frames")
        self.tracker.prompt_player(0, 1, (0, 0, 10, 20))
        self.tracker.prompt_player(5, 1, (2, 0, 12, 20))
        self.tracker.prompt_player(5, 2, (20, 0, 30, 20))

        self.assertEqual(self.tracker.track_ids, frozenset({1, 2}))
        self.assertEqual(
            [call["obj_id"] for call in self.predictor.prompt_calls],
            [1, 1, 2],
        )

    def test_validates_lifecycle_and_box_geometry(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "start_segment"):
            self.tracker.prompt_player(0, 1, (0, 0, 10, 20))

        self.tracker.start_segment("frames")
        with self.assertRaisesRegex(ValueError, "positive width"):
            self.tracker.prompt_player(0, 1, (10, 0, 10, 20))


if __name__ == "__main__":
    unittest.main()
