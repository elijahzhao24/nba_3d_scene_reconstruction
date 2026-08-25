from __future__ import annotations

import unittest

import numpy as np

from nba_3d_scene_reconstruction.tracking.association_engine import (
    PlayerAssociationEngine,
)
from nba_3d_scene_reconstruction.tracking.schemas import (
    PlayerDetection,
    SamMaskPrediction,
)


def detection(
    frame_idx: int,
    box: tuple[float, float, float, float],
) -> PlayerDetection:
    return PlayerDetection(
        frame_idx=frame_idx,
        bbox_xyxy=box,
        confidence=0.9,
        class_id=1,
    )


def sam_mask(
    frame_idx: int,
    track_id: int,
    box: tuple[int, int, int, int] | None,
) -> SamMaskPrediction:
    mask = np.zeros((100, 100), dtype=np.bool_)
    if box is not None:
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = True
    return SamMaskPrediction(
        frame_idx=frame_idx,
        track_id=track_id,
        mask=mask,
    )


class PlayerAssociationEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PlayerAssociationEngine()

    def test_matches_overlapping_detection_and_mask(self) -> None:
        result = self.engine.associate(
            detections=(detection(5, (10, 20, 30, 60)),),
            sam_masks=(sam_mask(5, 7, (10, 20, 30, 60)),),
        )

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].detection_index, 0)
        self.assertEqual(result.matches[0].track_id, 7)
        self.assertEqual(result.matches[0].score, 1.0)
        self.assertEqual(result.unmatched_detection_indices, ())
        self.assertEqual(result.unmatched_track_ids, ())

    def test_reports_distant_inputs_as_unmatched(self) -> None:
        result = self.engine.associate(
            detections=(detection(5, (70, 20, 90, 60)),),
            sam_masks=(sam_mask(5, 7, (10, 20, 30, 60)),),
        )

        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_detection_indices, (0,))
        self.assertEqual(result.unmatched_track_ids, (7,))

    def test_matching_is_one_to_one(self) -> None:
        result = self.engine.associate(
            detections=(
                detection(5, (10, 20, 30, 60)),
                detection(5, (11, 20, 31, 60)),
            ),
            sam_masks=(sam_mask(5, 7, (10, 20, 30, 60)),),
        )

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].detection_index, 0)
        self.assertEqual(result.matches[0].track_id, 7)
        self.assertEqual(result.unmatched_detection_indices, (1,))
        self.assertEqual(result.ignored_duplicate_detection_indices, ())

    def test_ignores_unmatched_detection_overlapping_a_matched_one(self) -> None:
        result = self.engine.associate(
            detections=(
                detection(5, (10, 20, 30, 60)),
                detection(5, (10.2, 20.2, 30.2, 60.2)),
            ),
            sam_masks=(sam_mask(5, 7, (10, 20, 30, 60)),),
        )

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].detection_index, 0)
        self.assertEqual(result.unmatched_detection_indices, ())
        self.assertEqual(result.ignored_duplicate_detection_indices, (1,))

    def test_empty_mask_is_unmatched(self) -> None:
        result = self.engine.associate(
            detections=(detection(5, (10, 20, 30, 60)),),
            sam_masks=(sam_mask(5, 7, None),),
        )

        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_detection_indices, (0,))
        self.assertEqual(result.unmatched_track_ids, (7,))

    def test_empty_inputs_are_supported(self) -> None:
        result = self.engine.associate(detections=(), sam_masks=())

        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_detection_indices, ())
        self.assertEqual(result.unmatched_track_ids, ())

    def test_rejects_different_frames_and_duplicate_track_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "one frame"):
            self.engine.associate(
                detections=(detection(5, (10, 20, 30, 60)),),
                sam_masks=(sam_mask(6, 7, (10, 20, 30, 60)),),
            )

        with self.assertRaisesRegex(ValueError, "unique track IDs"):
            self.engine.associate(
                detections=(),
                sam_masks=(
                    sam_mask(5, 7, (10, 20, 30, 60)),
                    sam_mask(5, 7, (10, 20, 30, 60)),
                ),
            )


if __name__ == "__main__":
    unittest.main()
