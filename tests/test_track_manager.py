from __future__ import annotations

import unittest

from nba_3d_scene_reconstruction.tracking.schemas import (
    PlayerDetection,
    TrackStatus,
)
from nba_3d_scene_reconstruction.tracking.track_manager import PlayerTrackManager


def detection(frame_idx: int = 0) -> PlayerDetection:
    return PlayerDetection(
        frame_idx=frame_idx,
        bbox_xyxy=(10, 20, 30, 60),
        confidence=0.9,
        class_id=1,
    )


class PlayerTrackManagerTest(unittest.TestCase):
    def test_creates_monotonically_increasing_ids(self) -> None:
        manager = PlayerTrackManager("segment-1")

        first = manager.create_track(detection())
        second = manager.create_track(detection())

        self.assertEqual((first.track_id, second.track_id), (1, 2))

    def test_missing_track_can_become_visible_again(self) -> None:
        manager = PlayerTrackManager("segment-1")
        track = manager.create_track(detection())

        manager.mark_missing(track.track_id, frame_idx=1)
        manager.mark_visible(track.track_id, frame_idx=2)

        self.assertEqual(track.status, TrackStatus.ACTIVE)
        self.assertEqual(track.missing_frame_count, 0)
        self.assertEqual(track.last_seen_frame, 2)

    def test_retires_track_after_missing_timeout(self) -> None:
        manager = PlayerTrackManager("segment-1")
        manager.MAX_MISSING_FRAMES = 1
        track = manager.create_track(detection())

        self.assertIsNone(manager.mark_missing(track.track_id, frame_idx=1))
        retired = manager.mark_missing(track.track_id, frame_idx=2)

        self.assertIs(retired, track)
        self.assertEqual(retired.status, TrackStatus.RETIRED)
        self.assertEqual(retired.end_frame, 2)
        self.assertNotIn(track.track_id, manager.tracks)


if __name__ == "__main__":
    unittest.main()
