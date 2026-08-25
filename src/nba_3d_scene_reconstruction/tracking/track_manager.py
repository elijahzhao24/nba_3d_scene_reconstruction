from __future__ import annotations

from .schemas import (
    BoundingBox,
    PlayerDetection,
    TrackState,
    TrackStatus,
)

'''
Responsbile for creating and keeping track of unique ID's for players
'''
class PlayerTrackManager:
    MAX_MISSING_FRAMES = 30

    def __init__(self, segment_id: str) -> None:
        self.segment_id = segment_id
        self.tracks: dict[int, TrackState] = {}
        self._next_track_id = 1

    def create_track(
            self,
            detection: PlayerDetection
    ) -> TrackState:
        track = TrackState(
        track_id=self._next_track_id,
        segment_id=self.segment_id,
        status=TrackStatus.ACTIVE,
        start_frame=detection.frame_idx,
        last_seen_frame=detection.frame_idx,
        latest_bbox_xyxy=detection.bbox_xyxy,
    )

        self.tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def mark_visible(
        self,
        track_id: int,
        frame_idx: int,
        bbox_xyxy: BoundingBox | None = None,
    ) -> None:
        track = self.tracks[track_id]
        track.status = TrackStatus.ACTIVE
        track.last_seen_frame = frame_idx
        track.missing_frame_count = 0

        if bbox_xyxy is not None:
            track.latest_bbox_xyxy = bbox_xyxy

    def mark_missing(
            self,
            track_id: int,
            frame_idx: int,
            bbox_xyxy: BoundingBox | None = None,
        ) -> None:
            track = self.tracks[track_id]
            track.status = TrackStatus.MISSING
            track.missing_frame_count += 1

            if track.missing_frame_count <= self.MAX_MISSING_FRAMES:
                return None

            track.status = TrackStatus.RETIRED
            track.end_frame = frame_idx
            return self.tracks.pop(track_id)
