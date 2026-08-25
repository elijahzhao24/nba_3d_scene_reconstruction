"""Associate RF-DETR boxes with SAM 2 player masks."""

from __future__ import annotations

import math

import numpy as np

from .schemas import (
    AssociationMatch,
    AssociationResult,
    BoundingBox,
    PlayerDetection,
    SamMaskPrediction,
)


class PlayerAssociationEngine:
    """Perform stateless, one-to-one matching at detector checkpoints."""

    MINIMUM_SCORE = 0.4
    IOU_WEIGHT = 0.5
    CENTER_WEIGHT = 0.5

    '''
    Matches new RF-DETR boxes to exisiting SAM IDS.
    returns a list of matched and unmatched indices.
    '''
    def associate(
        self,
        *,
        detections: tuple[PlayerDetection, ...],
        sam_masks: tuple[SamMaskPrediction, ...],
    ) -> AssociationResult:
        """Match detections to SAM track IDs and report unmatched inputs."""
        self._validate_inputs(detections, sam_masks)

        candidates: list[AssociationMatch] = []
        for detection_index, detection in enumerate(detections):
            for sam_mask in sam_masks:
                mask_box = self._mask_to_bbox(sam_mask.mask)
                if mask_box is None:
                    continue

                score = self._calculate_match_score(
                    detection.bbox_xyxy,
                    mask_box,
                )
                if score >= self.MINIMUM_SCORE:
                    candidates.append(
                        AssociationMatch(
                            detection_index=detection_index,
                            track_id=sam_mask.track_id,
                            score=score,
                        )
                    )

        # Consider the strongest pairs first. The ID/index tie breakers keep
        # results deterministic when two candidates have the same score.
        candidates.sort(
            key=lambda match: (
                -match.score,
                match.detection_index,
                match.track_id,
            )
        )

        matches: list[AssociationMatch] = []
        matched_detection_indices: set[int] = set()
        matched_track_ids: set[int] = set()

        for candidate in candidates:
            if candidate.detection_index in matched_detection_indices:
                continue
            if candidate.track_id in matched_track_ids:
                continue

            matches.append(candidate)
            matched_detection_indices.add(candidate.detection_index)
            matched_track_ids.add(candidate.track_id)

        matches.sort(key=lambda match: match.detection_index)
        return AssociationResult(
            matches=tuple(matches),
            unmatched_detection_indices=tuple(
                index
                for index in range(len(detections))
                if index not in matched_detection_indices
            ),
            unmatched_track_ids=tuple(
                sam_mask.track_id
                for sam_mask in sam_masks
                if sam_mask.track_id not in matched_track_ids
            ),
        )

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> BoundingBox | None:
        """Return the smallest xyxy box containing a binary mask."""
        if mask.ndim != 2:
            raise ValueError("SAM mask must be a two-dimensional array")

        y_coordinates, x_coordinates = np.nonzero(mask)
        if len(x_coordinates) == 0:
            return None

        return (
            float(x_coordinates.min()),
            float(y_coordinates.min()),
            float(x_coordinates.max() + 1),
            float(y_coordinates.max() + 1),
        )

    @staticmethod
    def _calculate_iou(
        first_box: BoundingBox,
        second_box: BoundingBox,
    ) -> float:
        """Calculate intersection over union for two xyxy boxes."""
        first_x1, first_y1, first_x2, first_y2 = first_box
        second_x1, second_y1, second_x2, second_y2 = second_box

        intersection_width = max(
            0.0,
            min(first_x2, second_x2) - max(first_x1, second_x1),
        )
        intersection_height = max(
            0.0,
            min(first_y2, second_y2) - max(first_y1, second_y1),
        )
        intersection_area = intersection_width * intersection_height

        first_area = max(0.0, first_x2 - first_x1) * max(
            0.0, first_y2 - first_y1
        )
        second_area = max(0.0, second_x2 - second_x1) * max(
            0.0, second_y2 - second_y1
        )
        union_area = first_area + second_area - intersection_area

        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def _calculate_center_score(
        detection_box: BoundingBox,
        mask_box: BoundingBox,
    ) -> float:
        """Score box-center proximity relative to the players' heights."""
        detection_x1, detection_y1, detection_x2, detection_y2 = detection_box
        mask_x1, mask_y1, mask_x2, mask_y2 = mask_box

        detection_center = (
            (detection_x1 + detection_x2) / 2.0,
            (detection_y1 + detection_y2) / 2.0,
        )
        mask_center = (
            (mask_x1 + mask_x2) / 2.0,
            (mask_y1 + mask_y2) / 2.0,
        )
        distance = math.dist(detection_center, mask_center)

        detection_height = max(0.0, detection_y2 - detection_y1)
        mask_height = max(0.0, mask_y2 - mask_y1)
        distance_limit = 2.0 * max(detection_height, mask_height, 1.0)

        return max(0.0, 1.0 - distance / distance_limit)

    def _calculate_match_score(
        self,
        detection_box: BoundingBox,
        mask_box: BoundingBox,
    ) -> float:
        iou = self._calculate_iou(detection_box, mask_box)
        center_score = self._calculate_center_score(detection_box, mask_box)
        return self.IOU_WEIGHT * iou + self.CENTER_WEIGHT * center_score

    @staticmethod
    def _validate_inputs(
        detections: tuple[PlayerDetection, ...],
        sam_masks: tuple[SamMaskPrediction, ...],
    ) -> None:
        track_ids = [sam_mask.track_id for sam_mask in sam_masks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("sam_masks must contain unique track IDs")

        frame_indices = {
            *(detection.frame_idx for detection in detections),
            *(sam_mask.frame_idx for sam_mask in sam_masks),
        }
        if len(frame_indices) > 1:
            raise ValueError("detections and SAM masks must come from one frame")
