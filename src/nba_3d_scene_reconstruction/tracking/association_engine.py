"""Associate RF-DETR boxes with SAM 2 player masks."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schemas import (
    AssociationMatch,
    AssociationResult,
    BoundingBox,
    PlayerDetection,
    SamMaskPrediction,
)


class PlayerAssociationEngine:
    """Perform stateless, one-to-one matching at detector checkpoints."""

    # RF-DETR can emit both the generic player class and an action-specific
    # player class around the same body. Treat a strong overlap as duplicate
    # evidence instead of starting another SAM object.
    DUPLICATE_IOU_THRESHOLD = 0.85
    MINIMUM_SCORE = 0.45
    IOU_WEIGHT = 0.35
    COVERAGE_WEIGHT = 0.20
    CENTER_WEIGHT = 0.30
    CONTINUITY_WEIGHT = 0.15

    """
    Matches new RF-DETR boxes to exisiting SAM IDS.
    returns a list of matched and unmatched indices.
    """

    def associate(
        self,
        *,
        detections: tuple[PlayerDetection, ...],
        sam_masks: tuple[SamMaskPrediction, ...],
        track_bboxes: Mapping[int, BoundingBox | None] | None = None,
    ) -> AssociationResult:
        """Match detections to SAM track IDs and report unmatched inputs."""
        self._validate_inputs(detections, sam_masks)

        # A greedy best-first loop can take the strongest local pair and leave
        # two otherwise good pairs unmatched.  This happens regularly in the
        # paint.  Solve the complete gated matrix instead.
        scores = np.full((len(detections), len(sam_masks)), -np.inf)
        for detection_index, detection in enumerate(detections):
            for mask_index, sam_mask in enumerate(sam_masks):
                mask_box = self._mask_to_bbox(sam_mask.mask)
                if mask_box is None:
                    continue

                score = self._calculate_match_score(
                    detection.bbox_xyxy,
                    mask_box,
                    previous_box=(track_bboxes or {}).get(sam_mask.track_id),
                )
                if score >= self.MINIMUM_SCORE and self._passes_geometry_gate(
                    detection.bbox_xyxy, mask_box
                ):
                    scores[detection_index, mask_index] = score

        matches: list[AssociationMatch] = []
        if scores.size:
            # Invalid edges have a cost larger than any valid score.  They are
            # discarded after the assignment rather than permitted as matches.
            costs = np.where(np.isfinite(scores), 1.0 - scores, 2.0)
            row_indices, column_indices = linear_sum_assignment(costs)
            for detection_index, mask_index in zip(row_indices, column_indices):
                score = scores[detection_index, mask_index]
                if not np.isfinite(score):
                    continue
                matches.append(
                    AssociationMatch(
                        detection_index=int(detection_index),
                        track_id=sam_masks[int(mask_index)].track_id,
                        score=float(score),
                    )
                )

        matched_detection_indices = {match.detection_index for match in matches}
        matched_track_ids = {match.track_id for match in matches}

        matches.sort(key=lambda match: match.detection_index)
        ignored_duplicate_detection_indices = tuple(
            detection_index
            for detection_index, detection in enumerate(detections)
            if detection_index not in matched_detection_indices
            and any(
                self._calculate_iou(
                    detection.bbox_xyxy,
                    detections[match.detection_index].bbox_xyxy,
                )
                >= self.DUPLICATE_IOU_THRESHOLD
                for match in matches
            )
        )
        ignored_duplicate_indices = set(ignored_duplicate_detection_indices)

        return AssociationResult(
            matches=tuple(matches),
            unmatched_detection_indices=tuple(
                index
                for index in range(len(detections))
                if index not in matched_detection_indices
                and index not in ignored_duplicate_indices
            ),
            unmatched_track_ids=tuple(
                sam_mask.track_id
                for sam_mask in sam_masks
                if sam_mask.track_id not in matched_track_ids
            ),
            ignored_duplicate_detection_indices=(ignored_duplicate_detection_indices),
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

        first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
        second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
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

    @classmethod
    def _calculate_coverage_score(
        cls,
        detection_box: BoundingBox,
        mask_box: BoundingBox,
    ) -> float:
        """Intersection divided by the smaller box, robust to loose masks."""
        first_x1, first_y1, first_x2, first_y2 = detection_box
        second_x1, second_y1, second_x2, second_y2 = mask_box
        intersection = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
        intersection *= max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
        first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
        second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
        smaller_area = min(first_area, second_area)
        return 0.0 if smaller_area <= 0.0 else intersection / smaller_area

    @staticmethod
    def _passes_geometry_gate(
        detection_box: BoundingBox,
        mask_box: BoundingBox,
    ) -> bool:
        """Allow modest movement but never match unrelated distant objects."""
        if PlayerAssociationEngine._calculate_iou(detection_box, mask_box) >= 0.05:
            return True
        detection_height = max(1.0, detection_box[3] - detection_box[1])
        mask_height = max(1.0, mask_box[3] - mask_box[1])
        detection_foot = ((detection_box[0] + detection_box[2]) / 2.0, detection_box[3])
        mask_foot = ((mask_box[0] + mask_box[2]) / 2.0, mask_box[3])
        return math.dist(detection_foot, mask_foot) <= 0.75 * max(
            detection_height, mask_height
        )

    def _calculate_match_score(
        self,
        detection_box: BoundingBox,
        mask_box: BoundingBox,
        *,
        previous_box: BoundingBox | None = None,
    ) -> float:
        iou = self._calculate_iou(detection_box, mask_box)
        coverage = self._calculate_coverage_score(detection_box, mask_box)
        center_score = self._calculate_center_score(detection_box, mask_box)
        if previous_box is None:
            # Preserve the intuitive [0, 1] score for callers that do not
            # provide motion history (and for the original public API).
            return 0.4 * iou + 0.25 * coverage + 0.35 * center_score
        continuity = self._calculate_iou(detection_box, previous_box)
        return (
            self.IOU_WEIGHT * iou
            + self.COVERAGE_WEIGHT * coverage
            + self.CENTER_WEIGHT * center_score
            + self.CONTINUITY_WEIGHT * continuity
        )

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
