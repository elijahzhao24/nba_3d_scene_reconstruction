"""Offline cleanup for projected player trajectories.

The numerical defaults mirror the ``clean_paths`` call in Roboflow's
basketball player notebook. That notebook works in feet; this project stores
court positions in centimeters, so 0.6 feet becomes 18.288 centimeters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from scipy.signal import savgol_filter

from .schemas import PlayerCourtPosition

PointArray: TypeAlias = np.ndarray

FEET_TO_CENTIMETERS = 30.48
DEFAULT_JUMP_SIGMA = 3.5
DEFAULT_MINIMUM_JUMP_DISTANCE_CM = 0.6 * FEET_TO_CENTIMETERS
DEFAULT_MAXIMUM_JUMP_RUN = 18
DEFAULT_JUMP_PADDING_FRAMES = 2
DEFAULT_SMOOTHING_WINDOW = 9
DEFAULT_SMOOTHING_POLYNOMIAL = 2
DEFAULT_MAXIMUM_INTERPOLATION_GAP = 18


@dataclass(frozen=True)
class TrajectorySmoothingConfiguration:
    """Thresholds for robust interpolation and Savitzky-Golay smoothing."""

    jump_sigma: float = DEFAULT_JUMP_SIGMA
    minimum_jump_distance_cm: float = DEFAULT_MINIMUM_JUMP_DISTANCE_CM
    maximum_jump_run: int = DEFAULT_MAXIMUM_JUMP_RUN
    jump_padding_frames: int = DEFAULT_JUMP_PADDING_FRAMES
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW
    smoothing_polynomial: int = DEFAULT_SMOOTHING_POLYNOMIAL
    maximum_interpolation_gap: int = DEFAULT_MAXIMUM_INTERPOLATION_GAP

    def __post_init__(self) -> None:
        if not math.isfinite(self.jump_sigma) or self.jump_sigma < 0.0:
            raise ValueError("jump_sigma must be finite and non-negative")
        if (
            not math.isfinite(self.minimum_jump_distance_cm)
            or self.minimum_jump_distance_cm < 0.0
        ):
            raise ValueError("minimum_jump_distance_cm must be finite and non-negative")
        for name, value in (
            ("maximum_jump_run", self.maximum_jump_run),
            ("jump_padding_frames", self.jump_padding_frames),
            ("maximum_interpolation_gap", self.maximum_interpolation_gap),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.smoothing_window, int)
            or self.smoothing_window <= 0
            or self.smoothing_window % 2 == 0
        ):
            raise ValueError("smoothing_window must be a positive odd integer")
        if (
            not isinstance(self.smoothing_polynomial, int)
            or self.smoothing_polynomial < 0
            or self.smoothing_polynomial >= self.smoothing_window
        ):
            raise ValueError(
                "smoothing_polynomial must be non-negative and smaller than "
                "smoothing_window"
            )


def _median_absolute_deviation(values: PointArray) -> float:
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _true_runs(mask: PointArray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    return [(int(group[0]), int(group[-1])) for group in np.split(indices, splits)]


def _linear_interpolate(values: PointArray) -> PointArray:
    result = values.astype(float).copy()
    missing = np.isnan(result)
    if missing.all():
        return result
    indices = np.arange(len(result))
    present_indices = np.flatnonzero(~missing)
    first = int(present_indices[0])
    last = int(present_indices[-1])
    result[:first] = result[first]
    result[last + 1 :] = result[last]
    missing = np.isnan(result)
    if missing.any():
        result[missing] = np.interp(
            indices[missing], indices[~missing], result[~missing]
        )
    return result


def _smooth(values: PointArray, window: int, polynomial: int) -> PointArray:
    values = values.astype(float)
    size = len(values)
    effective_window = min(window, size if size % 2 else size - 1)
    if effective_window < polynomial + 2:
        kernel_size = min(5, size)
        if kernel_size < 2:
            return values
        padding = kernel_size // 2
        padded = np.pad(values, (padding, padding), mode="edge")
        smoothed = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode="valid")
        # An even moving-average kernel produces one extra value.
        return smoothed[:size]
    return savgol_filter(
        values,
        window_length=effective_window,
        polyorder=polynomial,
        mode="interp",
    )


def clean_paths(
    video_xy: PointArray,
    *,
    configuration: TrajectorySmoothingConfiguration | None = None,
) -> tuple[PointArray, PointArray]:
    """Clean a dense ``(frames, players, 2)`` court-coordinate array.

    This is the same robust-speed, short-jump removal, interpolation, and
    Savitzky-Golay sequence used by the reference notebook. Inputs to this
    low-level function must be finite. Use ``clean_player_position_frames``
    for sparse, dynamically created tracks.
    """
    config = configuration or TrajectorySmoothingConfiguration()
    values = np.asarray(video_xy, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("video_xy must have shape (frames, players, 2)")
    if not np.isfinite(values).all():
        raise ValueError("video_xy must contain only finite coordinates")

    frame_count, player_count, _ = values.shape
    cleaned = values.copy()
    edited = np.zeros((frame_count, player_count), dtype=bool)

    for player_index in range(player_count):
        trajectory = cleaned[:, player_index, :]
        speeds = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        if speeds.size:
            median_speed = float(np.median(speeds))
            speed_scale = max(_median_absolute_deviation(speeds), 1e-6)
            jump_mask = (speeds > median_speed + config.jump_sigma * speed_scale) & (
                speeds > config.minimum_jump_distance_cm
            )

            remove = np.zeros(frame_count, dtype=bool)
            for start, end in _true_runs(jump_mask):
                if end - start + 1 > config.maximum_jump_run:
                    continue
                remove_start = max(0, start - config.jump_padding_frames)
                remove_end = min(
                    frame_count - 1,
                    end + 1 + config.jump_padding_frames,
                )
                remove[remove_start : remove_end + 1] = True

            if remove.any():
                edited[:, player_index] |= remove
                trajectory = trajectory.copy()
                trajectory[remove] = np.nan
                for dimension in range(2):
                    trajectory[:, dimension] = _linear_interpolate(
                        trajectory[:, dimension]
                    )

        for dimension in range(2):
            trajectory[:, dimension] = _smooth(
                trajectory[:, dimension],
                config.smoothing_window,
                config.smoothing_polynomial,
            )
        cleaned[:, player_index, :] = trajectory

    return cleaned, edited


def clean_player_position_frames(
    frames: tuple[tuple[PlayerCourtPosition, ...], ...],
    *,
    configuration: TrajectorySmoothingConfiguration | None = None,
) -> tuple[tuple[PlayerCourtPosition, ...], ...]:
    """Clean dynamic tracks while preserving raw positions and long gaps."""
    config = configuration or TrajectorySmoothingConfiguration()
    cleaned_frames = [list(frame) for frame in frames]
    track_records: dict[
        tuple[str, int], list[tuple[int, int, PlayerCourtPosition]]
    ] = {}
    for frame_slot, frame in enumerate(frames):
        for position_slot, position in enumerate(frame):
            track_records.setdefault(
                (position.segment_id, position.track_id), []
            ).append((frame_slot, position_slot, position))

    for records in track_records.values():
        records.sort(key=lambda item: item[2].frame_idx)
        for record_group in _continuous_record_groups(records):
            _clean_record_group(
                record_group,
                cleaned_frames=cleaned_frames,
                configuration=config,
            )

    return tuple(tuple(frame) for frame in cleaned_frames)


def _continuous_record_groups(
    records: list[tuple[int, int, PlayerCourtPosition]],
) -> list[list[tuple[int, int, PlayerCourtPosition]]]:
    """Split at absent frames and at missing-coordinate gaps that are too long."""
    if not records:
        return []
    groups: list[list[tuple[int, int, PlayerCourtPosition]]] = []
    current = [records[0]]
    for record in records[1:]:
        if record[2].frame_idx != current[-1][2].frame_idx + 1:
            groups.append(current)
            current = [record]
        else:
            current.append(record)
    groups.append(current)
    return groups


def _clean_record_group(
    records: list[tuple[int, int, PlayerCourtPosition]],
    *,
    cleaned_frames: list[list[PlayerCourtPosition]],
    configuration: TrajectorySmoothingConfiguration,
) -> None:
    raw = np.full((len(records), 2), np.nan, dtype=float)
    for index, (_, _, position) in enumerate(records):
        if position.raw_court_xy is not None:
            raw[index] = position.raw_court_xy
    valid_indices = np.flatnonzero(np.isfinite(raw).all(axis=1))
    if valid_indices.size == 0:
        return

    # A long missing interval is a genuine loss of evidence. Do not invent a
    # path through it or let smoothing bridge unrelated appearances.
    splits = (
        np.where(np.diff(valid_indices) - 1 > configuration.maximum_interpolation_gap)[
            0
        ]
        + 1
    )
    for valid_group in np.split(valid_indices, splits):
        first = int(valid_group[0])
        last = int(valid_group[-1])
        dense = raw[first : last + 1].copy()
        originally_missing = ~np.isfinite(dense).all(axis=1)
        for dimension in range(2):
            dense[:, dimension] = _linear_interpolate(dense[:, dimension])
        cleaned, outlier_mask = clean_paths(
            dense[:, None, :], configuration=configuration
        )
        for offset, xy in enumerate(cleaned[:, 0, :]):
            record_index = first + offset
            frame_slot, position_slot, position = records[record_index]
            flags = list(position.quality_flags)
            was_edited = bool(outlier_mask[offset, 0])
            if originally_missing[offset]:
                flags.append("trajectory_gap_interpolated")
                was_edited = True
            if outlier_mask[offset, 0]:
                flags.append("trajectory_jump_interpolated")
            cleaned_frames[frame_slot][position_slot] = replace(
                position,
                clean_court_xy=(float(xy[0]), float(xy[1])),
                position_was_edited=was_edited,
                quality_flags=tuple(dict.fromkeys(flags)),
            )
