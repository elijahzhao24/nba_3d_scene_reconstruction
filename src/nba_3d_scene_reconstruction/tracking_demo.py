"""Run the tracking pipeline and render SAM masks over a source video."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .ingest_video import processNewVideo
from .tracking.association_engine import PlayerAssociationEngine
from .tracking.pipeline import PlayerTrackingPipeline
from .tracking.player_detector import RoboflowPlayerDetector
from .tracking.sam2_tracker import Sam2PlayerTracker
from .tracking.schemas import SamMaskPrediction, VideoManifest
from .tracking.track_manager import PlayerTrackManager


TRACK_COLORS = (
    (66, 135, 245),
    (80, 200, 120),
    (235, 90, 90),
    (80, 210, 230),
    (210, 110, 220),
    (230, 170, 70),
)


def draw_tracking_overlay(
    frame: np.ndarray,
    masks: tuple[SamMaskPrediction, ...],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Draw colored masks, outlines, and track IDs on one video frame."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    result = frame.copy()
    expected_shape = frame.shape[:2]

    for prediction in masks:
        mask = np.asarray(prediction.mask, dtype=np.bool_)
        if mask.shape != expected_shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match frame shape {expected_shape}"
            )
        if not mask.any():
            continue

        color = TRACK_COLORS[prediction.track_id % len(TRACK_COLORS)]
        color_layer = result.copy()
        color_layer[mask] = color
        result = cv2.addWeighted(color_layer, alpha, result, 1.0 - alpha, 0)

        mask_image = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(result, contours, -1, color, 2)

        y_coordinates, x_coordinates = np.nonzero(mask)
        label_position = (
            int(x_coordinates.min()),
            max(18, int(y_coordinates.min()) - 6),
        )
        cv2.putText(
            result,
            f"ID {prediction.track_id}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return result


def render_tracking_video(
    manifest: VideoManifest,
    pipeline: Any,
    output_path: str | Path,
    *,
    max_frames: int | None = None,
) -> Path:
    """Process extracted frames and write a mask-overlay debug video."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    frame_count = manifest.frame_count
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if frame_count == 0:
        raise ValueError("the video contains no frames")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = manifest.fps if manifest.fps > 0 else 30.0
    suffix = destination.suffix.lower()
    if suffix == ".webm":
        codec = "VP80"
    elif suffix == ".mp4":
        codec = "mp4v"
    else:
        raise ValueError("output path must end in .webm or .mp4")

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (manifest.width, manifest.height),
    )
    if not writer.isOpened():
        raise OSError(f"could not create output video: {destination}")

    pipeline.start_segment(manifest.frames_dir)
    try:
        for frame_idx in range(frame_count):
            frame_path = Path(manifest.frames_dir) / f"{frame_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise OSError(f"could not read extracted frame: {frame_path}")

            masks = pipeline.process_frame(frame, frame_idx)
            writer.write(draw_tracking_overlay(frame, masks))
    finally:
        writer.release()

    return destination


def run_tracking_demo(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    detector_interval: int = 5,
    max_frames: int | None = None,
    skip_frame_extraction: int = 0
) -> Path:
    """Ingest one video, run the real models, and render an overlay video."""
    if not skip_frame_extraction:
        manifest = processNewVideo(str(video_path))
    pipeline = PlayerTrackingPipeline(
        detector=RoboflowPlayerDetector(),
        sam_tracker=Sam2PlayerTracker(),
        association_engine=PlayerAssociationEngine(),
        track_manager=PlayerTrackManager(manifest.segment_id),
        detector_interval=detector_interval,
    )

    if output_path is None:
        output_path = (
            Path(manifest.frames_dir).parent
            / "debug"
            / "tracking_overlay.webm"
        )

    return render_tracking_video(
        manifest,
        pipeline,
        output_path,
        max_frames=max_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run player tracking and render SAM mask overlays.",
    )
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument(
        "--output",
        help="Output video path (.webm is recommended for browser playback)",
    )
    parser.add_argument(
        "--detector-interval",
        type=int,
        default=5,
        help="Run RF-DETR every N frames (default: 5)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process only the first N frames for a quick test",
    )
    parser.add_argument(
        "--skip-frame-extraction",
        type=int,
        help="Use whatever is already in artifacts",
    )
    args = parser.parse_args()

    output = run_tracking_demo(
        args.video,
        output_path=args.output,
        detector_interval=args.detector_interval,
        max_frames=args.max_frames,
        skip_frame_extraction = args.skip_frame_extraction,
    )
    print(f"Tracking overlay written to {output}")


if __name__ == "__main__":
    main()
