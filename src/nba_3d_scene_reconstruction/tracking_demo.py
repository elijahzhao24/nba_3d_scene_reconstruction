"""Run the tracking pipeline and render SAM masks over a source video."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from .court.debug import CourtDebugRenderer
from .court.detector import RoboflowCourtDetector
from .ingest_video import processNewVideo
from .pipeline import SceneFrame, SceneReconstructionPipeline
from .tracking.association_engine import PlayerAssociationEngine
from .tracking.debug import TRACK_COLORS, draw_tracking_overlay
from .tracking.pipeline import PlayerTrackingPipeline
from .tracking.player_detector import RoboflowPlayerDetector
from .tracking.sam2_tracker import Sam2PlayerTracker
from .tracking.schemas import VideoManifest
from .tracking.track_manager import PlayerTrackManager


def render_tracking_video(
    manifest: VideoManifest,
    pipeline: Any,
    output_path: str | Path,
    *,
    max_frames: int | None = None,
    records_dir: str | Path | None = None,
) -> Path:
    """Write the existing mask overlay, optionally persisting scene records."""
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

    writer: cv2.VideoWriter | None = None
    court_renderer = (
        CourtDebugRenderer(
            pipeline.calibrator.estimator.detector_configuration,
        )
        if isinstance(pipeline, SceneReconstructionPipeline)
        else None
    )

    try:
        with ExitStack() as stack:
            streams = {}
            if records_dir is not None:
                root = Path(records_dir)
                root.mkdir(parents=True, exist_ok=True)
                for name in ("observations", "court_detections", "calibrations",
                             "player_court_positions_raw"):
                    streams[name] = stack.enter_context(
                        (root / f"{name}.jsonl").open("w", encoding="utf-8")
                    )
            pipeline.start_segment(manifest.frames_dir)
            for frame_idx in range(frame_count):
                frame_path = Path(manifest.frames_dir) / f"{frame_idx:06d}.jpg"
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise OSError(f"could not read extracted frame: {frame_path}")

                result = pipeline.process_frame(frame, frame_idx)
                masks = result.masks if isinstance(result, SceneFrame) else result
                if streams:
                    if not isinstance(result, SceneFrame):
                        raise TypeError("records_dir requires a scene pipeline")
                    records = {
                        "observations": result.observations,
                        "court_detections": (() if result.court_detection is None
                                             else (result.court_detection,)),
                        "calibrations": (result.calibration,),
                        "player_court_positions_raw": result.positions,
                    }
                    for name, values in records.items():
                        for value in values:
                            streams[name].write(json.dumps(asdict(value), allow_nan=False) + "\n")
                rendered = (
                    court_renderer.render(frame, result, fps=fps)
                    if court_renderer is not None and isinstance(result, SceneFrame)
                    else draw_tracking_overlay(frame, masks)
                )
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(destination),
                        cv2.VideoWriter_fourcc(*codec),
                        fps,
                        (rendered.shape[1], rendered.shape[0]),
                    )
                    if not writer.isOpened():
                        raise OSError(f"could not create output video: {destination}")
                writer.write(rendered)
    finally:
        if writer is not None:
            writer.release()

    return destination


def run_tracking_demo(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    detector_interval: int = 5,
    max_frames: int | None = None,
    skip_frame_extraction: int = 0,
    court_projection: bool = False,
    court_interval: int = 1,
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
        fps=manifest.fps,
    )
    if court_projection:
        pipeline = SceneReconstructionPipeline(
            tracking_pipeline=pipeline,
            court_detector=RoboflowCourtDetector(),
            court_detector_interval=court_interval,
        )

    if output_path is None:
        filename = "court_debug.webm" if court_projection else "tracking_overlay.webm"
        output_path = (
            Path(manifest.frames_dir).parent
            / "debug"
            / filename
        )

    return render_tracking_video(
        manifest,
        pipeline,
        output_path,
        max_frames=max_frames,
        records_dir=Path(manifest.frames_dir).parent if court_projection else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run player tracking and render synchronized diagnostics.",
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
        "--court-projection",
        action="store_true",
        help=("Detect/calibrate the court, save raw records, "
              "and render a synchronized top-down court"),
    )
    parser.add_argument(
        "--court-interval",
        type=int,
        default=1,
        help="Run court detection every N frames (default: every frame)",
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
        court_projection=args.court_projection,
        court_interval=args.court_interval,
    )
    print(f"Debug video written to {output}")


if __name__ == "__main__":
    main()
