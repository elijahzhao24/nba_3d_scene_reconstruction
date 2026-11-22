import json
from dataclasses import asdict
from pathlib import Path

import cv2

from nba_3d_scene_reconstruction.tracking.schemas import VideoManifest


def _segment_directory(video_path: str | Path) -> Path:
    return Path("artifacts", Path(video_path).stem, "segment_001")


def read_saved_video_manifest(video_path: str | Path) -> VideoManifest:
    manifest_path = _segment_directory(video_path) / "video_manifest.json"
    manifest = VideoManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    frame_count = sum(
        path.suffix in {".jpg", ".jpeg", ".JPG", ".JPEG"}
        for path in Path(manifest.frames_dir).iterdir()
    )
    if frame_count != manifest.frame_count:
        raise ValueError(
            "saved frame directory does not match its manifest; rerun without "
            "--skip-frame-extraction"
        )
    return manifest


def processNewVideo(video_path: str, *, max_frames: int | None = None) -> VideoManifest:
    """Extract the source frames used by tracking and downstream stages."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    segment_dir = _segment_directory(video_path)
    frames_dir = segment_dir / "frames"
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open source video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        while max_frames is None or frame_count < max_frames:
            success, frame = capture.read()
            if not success:
                break

            frame_path = frames_dir / f"{frame_count:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise OSError(f"could not write frame: {frame_path}")
            frame_count += 1
    finally:
        capture.release()

    # SAM 2 loads the full frame directory, so drop the tail of earlier runs.
    for frame_path in frames_dir.glob("*.jpg"):
        if frame_path.stem.isdecimal() and int(frame_path.stem) >= frame_count:
            frame_path.unlink()

    manifest = VideoManifest(
        clip_id=segment_dir.parent.name,
        segment_id=segment_dir.name,
        source_path=video_path,
        frames_dir=str(frames_dir),
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    )
    manifest_path = segment_dir / "video_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(asdict(manifest), stream, indent=2)
        stream.write("\n")
    return manifest
