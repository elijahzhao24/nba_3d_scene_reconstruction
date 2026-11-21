"""Package saved reconstruction artifacts for offline Three.js replay."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypeVar

from .court.configuration import COURT_LANDMARK_POINTS_CM
from .court.pose_camera import estimate_pose_camera
from .court.schemas import CalibrationSource, Matrix3x3, PlayerCourtPosition
from .pose.io import read_manifest, read_observations, read_poses_3d
from .pose.schemas import Pose3D
from .tracking.schemas import PlayerObservation

_MANIFEST = "video_manifest.json"
_POSITIONS = "player_court_positions_clean.jsonl"
_OBSERVATIONS = "observations.jsonl"
_POSES = "poses_3d_local.jsonl"
_CALIBRATIONS = "calibrations.jsonl"
_TIMESTAMP_TOLERANCE_SECONDS = 1e-6
_CM_PER_METER = 100

_TrackRecord = TypeVar("_TrackRecord", PlayerCourtPosition, PlayerObservation, Pose3D)


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def _index_tracks(
    rows: Iterable[_TrackRecord],
    *,
    segment_id: str,
    timestamps: dict[int, float],
    name: str,
) -> dict[tuple[int, int], _TrackRecord]:
    indexed: dict[tuple[int, int], _TrackRecord] = {}
    for row in rows:
        key = (row.frame_idx, row.track_id)
        if row.segment_id != segment_id:
            raise ValueError(
                f"{name}: expected segment {segment_id!r}, got {row.segment_id!r}"
            )
        if any(type(value) is not int or value < 0 for value in key):
            raise ValueError(f"{name}: invalid frame or track ID {key}")
        if key in indexed:
            raise ValueError(f"{name}: duplicate frame and track {key}")
        t = row.timestamp_seconds
        if not math.isfinite(t) or t < 0:
            raise ValueError(f"{name}: invalid timestamp for frame and track {key}")
        known_time = timestamps.setdefault(row.frame_idx, t)
        if not math.isclose(
            t, known_time, rel_tol=0, abs_tol=_TIMESTAMP_TOLERANCE_SECONDS
        ):
            raise ValueError(f"{name}: timestamp mismatch at frame {row.frame_idx}")
        indexed[key] = row
    return indexed


def _read_legacy_video_metadata(source_video: Path) -> tuple[float, float, float]:
    import cv2

    capture = cv2.VideoCapture(str(source_video))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open source video: {source_video}")
        return (
            float(capture.get(cv2.CAP_PROP_FPS)),
            float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def _calibration_summary(path: Path, segment_id: str) -> tuple[int, list[Matrix3x3]]:
    if not path.exists():
        return 0, []
    count = 0
    homographies: list[Matrix3x3] = []
    for row in _read_rows(path):
        if row.get("segment_id") != segment_id or row.get("frame_idx") != count:
            raise ValueError(f"{path}: expected calibration for {segment_id} frame {count}")
        if (
            row.get("valid") is True
            and row.get("source") == CalibrationSource.DETECTED.value
            and (matrix := row.get("court_to_image")) is not None
        ):
            homographies.append(matrix)
        count += 1
    return count, homographies


def _pose_metadata(pose: Pose3D) -> dict[str, Any]:
    names = [joint.name for joint in pose.joints]
    if not names or len(set(names)) != len(names) or not all(names):
        raise ValueError("invalid pose joint names")
    if (pose.units, pose.axes, pose.coordinate_space) != (
        "model_relative",
        "motionbert_camera_axes",
        "pelvis_relative",
    ):
        raise ValueError("expected pelvis-relative MotionBERT poses")
    return {
        "layout": pose.layout,
        "units": pose.units,
        "axes": pose.axes,
        "coordinate_space": pose.coordinate_space,
        "joints": names,
    }


def build_scene(
    artifact_dir: str | Path, *, source_video: str | Path | None = None
) -> dict[str, Any]:
    """Assemble one processed segment without performing any reconstruction."""
    root = Path(artifact_dir)
    observations = read_observations(root / _OBSERVATIONS)
    manifest_path = root / _MANIFEST
    calibration_path = root / _CALIBRATIONS
    pose_path = root / _POSES
    manifest = read_manifest(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        clip_id, segment_id, fps = manifest.clip_id, manifest.segment_id, manifest.fps
        image_size = (manifest.width, manifest.height)
    elif source_video is not None:
        clip_id = root.resolve().parent.name
        segment_id = observations[0].segment_id if observations else root.resolve().name
        fps, width, height = _read_legacy_video_metadata(Path(source_video))
        image_size = (width, height)
    else:
        raise FileNotFoundError(
            f"missing {manifest_path}; pass --source-video for older artifacts"
        )
    if not clip_id or not segment_id or not math.isfinite(fps) or fps <= 0:
        raise ValueError("scene requires clip and segment IDs and a finite, positive source FPS")

    timestamps: dict[int, float] = {}
    observed = _index_tracks(
        observations, segment_id=segment_id, timestamps=timestamps, name=_OBSERVATIONS
    )
    positions = _index_tracks(
        (PlayerCourtPosition(**row) for row in _read_rows(root / _POSITIONS)),
        segment_id=segment_id,
        timestamps=timestamps,
        name=_POSITIONS,
    )
    # Tracking may have processed only a prefix of the source video. The
    # calibration log has a row for every processed frame, even empty ones.
    last_track_frame = max(timestamps, default=-1)
    calibration_count, homographies = _calibration_summary(calibration_path, segment_id)
    frame_count = calibration_count or last_track_frame + 1
    if frame_count == 0:
        raise ValueError("no processed frames found")
    if last_track_frame >= frame_count or (
        manifest is not None and frame_count > manifest.frame_count
    ):
        raise ValueError("processed frames exceed the calibration or source frame range")

    frames: list[dict[str, Any]] = []
    previous_time = -math.inf
    for frame_idx in range(frame_count):
        t = timestamps.get(frame_idx, frame_idx / fps)
        if t <= previous_time:
            raise ValueError(f"timestamps must increase at frame {frame_idx}")
        frames.append({"frame_idx": frame_idx, "t": t, "players": []})
        previous_time = t

    poses = read_poses_3d(pose_path) if pose_path.exists() else ()
    pose_index = _index_tracks(
        poses, segment_id=segment_id, timestamps=timestamps, name=_POSES
    )
    pose_metadata = _pose_metadata(poses[0]) if poses else None
    for key, pose in pose_index.items():
        if pose.clip_id != clip_id or key not in observed:
            raise ValueError(f"{_POSES}: no matching observation for {pose.clip_id!r} {key}")
        if _pose_metadata(pose) != pose_metadata:
            raise ValueError(f"{_POSES}: inconsistent skeleton layout at frame and track {key}")
        if any(
            len(joint.xyz) != 3 or not all(math.isfinite(value) for value in joint.xyz)
            for joint in pose.joints
        ):
            raise ValueError(f"{_POSES}: invalid joint coordinate at frame and track {key}")

    width_cm = max(y for _, y in COURT_LANDMARK_POINTS_CM)
    length_cm = max(x for x, _ in COURT_LANDMARK_POINTS_CM)
    for key, position in sorted(positions.items()):
        observation = observed.get(key)
        if observation is None:
            raise ValueError(
                f"{_POSITIONS}: no matching observation for frame and track {key}"
            )
        xy = position.clean_court_xy
        if xy is None or not observation.visible:
            continue
        if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
            raise ValueError(f"{_POSITIONS}: invalid court position at frame and track {key}")
        frame_idx, track_id = key
        x_cm, y_cm = xy
        player: dict[str, Any] = {
            "id": track_id,
            "p": [
                (x_cm - length_cm / 2) / _CM_PER_METER,
                0,
                (y_cm - width_cm / 2) / _CM_PER_METER,
            ],
        }
        pose = pose_index.get(key)
        if pose is not None:
            player["pose_sequence"] = pose.sequence_id
            player["joints"] = [list(joint.xyz) for joint in pose.joints]
        frames[frame_idx]["players"].append(player)

    scene: dict[str, Any] = {
        "version": 1,
        "clip_id": clip_id,
        "segment_id": segment_id,
        "fps": fps,
        "units": "meters",
        "court": {
            "length_m": length_cm / _CM_PER_METER,
            "width_m": width_cm / _CM_PER_METER,
        },
        "frames": frames,
    }
    if pose_metadata is not None:
        if homographies:
            camera = estimate_pose_camera(homographies, *image_size)
            if camera is not None:
                pose_metadata["camera"] = asdict(camera)
        scene["pose"] = pose_metadata
    return scene


def export_scene(
    artifact_dir: str | Path,
    *,
    source_video: str | Path | None = None,
    output: str | Path = "scene.json",
) -> Path:
    """Build the replay JSON and replace the output only after a complete write."""
    scene = build_scene(artifact_dir, source_video=source_video)
    destination = Path(output)
    root = Path(artifact_dir)
    input_paths = {
        root / name for name in (_MANIFEST, _POSITIONS, _OBSERVATIONS, _POSES, _CALIBRATIONS)
    }
    if source_video is not None:
        input_paths.add(Path(source_video))
    if destination.resolve() in {path.resolve() for path in input_paths}:
        raise ValueError(f"output would overwrite a scene input: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(scene, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export saved scene reconstruction artifacts to Three.js replay JSON."
    )
    parser.add_argument("artifact_dir", type=Path, help="Processed segment directory")
    parser.add_argument(
        "--source-video",
        type=Path,
        help="Source video for older artifacts without a manifest",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("scene.json"), help="Output path (default: scene.json)"
    )
    args = parser.parse_args(argv)
    try:
        output = export_scene(
            args.artifact_dir, source_video=args.source_video, output=args.output
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Scene written to {output}")


if __name__ == "__main__":
    main()
