"""Diagnostic renderers for raw 2D and local 3D pose artifacts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schemas import Pose2D, Pose3D
from .skeletons import H36M_17, HALPE_26


def draw_pose_2d(
    frame: np.ndarray, poses: tuple[Pose2D, ...], *, threshold: float = 0.3
) -> np.ndarray:
    result = frame.copy()
    bones = HALPE_26.bones
    for pose in poses:
        joints = {joint.name: joint for joint in pose.joints}
        color = (
            (53 * pose.track_id) % 180 + 50,
            (97 * pose.track_id) % 180 + 50,
            (151 * pose.track_id) % 180 + 50,
        )
        for bone in bones:
            a, b = joints.get(bone.parent), joints.get(bone.child)
            if a and b and a.score >= threshold and b.score >= threshold:
                cv2.line(
                    result,
                    tuple(map(round, a.image_xy)),
                    tuple(map(round, b.image_xy)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for joint in joints.values():
            if joint.score >= threshold:
                cv2.circle(
                    result, tuple(map(round, joint.image_xy)), 3, color, -1, cv2.LINE_AA
                )
        anchor = next((j for j in joints.values() if j.score >= threshold), None)
        if anchor:
            cv2.putText(
                result,
                f"ID {pose.track_id}",
                tuple(map(round, anchor.image_xy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    return result


def render_pose_2d_video(
    frames_dir: str | Path, poses: tuple[Pose2D, ...], output: str | Path, *, fps: float
) -> Path:
    grouped: dict[int, list[Pose2D]] = {}
    for pose in poses:
        grouped.setdefault(pose.frame_idx, []).append(pose)
    return _write_video(
        frames_dir,
        output,
        fps,
        lambda frame, index: draw_pose_2d(frame, tuple(grouped.get(index, ()))),
    )


def render_pose_3d_videos(
    poses: tuple[Pose3D, ...],
    output_dir: str | Path,
    *,
    fps: float,
    frame_count: int | None = None,
) -> tuple[Path, ...]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    by_track: dict[int, list[Pose3D]] = {}
    for pose in poses:
        by_track.setdefault(pose.track_id, []).append(pose)
    outputs = []
    for track_id, track in by_track.items():
        by_frame = {pose.frame_idx: pose for pose in track}
        diagnostic_frame_count = frame_count or (max(by_frame) + 1)
        destination = root / f"track_{track_id:03d}.mp4"
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 640)
        )
        if not writer.isOpened():
            raise OSError(f"could not create {destination}")
        for frame_idx in range(diagnostic_frame_count):
            canvas = np.full((640, 640, 3), 245, dtype=np.uint8)
            cv2.line(canvas, (40, 520), (600, 520), (180, 180, 180), 1)
            cv2.line(canvas, (320, 50), (320, 600), (180, 180, 180), 1)
            pose = by_frame.get(frame_idx)
            if pose is None:
                cv2.putText(
                    canvas,
                    f"track {track_id} frame {frame_idx} | missing",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (20, 20, 20),
                    1,
                )
                writer.write(canvas)
                continue
            joints = {j.name: j for j in pose.joints}
            points = {
                name: (round(320 + j.xyz[0] * 180), round(520 - j.xyz[1] * 180))
                for name, j in joints.items()
            }
            for bone in H36M_17.bones:
                cv2.line(
                    canvas,
                    points[bone.parent],
                    points[bone.child],
                    (50, 90, 180),
                    3,
                    cv2.LINE_AA,
                )
            for endpoint in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
                cv2.circle(canvas, points[endpoint], 7, (20, 20, 220), -1)
                cv2.putText(
                    canvas,
                    endpoint,
                    points[endpoint],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (20, 20, 20),
                    1,
                )
            cv2.putText(
                canvas,
                f"track {track_id} frame {frame_idx} | model-relative X/Y view",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
            )
            writer.write(canvas)
        writer.release()
        outputs.append(destination)
    return tuple(outputs)


def _write_video(
    frames_dir: str | Path, output: str | Path, fps: float, render
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(frames_dir).glob("*.jpg"))
    writer = None
    try:
        for index, path in enumerate(files):
            frame = cv2.imread(str(path))
            if frame is None:
                raise OSError(f"could not read {path}")
            rendered = render(frame, index)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(destination),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (rendered.shape[1], rendered.shape[0]),
                )
                if not writer.isOpened():
                    raise OSError(f"could not create {destination}")
            writer.write(rendered)
    finally:
        if writer is not None:
            writer.release()
    return destination
