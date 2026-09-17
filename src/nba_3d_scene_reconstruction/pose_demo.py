"""CLI for independently rerunnable tracked pose stages."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

from .pose.debug import render_pose_2d_video, render_pose_3d_videos
from .pose.estimator import MMPosePose2DEstimator
from .pose.io import (
    file_sha256,
    git_revision,
    read_manifest,
    read_observations,
    read_poses_2d,
    write_json,
    write_jsonl,
)
from .pose.lifter import MotionBertPoseLifter
from .pose.pipeline import estimate_tracked_poses, lift_prepared_sequences
from .pose.preparation import prepare_pose_sequences
from .pose.skeletons import H36M_17, HALPE_26
from .tracking.schemas import VideoManifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("2d", "lift", "all"):
        stage = subparsers.add_parser(name)
        stage.add_argument("artifact_root", type=Path)
        stage.add_argument("--manifest", type=Path)
        stage.add_argument("--frames-dir", type=Path)
        stage.add_argument("--source-video", type=Path)
        stage.add_argument("--device", default="cuda:0")
        if name in ("2d", "all"):
            stage.add_argument("--pose2d-python", default=sys.executable)
            stage.add_argument("--mmpose-config", required=True)
            stage.add_argument("--mmpose-checkpoint", required=True)
            stage.add_argument("--confidence-threshold", type=float, default=0.3)
            stage.add_argument("--maximum-gap-seconds", type=float, default=0.1)
        if name in ("lift", "all"):
            stage.add_argument("--pose3d-python", default=sys.executable)
            stage.add_argument("--motionbert-repository", required=True)
            stage.add_argument("--motionbert-config", required=True)
            stage.add_argument("--motionbert-checkpoint", required=True)
    worker2d = subparsers.add_parser("_worker_2d")
    worker2d.add_argument("payload", type=Path)
    worker3d = subparsers.add_parser("_worker_3d")
    worker3d.add_argument("payload", type=Path)
    return parser


def _manifest(args: argparse.Namespace) -> VideoManifest:
    path = args.manifest or args.artifact_root / "video_manifest.json"
    if path.exists():
        manifest = read_manifest(path)
    elif args.source_video:
        capture = cv2.VideoCapture(str(args.source_video))
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open source video: {args.source_video}")
        observations = read_observations(args.artifact_root / "observations.jsonl")
        segment = (
            observations[0].segment_id if observations else args.artifact_root.name
        )
        manifest = VideoManifest(
            args.artifact_root.parent.name,
            segment,
            str(args.source_video),
            str(args.frames_dir or args.artifact_root / "frames"),
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        capture.release()
    else:
        raise FileNotFoundError(
            f"missing {path}; pass --source-video for older artifacts"
        )
    if args.frames_dir:
        manifest = VideoManifest(
            **{**asdict(manifest), "frames_dir": str(args.frames_dir)}
        )
    return manifest


def _external(python: str, worker: str, payload: Path) -> None:
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = (
        source_root + os.pathsep + environment.get("PYTHONPATH", "")
    )
    subprocess.run(
        [python, "-m", "nba_3d_scene_reconstruction.pose_demo", worker, str(payload)],
        check=True,
        env=environment,
    )


def _run_2d(args: argparse.Namespace, manifest: VideoManifest) -> None:
    payload = args.artifact_root / ".pose2d_job.json"
    write_json(
        payload,
        {
            "artifact_root": str(args.artifact_root),
            "manifest": asdict(manifest),
            "config": args.mmpose_config,
            "checkpoint": args.mmpose_checkpoint,
            "device": args.device,
            "confidence_threshold": args.confidence_threshold,
            "maximum_gap_seconds": args.maximum_gap_seconds,
        },
    )
    try:
        _external(args.pose2d_python, "_worker_2d", payload)
    finally:
        payload.unlink(missing_ok=True)


def _worker_2d(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text())
    root = Path(payload["artifact_root"])
    manifest = VideoManifest(**payload["manifest"])
    observations = read_observations(root / "observations.jsonl")
    estimator = MMPosePose2DEstimator(
        payload["config"], payload["checkpoint"], device=payload["device"]
    )
    poses = estimate_tracked_poses(manifest, observations, estimator)
    prepared = prepare_pose_sequences(
        poses,
        fps=manifest.fps,
        confidence_threshold=payload["confidence_threshold"],
        maximum_gap_seconds=payload["maximum_gap_seconds"],
    )
    write_jsonl(root / "poses_2d.jsonl", poses)
    write_jsonl(root / "poses_2d_prepared.jsonl", prepared)
    write_json(root / "skeleton_halpe26.json", HALPE_26.to_dict())
    write_json(root / "skeleton_h36m17.json", H36M_17.to_dict())
    write_json(
        root / "pose_2d_run.json",
        {
            "model": "RTMPose-M HALPE-26",
            "config": payload["config"],
            "config_sha256": file_sha256(payload["config"]),
            "checkpoint": payload["checkpoint"],
            "checkpoint_sha256": file_sha256(payload["checkpoint"]),
            "coordinates": "original_image_pixels",
            "joint_layout": HALPE_26.id,
            "box_source": "finalized_player_observation",
            "confidence_threshold": payload["confidence_threshold"],
            "maximum_gap_seconds": payload["maximum_gap_seconds"],
        },
    )
    render_pose_2d_video(
        manifest.frames_dir, poses, root / "debug" / "pose_2d.mp4", fps=manifest.fps
    )


def _run_lift(args: argparse.Namespace, manifest: VideoManifest) -> None:
    payload = args.artifact_root / ".pose3d_job.json"
    write_json(
        payload,
        {
            "artifact_root": str(args.artifact_root),
            "manifest": asdict(manifest),
            "repository": args.motionbert_repository,
            "config": args.motionbert_config,
            "checkpoint": args.motionbert_checkpoint,
            "device": args.device.replace(":0", ""),
        },
    )
    try:
        _external(args.pose3d_python, "_worker_3d", payload)
    finally:
        payload.unlink(missing_ok=True)


def _worker_3d(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text())
    root = Path(payload["artifact_root"])
    manifest = VideoManifest(**payload["manifest"])
    poses = read_poses_2d(root / "poses_2d_prepared.jsonl", prepared=True)
    lifter = MotionBertPoseLifter(
        payload["repository"],
        payload["config"],
        payload["checkpoint"],
        device=payload["device"],
    )
    lifted = lift_prepared_sequences(poses, lifter)
    write_jsonl(root / "poses_3d_local.jsonl", lifted)
    write_json(
        root / "pose_3d_run.json",
        {
            "model": "MotionBERT-Lite MB_ft_h36m_global_lite",
            "repository": payload["repository"],
            "repository_revision": git_revision(payload["repository"]),
            "config": payload["config"],
            "config_sha256": file_sha256(payload["config"]),
            "checkpoint": payload["checkpoint"],
            "checkpoint_sha256": file_sha256(payload["checkpoint"]),
            "units": "model_relative",
            "coordinate_space": "pelvis_relative",
            "axes": "motionbert_camera_axes",
            "window_size": 243,
            "stride": 81,
            "normalization": "motionbert_crop_scale_once_per_sequence",
            "root_joint": "pelvis",
        },
    )
    render_pose_3d_videos(
        lifted,
        root / "debug" / "pose_3d",
        fps=manifest.fps,
        frame_count=manifest.frame_count,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.stage == "_worker_2d":
        _worker_2d(args.payload)
        return
    if args.stage == "_worker_3d":
        _worker_3d(args.payload)
        return
    manifest = _manifest(args)
    if args.stage in ("2d", "all"):
        _run_2d(args, manifest)
    if args.stage in ("lift", "all"):
        _run_lift(args, manifest)


if __name__ == "__main__":
    main()
