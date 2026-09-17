# Pose Estimation and Local 3D Lifting

This subsystem runs after tracking finalization. It consumes the extracted source
frames, `observations.jsonl`, and `video_manifest.json`; it does not use court
positions or homographies.

## Data flow

```text
finalized observations + source frames
  -> RTMPose-M HALPE-26
  -> poses_2d.jsonl
  -> confidence filtering, short-gap interpolation, HALPE-to-H36M conversion
  -> poses_2d_prepared.jsonl
  -> MotionBERT-Lite
  -> poses_3d_local.jsonl
```

All outputs preserve `clip_id`, `segment_id`, `frame_idx`, `timestamp_seconds`,
and `track_id`. Raw 2D coordinates are original-image pixels. Lifted coordinates
are pelvis-relative MotionBERT outputs in model-relative units and camera axes.
They are not court coordinates or meters.

The skeleton definition artifacts give every joint and bone a stable anatomical
name. `left_hand` and `right_hand` resolve to wrist joints for now. A future rig
will replace these endpoints with editable palm sockets; no palm rotation or
finger reconstruction is inferred here.

## Model environments

The base project deliberately does not install MMPose or MotionBERT. Create two
Python 3.11 environments and pass their executables to `pose-demo`.
The launcher adds this repository's `src` directory to each worker's
`PYTHONPATH`, so do not install the base project into these environments. Install
NumPy, OpenCV, and the model-specific packages directly instead; this also keeps
older model dependencies isolated from the base project's NumPy 2 environment.

For 2D inference, use MMPose revision
`759b39c13fea6ba094afc1fa932f51dc1b11cbf9`, its installation instructions for
your CUDA/PyTorch combination, and this config:

```text
configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py
```

Use the matching checkpoint:

```text
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth
```

For lifting, use MotionBERT revision
`705d3a95354db8bdb696b3492e47a3b5537174ff`, config
`configs/pose3d/MB_ft_h36m_global_lite.yaml`, and the released matching
`FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` checkpoint linked by the
upstream inference documentation. The checkpoint is separate from the
representation-only MotionBERT-Lite weights.

## Commands

Run both stages on a newly generated segment:

```bash
uv run pose-demo all artifacts/<clip>/segment_001 \
  --pose2d-python /path/to/pose2d-env/bin/python \
  --mmpose-config /path/to/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py \
  --mmpose-checkpoint /path/to/rtmpose-halpe26.pth \
  --pose3d-python /path/to/pose3d-env/bin/python \
  --motionbert-repository /path/to/MotionBERT \
  --motionbert-config /path/to/MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml \
  --motionbert-checkpoint /path/to/MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
```

Use `pose-demo 2d` or `pose-demo lift` to rerun only one stage. `lift` reads
`poses_2d_prepared.jsonl`, so it does not load tracking or RTMPose. For artifacts
created before manifest persistence was added, pass `--source-video` and
optionally `--frames-dir`.

Outputs include versioned skeleton JSON, checkpoint hashes, preprocessing
metadata, a synchronized `debug/pose_2d.mp4`, and one local-skeleton video per
track under `debug/pose_3d/`.
