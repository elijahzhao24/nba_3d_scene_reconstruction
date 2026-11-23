# Pipeline options and diagnostics

Use the [Quickstart](../README.md#quickstart) to run tracking, court projection,
and pose estimation in Docker and export `artifacts/<clip_id>/scene.json`.
Use `./reconstruct --help` to see the CLI options and `./reconstruct --build`
to refresh the Docker image.

## Run tracking and court projection directly

Follow the [native GPU setup](../LLDS/TRACKING.md#configuration) and configure
the player and court models in [`.env.example`](../.env.example). From the
repository root, run:

```bash
uv run --extra gpu --env-file .env tracking-demo path/to/clip.mp4 \
  --court-projection --max-frames 60
```

Court detection runs on every frame by default. Use `--court-interval 5` to
reduce hosted court-model calls. The pipeline can hold the last valid
calibration for up to 15 frames. Run each continuous camera segment separately
to reset calibration across cuts.

For standalone pose processing, see the [pose estimation guide](../LLDS/POSE_ESTIMATION.md#commands).
To replay saved segment artifacts, see [Export a saved segment](VIEWER.md#export-a-saved-segment).

## Inspect the artifacts

The tracking command writes these outputs under `artifacts/<clip_id>/segment_001/`:

| Output | Contents |
| --- | --- |
| `debug/court_debug.webm` | Source video with masks, track IDs, footpoints, and detected and reprojected court landmarks on the left. Cleaned trajectories on a top-down court on the right. |
| `observations.jsonl` | Final player observations. |
| `court_detections.jsonl`, `calibrations.jsonl` | Court detections and calibration metrics. |
| `player_court_positions_raw.jsonl`, `player_court_positions_clean.jsonl` | Raw and smoothed player positions in centimeters. Use the raw positions to diagnose errors or retune smoothing. |

Batch tracking finalizes masks and track IDs before rendering the debug video.
It saves raw detections, masks, and lifecycle events, suppresses sustained
duplicate masks, and can recover up to 30 earlier frames for a late-confirmed
player. Recovered observations carry `sam2_backward_recovery` provenance.

## Runtime notes

- SAM 2 keeps video frames and tracking history in CPU RAM by default while
  inference runs on the GPU. This saves GPU memory at the cost of host memory
  and processing time. CUDA autocast keeps cached state compatible with model
  weights across prompting and propagation. Retiring a track clears its SAM 2
  state. Custom runners can disable offloading with `offload_video_to_cpu=False`
  or `offload_state_to_cpu=False` on `Sam2PlayerTracker`.
- `PlayerTrackingPipeline.process_frame()` returns masks. Read
  `.observations` for the latest `PlayerObservation` records, and pass
  the source video FPS when creating the tracker. The standalone default is 30.
  The observation footpoint uses the largest mask component and the median
  x-coordinate in its lowest 5%. A missing mask falls back to the detector
  box’s bottom center and sets `bbox_footpoint_fallback` on the missing
  observation.
- Detection confidence is recorded on detector matches and new detections.
  Mask-derived observations carry `sam2_propagation` provenance, and re-prompts
  affect subsequent frames. Live observations leave `mask_ref` unset.
- RF-DETR accepts the five player and player-action classes. Confirmed tracks
  can be replaced if they go missing, lose detector support, or have low
  confidence. Repeated off-court mask rejections use a shorter timeout than
  ordinary mask loss. Unavailable calibration resets the off-court streak.
- `SceneReconstructionPipeline` joins observations with per-frame court
  calibrations and returns a `SceneFrame`. The projector records calibration
  source, age, and quality alongside `raw_court_xy` and `clean_court_xy`.
  Invalid calibration, missing footpoints, projection at infinity, or positions
  beyond the court margin (100 cm by default) yield a null position. The debug
  minimap uses cleaned positions after centered smoothing.

See the [tracking design](../LLDS/TRACKING.md#configuration) for detector,
association, and lifecycle thresholds, and the [court design](../LLDS/COURT_HOMOGRAPHY_CALIBRATION.md#3-reduce-noise-and-smooth-trajectories)
for calibration and smoothing defaults.
