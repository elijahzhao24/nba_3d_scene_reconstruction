# Court Homography and Player Projection

## Goal

Convert each tracked player's image-space floor contact point into a stable
position on a canonical NBA court.

The court model and player tracker solve different parts of this:

- the court model estimates how the camera sees the court;
- the player tracker supplies a floor contact point for every visible player;
- the homography joins them by mapping pixels onto the court plane;
- temporal cleanup removes calibration jitter and implausible player movement.

## End-to-end flow

```text
Video frame
    |
    +--> every K frames: court keypoint detection
    |          |
    |          v
    |     validate landmark schema and confidence
    |          |
    |          v
    |     stabilize landmark pixel positions
    |          |
    |          v
    |     RANSAC homography + quality checks
    |          |
    |          v
    |     most recent valid CourtCalibration
    |
    +--> every frame: tracked player footpoints
               |
               v
        image-to-court projection
               |
               v
        raw player court positions
               |
               v
        per-track outlier removal, interpolation, and smoothing
               |
               v
        saved positions + debug court video + Three.js output
```

The MVP runs court inference on frame `0` and every fifth frame. Intermediate
frames use the latest valid homography. If camera motion makes that visibly
lag, optical flow can update the landmarks between model checkpoints later.

## 1. Detect and calibrate the court

### Model output and version safety

The detector returns one parent prediction with class `court`. Its 33 court
landmarks are nested inside that prediction; they are not separate object
detections.

The expected label order is:

```text
01 02 04 05 07 08 09 10 11 12 13 14 15 16 17 19 21
23 25 26 27 28 29 30 31 32 33 34 35 37 38 40 41
```

Raw nested keypoints contain the landmark `class_id` and label. After
`sv.KeyPoints.from_inference`, the normalized `class_id` describes the parent
court object, not each landmark. Therefore:

- validate landmark IDs and labels from the raw response;
- densify them into the configured 33-element order if necessary;
- use Supervision's `xy` and confidence arrays by array index only after that
  model version's ordering has been validated;
- apply the identical confidence mask to detected image points and canonical
  court vertices.

`sports.basketball.CourtConfiguration` supplies the canonical NBA geometry.
Its coordinate plane is 94 by 50 feet, with `x` running between baselines and
`y` running between sidelines.

### Homography estimation

For each checkpoint:

1. Remove landmarks below a confidence threshold
2. Require at least four non-collinear correspondences; use at least six
   inliers for an accepted production calibration.
3. Optionally update an exponential moving average for each landmark's image position.
4. Fit `court_to_image` using `cv2.findHomography` with RANSAC.
5. Validate the result, then invert it to obtain `image_to_court` (to map player image pixel -> court positions).

Estimate court-to-image first so the robust fitting threshold can use pixels.

For RANSACE filtering, A calibration is accepted only if it has enough well-distributed inliers, is
finite and invertible, and preserves court orientation. A failed checkpoint can reuse the last good calibration briefly. 

Initial values to tune from debug footage:

```text
checkpoint interval       5 frames
court confidence          0.30
keypoint confidence       0.50
RANSAC threshold          5-6 px
minimum inliers           6
minimum inlier ratio      0.60
maximum calibration age   10 frames
```

## 2. Project each player's floor position

Every visible `PlayerObservation` already contains a `footpoint_xy` in the
original video resolution. Prefer anchors in this order:

1. detected ankle or foot contact points;
2. bottom of the cleaned SAM2 mask;
3. bounding-box bottom-center as a fallback.

Given image point `(u, v)` and `image_to_court` matrix `H`:

```python
point = np.array([[[u, v]]], dtype=np.float32)
court_xy = cv2.perspectiveTransform(point, H)[0, 0]
```

Reject positions outside the court plus a small margin.
The homography maps only the floor plane, so it should not be used to predict the 3d position of something off the ground (head, hand, jumping player, basketball)

If court coordinates are stored in feet, convert them to centered Three.js
meters with:

```python
world_x = court_x_ft * 0.3048 - 14.325
world_y = 0.0
world_z = court_y_ft * 0.3048 - 7.620
```

The projection must record the calibration frame and age used. If no valid
calibration exists, store a missing position rather than inventing one.

## 3. Reduce noise and smooth trajectories

Use two separate cleanup layers.

### Calibration stabilization

- Smooth landmark pixel positions by landmark index before fitting `H`.
- Use RANSAC/MAGSAC to remove individual incorrect landmarks.
- Reject high reprojection error, orientation flips, and implausible jumps.
- Hold the last good calibration across a short detection failure.
- Reset landmark filters and homography state on a camera cut.

This corrects movement shared by every projected player. 

### Per-player path cleanup

After projection, group positions by `track_id` and process each continuous
track independently:

1. Calculate frame-to-frame court speed.
2. Detect "teleports" using a threshold such as median speed plus a
   multiple of median absolute deviation.
3. Mark suspicious frames as 'missing'. Pad their boundries by removing a small number of neighboring frames as well.
4. Linearly interpolate only short gaps with valid positions on both sides.
5. Apply a light Savitzky-Golay filter independently to court `x` and `y`.

Default with a  nine-frame, second-order Savitzky-Golay filter after jump/suspicious frame removal. This is an offline
centered filter and uses roughly four future frames. 

Never smooth across retired/reassigned track IDs, or long missing
intervals. Preserve raw and cleaned positions so smoothing can be tuned
without rerunning the models.

If a live output is added, replace smoothing  with a causal EMA, One Euro filter, or Kalman filter.

## Runtime ownership

```text
src/nba_3d_scene_reconstruction/court/
├── configuration.py      # model/schema pin, NBA vertices, thresholds
├── detector.py           # inference and raw-response validation
├── calibration.py        # landmark filtering, RANSAC, hold/reset state
├── projector.py          # footpoint -> court/world position
├── smoothing.py          # offline per-track path cleanup
├── schemas.py            # persisted data contracts
└── debug.py              # frame overlay and sports.basketball minimap
```

A segment-level pipeline runs player tracking and court calibration against
the same original frame and joins their results by `(segment_id, frame_idx)`.

Minimal control flow:

```python
for frame_idx, frame in frames:
    if frame_idx == 0 or frame_idx % checkpoint_interval == 0:
        calibration = calibrator.detect_and_update(frame, frame_idx)
    else:
        calibration = calibrator.current(frame_idx)

    observations = player_pipeline.process_frame(frame, frame_idx)
    raw_positions += projector.project(observations, calibration)

clean_positions = path_smoother.clean(raw_positions)
```

## Data and artifacts

Keep calibration, raw projection, and cleaned output separate.

```python
@dataclass(frozen=True)
class CourtCalibration:
    segment_id: str
    frame_idx: int
    source_frame_idx: int | None
    valid: bool
    source: str                 # detected, held, invalid
    image_to_court: tuple[float, ...] | None
    court_to_image: tuple[float, ...] | None
    keypoint_count: int
    inlier_count: int
    median_error_px: float | None
    age_frames: int | None
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlayerCourtPosition:
    segment_id: str
    frame_idx: int
    track_id: int
    footpoint_image_xy: tuple[float, float]
    raw_court_xy: tuple[float, float] | None
    clean_court_xy: tuple[float, float] | None
    world_xz_m: tuple[float, float] | None
    calibration_source_frame_idx: int | None
    quality_flags: tuple[str, ...] = ()
```

```text
artifacts/<clip_id>/<segment_id>/
├── calibrations.jsonl
├── player_court_positions_raw.jsonl
├── player_court_positions_clean.jsonl
└── debug/
    ├── court_reprojection.mp4
    └── court_minimap.mp4
```

Write calibration records for detected, held, and invalid frames. Flatten
NumPy matrices only when serializing them.

## Birds eye court debug with `sports.basketball`

Use `CourtConfiguration`, `draw_court`, and `draw_points_on_court` to render a
canonical NBA minimap with tracker IDs:

```python
from sports import MeasurementUnit
from sports.basketball import (
    CourtConfiguration,
    League,
    draw_court,
    draw_points_on_court,
)

config = CourtConfiguration(
    league=League.NBA,
    measurement_unit=MeasurementUnit.FEET,
)

court = draw_court(config=config)
court = draw_points_on_court(
    config=config,
    xy=player_court_xy,
    labels=[str(track_id) for track_id in tracker_ids],
    court=court,
)
```

Render raw positions in red and cleaned positions in green. On the original
video, also draw detected court landmarks and canonical landmarks reprojected
through `court_to_image`; the two sets should nearly overlap.

Useful failure patterns:

- all players jump together: court calibration failure;
- one player jumps: footpoint, occlusion, or player-track failure;
- players are mirrored: landmark order or homography direction is wrong;
- center is accurate but edges drift: weak landmark coverage/extrapolation;
- dots lag during a pan: landmark smoothing is too aggressive;
- a huge cut-frame jump: calibration state was not reset.

## Implementation order

1. Pin model version, expected raw landmark schema, NBA geometry, and tests.
2. Parse/densify raw keypoints and build matching image/court arrays.
3. Implement stabilized RANSAC homography estimation and quality metrics.
4. Add five-frame checkpoint, short hold, expiration, and cut reset behavior.
5. Project `PlayerObservation.footpoint_xy` and persist raw positions.
6. Implement per-track jump removal, short-gap interpolation, and smoothing.
7. Add original-frame reprojection and `sports.basketball` minimap videos.
8. Tune thresholds on clips containing pans, zooms, occlusion, and cuts.
