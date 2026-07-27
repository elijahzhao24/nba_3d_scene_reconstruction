# Tracking Subsystem

## Purpose

The tracking subsystem converts one continuous basketball video segment into
stable player tracks.

**Input:** a video path, `clip_id`, and `segment_id`.

**Output:** video metadata, track metadata, and a per-frame observation for each
tracked player. Later pipeline stages use the key
`(segment_id, frame_idx, track_id)` and never call RF-DETR or SAM 2 directly.


## Runtime flow

```text
Video segment
    |
    v
Frame extraction
    |
    v
RF-DETR detections
    |
    v 
SAM 2 mask predictions
    |
    v 
Association engine
    |
    v 
Track lifecycle updates (keep, create, or retire ID)
    |
    v 
Player observations
    |
    v 
Artifacts + debug video
```

RF-DETR runs on the first frame and periodically afterward, initially every
five frames. SAM 2 propagates existing tracks on every frame between those
detector checkpoints.

## Components

### Video ingestion and frame extraction

The video reader decodes the segment at its original resolution and writes
frames with deterministic, zero-based names such as `000000.jpg`. It produces
a `VideoManifest` containing FPS, dimensions, frame count, and artifact paths.
Frame indices and timestamps must remain aligned with the source video.

### RF-DETR player detection

The RF-DETR adapter detects on-court players in a frame. It filters by class and
confidence, clips boxes to image bounds, and converts model output into
`PlayerDetection` records.

RF-DETR does not assign or preserve track IDs. Running it periodically instead
of every frame reduces compute while still discovering entering players and
checking SAM 2 tracks. 

### SAM 2 object and mask tracking

The SAM 2 adapter uses an RF-DETR box to initialize or correct a tracked object.
The SAM object ID must equal the subsystem `track_id`. The adapter returns one
binary mask per visible object and hides model tensors and device handling from
the rest of the pipeline.

SAM 2 is the main frame-to-frame tracker. Its inference state and memory bank
carry information from earlier frames to propagate each object's mask. The
application does not manage this memory directly.

How the memory bank works at a high level is that it will store visual and mask features for a specific object. SAM 2 then uses **spatial cross-attention** to retrieve relevant information from previous object memories to locate the object in the next frame.

Mask is cleaned before use such as small
disconnected regions and derives the current bounding box, centroid, mask area,
and bottom-center footpoint.

### Association engine

At each RF-DETR checkpoint, the association engine reconciles current
detections with existing SAM tracks. It is not the frame-to-frame tracker;
instead, it preserves IDs, identifies unmatched detections as possible new
players, and detects masks that may need correction. Matching may use:

- bounding-box IoU;
- detection coverage by the mask;
- normalized center distance.

The engine combines these values into a score and performs one-to-one
assignment. It only returns matches, unmatched detections, and unmatched
tracks; it does not create IDs or change lifecycle state.

### Track ID and lifecycle management

The track manager is the only component allowed to create `track_id` values.
IDs are unique within a segment and increase monotonically.

```text
unmatched detection -> ACTIVE
ACTIVE -> MISSING    when neither a valid mask nor detection is available
MISSING -> ACTIVE    when matched again before the timeout
MISSING -> RETIRED   after max_missing_frames
```

- A match preserves the existing ID and may re-prompt SAM 2 if its mask drifted.
- An unmatched detection creates a new ID and SAM 2 object.
- An unmatched track with a valid SAM mask remains active but is flagged for
  detector disagreement.
- A retired track is never reused.

A player returning after retirement receives a new `track_id`.

### Player observations

The observation builder creates a `PlayerObservation` for every active or
missing track on each processed frame.

Each visible observation contains:

- `segment_id`, `frame_idx`, timestamp, and `track_id`;
- mask reference and mask-derived bounding box;
- centroid, footpoint, and mask area;
- RF-DETR confidence;
- source and quality flags.

Masks are stored separately from the observation records. Coordinates always
refer to the original video resolution and boxes use `[x1, y1, x2, y2]`.

The concrete data contracts are defined in
[`src/nba_3d_scene_reconstruction/tracking/schemas.py`](src/nba_3d_scene_reconstruction/tracking/schemas.py).

### Artifact storage and rendering

```text
artifacts/<clip_id>/<segment_id>/
├── manifest.json
├── tracks.json
├── detections.jsonl
├── observations.jsonl
├── masks/<frame_idx>/<track_id>.png
└── debug/tracking.mp4
```

The debug video overlays each mask, box, `track_id`, centroid, footpoint, and
quality warnings. It must be renderable from saved artifacts without loading
RF-DETR or SAM 2.

## Configuration

- RF-DETR confidence threshold and checkpoint interval;
- association score threshold and weights;
- maximum missing frames;
- minimum and maximum valid mask area.
