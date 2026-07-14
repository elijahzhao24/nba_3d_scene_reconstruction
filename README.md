# nba_3d_scene_reconstruction

## Summary

The goal of this project is too create a CV pipeline that can ingest a basketball clip, and reconstruct the scene in 3js with human meshes. The 3d scene should accurately recreate the ingestted clip, and allow replay from any angle or perspective.

This can be done by using multiple models to solve sub problems, such as player detection, team clustering, court keypoint detection, ect. When combined together, enough information can be gathered to recreate the scene

## Requirements/Goals


#### Functional:

**1. Player detection + tracking:**
- The system should dtect basketball players in each processed frame. (should be able to distinguish crowd and referees from active players)
- The system should continue a player track even across a limited number of missing detections
- Persistent `track_id` should be assigned to each detected player to uniquely identify them
- Maintain identity over frames, and attempt to preserve the same track id trhough rapid movements, temporary occlusion, player overlap.

**2. Team assignment:**
- The system should be able to cluster players into two teams, which shall be represented be colors
```
cluster 0 → red
cluster 1 → blue
```

**3. Court keypoint detection + Homography:**

- The system shall detect predefined basketball-court landmarks in each relevant frame.
- The system shall convert each player’s image-space ground point into a court-space using an image-to-court homography for each calibrated frame.

**4. Trajectory correction:**

- The system shall identify implausible frame-to-frame player movements, and smooth movement to be realistic
- The system should be able to interpolate missing player positions for short gaps.

**5. Pose estimation:**

- The system shall estimate body joints for each tracked player.

A basic skeleton should include:
```
head or nose
shoulders
elbows
wrists
hips
knees
ankles
```
- Every pose estimate shall be attached to an existing player track.

**6. Three.js visualization**

- Frontend should display a correctly scaled 3D basketball court
- Create a generic rigged human for each active player.
- Players shall be displayed as red, blue to distunguish teams
- Animate player positions and poses based on animation data
- The viewer shall support camera controls, such as zooming, and viewing from others perspective

-  Support manual correction, like player ID merge.
- Create Ball events where user can annoate things like:
```
left-hand contact
right-hand contact
dribble
pass
catch
```

- The system shall generate ball paths for dribbles, passes, and shots.

#### Non Functional:

- The player detector should achieve an acceptable precision/recall level on representative broadcast clips.
- Team assignment should achieve: 95% > accuracy
- viewer frame rate should be at least 30 FPS
- Video processing jobs shall be independently executable by workers.


## High Level Design

This whole pipeline can be seen as a bunch of subcomponents that are combined together to gain all the nesscary info to recreate the scene. We will combine multiple Computer Vision techniques and models to solve these subproblems.

![High Level Diagram](high_level_diagram.png)

### Player Detection

We need to find every active player in the video frame.

**Output:** one bounding box and confidence score per detected player.

**Approach:** use a fine-tuned RF-DETR model trained
on basketball footage so it can separate players from crowd members, referees,
and background people.

### Player Segmentation and ID'ing

keep the same player identity across frames, and store a persistent `track_id` and optional player mask for each tracked player.

Approach: use detector boxes to initialize a tracker, with SAM2-style video
segmentation for masks. The masks help separate overlapping players and produce
cleaner crops for pose and team assignment.

Pixel-level masks are also preferred for temporal memory as there is much less noise compared to bounding boxes.

### Team Assignment

Subproblem: assign each tracked player to one of the two teams.

Output: `team_id = 0` or `team_id = 1`, later rendered as red or blue.

Approach: crop each player's torso, embed the crop with SigLIP, cluster the
embeddings into two groups, then use cosine similarity to the team centroids to
assign each track. This replaces jersey OCR and real player identification.

### Court Keypoint and Homography

Subproblem: map 2D image positions onto the real court.

Output: a per-frame homography that converts image points into court-space
coordinates.

Approach: use a court keypoint model to detect known court landmarks, match
those landmarks to a court template, and compute the image-to-court homography.
This is how each player's foot point becomes a Three.js court position.

### 2D Pose Estimation and 3D Skeleton

Subproblem: estimate each player's body pose.

Output: 2D body joints in image space and a root-relative 3D skeleton.

Approach: run a top-down 2D pose model, such as RTMPose/MMPose, on each tracked
player crop. Then use a temporal 2D-to-3D lifting model (MotionBERT), to
estimate the player's local 3D skeleton.

### Temporal Corrections

At the end, with all the jumping and sharp movements, we may need to clean noisy frame-by-frame using predictions.

Approach: use interpolation, outlier removal, trajectory smoothing, pose
smoothing before
exporting animation data.

### Three.js Render and Scene

Finally we will need to visualize the reconstructed play in 3D.

We will do this with an interactive Three.js replay with a court, red/blue generic players, timeline playback, and camera controls.

### Ball Handling Decision

Ball pose will be handled in the editor rather than inferred fully from the
single source video. In monocular broadcast footage, the ball is small, often
blurred or occluded, and its depth is extremely ambiguous. A detector may find
the ball in some frames, but reliable 3D ball positioning, especially for
passes, shots, rebounds, and dribbles, is too noisy for the first version.

Instead, the editor should let a user manually annotate ball events:

```text
hand contact
dribble
pass
catch
shot
```

Those annotations can be converted into simple ball trajectories for the
Three.js replay. This keeps the main player reconstruction pipeline focused on
the parts that are more reliable from one camera: player tracks, team colors,
court position, and approximate body pose.

### Coordinate Systems

The pipeline should keep coordinate spaces explicit:

| Space | Description |
| --- | --- |
| Image space | Original video pixels, usually `[x, y]` with origin at the top-left. Detector boxes, masks, court keypoints, and 2D pose keypoints live here. |
| Court/world space | Three.js-compatible court coordinates in meters. `Y` is vertical height and the floor plane is `X/Z`. Player roots and ball positions are rendered here. |
| Local skeleton space | Root-relative 3D pose, usually pelvis-relative. This describes body shape but not global court location. |
| Rig bone space | Bone-local rotations for the generic humanoid model. This is the compact form the viewer needs for animation. |

### High-Level Internal Data Schema

The inference pipeline should keep a processing schema. This schema stores raw model
outputs, confidence values, calibration data, intermediate references, cleaned
states, and manual annotations. It will be useful for debugging and reprocessing.


```json
{
  "schema_version": "1.0",
  "clip": {
    "clip_id": "clip_001",
    "source_uri": "uploads/clip_001.mp4",
    "fps": 29.97,
    "frame_count": 420,
    "width": 1920,
    "height": 1080
  },
  "segments": [
    {
      "segment_id": "segment_001",
      "start_frame": 0,
      "end_frame": 419,
      "camera_view": "broadcast_side"
    }
  ],
  "court": {
    "units": "meters",
    "up_axis": "Y",
    "floor_axes": ["X", "Z"],
    "length": 28.65,
    "width": 15.24,
    "landmarks": [
      {
        "landmark_id": 0,
        "name": "court_corner_left_baseline",
        "position": [-14.325, 0.0, -7.62]
      }
    ]
  },
  "teams": [
    {
      "team_id": 0,
      "display_color": "#e5484d",
      "embedding_centroid_ref": "embeddings/team_0.npy"
    },
    {
      "team_id": 1,
      "display_color": "#3b82f6",
      "embedding_centroid_ref": "embeddings/team_1.npy"
    }
  ],
  "tracks": [
    {
      "track_id": 7,
      "role": "player",
      "start_frame": 14,
      "end_frame": 419,
      "team_id": 0,
      "team_confidence": 0.96,
      "appearance_embedding_ref": "embeddings/tracks/7.npy"
    }
  ],
  "frames": [
    {
      "frame_index": 120,
      "timestamp_seconds": 4.004,
      "camera": {
        "segment_id": "segment_001",
        "homography_image_to_court": [
          [0.021, -0.004, -8.42],
          [0.001, 0.018, -5.31],
          [0.00001, -0.00002, 1.0]
        ],
        "calibration_confidence": 0.92,
        "reprojection_error": 2.7
      },
      "players": [
        {
          "track_id": 7,
          "observation": {
            "bbox_xyxy": [822, 315, 946, 708],
            "detection_confidence": 0.96,
            "tracking_confidence": 0.91,
            "mask_ref": "masks/120/7.rle"
          },
          "grounding": {
            "image_footpoint": [873.2, 694.8],
            "footpoint_source": "ankle_midpoint",
            "court_position": [8.21, 0.0, -3.44],
            "court_velocity": [1.18, 0.0, 0.35]
          },
          "pose_2d": {
            "skeleton": "coco17",
            "keypoints": [
              {
                "joint": "left_shoulder",
                "position": [854.2, 397.7],
                "confidence": 0.94
              }
            ]
          },
          "pose_3d_local": {
            "root_joint": "pelvis",
            "joints": [
              {
                "joint": "left_shoulder",
                "position": [-0.20, 0.57, 0.02],
                "confidence": 0.88
              }
            ]
          },
          "world_transform": {
            "root_position": [8.21, 0.0, -3.44],
            "root_rotation_xyzw": [0.0, 0.707, 0.0, 0.707]
          },
          "rig_pose": {
            "bone_rotations_xyzw": {
              "Hips": [0.0, 0.0, 0.0, 1.0],
              "LeftUpperArm": [0.14, -0.21, 0.04, 0.96]
            }
          },
          "quality": {
            "pose_valid": true,
            "court_position_valid": true,
            "interpolated": false,
            "occlusion_score": 0.18
          }
        }
      ]
    }
  ],
  "ball_annotations": [
    {
      "event_id": "ball_evt_001",
      "type": "pass",
      "start_frame": 150,
      "end_frame": 171,
      "from_track_id": 7,
      "to_track_id": 12,
      "control_points": [
        [8.21, 1.8, -3.44],
        [4.10, 2.6, -1.20],
        [1.05, 1.6, 0.80]
      ]
    }
  ]
}
```

### Three.js Animation Schema

The viewer should receive a compact animation schema, not the full processing
schema. It should only include assets, player identities, team colors, root
transforms, rig bone rotations, and any generated ball animation. Detector
boxes, masks, embeddings, homographies, and raw model confidences should stay on
the backend.

```json
{
  "version": "1.0",
  "fps": 29.97,
  "units": "meters",
  "assets": {
    "court": "/models/basketball-court.glb",
    "player": "/models/generic-player.glb"
  },
  "teams": [
    {
      "id": 0,
      "color": "#e5484d"
    },
    {
      "id": 1,
      "color": "#3b82f6"
    }
  ],
  "players": [
    {
      "id": 7,
      "team": 0
    },
    {
      "id": 12,
      "team": 1
    }
  ],
  "frames": [
    {
      "t": 4.004,
      "players": [
        {
          "id": 7,
          "p": [8.21, 0.0, -3.44],
          "q": [0.0, 0.707, 0.0, 0.707],
          "bones": {
            "Hips": [0.0, 0.0, 0.0, 1.0],
            "Spine": [0.03, 0.01, -0.02, 0.999],
            "LeftUpperArm": [0.14, -0.21, 0.04, 0.96],
            "LeftLowerArm": [0.06, 0.12, -0.33, 0.93]
          }
        }
      ],
      "ball": {
        "visible": true,
        "p": [4.10, 2.6, -1.20]
      }
    }
  ],
  "ball_events": [
    {
      "id": "ball_evt_001",
      "type": "pass",
      "start_t": 5.005,
      "end_t": 5.706,
      "from_player": 7,
      "to_player": 12
    }
  ]
}
```

For longer clips, this schema can be chunked by time range or by player track.
It may also move from plain JSON to compressed JSON or a binary format after the
data contract is stable.

## Low Level Design

- Video ingestion and frame extraction
- Player detector interface
- Tracker and segmentation interface
- Team assignment pipeline
- Court landmark and homography calibration
- 2D pose estimator interface
- 3D pose lifter interface
- Temporal correction algorithms
- Skeleton-to-rig retargeting
- Animation export API
- Three.js viewer runtime
- Manual ball annotation editor
