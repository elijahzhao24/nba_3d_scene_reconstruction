# Three.js viewer

Follow the [Quickstart](../README.md#2-open-the-scene-in-the-viewer) to start
the viewer and use **Open scene** to load a `scene.json` file. The viewer reads
the file locally and uses its frame times for playback.

## Playback controls

- **Top down** provides a view like the tracking minimap. **Orbit** lets you
  rotate and zoom around the court.
- **Players** animates the bundled rig when poses are available.
  **Skeletons** shows the 17-joint poses. **Markers** shows court positions.
  Tracks with missing or implausibly scaled poses keep their floor markers.
- **Pose alignment** adjusts yaw and down tilt. The viewer uses the estimated
  source camera angle when the export includes one. You can align older scenes
  manually.
- **Fullscreen** keeps the camera controls and timeline available. Click
  **Exit fullscreen** or press `Esc` to return to editing.

### Player POV

Open a scene with poses, select **Player POV**, and choose a track under
**Follow player**. The camera uses the estimated head position and body facing
direction with a level base angle and a 95° field of view. It follows authored
gaze targets when available. If the track’s pose drops out, the viewer switches
to Orbit and resumes POV when the pose returns.

## Add ball keyframes

Open **Ball editor** below the timeline. Use **Hold in hand** to attach the ball
to a player’s left or right hand until the next action.

At the release frame, choose **Start dribble**, **Start pass**, or **Start shot**
and select the releasing hand. Scrub to a later frame and add the corresponding
end marker as the next keyframe.

| End marker | Select | Ball path |
| --- | --- | --- |
| **End dribble** | Receiving hand | One bounce halfway between the keyframes. |
| **End pass** | Receiving hand | A straight path between the hands. |
| **End shot** | Hoop | A high arc through the rim, ending below it. |

After a dribble or pass, the ball follows the receiving hand until the next
action. Click a keyframe row to seek to its frame. **Export ball JSON** saves
edits separately. Reopen the same scene and use **Import ball JSON** to load
them. The import also converts older ball files to start and end markers.

## Export a saved segment

The Docker pipeline exports a scene automatically. For a segment generated
with [direct court projection](PIPELINE.md#run-tracking-and-court-projection-directly),
run this from the repository root, replacing `clip` with the video’s filename
stem:

```bash
uv run export-scene artifacts/clip/segment_001 \
  --output artifacts/clip/scene.json
```

For older segments without `video_manifest.json`, pass the original video with
`--source-video`. In the viewer, use **Open scene** to select the exported file.

## Scene format

The exporter writes each visible player’s track ID and cleaned court position
as centered `[x, 0, z]` coordinates in meters. When available, it includes
joints from `poses_3d_local.jsonl` with metadata for their pelvis-relative
camera axes and model units. Saved court calibration can supply one estimated
source camera orientation per segment.

The viewer maps joint directions to the bundled gray player model with a
yellow uniform, preserving each track’s rig proportions. It normalizes scale
once per lifting sequence and grounds the lower foot, so jump height is unresolved.
Source camera alignment uses one orientation through broadcast pans.

After editing [the player model generator](../viewer/tools/generate_player.py),
run `python3 viewer/tools/generate_player.py` from the repository root to
regenerate the GLB.
