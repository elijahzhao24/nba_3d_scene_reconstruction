import assert from 'node:assert/strict';
import test from 'node:test';
import {
  estimateTrackScales, getPoseLayout, groundPose, makeCameraBasis, poseScaleKey,
} from '../src/player-pose.js';
import { readScene } from '../src/scene-data.js';

const jointNames = [
  'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
  'left_ankle', 'spine', 'neck', 'nose', 'head', 'left_shoulder', 'left_elbow',
  'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist',
];

const poseMetadata = {
  layout: 'h36m17',
  units: 'model_relative',
  axes: 'motionbert_camera_axes',
  coordinate_space: 'pelvis_relative',
  joints: jointNames,
};

const standingJoints = [
  [0, 0, 0],
  [-0.1, 0, 0], [-0.1, 0.25, 0], [-0.1, 0.5, 0],
  [0.1, 0, 0], [0.1, 0.25, 0], [0.1, 0.5, 0],
  [0, -0.2, 0], [0, -0.4, 0], [0, -0.45, -0.04], [0, -0.5, 0],
  [0.2, -0.35, 0], [0.3, -0.2, 0], [0.4, 0, 0],
  [-0.2, -0.35, 0], [-0.3, -0.2, 0], [-0.4, 0, 0],
];

function makeScene(players) {
  return {
    version: 1,
    clip_id: 'demo',
    segment_id: 'segment_001',
    fps: 25,
    units: 'meters',
    court: { length_m: 28.65, width_m: 15.24 },
    frames: [{ frame_idx: 0, t: 0, players }],
  };
}

function close(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} differs from ${expected}`);
}

function rotate(matrix, [x, y, z]) {
  return matrix.map(([a, b, c]) => a * x + b * y + c * z);
}

function scaleJoints(factor) {
  return standingJoints.map(joint => joint.map(value => value * factor));
}

test('interprets heading and tilt as a right-handed camera-to-court rotation', () => {
  const level = makeCameraBasis();
  const basis = makeCameraBasis(90, 30);
  rotate(level, [0, -1, 0]).forEach((value, axis) => close(value, [0, 1, 0][axis]));
  rotate(level, [0, 0, 1]).forEach((value, axis) => close(value, [0, 0, -1][axis]));
  const right = rotate(basis, [1, 0, 0]);
  const down = rotate(basis, [0, 1, 0]);
  const forward = rotate(basis, [0, 0, 1]);
  forward.forEach((value, axis) => close(value, [Math.sqrt(3) / 2, -0.5, 0][axis]));
  right.forEach((value, axis) => close(value, [0, 0, 1][axis]));
  const cross = [
    right[1] * down[2] - right[2] * down[1],
    right[2] * down[0] - right[0] * down[2],
    right[0] * down[1] - right[1] * down[0],
  ];
  cross.forEach((value, axis) => close(value, forward[axis]));
});

test('scales a track using its median body-chain length across frames', () => {
  const layout = getPoseLayout(poseMetadata);
  const frames = [
    { players: [{ id: 1, joints: scaleJoints(0.5) }, { id: 2, joints: scaleJoints(2) }] },
    { players: [{ id: 1, joints: standingJoints }, { id: 2, joints: scaleJoints(2) }] },
    { players: [{ id: 1, joints: scaleJoints(2) }, { id: 2 }] },
  ];
  const scales = estimateTrackScales(frames, layout);
  close(scales.get(1), 1.95 - 0.07);
  close(scales.get(2), (1.95 - 0.07) / 2);
});

test('estimates independent scales when a track has multiple lifting sequences', () => {
  const firstRun = { id: 9, pose_sequence: 'test_video:segment_001:9:0', joints: standingJoints };
  const secondRun = { id: 9, pose_sequence: 'test_video:segment_001:9:1', joints: scaleJoints(4) };
  const fallback = { id: 9, joints: scaleJoints(2) };
  const frames = [firstRun, firstRun, secondRun, fallback].map(player => ({ players: [player] }));
  const scales = estimateTrackScales(frames, getPoseLayout(poseMetadata));

  assert.equal(poseScaleKey(fallback), 9);
  assert.notEqual(poseScaleKey(firstRun), poseScaleKey(secondRun));
  assert.notEqual(poseScaleKey({ id: 1, pose_sequence: '23' }), poseScaleKey({ id: 12, pose_sequence: '3' }));
  close(scales.get(poseScaleKey(firstRun)), 1.95 - 0.07);
  close(scales.get(poseScaleKey(secondRun)), (1.95 - 0.07) / 4);
  close(scales.get(poseScaleKey(fallback)), (1.95 - 0.07) / 2);
});

test('anchors the lower ankle and keeps the feet centered over the floor marker', () => {
  const joints = structuredClone(standingJoints);
  joints[jointNames.indexOf('right_ankle')] = [-0.1, 0.5, 0.15];
  joints[jointNames.indexOf('left_ankle')] = [0.15, 0.32, -0.05];
  const placed = groundPose(joints, getPoseLayout(poseMetadata), 2, makeCameraBasis());
  const [right, left, head, pelvis] = ['right_ankle', 'left_ankle', 'head', 'pelvis']
    .map(name => placed[jointNames.indexOf(name)]);
  close(right[1], 0.07);
  close(left[1], 0.43);
  close((left[0] + right[0]) / 2, 0);
  close((left[2] + right[2]) / 2, 0);
  close(head[1] - pelvis[1], 1);
  assert.ok(left[0] > right[0]);
});

test('respects joint order and declines unsupported or unusable poses', () => {
  const layout = getPoseLayout(poseMetadata);
  assert.equal(layout.bones.length, 16);
  assert.equal(getPoseLayout({ ...poseMetadata, units: 'meters' }), null);
  assert.equal(getPoseLayout({ ...poseMetadata, joints: Array(17).fill('head') }), null);
  const basis = makeCameraBasis();
  assert.equal(groundPose(undefined, layout, 1, basis), null);
  assert.equal(groundPose(standingJoints.slice(0, 16), layout, 1, basis), null);
  assert.equal(groundPose(jointNames.map(() => [0, 0, 0]), layout, 1, basis), null);
  assert.equal(groundPose(standingJoints, layout, 4, basis), null);
  assert.equal(groundPose(standingJoints, layout, 0.25, basis), null);
  const upsideDown = standingJoints.map(([x, y, z]) => [x, -y, z]);
  assert.equal(groundPose(upsideDown, layout, 1, basis), null);

  const reversed = getPoseLayout({ ...poseMetadata, joints: [...jointNames].reverse() });
  const placed = groundPose(standingJoints, layout, 1, basis);
  const placedReversed = groundPose([...standingJoints].reverse(), reversed, 1, basis);
  assert.deepEqual(placedReversed, [...placed].reverse());
});

test('reads mixed skeleton and marker rows in a single frame', () => {
  const playerWithPose = { id: 1, p: [2, 0, 3], joints: standingJoints };
  const markerOnly = { id: 2, p: [4, 0, 5] };
  const scene = { ...makeScene([playerWithPose, markerOnly]), pose: poseMetadata };
  const parsed = readScene(JSON.stringify(scene));
  assert.deepEqual(parsed.frames[0].players[0].joints, standingJoints);
  assert.deepEqual(parsed.frames[0].players[1], markerOnly);
});

test('continues to load marker-only scene exports', () => {
  const markerOnly = { id: 1, p: [2, 0, 3] };
  const parsed = readScene(JSON.stringify(makeScene([markerOnly])));
  assert.deepEqual(parsed.frames[0].players, [markerOnly]);
});
