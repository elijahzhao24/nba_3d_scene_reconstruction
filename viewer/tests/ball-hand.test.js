import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone } from 'three/addons/utils/SkeletonUtils.js';
import { createBallHandResolver, createRigBallHandResolver } from '../src/ball-hand.js';
import { BALL_RADIUS_M } from '../src/ball-keyframes.js';
import { estimateTrackScales, getPoseLayout, makeCameraBasis } from '../src/player-pose.js';
import { bindPlayerRig } from '../src/player-rig.js';

const JOINTS = {
  pelvis: [0, 0, 0], right_hip: [-0.1, 0, 0], right_knee: [-0.1, 0.25, 0],
  right_ankle: [-0.1, 0.5, 0], left_hip: [0.1, 0, 0], left_knee: [0.1, 0.25, 0],
  left_ankle: [0.1, 0.5, 0], spine: [0, -0.2, 0], neck: [0, -0.4, 0],
  nose: [0, -0.45, -0.04], head: [0, -0.5, 0], left_shoulder: [0.2, -0.35, 0],
  left_elbow: [0.3, -0.2, 0], left_wrist: [0.4, -0.1, 0], right_shoulder: [-0.2, -0.35, 0],
  right_elbow: [-0.3, -0.2, 0], right_wrist: [-0.4, -0.1, 0],
};
const layout = getPoseLayout({
  layout: 'h36m17',
  units: 'model_relative',
  axes: 'motionbert_camera_axes',
  coordinate_space: 'pelvis_relative',
  joints: Object.keys(JOINTS),
});

function player(x) {
  return { id: 7, p: [x, 0, 3], joints: Object.values(JOINTS) };
}

function makePose(frames) {
  return { layout, scales: estimateTrackScales(frames, layout), basis: makeCameraBasis() };
}

test('resolves either wrist and interpolates movement between adjacent frames', () => {
  const frames = [
    { frame_idx: 4, t: 1, players: [player(2)] },
    { frame_idx: 5, t: 1.5, players: [player(4)] },
  ];
  const resolve = createBallHandResolver({ frames }, makePose(frames));
  const left = { track_id: 7, hand: 'left' };
  const right = { track_id: 7, hand: 'right' };
  assert.ok(resolve(left, 1)[0] > resolve(right, 1)[0]);
  const halfway = resolve(left, 1.25);
  assert.deepEqual(halfway, resolve(left, 1).map((value, axis) => value + (axis === 0 ? 1 : 0)));
  assert.equal(resolve(left, 0.9), null);
  assert.equal(resolve({ ...right, track_id: 8 }, 1), null);
});

test('only uses neighboring known wrists and prefers a rig socket when available', () => {
  const frames = [
    { frame_idx: 4, t: 1, players: [player(2)] },
    { frame_idx: 5, t: 2, players: [] },
    { frame_idx: 6, t: 3, players: [player(4)] },
    { frame_idx: 8, t: 4, players: [player(6)] },
  ];
  let rigCalls = 0;
  const resolve = createBallHandResolver({ frames }, makePose(frames), (p, points, jointLayout, hand) => {
    rigCalls += 1;
    assert.equal(jointLayout, layout);
    assert.equal(hand, 'left');
    return [p[0], points[layout.index.left_wrist][1], p[2]];
  });
  const left = { track_id: 7, hand: 'left' };
  assert.equal(resolve(left, 1)[0], 2);
  assert.equal(resolve(left, 1)[0], 2);
  assert.equal(rigCalls, 1);
  assert.equal(resolve(left, 1.5), null);
  assert.equal(resolve(left, 2), null);
  assert.equal(resolve(left, 3.5), null);
  assert.equal(resolve(left, 4)[0], 6);
});

test('rig ball sockets follow either posed palm without moving the visible player', async () => {
  const file = await readFile(new URL('../assets/player.glb', import.meta.url));
  const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const { scene } = await new GLTFLoader().parseAsync(buffer, '');
  const visible = bindPlayerRig(scene);
  const rig = bindPlayerRig(clone(scene));
  assert.ok(visible && rig);
  const resolve = createRigBallHandResolver(rig);
  const position = [2, 0, 4];
  const points = layout.names.map((name) => (visible.bones[name] ?? visible.bones.head)
    .getWorldPosition(new Vector3()).toArray());
  const leftBall = resolve(position, points, layout, 'left');
  const rightBall = resolve(position, points, layout, 'right');
  assert.ok(leftBall[0] > rightBall[0]);
  assert.ok(leftBall[1] > BALL_RADIUS_M);
  assert.ok(rightBall[1] > BALL_RADIUS_M);
  const wrist = rig.bones.left_wrist.getWorldPosition(new Vector3());
  assert.ok(wrist.distanceTo(new Vector3(...leftBall)) > BALL_RADIUS_M);
  assert.ok(wrist.distanceTo(new Vector3(...leftBall)) < 0.2);
  assert.deepEqual(scene.position.toArray(), [0, 0, 0]);
  assert.deepEqual(visible.bones.left_wrist.getWorldPosition(new Vector3()).toArray(), points[layout.index.left_wrist]);
});
