import assert from 'node:assert/strict';
import test from 'node:test';
import { getPoseLayout, makeCameraBasis } from '../src/player-pose.js';
import { playerPov } from '../src/player-pov.js';

const joints = {
  pelvis: [0, 0, 0],
  right_hip: [-0.1, 0, 0],
  right_knee: [-0.1, 0.25, 0],
  right_ankle: [-0.1, 0.5, 0],
  left_hip: [0.1, 0, 0],
  left_knee: [0.1, 0.25, 0],
  left_ankle: [0.1, 0.5, 0],
  spine: [0, -0.2, 0],
  neck: [0, -0.4, 0],
  nose: [0, -0.45, -0.04],
  head: [0, -0.5, 0],
  left_shoulder: [0.2, -0.35, 0],
  left_elbow: [0.3, -0.2, 0],
  left_wrist: [0.4, 0, 0],
  right_shoulder: [-0.2, -0.35, 0],
  right_elbow: [-0.3, -0.2, 0],
  right_wrist: [-0.4, 0, 0],
};

const layout = getPoseLayout({
  layout: 'h36m17',
  units: 'model_relative',
  axes: 'motionbert_camera_axes',
  coordinate_space: 'pelvis_relative',
  joints: Object.keys(joints),
});

function makeFrame(overrides = {}) {
  const p = [2, 0, 6];
  const players = [{ id: 7, p, joints: Object.values({ ...joints, ...overrides }) }];
  return { frame_idx: 40, t: 40 / 25, players };
}

function makePose(yaw = 0) {
  return { layout, scales: new Map([[7, 1]]), basis: makeCameraBasis(yaw) };
}

function assertVector(actual, expected) {
  for (const [i, coordinate] of expected.entries()) {
    assert.ok(Math.abs(actual[i] - coordinate) < 1e-9, `coordinate ${i} is ${actual[i]}, expected ${coordinate}`);
  }
}

test('places a level POV just in front of the reconstructed head', () => {
  const view = playerPov(makeFrame(), 7, makePose());
  assertVector(view.eye, [2, 0.99, 6.1]);
  assertVector(view.direction, [0, 0, 1]);
});

test('turns the POV with the reconstructed player in court coordinates', () => {
  const view = playerPov(makeFrame(), 7, makePose(90));
  assertVector(view.eye, [1.9, 0.99, 6]);
  assertVector(view.direction, [-1, 0, 0]);
});

test('passes authored look targets to the POV camera', () => {
  const frame = makeFrame();
  frame.players[0].look_at = { target: [12.725, 3.048, 0], weight: 0.5 };
  const view = playerPov(frame, 7, makePose());
  assertVector(view.eye, [2, 0.99, 6.1]);
  assert.deepEqual(view.lookAt, frame.players[0].look_at);
});

test('falls back to the hips when the shoulders collapse in the pose', () => {
  const frame = makeFrame({ left_shoulder: joints.right_shoulder });
  assertVector(playerPov(frame, 7, makePose()).direction, [0, 0, 1]);
});

test('returns no POV for an absent track or an unusable pose', () => {
  assert.equal(playerPov(makeFrame(), 8, makePose()), null);
  assert.equal(playerPov(makeFrame(), 7, null), null);
  assert.equal(playerPov({ players: [{ id: 7, p: [0, 0, 0] }] }, 7, makePose()), null);
  const frame = makeFrame({
    left_shoulder: joints.right_shoulder,
    left_hip: joints.right_hip,
  });
  assert.equal(playerPov(frame, 7, makePose()), null);
});
