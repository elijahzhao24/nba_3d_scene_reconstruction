import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BALL_RADIUS_M,
  readBallKeyframes,
  sampleBall,
  validateBallKeyframes,
  writeBallKeyframes,
} from '../src/ball-keyframes.js';

const scene = {
  clip_id: 'clip_001',
  segment_id: 'segment_001',
  frames: [
    { frame_idx: 0, t: 0, players: [{ id: 1 }] },
    { frame_idx: 6, t: 0.2, players: [{ id: 1 }, { id: 2 }] },
    { frame_idx: 12, t: 0.9, players: [{ id: 1 }, { id: 2 }] },
    { frame_idx: 16, t: 1.7, players: [{ id: 1 }, { id: 2 }] },
    { frame_idx: 32, t: 2.9, players: [{ id: 1 }, { id: 2 }] },
    { frame_idx: 48, t: 4.8, players: [{ id: 1 }, { id: 2 }] },
  ],
};

const hand = (track_id = 1, hand = 'right') => ({ track_id, hand });
const hold = (frame_idx, anchor = hand()) => ({ kind: 'hold', frame_idx, anchor });
const motion = (kind, start_frame, end_frame, to, from = hand()) => ({
  kind, start_frame, end_frame, from, to,
});
const shot = (start, end, hoop = 'right', from = hand()) => motion('shot', start, end, { hoop }, from);
const rim = (side) => [side === 'left' ? -12 : 12, 3.048, 0];
const atTime = (time, progress, endTime) => time + (endTime - time) * progress;

function assertPoint(actual, expected) {
  assert.ok(actual);
  for (const [index, coordinate] of expected.entries()) {
    assert.ok(Math.abs(actual[index] - coordinate) < 1e-9, `${actual} ≠ ${expected}`);
  }
}

test('round-trips paired sidecars sorted by start frame', () => {
  const first = hold(6);
  const second = motion('dribble', 12, 16, hand(1, 'left'));
  const third = shot(16, 32, 'left');
  const unordered = [third, first, second];
  const saved = writeBallKeyframes(unordered, scene);
  assert.deepEqual(JSON.parse(saved), {
    version: 2,
    clip_id: scene.clip_id,
    segment_id: scene.segment_id,
    keyframes: [first, second, third],
  });
  assert.deepEqual(readBallKeyframes(saved, scene), [first, second, third]);
  assert.deepEqual(unordered, [third, first, second]);
  assert.deepEqual(readBallKeyframes(writeBallKeyframes([], scene), scene), []);
});

test('rejects incompatible and malformed ball files without changing the scene', () => {
  const validFile = JSON.parse(writeBallKeyframes([hold(6)], scene));
  assert.throws(() => readBallKeyframes('}', scene), /valid JSON/);
  assert.throws(() => readBallKeyframes(JSON.stringify({ ...validFile, version: 3 }), scene), /version 1 or version 2/);
  assert.throws(() => readBallKeyframes(JSON.stringify({ ...validFile, clip_id: 'other' }), scene), /different clip/);
  assert.throws(() => readBallKeyframes(JSON.stringify({ ...validFile, keyframes: null }), scene), /list of keyframes/);
});

test('requires known frames, valid anchors, and a strictly later endpoint for all motion', () => {
  assert.throws(() => validateBallKeyframes([hold(7)], scene), /unknown frame/);
  assert.throws(() => validateBallKeyframes([hold(6), motion('pass', 6, 12, hand(2))], scene), /already holds/);
  assert.throws(() => validateBallKeyframes([{ kind: 'fly' }], scene), /supported ball movement/);
  assert.throws(() => validateBallKeyframes([hold(0, hand(2))], scene), /player visible/);
  assert.throws(() => validateBallKeyframes([hold(6, hand(1, 'wrong'))], scene), /left or right hand/);
  assert.throws(() => validateBallKeyframes([shot(6, 32, 'wrong')], scene), /Choose a hoop/);
  assert.throws(() => validateBallKeyframes([motion('dribble', 12, 6, hand())], scene), /end frame after/);
  assert.throws(() => validateBallKeyframes([motion('dribble', 6, 6, hand())], scene), /end frame after/);
  assert.throws(() => validateBallKeyframes([motion('pass', 0, 16, hand(), hand(2))], scene), /visible at frame 0/);
});

test('a motion allows a shared boundary and rejects any intervening hold or action', () => {
  const bounce = motion('dribble', 6, 32, hand(1, 'left'));
  assert.throws(() => validateBallKeyframes([hold(16), bounce], scene), /inside the dribble/);
  assert.throws(() => validateBallKeyframes([bounce, shot(12, 48)], scene), /inside the dribble/);
  assert.deepEqual(validateBallKeyframes([bounce, hold(32, hand(1, 'left'))], scene), [bounce, hold(32, hand(1, 'left'))]);
});

test('converts linked version 1 ends to adjacent pairs while keeping their timing', () => {
  const v1 = {
    version: 1,
    clip_id: scene.clip_id,
    segment_id: scene.segment_id,
    keyframes: [
      hold(0),
      hold(6),
      { kind: 'dribble', frame_idx: 12, anchor: hand(1, 'left') },
      { kind: 'pass', frame_idx: 16, anchor: hand(2) },
      { kind: 'shot', frame_idx: 32, anchor: { hoop: 'right' } },
      hold(48, hand(2, 'left')),
    ],
  };
  const entries = readBallKeyframes(JSON.stringify(v1), scene);
  assert.deepEqual(entries, [
    hold(0),
    motion('dribble', 6, 12, hand(1, 'left')),
    motion('pass', 12, 16, hand(2), hand(1, 'left')),
    shot(16, 32, 'right', hand(2)),
    hold(48, hand(2, 'left')),
  ]);
  const movingHand = ({ track_id, hand: side }, t) => [t, track_id, side === 'left' ? 1 : -1];
  assertPoint(sampleBall(entries, scene, 0.1, movingHand, rim), [0.1, 1, -1]);
  assertPoint(sampleBall(entries, scene, 0.9, movingHand, rim), [0.9, 1, 1]);
  assertPoint(sampleBall(entries, scene, 1.7, movingHand, rim), [1.7, 2, -1]);
  assert.ok(sampleBall(entries, scene, 2.9, movingHand, rim)[1] < rim('right')[1]);
  assert.equal(sampleBall(entries, scene, 3, movingHand, rim), null);
  assertPoint(sampleBall(entries, scene, 4.8, movingHand, rim), [4.8, 2, 1]);
  assert.throws(() => readBallKeyframes(JSON.stringify({ ...v1, keyframes: v1.keyframes.slice(2) }), scene),
    /before the dribble/);
});

test('holds follow the selected animated hand until an explicit action begins', () => {
  const keys = [hold(6), motion('dribble', 12, 16, hand(2, 'left')), hold(32, hand(1, 'left'))];
  const movingHand = ({ track_id, hand: side }, t) => [t, track_id, side === 'left' ? 1 : -1];
  assert.equal(sampleBall(keys, scene, 0.1, movingHand, rim), null);
  assertPoint(sampleBall(keys, scene, 0.2, movingHand, rim), [0.2, 1, -1]);
  assertPoint(sampleBall(keys, scene, 0.8, movingHand, rim), [0.8, 1, -1]);
  assertPoint(sampleBall(keys, scene, 0.9, movingHand, rim), [0.9, 1, -1]);
  assertPoint(sampleBall(keys, scene, atTime(0.9, 0.5, 1.7), movingHand, rim), [1.3, BALL_RADIUS_M, 0]);
  assertPoint(sampleBall(keys, scene, 2, movingHand, rim), [2, 2, 1]);
  assertPoint(sampleBall(keys, scene, 2.9, movingHand, rim), [2.9, 1, 1]);
  assertPoint(sampleBall(keys, scene, 4, movingHand, rim), [4, 1, 1]);
  assert.equal(sampleBall(keys, scene, 4, () => null, rim), null);
});

test('passes linearly between hands sampled at release and arrival timestamps', () => {
  const keys = [motion('pass', 6, 32, hand(2, 'left'))];
  const samples = [];
  function movingHand({ track_id, hand: side }, time) {
    samples.push([track_id, side, time]);
    return track_id === 1 ? [time, 1, 2] : [10 + time, 3, -2];
  }
  assertPoint(sampleBall(keys, scene, atTime(0.2, 0.25, 2.9), movingHand, rim), [3.375, 1.5, 1]);
  assert.deepEqual(samples, [[1, 'right', 0.2], [2, 'left', 2.9]]);
  assertPoint(sampleBall(keys, scene, 2.9, movingHand, rim), [12.9, 3, -2]);
  assertPoint(sampleBall(keys, scene, 3.4, movingHand, rim), [13.4, 3, -2]);

  const missingReceiver = ({ track_id }) => track_id === 1 ? [0, 1, 0] : null;
  assertPoint(sampleBall(keys, scene, 0.2, missingReceiver, rim), [0, 1, 0]);
  assert.equal(sampleBall(keys, scene, 1.1, missingReceiver, rim), null);
});

test('dribbles along two straight segments with the ball surface meeting the court', () => {
  const keys = [motion('dribble', 6, 32, hand(1, 'left'))];
  const resolveHand = ({ hand: side }) => side === 'right' ? [0, 1.2, 0] : [4, 1.6, 2];
  assertPoint(
    sampleBall(keys, scene, atTime(0.2, 0.25, 2.9), resolveHand, rim),
    [1, (1.2 + BALL_RADIUS_M) / 2, 0.5],
  );
  assertPoint(
    sampleBall(keys, scene, atTime(0.2, 0.5, 2.9), resolveHand, rim),
    [2, BALL_RADIUS_M, 1],
  );
  assertPoint(
    sampleBall(keys, scene, atTime(0.2, 0.75, 2.9), resolveHand, rim),
    [3, (1.6 + BALL_RADIUS_M) / 2, 1.5],
  );
  assertPoint(sampleBall(keys, scene, 2.9, resolveHand, rim), [4, 1.6, 2]);
});

test('a shot descends through its chosen hoop and hides after the net drop', () => {
  const keys = [shot(6, 32, 'left'), hold(48, hand(2))];
  const resolveHand = () => [1, 2, 1.5];
  const nearRim = sampleBall(keys, scene, atTime(0.2, 0.78, 2.9), resolveHand, rim);
  assert.ok(nearRim[1] > rim('left')[1]);
  assertPoint(sampleBall(keys, scene, atTime(0.2, 0.8, 2.9), resolveHand, rim), rim('left'));
  const throughNet = sampleBall(keys, scene, atTime(0.2, 0.9, 2.9), resolveHand, rim);
  assertPoint([throughNet[0], throughNet[2]], [-12, 0]);
  assert.ok(throughNet[1] < rim('left')[1]);
  assert.ok(sampleBall(keys, scene, 2.9, resolveHand, rim)[1] < throughNet[1]);
  assert.equal(sampleBall(keys, scene, 3, resolveHand, rim), null);
  assertPoint(sampleBall(keys, scene, 4.8, resolveHand, rim), [1, 2, 1.5]);
  assert.equal(sampleBall(keys, scene, 2, resolveHand, () => null), null);
});

test('imports an older shot with an arc setting and saves it using the shared arc', () => {
  const entry = shot(6, 32);
  const saved = JSON.parse(writeBallKeyframes([entry], scene));
  saved.keyframes[0].extra_arc_m = 2;
  const imported = readBallKeyframes(JSON.stringify(saved), scene);
  assert.deepEqual(imported, [entry]);
  assert.deepEqual(JSON.parse(writeBallKeyframes(imported, scene)).keyframes, [entry]);
});

test('every shot gets 1.5 m more lift halfway through its flight', () => {
  const keys = [shot(6, 32)];
  const resolveHand = () => [1, 2, 0];
  const position = (progress) => sampleBall(keys, scene, atTime(0.2, progress, 2.9), resolveHand, rim);

  assertPoint(position(0), [1, 2, 0]);
  assertPoint(position(0.4), [6.5, 4.904, 0]);
  assertPoint(position(0.8), [12, 3.048, 0]);
  assertPoint(position(1), [12, 2.398, 0]);
});
