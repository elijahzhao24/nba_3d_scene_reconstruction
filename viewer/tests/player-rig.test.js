import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { Bone, Group, Quaternion, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { bindPlayerRig, placePlayerRig } from '../src/player-rig.js';

const restPositions = {
  pelvis: [null, [0, 1, 0]],
  spine: ['pelvis', [0, 0.22, 0]],
  neck: ['spine', [0, 0.35, 0]],
  head: ['neck', [0, 0.18, 0]],
  left_hip: ['pelvis', [0.15, -0.02, 0]],
  left_knee: ['left_hip', [0, -0.42, 0]],
  left_ankle: ['left_knee', [0, -0.46, 0]],
  right_hip: ['pelvis', [-0.15, -0.02, 0]],
  right_knee: ['right_hip', [0, -0.42, 0]],
  right_ankle: ['right_knee', [0, -0.46, 0]],
  left_shoulder: ['neck', [0.2, -0.04, 0]],
  left_elbow: ['left_shoulder', [0.22, -0.23, 0]],
  left_wrist: ['left_elbow', [0.14, -0.25, 0]],
  right_shoulder: ['neck', [-0.2, -0.04, 0]],
  right_elbow: ['right_shoulder', [-0.22, -0.23, 0]],
  right_wrist: ['right_elbow', [-0.14, -0.25, 0]],
};

const names = Object.keys(restPositions);
const layout = { index: Object.fromEntries(names.map((name, index) => [name, index])) };
const UP = new Vector3(0, 1, 0);

function fixture() {
  const object = new Group();
  const bones = {};
  for (const name of names) {
    const [parent, position] = restPositions[name];
    const bone = new Bone();
    bone.name = name;
    bone.position.fromArray(position);
    (parent ? bones[parent] : object).add(bone);
    bones[name] = bone;
  }
  object.updateWorldMatrix(true, true);
  const rig = bindPlayerRig(object);
  assert.ok(rig);
  const points = names.map((name) => position(bones[name]).toArray());
  return { object, bones, rig, points };
}

function position(bone) {
  return bone.getWorldPosition(new Vector3());
}

function assertNear(actual, expected, tolerance = 1e-8) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);
}

function assertVector(vector, expected) {
  vector.toArray().forEach((value, i) => assertNear(value, expected[i]));
}

function assertDirection(bones, points, parent, child) {
  const expected = new Vector3().fromArray(points[layout.index[child]])
    .sub(new Vector3().fromArray(points[layout.index[parent]])).normalize();
  const actual = position(bones[child]).sub(position(bones[parent])).normalize();
  assertVector(actual, expected.toArray());
}

test('preserves the neutral rig and translates it to the floor marker', () => {
  const { rig, object, bones, points } = fixture();
  assert.equal(placePlayerRig(rig, [3, 0.2, -2], points, layout), true);
  assertVector(object.position, [3, 0.2, -2]);
  assertVector(position(bones.left_ankle), [3.15, 0.3, -2]);
  assertVector(position(bones.right_ankle), [2.85, 0.3, -2]);
  for (const bone of Object.values(bones)) {
    assertNear(bone.quaternion.angleTo(new Quaternion()), 0, 1e-7);
  }
});

test('aims bone rotations while retaining bind lengths and grounding the lowest sole', () => {
  const { rig, bones, points } = fixture();
  points[layout.index.left_knee] = [0.25, 0.65, 0.55];
  points[layout.index.left_ankle] = [0.25, 0.25, 0.23];
  points[layout.index.right_elbow] = [-0.7, 1.35, 0.26];
  points[layout.index.right_wrist] = [-0.65, 1.42, 0.88];
  assert.equal(placePlayerRig(rig, [4, 0, -3], points, layout), true);

  for (const [parent, child] of [
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
  ]) {
    assertDirection(bones, points, parent, child);
    const [, bindTranslation] = restPositions[child];
    assertVector(bones[child].position, bindTranslation);
    assertNear(position(bones[child]).distanceTo(position(bones[parent])), Math.hypot(...bindTranslation));
  }
  const left = position(bones.left_ankle);
  const right = position(bones.right_ankle);
  assertNear(Math.min(left.y, right.y), 0.1);
  assertNear((left.x + right.x) / 2, 4);
  assertNear((left.z + right.z) / 2, -3);
  for (const ankle of [bones.left_ankle, bones.right_ankle]) {
    assertVector(UP.clone().applyQuaternion(ankle.getWorldQuaternion(new Quaternion())), [0, 1, 0]);
  }
});

test('uses anatomical left and right to turn the body and flat feet together', () => {
  const { rig, bones, points } = fixture();
  const turn = new Quaternion().setFromAxisAngle(UP, Math.PI / 2);
  const turned = points.map((point) => new Vector3().fromArray(point).applyQuaternion(turn).toArray());
  assert.equal(placePlayerRig(rig, [0, 0, 0], turned, layout), true);
  for (const name of ['pelvis', 'neck', 'left_ankle', 'right_ankle']) {
    const rotation = bones[name].getWorldQuaternion(new Quaternion());
    assertVector(new Vector3(0, 0, 1).applyQuaternion(rotation), [1, 0, 0]);
  }
  assertVector(position(bones.left_ankle), [0, 0.1, -0.15]);
});

test('scrubbing to a pose is deterministic and degenerate limbs reset locally', () => {
  const { rig, object, bones, points } = fixture();
  const during = structuredClone(points);
  during[layout.index.left_elbow] = [1.4, 1.5, -0.2];
  during[layout.index.left_wrist] = [1.2, 1.8, -0.9];
  const repeated = structuredClone(points);
  repeated[layout.index.left_wrist] = repeated[layout.index.left_elbow];
  assert.equal(placePlayerRig(rig, [3, 0, 7], repeated, layout), true);
  const recorded = names.map((name) => bones[name].quaternion.clone());
  const originalPosition = object.position.toArray();
  assert.equal(placePlayerRig(rig, [-2, 0.5, 5], during, layout), true);
  assert.equal(placePlayerRig(rig, [3, 0, 7], repeated, layout), true);
  assertVector(object.position, originalPosition);
  names.forEach((name, i) => assertNear(bones[name].quaternion.angleTo(recorded[i]), 0, 1e-7));
});

test('declines incompatible rigs and unusable source poses', () => {
  const { rig, object, bones, points } = fixture();
  assert.equal(bindPlayerRig(new Group()), null);
  assert.equal(placePlayerRig(rig, [0, 0], points, layout), false);
  assert.equal(placePlayerRig(rig, [0, 0, 0], points, { index: {} }), false);
  const invalid = points.map(() => [0, 0, 0]);
  assert.equal(placePlayerRig(rig, [0, 0, 0], invalid, layout), false);
  object.add(bones.neck);
  assert.equal(bindPlayerRig(object), null);
});

test('the retargeted mesh grounds an actual shoe after the legs bend', async () => {
  const file = await readFile(new URL('../assets/player.glb', import.meta.url));
  const glb = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const { scene } = await new GLTFLoader().parseAsync(glb, '');
  const rig = bindPlayerRig(scene);
  assert.ok(rig);
  const mesh = scene.getObjectByProperty('isSkinnedMesh', true);
  assert.ok(mesh);

  const points = names.map((name) => position(rig.bones[name]).toArray());
  points[layout.index.left_knee] = [0.3, 0.7, 0.52];
  points[layout.index.left_ankle] = [0.3, 0.33, 0.28];
  points[layout.index.right_knee] = [-0.16, 0.67, 0.27];
  points[layout.index.right_ankle] = [-0.16, 0.3, 0.1];
  assert.equal(placePlayerRig(rig, [2, 0, -1], points, layout), true);
  assert.ok(scene.position.y < -0.03, 'the bent legs require a vertical offset');
  scene.updateMatrixWorld(true);
  const vertex = new Vector3();
  let floor = Infinity;
  for (let i = 0; i < mesh.geometry.attributes.position.count; i += 1) {
    mesh.getVertexPosition(i, vertex);
    floor = Math.min(floor, mesh.localToWorld(vertex).y);
  }
  assertNear(floor, 0, 1e-6);
});
