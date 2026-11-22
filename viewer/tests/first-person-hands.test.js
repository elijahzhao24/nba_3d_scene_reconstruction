import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { createHandsGeometry } from '../src/first-person-hands.js';

test('extracts closed steel arms and hands without the uniform, head, or torso', async () => {
  const file = await readFile(new URL('../assets/player.glb', import.meta.url));
  const glb = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const { scene } = await new GLTFLoader().parseAsync(glb, '');
  const meshes = scene.getObjectsByProperty('isSkinnedMesh', true);
  const mesh = meshes.find(({ material }) => material.metalness > 0);
  const uniform = meshes.find(({ material }) => material.metalness === 0);
  assert.ok(mesh && uniform);
  assert.ok(mesh.material.roughness < uniform.material.roughness);
  assert.equal(createHandsGeometry(uniform), null);

  const originalIndex = mesh.geometry.getIndex();
  const originalTriangles = Array.from(originalIndex.array);
  const hands = createHandsGeometry(mesh);
  assert.ok(hands);
  assert.notEqual(hands, mesh.geometry);
  assert.notEqual(hands.getIndex(), originalIndex);
  assert.deepEqual(Array.from(mesh.geometry.getIndex().array), originalTriangles);
  const triangles = hands.getIndex().array;
  assert.ok(triangles.length > 0 && triangles.length < originalTriangles.length);
  assert.equal(triangles.length % 3, 0);
  assert.equal(hands.drawRange.count, triangles.length);

  const joints = hands.getAttribute('skinIndex');
  const weights = hands.getAttribute('skinWeight');
  const wrists = new Set();
  for (const vertex of new Set(triangles)) {
    let armWeight = 0;
    for (let influence = 0; influence < 4; influence += 1) {
      const bone = mesh.skeleton.bones[joints.getComponent(vertex, influence)].name;
      const weight = weights.getComponent(vertex, influence);
      if (/^(left|right)_(shoulder|elbow|wrist)$/.test(bone)) armWeight += weight;
      if (weight > 0 && bone.endsWith('_wrist')) wrists.add(bone);
    }
    assert.ok(armWeight >= 0.5, `vertex ${vertex} lies outside the arms and hands`);
  }
  assert.deepEqual(wrists, new Set(['left_wrist', 'right_wrist']));

  const edgeCounts = new Map();
  for (let offset = 0; offset < triangles.length; offset += 3) {
    const [a, b, c] = triangles.subarray(offset, offset + 3);
    for (const [from, to] of [[a, b], [b, c], [c, a]]) {
      const edge = from < to ? `${from}:${to}` : `${to}:${from}`;
      edgeCounts.set(edge, (edgeCounts.get(edge) ?? 0) + 1);
    }
  }
  assert.ok([...edgeCounts.values()].every((count) => count === 2), 'arm surfaces should have no open edges');

  mesh.geometry = mesh.geometry.toNonIndexed();
  assert.equal(createHandsGeometry(mesh), null);
});
