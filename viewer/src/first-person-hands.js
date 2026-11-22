const ARM_BONES = new Set([
  'left_shoulder', 'left_elbow', 'left_wrist',
  'right_shoulder', 'right_elbow', 'right_wrist',
]);
const MIN_ARM_WEIGHT = 0.5;

export function createHandsGeometry(mesh) {
  if (!mesh?.isSkinnedMesh || Array.isArray(mesh.material)) return null;

  const geometry = mesh.geometry;
  const index = geometry.getIndex();
  const position = geometry.getAttribute('position');
  const joints = geometry.getAttribute('skinIndex');
  const weights = geometry.getAttribute('skinWeight');
  if (!index || index.count % 3 !== 0 || !position || !joints || !weights
      || joints.itemSize !== 4 || weights.itemSize !== 4
      || joints.count !== position.count || weights.count !== position.count) return null;

  const armIndices = new Set();
  mesh.skeleton?.bones.forEach((bone, joint) => {
    if (ARM_BONES.has(bone.name)) armIndices.add(joint);
  });
  if (!armIndices.size) return null;

  const isArmVertex = new Uint8Array(position.count);
  for (let vertex = 0; vertex < position.count; vertex += 1) {
    let armWeight = 0;
    for (let influence = 0; influence < 4; influence += 1) {
      if (armIndices.has(joints.getComponent(vertex, influence))) {
        armWeight += weights.getComponent(vertex, influence);
      }
    }
    isArmVertex[vertex] = Number(armWeight >= MIN_ARM_WEIGHT);
  }

  // Keep complete triangles so the extracted hands still deform with the same rig.
  const armTriangles = [];
  for (let offset = 0; offset < index.count; offset += 3) {
    const a = index.getX(offset);
    const b = index.getX(offset + 1);
    const c = index.getX(offset + 2);
    if (isArmVertex[a] && isArmVertex[b] && isArmVertex[c]) {
      armTriangles.push(a, b, c);
    }
  }
  if (!armTriangles.length) return null;

  const hands = geometry.clone();
  hands.setIndex(armTriangles);
  hands.setDrawRange(0, armTriangles.length);
  return hands;
}
