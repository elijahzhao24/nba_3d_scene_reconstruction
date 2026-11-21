const CHAINS = [
  ['pelvis', 'right_hip', 'right_knee', 'right_ankle'],
  ['pelvis', 'left_hip', 'left_knee', 'left_ankle'],
  ['pelvis', 'spine', 'neck', 'nose', 'head'],
  ['neck', 'left_shoulder', 'left_elbow', 'left_wrist'],
  ['neck', 'right_shoulder', 'right_elbow', 'right_wrist'],
];

const JOINT_NAMES = new Set(CHAINS.flat());
const REFERENCE_HEIGHT_M = 1.95;
const ANKLE_HEIGHT_M = 0.07;
const REFERENCE_CHAIN_M = REFERENCE_HEIGHT_M - ANKLE_HEIGHT_M;

export function getPoseLayout(metadata) {
  if (metadata?.layout !== 'h36m17' || metadata.units !== 'model_relative'
      || metadata.axes !== 'motionbert_camera_axes' || metadata.coordinate_space !== 'pelvis_relative') {
    return null;
  }

  const names = metadata.joints;
  if (!Array.isArray(names) || names.length !== JOINT_NAMES.size
      || new Set(names).size !== names.length || !names.every((name) => JOINT_NAMES.has(name))) {
    return null;
  }

  const index = Object.fromEntries(names.map((name, i) => [name, i]));
  const bones = CHAINS.flatMap((chain) => chain.slice(1).map((name, i) => [index[chain[i]], index[name]]));
  return { names, index, bones };
}

// The model looks along camera +Z with +Y down. Court +Y is up.
// Yaw zero points toward court -Z, and a positive tilt looks down at the floor.
export function makeCameraBasis(yawDegrees = 0, pitchDegrees = 0) {
  const yaw = yawDegrees * Math.PI / 180;
  const pitch = pitchDegrees * Math.PI / 180;
  const sinYaw = Math.sin(yaw);
  const cosYaw = Math.cos(yaw);
  const sinPitch = Math.sin(pitch);
  const cosPitch = Math.cos(pitch);
  return [
    [cosYaw, -sinYaw * sinPitch, sinYaw * cosPitch],
    [0, -cosPitch, -sinPitch],
    [sinYaw, cosYaw * sinPitch, -cosYaw * cosPitch],
  ];
}

function isValidPose(joints, layout) {
  return Array.isArray(joints) && joints.length === layout.names.length
    && joints.every((joint) => Array.isArray(joint) && joint.length === 3 && joint.every(Number.isFinite));
}

function measurePoseHeight(joints, index) {
  function distance(a, b) {
    const [ax, ay, az] = joints[index[a]];
    const [bx, by, bz] = joints[index[b]];
    return Math.hypot(ax - bx, ay - by, az - bz);
  }

  const torso = distance('pelvis', 'spine') + distance('spine', 'neck') + distance('neck', 'head');
  const leftLeg = distance('left_hip', 'left_knee') + distance('left_knee', 'left_ankle');
  const rightLeg = distance('right_hip', 'right_knee') + distance('right_knee', 'right_ankle');
  return torso + (leftLeg + rightLeg) / 2;
}

function usableHeight(height) {
  return Number.isFinite(height) && height > 1e-6;
}

export function poseScaleKey({ id, pose_sequence: sequence }) {
  return typeof sequence === 'string' && sequence.length > 0 ? JSON.stringify([id, sequence]) : id;
}

export function estimateTrackScales(frames, layout) {
  const samples = new Map();
  for (const frame of frames) {
    for (const player of frame.players) {
      const { joints } = player;
      if (!isValidPose(joints, layout)) continue;
      const height = measurePoseHeight(joints, layout.index);
      if (!usableHeight(height)) continue;
      const key = poseScaleKey(player);
      if (!samples.has(key)) samples.set(key, []);
      samples.get(key).push(height);
    }
  }

  const scales = new Map();
  for (const [key, heights] of samples) {
    heights.sort((a, b) => a - b);
    const right = Math.floor(heights.length / 2);
    const left = Math.floor((heights.length - 1) / 2);
    const medianHeight = heights[left] / 2 + heights[right] / 2;
    scales.set(key, REFERENCE_CHAIN_M / medianHeight);
  }
  return scales;
}

export function groundPose(joints, layout, scale, basis) {
  if (!Number.isFinite(scale) || scale <= 0 || !isValidPose(joints, layout)) {
    return null;
  }
  const chainRatio = measurePoseHeight(joints, layout.index) * scale / REFERENCE_CHAIN_M;
  // A failed lift can stretch the skeleton. Keep the floor marker for extreme outliers.
  if (!(chainRatio >= 0.45 && chainRatio <= 1.6)) {
    return null;
  }

  const points = joints.map(([x, y, z]) => basis.map(
    ([a, b, c]) => (a * x + b * y + c * z) * scale,
  ));
  const left = points[layout.index.left_ankle];
  const right = points[layout.index.right_ankle];
  const offsetX = left[0] / 2 + right[0] / 2;
  const lowerAnkleY = Math.min(left[1], right[1]);
  if (points[layout.index.head][1] <= lowerAnkleY) return null;
  const offsetY = lowerAnkleY - ANKLE_HEIGHT_M;
  const offsetZ = left[2] / 2 + right[2] / 2;
  const grounded = points.map(([x, y, z]) => [x - offsetX, y - offsetY, z - offsetZ]);
  return grounded.every((point) => point.every(Number.isFinite)) ? grounded : null;
}
