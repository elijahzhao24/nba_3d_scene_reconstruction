import { groundPose, poseScaleKey } from './player-pose.js';

const MIN_BODY_WIDTH_M = 0.05;
const EYE_FORWARD_M = 0.1;
const EYE_BELOW_HEAD_M = 0.08;

function facingFromPair(left, right) {
  // Cross the anatomical right-to-left axis with court up to get forward.
  const x = right[2] - left[2];
  const z = left[0] - right[0];
  const length = Math.hypot(x, z);
  return length >= MIN_BODY_WIDTH_M ? [x / length, z / length] : null;
}

function blendFacing(shoulders, hips) {
  if (!shoulders) return hips;
  if (!hips || shoulders[0] * hips[0] + shoulders[1] * hips[1] <= 0) return shoulders;

  const x = 2 * shoulders[0] + hips[0];
  const z = 2 * shoulders[1] + hips[1];
  const length = Math.hypot(x, z);
  return [x / length, z / length];
}

// This estimates body facing, since the exported skeleton does not capture gaze.
export function playerPov(frame, trackId, pose) {
  if (!pose?.layout || !Number.isSafeInteger(trackId)) return null;
  const player = frame?.players?.find(({ id }) => id === trackId);
  if (!player) return null;

  const { layout, scales, basis } = pose;
  const points = groundPose(player.joints, layout, scales.get(poseScaleKey(player)), basis);
  if (!points) return null;

  const { left_shoulder, right_shoulder, left_hip, right_hip, head } = layout.index;
  const shoulders = facingFromPair(points[left_shoulder], points[right_shoulder]);
  const hips = facingFromPair(points[left_hip], points[right_hip]);
  const facing = blendFacing(shoulders, hips);
  if (!facing) return null;

  const [x, z] = facing;
  const [rootX, rootY, rootZ] = player.p;
  const [headX, headY, headZ] = points[head];
  return {
    eye: [
      rootX + headX + x * EYE_FORWARD_M,
      rootY + headY - EYE_BELOW_HEAD_M,
      rootZ + headZ + z * EYE_FORWARD_M,
    ],
    direction: [x, 0, z],
  };
}
