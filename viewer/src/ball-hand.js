import { Vector3 } from 'three';
import { BALL_RADIUS_M } from './ball-keyframes.js';
import { groundPose, poseScaleKey } from './player-pose.js';
import { placePlayerRig } from './player-rig.js';
import { frameAtTime } from './scene-data.js';

const HANDS = ['left', 'right'];
const PALM_CENTER_FRACTION = 0.14;
const PALM_THICKNESS_M = 0.03;

// The bundled model's hands extend the forearms past the wrist joints, with
// their front surface facing model +Z in the bind pose.
export function createRigBallHandResolver(rig) {
  const sockets = new Map();
  for (const hand of HANDS) {
    const wrist = rig.bones[`${hand}_wrist`];
    const elbow = rig.bones[`${hand}_elbow`];
    const extension = wrist.worldToLocal(elbow.getWorldPosition(new Vector3())).negate();
    const palmNormal = new Vector3(0, 0, 1).projectOnPlane(extension.clone().normalize()).normalize();
    const socket = extension.multiplyScalar(PALM_CENTER_FRACTION)
      .addScaledVector(palmNormal, BALL_RADIUS_M + PALM_THICKNESS_M);
    sockets.set(hand, { wrist, socket });
  }

  const world = new Vector3();
  let previousPose;
  let placed = false;
  return (position, points, layout, hand) => {
    const handSocket = sockets.get(hand);
    if (!handSocket) return null;
    if (!previousPose || previousPose.points !== points || previousPose.position !== position || previousPose.layout !== layout) {
      placed = placePlayerRig(rig, position, points, layout);
      previousPose = { position, points, layout };
    }
    if (!placed) return null;
    const location = handSocket.wrist.localToWorld(world.copy(handSocket.socket));
    return [location.x, Math.max(location.y, BALL_RADIUS_M), location.z];
  };
}

export function createBallHandResolver(scene, pose, resolveRigHand) {
  if (!scene?.frames?.length || !pose) return () => null;

  const { frames } = scene;
  const { layout, scales, basis } = pose;
  const cachedPoses = new WeakMap();

  function resolveInFrame(frame, anchor) {
    const player = frame.players.find(({ id }) => id === anchor.track_id);
    if (!player) return null;
    let sample = cachedPoses.get(player);
    if (!sample) {
      const { joints } = player;
      const points = groundPose(joints, layout, scales.get(poseScaleKey(player)), basis);
      sample = { points, hands: new Map() };
      cachedPoses.set(player, sample);
    }
    if (!sample.points) return null;
    if (!sample.hands.has(anchor.hand)) {
      const wrist = sample.points[layout.index[`${anchor.hand}_wrist`]];
      const position = resolveRigHand?.(player.p, sample.points, layout, anchor.hand)
        ?? [player.p[0] + wrist[0], Math.max(player.p[1] + wrist[1], BALL_RADIUS_M), player.p[2] + wrist[2]];
      sample.hands.set(anchor.hand, position);
    }
    return sample.hands.get(anchor.hand);
  }

  return (anchor, time) => {
    if (!HANDS.includes(anchor?.hand) || !Number.isFinite(time)
        || time < frames[0].t || time > frames.at(-1).t) return null;
    const i = frameAtTime(frames, time);
    const frame = frames[i];
    if (frame.t === time) return resolveInFrame(frame, anchor);
    const next = frames[i + 1];
    if (!next || next.frame_idx !== frame.frame_idx + 1) return null;
    const start = resolveInFrame(frame, anchor);
    const end = resolveInFrame(next, anchor);
    if (!start || !end) return null;
    const amount = (time - frame.t) / (next.t - frame.t);
    return start.map((coordinate, axis) => coordinate + (end[axis] - coordinate) * amount);
  };
}
