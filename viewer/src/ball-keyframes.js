export const BALL_RADIUS_M = 0.12;

const MOTIONS = new Set(['dribble', 'pass', 'shot']);
const SIDES = new Set(['left', 'right']);
const SHOT_RIM_PROGRESS = 0.8;
const SHOT_ARC_LIFT_M = 1.5;
const NET_DROP_M = 0.65;
const FILE_VERSION = 2;
const frameMaps = new WeakMap();

function framesByIndex(scene) {
  if (!frameMaps.has(scene.frames)) {
    frameMaps.set(scene.frames, new Map(scene.frames.map((frame) => [frame.frame_idx, frame])));
  }
  return frameMaps.get(scene.frames);
}

function entryStartFrame(entry) {
  return entry.kind === 'hold' ? entry.frame_idx : entry.start_frame;
}

function requireFrame(frames, frameIndex) {
  const frame = frames.get(frameIndex);
  if (!frame) throw new Error(`Ball keyframe refers to an unknown frame (${frameIndex}).`);
  return frame;
}

function requireHand(anchor, frame) {
  if (!SIDES.has(anchor?.hand)) throw new Error(`Choose a left or right hand for frame ${frame.frame_idx}.`);
  if (!frame.players.some(({ id }) => id === anchor.track_id)) {
    throw new Error(`Choose a player visible at frame ${frame.frame_idx}.`);
  }
  return { track_id: anchor.track_id, hand: anchor.hand };
}

function requireHoop(anchor, frameIndex) {
  if (!SIDES.has(anchor?.hoop)) throw new Error(`Choose a hoop for the shot ending at frame ${frameIndex}.`);
  return { hoop: anchor.hoop };
}

export function validateBallKeyframes(keyframes, scene) {
  if (!Array.isArray(keyframes)) throw new Error('The ball file needs a list of keyframes.');
  const frames = framesByIndex(scene);
  const ordered = keyframes.map((keyframe) => {
    const { frame_idx, kind, anchor, start_frame, end_frame, from, to } = keyframe ?? {};
    if (kind === 'hold') {
      return { kind, frame_idx, anchor: requireHand(anchor, requireFrame(frames, frame_idx)) };
    }
    if (!MOTIONS.has(kind)) throw new Error('Choose a supported ball movement.');
    const start = requireFrame(frames, start_frame);
    const end = requireFrame(frames, end_frame);
    if (end_frame <= start_frame) {
      throw new Error(`Choose an end frame after the ${kind} starts at frame ${start_frame}.`);
    }
    return {
      kind,
      start_frame,
      end_frame,
      from: requireHand(from, start),
      to: kind === 'shot' ? requireHoop(to, end_frame) : requireHand(to, end),
    };
  }).sort((a, b) => entryStartFrame(a) - entryStartFrame(b));

  for (let index = 1; index < ordered.length; index += 1) {
    const previous = ordered[index - 1];
    const start = entryStartFrame(ordered[index]);
    if (start === entryStartFrame(previous)) {
      throw new Error(`Frame ${start} already holds the ball or starts an action.`);
    }
    if (previous.kind !== 'hold' && start < previous.end_frame) {
      throw new Error(`Frame ${start} is inside the ${previous.kind} from frame ${previous.start_frame} to ${previous.end_frame}. End that action before starting another keyframe.`);
    }
  }
  return ordered;
}

function migrateV1(keyframes, scene) {
  if (!Array.isArray(keyframes)) throw new Error('The ball file needs a list of keyframes.');
  const frames = framesByIndex(scene);
  const seen = new Set();
  const ordered = keyframes.map((keyframe) => {
    const { frame_idx, kind, anchor } = keyframe ?? {};
    const frame = requireFrame(frames, frame_idx);
    if (seen.has(frame_idx)) throw new Error(`Frame ${frame_idx} already has a ball keyframe.`);
    seen.add(frame_idx);
    if (kind !== 'hold' && !MOTIONS.has(kind)) throw new Error('Choose a supported ball movement.');
    return {
      frame_idx,
      kind,
      anchor: kind === 'shot' ? requireHoop(anchor, frame_idx) : requireHand(anchor, frame),
    };
  }).sort((a, b) => a.frame_idx - b.frame_idx);

  const entries = [];
  for (const [index, keyframe] of ordered.entries()) {
    if (keyframe.kind === 'hold') {
      const next = ordered[index + 1];
      if (!next || next.kind === 'hold') entries.push(keyframe);
      continue;
    }
    const previous = ordered[index - 1];
    if (!previous || previous.kind === 'shot') {
      throw new Error(`Add a hand keyframe before the ${keyframe.kind} at frame ${keyframe.frame_idx}.`);
    }
    entries.push({
      kind: keyframe.kind,
      start_frame: previous.frame_idx,
      end_frame: keyframe.frame_idx,
      from: previous.anchor,
      to: keyframe.anchor,
    });
  }
  return validateBallKeyframes(entries, scene);
}

export function readBallKeyframes(text, scene) {
  let sidecar;
  try {
    sidecar = JSON.parse(text);
  } catch {
    throw new Error('The ball file is not valid JSON.');
  }
  if (sidecar?.version !== 1 && sidecar?.version !== FILE_VERSION) {
    throw new Error('Choose a version 1 or version 2 ball keyframe file.');
  }
  if (sidecar.clip_id !== scene.clip_id || sidecar.segment_id !== scene.segment_id) {
    throw new Error('The ball file belongs to a different clip or segment.');
  }
  return sidecar.version === 1
    ? migrateV1(sidecar.keyframes, scene)
    : validateBallKeyframes(sidecar.keyframes, scene);
}

export function writeBallKeyframes(keyframes, scene) {
  return `${JSON.stringify({
    version: FILE_VERSION,
    clip_id: scene.clip_id,
    segment_id: scene.segment_id,
    keyframes: validateBallKeyframes(keyframes, scene),
  }, null, 2)}\n`;
}

const validPoint = (value) => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const lerp = (a, b, progress) => a + (b - a) * progress;

function shotPoint(start, rim, progress) {
  if (progress >= SHOT_RIM_PROGRESS) {
    const fall = (progress - SHOT_RIM_PROGRESS) / (1 - SHOT_RIM_PROGRESS);
    return [rim[0], rim[1] - NET_DROP_M * fall, rim[2]];
  }

  const flight = progress / SHOT_RIM_PROGRESS;
  const distance = Math.hypot(rim[0] - start[0], rim[2] - start[2]);
  // Ensure the ball is descending when it reaches the rim, even on a short shot.
  const arc = Math.max(0.85, (rim[1] - start[1]) * 0.5 + 0.3, distance * 0.08) + SHOT_ARC_LIFT_M;
  const height = lerp(start[1], rim[1], flight) + 4 * arc * flight * (1 - flight);
  return [
    lerp(start[0], rim[0], flight),
    height,
    lerp(start[2], rim[2], flight),
  ];
}

function sampleFlight(entry, startTime, endTime, time, resolveHand, resolveHoop) {
  const start = resolveHand(entry.from, startTime);
  if (!validPoint(start)) return null;
  const progress = (time - startTime) / (endTime - startTime);

  if (entry.kind === 'shot') {
    const rim = resolveHoop(entry.to.hoop);
    if (!validPoint(rim)) return null;
    const position = shotPoint(start, rim, progress);
    return validPoint(position) ? position : null;
  }

  const end = resolveHand(entry.to, endTime);
  if (!validPoint(end)) return null;
  const position = start.map((value, axis) => lerp(value, end[axis], progress));
  if (entry.kind === 'dribble') {
    position[1] = progress <= 0.5
      ? lerp(start[1], BALL_RADIUS_M, progress * 2)
      : lerp(BALL_RADIUS_M, end[1], progress * 2 - 1);
  }
  return position;
}

export function sampleBall(keyframes, scene, time, resolveHand, resolveHoop) {
  if (!keyframes.length || !Number.isFinite(time)) return null;
  const frames = framesByIndex(scene);
  let lower = 0;
  let upper = keyframes.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (frames.get(entryStartFrame(keyframes[middle])).t <= time) lower = middle + 1;
    else upper = middle;
  }
  if (lower === 0) return null;

  const current = keyframes[lower - 1];
  if (current.kind === 'hold') {
    return sampleHand(current.anchor, time, resolveHand);
  }

  const startTime = frames.get(current.start_frame).t;
  const endTime = frames.get(current.end_frame).t;
  if (time === startTime) return sampleHand(current.from, time, resolveHand);
  if (time < endTime || (current.kind === 'shot' && time === endTime)) {
    return sampleFlight(current, startTime, endTime, time, resolveHand, resolveHoop);
  }
  return current.kind === 'shot' ? null : sampleHand(current.to, time, resolveHand);
}

function sampleHand(anchor, time, resolveHand) {
  const hand = resolveHand(anchor, time);
  return validPoint(hand) ? hand : null;
}
