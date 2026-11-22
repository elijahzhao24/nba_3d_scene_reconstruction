export function readScene(text) {
  let scene;
  try {
    scene = JSON.parse(text);
  } catch {
    throw new Error('The file is not valid JSON.');
  }

  if (scene?.version !== 1 || scene.units !== 'meters') {
    throw new Error('Choose a version 1 scene export in meters.');
  }
  const { clip_id: clip, segment_id: segment, fps, court, frames } = scene;
  if (typeof clip !== 'string' || !clip || typeof segment !== 'string' || !segment) {
    throw new Error('The scene is missing a clip or segment ID.');
  }
  if (![fps, court?.length_m, court?.width_m].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error('Frame rate and court dimensions must be positive numbers.');
  }
  if (!Array.isArray(frames) || frames.length === 0) {
    throw new Error('The scene contains no frames.');
  }

  let previousIndex = -1;
  let previousTime = -1;
  for (const frame of frames) {
    if (!Number.isSafeInteger(frame?.frame_idx) || frame.frame_idx <= previousIndex
        || !Number.isFinite(frame.t) || frame.t < 0 || frame.t <= previousTime
        || !Array.isArray(frame.players)) {
      throw new Error('Scene frames must have ordered indices, timestamps, and player lists.');
    }
    const ids = new Set();
    for (const player of frame.players) {
      if (!Number.isSafeInteger(player?.id) || player.id < 0 || ids.has(player.id)
          || !Array.isArray(player.p) || player.p.length !== 3 || !player.p.every(Number.isFinite)) {
        throw new Error(`Frame ${frame.frame_idx} has an invalid or duplicate player.`);
      }
      if (player.look_at !== undefined) {
        const { target, weight } = player.look_at ?? {};
        if (!Array.isArray(target) || target.length !== 3 || !target.every(Number.isFinite)
            || !Number.isFinite(weight) || weight < 0 || weight > 1) {
          throw new Error(`Frame ${frame.frame_idx} has an invalid look target for player ${player.id}.`);
        }
      }
      ids.add(player.id);
    }
    previousIndex = frame.frame_idx;
    previousTime = frame.t;
  }
  return scene;
}

export function frameAtTime(frames, time) {
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (frames[mid].t <= time) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return low;
}
