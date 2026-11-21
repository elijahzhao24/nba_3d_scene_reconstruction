import { BallEditor } from './ball-editor.js';
import { createBallHandResolver } from './ball-hand.js';
import { sampleBall } from './ball-keyframes.js';
import { CourtView } from './court-view.js';
import { setupFullscreen } from './fullscreen.js';
import { estimateTrackScales, getPoseLayout, makeCameraBasis, poseScaleKey } from './player-pose.js';
import { playerPov } from './player-pov.js';
import { frameAtTime, readScene } from './scene-data.js';
import './style.css';

const elements = Object.fromEntries([
  'ball-editor', 'clip-name', 'court-size', 'duration', 'elapsed', 'frame', 'frame-index', 'fullscreen',
  'load-message', 'markers', 'model-message', 'next', 'open', 'orbit', 'play', 'player-count',
  'players', 'previous',
  'pose-alignment', 'pose-alignment-status', 'pose-pitch', 'pose-yaw', 'pitch-value',
  'pov', 'pov-controls', 'pov-status', 'pov-track',
  'renderer-error', 'replay', 'reset-pose', 'scene-file', 'skeletons', 'top-down',
  'view-hint', 'viewport', 'yaw-value',
].map((id) => [id, document.getElementById(id)]));

const PLAYER_DISPLAYS = ['players', 'skeletons', 'markers'];

function start(view) {
  const { frame, next, play, previous } = elements;
  setupFullscreen(elements.replay, elements.fullscreen);
  let scene;
  let index = 0;
  let clockOrigin = null;
  let loadVersion = 0;
  let pose;
  let cameraMode = 'top';
  let povTrack = null;
  let playerDisplay = 'markers';
  let playerDisplayChosen = false;
  let ballKeyframes = [];
  let resolveBallHand = () => null;
  const yaw = elements['pose-yaw'];
  const pitch = elements['pose-pitch'];
  const resolveHand = (anchor, time) => resolveBallHand(anchor, time);
  const resolveHoop = (hoop) => view.hoopPosition(hoop);
  const resolveRigHand = (...args) => view.ballHandPosition(...args);
  const ballEditor = new BallEditor(elements['ball-editor'], {
    seekFrame(frameIdx) {
      const target = scene.frames.findIndex((frame) => frame.frame_idx === frameIdx);
      if (target >= 0) scrubTo(target);
    },
    onChange(keyframes) {
      ballKeyframes = keyframes;
      drawBall();
    },
    resolveHand,
    pausePlayback: pause,
  });

  function setMode(mode) {
    cameraMode = mode;
    view.setMode(mode);
    elements.orbit.setAttribute('aria-pressed', String(mode === 'orbit'));
    elements['top-down'].setAttribute('aria-pressed', String(mode === 'top'));
    elements.pov.setAttribute('aria-pressed', String(mode === 'pov'));
    elements['view-hint'].hidden = mode !== 'orbit';
    elements['pov-controls'].hidden = mode !== 'pov';
    if (scene) drawPlayers();
  }

  function drawPlayers() {
    const current = scene.frames[index];
    const inPovMode = cameraMode === 'pov';
    const pov = inPovMode ? playerPov(current, povTrack, pose) : null;
    view.setPlayerPov(pov);
    const status = elements['pov-status'];
    const unavailable = inPovMode && !pov;
    status.hidden = !unavailable;
    if (unavailable) {
      const message = `Track ${povTrack} has no usable pose on this frame. Showing Orbit.`;
      if (status.textContent !== message) status.textContent = message;
    }
    view.showPlayers(current.players, pose, playerDisplay, pov ? povTrack : null);
    elements['player-count'].textContent = current.players.length - Number(Boolean(pov));
  }

  function setPlayerDisplay(mode) {
    playerDisplay = mode;
    for (const name of PLAYER_DISPLAYS) {
      elements[name].setAttribute('aria-pressed', String(name === mode));
    }
    elements['pose-alignment'].hidden = mode === 'markers';
  }

  function drawBall(time = scene?.frames[index].t) {
    const position = scene && ballKeyframes.length
      ? sampleBall(ballKeyframes, scene, time, resolveHand, resolveHoop)
      : null;
    view.showBall(position);
  }

  function refreshBallHands() {
    resolveBallHand = createBallHandResolver(scene, pose, resolveRigHand);
    ballEditor.refresh();
    drawBall();
  }

  function setFrame(newIndex, playbackTime = scene.frames[newIndex].t) {
    index = newIndex;
    const current = scene.frames[index];
    drawPlayers();
    ballEditor.setFrame(current);
    drawBall(playbackTime);
    frame.value = String(index);
    frame.setAttribute('aria-valuetext', `Frame ${current.frame_idx}`);
    elements['frame-index'].textContent = current.frame_idx;
    elements.elapsed.textContent = current.t.toFixed(2);
    previous.disabled = index === 0;
    next.disabled = index === scene.frames.length - 1;
  }

  function pause() {
    const wasPlaying = clockOrigin !== null;
    clockOrigin = null;
    play.textContent = 'Play';
    play.dataset.playing = 'false';
    if (wasPlaying) drawBall();
  }

  function scrubTo(newIndex) {
    pause();
    setFrame(newIndex);
  }

  function togglePlayback() {
    if (clockOrigin !== null) {
      pause();
      return;
    }
    if (index === scene.frames.length - 1) setFrame(0);
    clockOrigin = performance.now() - scene.frames[index].t * 1000;
    play.textContent = 'Pause';
    play.dataset.playing = 'true';
  }

  function updateAlignment() {
    pose.basis = makeCameraBasis(yaw.valueAsNumber, pitch.valueAsNumber);
    elements['yaw-value'].textContent = `${yaw.valueAsNumber.toFixed(1)}°`;
    elements['pitch-value'].textContent = `${pitch.valueAsNumber.toFixed(1)}°`;
  }

  function resetAlignment() {
    yaw.value = pose.yaw;
    pitch.value = pose.pitch;
    updateAlignment();
  }

  function loadPoses(data) {
    const layout = getPoseLayout(data.pose);
    const scales = layout && estimateTrackScales(data.frames, layout);
    pose = null;
    if (scales?.size) {
      const camera = data.pose.camera;
      const estimated = Number.isFinite(camera?.yaw_deg) && Math.abs(camera.yaw_deg) <= 180
        && Number.isFinite(camera?.pitch_deg) && Math.abs(camera.pitch_deg) <= 90;
      pose = { layout, scales, yaw: estimated ? camera.yaw_deg : 0, pitch: estimated ? camera.pitch_deg : 0 };
      elements['pose-alignment-status'].textContent = estimated
        ? 'Source camera angle estimated from the saved court calibration.'
        : 'Set the source camera angle to align the skeletons with the court.';
      resetAlignment();
    }
    const maxPlayers = data.frames.reduce((max, current) => Math.max(max, current.players.length), 0);
    view.configurePoses(pose?.layout, maxPlayers);
    elements.skeletons.disabled = !pose;
    elements.players.disabled = !pose || !view.playerModelReady;
    elements.markers.disabled = false;
    elements['model-message'].hidden = true;
    const defaultPoseDisplay = view.playerModelReady ? 'players' : 'skeletons';
    setPlayerDisplay(pose ? defaultPoseDisplay : 'markers');
    playerDisplayChosen = false;
  }

  function loadPlayerModel(data) {
    if (!pose || view.playerModelReady) return;
    view.loadPlayerModel().then(() => {
      if (scene !== data) return;
      elements.players.disabled = false;
      if (!playerDisplayChosen) {
        setPlayerDisplay('players');
        drawPlayers();
      }
      refreshBallHands();
    }, () => {
      if (scene !== data) return;
      elements['model-message'].textContent = 'The player model could not load. Skeletons are available.';
      elements['model-message'].hidden = false;
    });
  }

  function loadPovTracks() {
    const tracks = new Set();
    if (pose) {
      for (const { players } of scene.frames) {
        for (const player of players) {
          if (player.joints && pose.scales.has(poseScaleKey(player))) tracks.add(player.id);
        }
      }
    }
    const ids = [...tracks].sort((a, b) => a - b);
    povTrack = scene.frames[0].players.find(({ id }) => tracks.has(id))?.id ?? ids[0] ?? null;
    const select = elements['pov-track'];
    select.replaceChildren(...ids.map((id) => new Option(`Track ${id}`, String(id))));
    if (povTrack !== null) select.value = String(povTrack);
    elements.pov.disabled = !ids.length;
    if (!ids.length && cameraMode === 'pov') setMode('orbit');
  }

  function loadScene(data) {
    pause();
    view.setCourt(data.court);
    view.clearMarkers();
    loadPoses(data);
    if (pose && !scene) setMode('orbit');
    scene = data;
    index = 0;
    loadPovTracks();
    ballKeyframes = [];
    resolveBallHand = createBallHandResolver(scene, pose, resolveRigHand);
    ballEditor.setScene(scene);
    elements['clip-name'].textContent = `${data.clip_id} / ${data.segment_id}`;
    elements['court-size'].textContent = `${data.court.length_m} × ${data.court.width_m} m`;
    frame.max = String(data.frames.length - 1);
    frame.disabled = false;
    play.disabled = data.frames.length < 2;
    elements.fullscreen.disabled = false;
    elements.duration.textContent = `/ ${data.frames.at(-1).t.toFixed(2)} s`;
    setFrame(0);
    loadPlayerModel(data);
  }

  const input = elements['scene-file'];
  const message = elements['load-message'];
  elements.open.addEventListener('click', () => input.click());
  input.addEventListener('change', async () => {
    const [file] = input.files;
    input.value = '';
    if (!file) return;
    const version = ++loadVersion;
    message.dataset.state = 'loading';
    message.textContent = 'Reading scene…';
    try {
      const data = readScene(await file.text());
      if (version !== loadVersion) return;
      loadScene(data);
      message.dataset.state = 'loaded';
      message.textContent = `${data.frames.length.toLocaleString()} frames · ${data.fps.toFixed(2)} fps`;
    } catch (error) {
      if (version !== loadVersion) return;
      message.dataset.state = 'error';
      message.textContent = error.message;
    }
  });

  elements['top-down'].addEventListener('click', () => setMode('top'));
  elements.orbit.addEventListener('click', () => setMode('orbit'));
  elements.pov.addEventListener('click', () => setMode('pov'));
  elements['pov-track'].addEventListener('change', ({ target }) => {
    povTrack = Number(target.value);
    drawPlayers();
  });
  for (const mode of PLAYER_DISPLAYS) {
    elements[mode].addEventListener('click', () => {
      playerDisplayChosen = true;
      setPlayerDisplay(mode);
      drawPlayers();
    });
  }
  for (const input of [yaw, pitch]) {
    input.addEventListener('input', () => {
      updateAlignment();
      drawPlayers();
      refreshBallHands();
    });
  }
  elements['reset-pose'].addEventListener('click', () => {
    resetAlignment();
    drawPlayers();
    refreshBallHands();
  });
  play.addEventListener('click', togglePlayback);
  previous.addEventListener('click', () => scrubTo(index - 1));
  next.addEventListener('click', () => scrubTo(index + 1));
  frame.addEventListener('input', () => scrubTo(frame.valueAsNumber));

  function tick(now) {
    if (clockOrigin !== null) {
      const time = Math.min(scene.frames.at(-1).t, (now - clockOrigin) / 1000);
      const current = frameAtTime(scene.frames, time);
      if (current !== index) setFrame(current, time);
      else drawBall(time);
      if (current === scene.frames.length - 1) pause();
    }
    view.render();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

let view;
try {
  view = new CourtView(elements.viewport);
} catch {
  elements['renderer-error'].hidden = false;
  elements['renderer-error'].textContent = 'WebGL 2 could not start. Open the viewer in a browser with WebGL 2 enabled.';
  elements['load-message'].hidden = true;
  elements.open.disabled = true;
  elements.orbit.disabled = true;
  elements['top-down'].disabled = true;
}
if (view) start(view);
