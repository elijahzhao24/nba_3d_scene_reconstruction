import { BallEditor } from './ball-editor.js';
import { createBallHandResolver } from './ball-hand.js';
import { readBallKeyframes, sampleBall } from './ball-keyframes.js';
import { CourtView } from './court-view.js';
import { demoClips } from './demo-clips.js';
import { DemoPovViews } from './demo-pov-views.js';
import { setupFullscreen } from './fullscreen.js';
import { estimateTrackScales, getPoseLayout, makeCameraBasis, poseScaleKey } from './player-pose.js';
import { playerPov } from './player-pov.js';
import { frameAtTime, readScene } from './scene-data.js';
import './style.css';

const POV_PANELS = ['pov-first', 'pov-second'];

const elements = Object.fromEntries([
  'ball-editor', 'clip-name', 'compare', 'court-heading', 'court-size', 'demo-clips',
  'duration', 'elapsed', 'frame', 'frame-index', 'fullscreen',
  'load-message', 'markers', 'model-message', 'next', 'open', 'orbit', 'play', 'player-count',
  'players', 'previous',
  ...POV_PANELS.flatMap((name) => [`${name}-panel`, `${name}-viewport`, `${name}-status`, `${name}-id`]),
  'pose-alignment', 'pose-alignment-status', 'pose-pitch', 'pose-yaw', 'pitch-value',
  'pov', 'pov-controls', 'pov-status', 'pov-track',
  'renderer-error', 'replay', 'reset-pose', 'scene-file', 'skeletons', 'source-panel',
  'source-video', 'top-down', 'video-status', 'view-grid', 'view-hint', 'viewport', 'yaw-value',
].map((id) => [id, document.getElementById(id)]));

const PLAYER_DISPLAYS = ['players', 'skeletons', 'markers'];
const COMPARISON_PANELS = ['source-panel', 'court-heading', ...POV_PANELS.map((name) => `${name}-panel`)];

async function fetchDemoFile(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('A demo file could not load.');
  return response.text();
}

function start(view) {
  const { frame, next, play, previous } = elements;
  setupFullscreen(elements.replay, elements.fullscreen);
  let scene;
  let index = 0;
  let clockOrigin = null;
  let loadVersion = 0;
  let activeDemo = null;
  let demoPovs;
  let videoClock = false;
  let videoPlayVersion = 0;
  let pose;
  let cameraMode = 'top';
  let povTrack = null;
  let playerDisplay = 'markers';
  let playerDisplayChosen = false;
  let ballKeyframes = [];
  let resolveBallHand = () => null;
  const sourceVideo = elements['source-video'];
  const videoStatus = elements['video-status'];
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
    const comparing = mode === 'compare';
    const layoutChanged = comparing !== (cameraMode === 'compare');
    if (layoutChanged) pause();
    cameraMode = mode;
    view.setMode(comparing ? 'orbit' : mode);
    elements.compare.setAttribute('aria-pressed', String(comparing));
    elements.orbit.setAttribute('aria-pressed', String(mode === 'orbit'));
    elements['top-down'].setAttribute('aria-pressed', String(mode === 'top'));
    elements.pov.setAttribute('aria-pressed', String(mode === 'pov'));
    elements['view-hint'].hidden = mode !== 'orbit' && !comparing;
    elements['pov-controls'].hidden = mode !== 'pov';
    elements['view-grid'].classList.toggle('is-comparing', comparing);
    for (const name of COMPARISON_PANELS) elements[name].hidden = !comparing;
    if (layoutChanged) {
      view.resize();
      if (comparing) view.frameOrbitCourt();
    }
    if (scene) {
      drawPlayers();
      drawBall();
      seekVideo();
    }
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
    if (cameraMode === 'compare') demoPovs.setFrame(current, pose, playerDisplay);
    elements['player-count'].textContent = current.players.length - Number(Boolean(pov));
  }

  function setPlayerDisplay(mode) {
    playerDisplay = mode;
    for (const name of PLAYER_DISPLAYS) {
      elements[name].setAttribute('aria-pressed', String(name === mode));
    }
    elements['pose-alignment'].hidden = Boolean(activeDemo) || mode === 'markers';
  }

  function drawBall(time = scene?.frames[index].t) {
    const position = scene && ballKeyframes.length
      ? sampleBall(ballKeyframes, scene, time, resolveHand, resolveHoop)
      : null;
    view.showBall(position);
    if (cameraMode === 'compare') demoPovs.showBall(position);
  }

  function refreshBallHands() {
    resolveBallHand = createBallHandResolver(scene, pose, resolveRigHand);
    ballEditor.refresh();
    drawBall();
  }

  function setFrame(newIndex, playbackTime = scene.frames[newIndex].t) {
    index = newIndex;
    const current = scene.frames[index];
    const lastIndex = scene.frames.length - 1;
    drawPlayers();
    ballEditor.setFrame(current);
    drawBall(playbackTime);
    if (clockOrigin === null) seekVideo();
    frame.value = String(index);
    frame.style.setProperty('--progress', lastIndex ? `${(index / lastIndex) * 100}%` : '0%');
    frame.setAttribute('aria-valuetext', `Frame ${current.frame_idx}`);
    elements['frame-index'].textContent = current.frame_idx;
    elements.elapsed.textContent = current.t.toFixed(2);
    previous.disabled = index === 0;
    next.disabled = index === lastIndex;
  }

  function pause() {
    const wasPlaying = clockOrigin !== null;
    clockOrigin = null;
    videoClock = false;
    videoPlayVersion += 1;
    sourceVideo.pause();
    play.textContent = 'Play';
    play.dataset.playing = 'false';
    if (wasPlaying) {
      seekVideo();
      drawBall();
    }
  }

  function seekVideo() {
    if (cameraMode !== 'compare' || !activeDemo?.videoUrl || sourceVideo.error) return;
    const time = scene.frames[index].t;
    if (Math.abs(sourceVideo.currentTime - time) > 0.001) sourceVideo.currentTime = time;
  }

  function playVideo() {
    if (cameraMode !== 'compare' || !activeDemo?.videoUrl || sourceVideo.error) return;
    seekVideo();
    const version = ++videoPlayVersion;
    videoClock = true;
    sourceVideo.play().catch(() => {
      if (version !== videoPlayVersion) return;
      videoClock = false;
      clockOrigin = performance.now() - scene.frames[index].t * 1000;
      videoStatus.textContent = 'Original video could not play.';
      videoStatus.hidden = false;
    });
  }

  function timeAtPlayback(now) {
    if (!videoClock) return (now - clockOrigin) / 1000;
    const time = sourceVideo.currentTime;
    if (sourceVideo.error || sourceVideo.ended) {
      videoClock = false;
      clockOrigin = now - time * 1000;
    }
    return time;
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
    playVideo();
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
    return maxPlayers;
  }

  function loadPlayerModel(data) {
    if (!pose || view.playerModelReady) return;
    view.loadPlayerModel().then(() => {
      if (scene !== data) return;
      elements.players.disabled = false;
      if (!playerDisplayChosen) {
        setPlayerDisplay('players');
      }
      drawPlayers();
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

  function loadScene(data, demo, initialKeyframes) {
    pause();
    const firstScene = !scene;
    const isDemo = Boolean(demo);
    activeDemo = demo;
    if (isDemo) elements.replay.dataset.mode = 'demo';
    else delete elements.replay.dataset.mode;
    document.documentElement.classList.toggle('demo-active', isDemo);
    elements['ball-editor'].hidden = isDemo;
    view.setCourt(data.court);
    view.clearMarkers();
    const maxPlayers = loadPoses(data);
    scene = data;
    index = 0;
    ballKeyframes = [];
    resolveBallHand = createBallHandResolver(scene, pose, resolveRigHand);
    loadPovTracks();
    if (demo) {
      demoPovs ??= new DemoPovViews(POV_PANELS.map((name) => ({
        viewport: elements[`${name}-viewport`], status: elements[`${name}-status`],
      })));
      for (const [i, name] of POV_PANELS.entries()) elements[`${name}-id`].textContent = demo.povTracks[i];
      demoPovs.setScene(data.court, pose?.layout, maxPlayers, demo.povTracks);
      if (pose) demoPovs.loadPlayerModel().then(() => {
        if (scene === data && cameraMode === 'compare') drawPlayers();
      });
    }
    videoStatus.hidden = Boolean(demo?.videoUrl);
    videoStatus.textContent = 'Original video unavailable.';
    if (demo?.videoUrl) sourceVideo.src = demo.videoUrl;
    else if (sourceVideo.hasAttribute('src')) {
      sourceVideo.removeAttribute('src');
      sourceVideo.load();
    }
    elements.compare.hidden = !demo;
    for (const [id, button] of demoButtons) button.setAttribute('aria-pressed', String(id === demo?.id));
    if (demo) setMode('compare');
    else if (cameraMode === 'compare' || (firstScene && pose)) setMode('orbit');
    ballEditor.setScene(scene, { demoId: demo?.id, initialKeyframes });
    elements['clip-name'].textContent = demo?.title ?? `${data.clip_id} / ${data.segment_id}`;
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
  const demoButtons = new Map(demoClips.map((demo) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = `${demo.title} `;
    const description = document.createElement('span');
    description.className = 'demo-description';
    description.textContent = `(${demo.description})`;
    button.append(description);
    button.setAttribute('aria-pressed', 'false');
    button.addEventListener('click', () => loadDemo(demo));
    return [demo.id, button];
  }));
  elements['demo-clips'].append(...demoButtons.values());

  async function loadClip(readFiles, demo = null) {
    pause();
    const version = ++loadVersion;
    message.dataset.state = 'loading';
    message.textContent = demo ? `Loading ${demo.title}…` : 'Reading scene…';
    try {
      const [sceneText, ballText] = await readFiles();
      if (version !== loadVersion) return;
      const data = readScene(sceneText);
      const keyframes = ballText ? readBallKeyframes(ballText, data) : [];
      loadScene(data, demo, keyframes);
      message.dataset.state = 'loaded';
      message.textContent = `${data.frames.length.toLocaleString()} frames · ${data.fps.toFixed(2)} fps`;
    } catch (error) {
      if (version !== loadVersion) return;
      message.dataset.state = 'error';
      message.textContent = error.message;
    }
  }

  function loadDemo(demo) {
    return loadClip(() => Promise.all([
      fetchDemoFile(demo.sceneUrl),
      demo.ballUrl ? fetchDemoFile(demo.ballUrl) : null,
    ]), demo);
  }

  sourceVideo.addEventListener('error', () => {
    if (activeDemo?.videoUrl && sourceVideo.error) {
      videoStatus.textContent = 'Original video could not load.';
      videoStatus.hidden = false;
    }
  });
  elements.open.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    const [file] = input.files;
    input.value = '';
    if (file) loadClip(async () => [await file.text()]);
  });

  elements.compare.addEventListener('click', () => setMode('compare'));

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
      const time = Math.min(scene.frames.at(-1).t, timeAtPlayback(now));
      const current = frameAtTime(scene.frames, time);
      if (current !== index) setFrame(current, time);
      else drawBall(time);
      if (current === scene.frames.length - 1) pause();
    }
    view.render();
    if (cameraMode === 'compare') demoPovs.render();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  if (demoClips.length) loadDemo(demoClips[0]);
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
