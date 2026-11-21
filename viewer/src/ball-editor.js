import { readBallKeyframes, validateBallKeyframes, writeBallKeyframes } from './ball-keyframes.js';
import './ball-editor.css';

const ACTIONS = ['hold', 'dribble', 'pass', 'shot'];
const HANDS = { left: 'Left hand', right: 'Right hand' };
const HOOPS = { left: 'Left hoop (−X)', right: 'Right hoop (+X)' };
const MOTION_HELP = {
  dribble: 'The ball will bounce once between those hands.',
  pass: 'The ball will travel straight between those hands.',
  shot: 'The ball will arc through the selected hoop.',
};

function actionLabel(kind, phase = 'start') {
  if (kind === 'hold') return 'Hold in hand';
  return `${phase === 'end' ? 'End' : 'Start'} ${kind}`;
}

function startFrame(entry) {
  return entry.kind === 'hold' ? entry.frame_idx : entry.start_frame;
}

function initialDraft() {
  return { kind: 'hold', trackId: null, hand: 'right', hoop: 'right' };
}

function fillOptions(select, options, value) {
  select.replaceChildren(...options.map(([key, label]) => new Option(label, String(key))));
  if (value !== null) select.value = String(value);
}

export class BallEditor {
  constructor(container, { seekFrame, onChange, resolveHand, pausePlayback }) {
    this.seekFrame = seekFrame;
    this.onChange = onChange;
    this.resolveHand = resolveHand;
    this.pausePlayback = pausePlayback;
    this.edits = new Map();
    this.importVersion = 0;
    this.keyframes = [];
    this.draft = initialDraft();
    this.pending = null;
    this.editing = null;
    container.innerHTML = `
      <details class="ball-editor" data-ball="panel" hidden>
        <summary><span>Ball editor</span> <span class="ball-count" data-ball="count">0 keyframes</span></summary>
        <div class="ball-editor-body">
          <form class="ball-form" data-ball="form">
            <div class="ball-frame">Frame <span data-ball="frame">—</span></div>
            <p class="ball-context" data-ball="context"></p>
            <div class="ball-fields">
              <div class="ball-field">
                <label for="ball-action">Action</label>
                <select id="ball-action" name="kind" data-ball="kind"></select>
              </div>
              <div class="ball-field" data-ball="playerLabel">
                <label for="ball-player">Player</label>
                <select id="ball-player" name="trackId" data-ball="player"></select>
              </div>
              <div class="ball-field" data-ball="handLabel">
                <label for="ball-hand">Hand</label>
                <select id="ball-hand" name="hand" data-ball="hand"></select>
              </div>
              <div class="ball-field" data-ball="hoopLabel" hidden>
                <label for="ball-hoop">Hoop</label>
                <select id="ball-hoop" name="hoop" data-ball="hoop">
                  <option value="left">Left hoop (−X)</option>
                  <option value="right">Right hoop (+X)</option>
                </select>
              </div>
            </div>
            <div class="ball-actions">
              <button class="ball-add" type="submit" data-ball="save" disabled>Add hold</button>
              <button class="ball-secondary" type="button" data-ball="cancel" hidden>Cancel start</button>
              <button class="ball-secondary" type="button" data-ball="remove" hidden>Delete action</button>
              <button class="ball-secondary" type="button" data-ball="addAnother" hidden>New keyframe</button>
            </div>
          </form>
          <section class="ball-keyframe-panel" aria-label="Ball keyframe files and navigation">
            <div class="ball-files">
              <button class="ball-secondary" type="button" data-ball="importButton">Import ball JSON</button>
              <button class="ball-secondary" type="button" data-ball="exportButton">Export ball JSON</button>
              <input type="file" data-ball="file" accept=".json,application/json" hidden />
            </div>
            <p class="ball-empty" data-ball="empty">No ball keyframes yet.</p>
            <ol class="ball-keyframe-list" data-ball="list" aria-label="Ball keyframes"></ol>
          </section>
        </div>
        <p class="ball-feedback" data-ball="feedback" role="status" hidden></p>
      </details>`;
    this.ui = Object.fromEntries(
      Array.from(container.querySelectorAll('[data-ball]'), (node) => [node.dataset.ball, node]),
    );
    this.rows = new Map();
    this.ui.panel.addEventListener('toggle', () => {
      if (!this.ui.panel.open) return;
      this.pausePlayback?.();
      this.refresh();
    });
    this.ui.form.addEventListener('pointerdown', () => this.pausePlayback?.());
    this.ui.form.addEventListener('focusin', () => this.pausePlayback?.());
    this.ui.form.addEventListener('change', ({ target }) => {
      this.draft[target.name] = target.name === 'trackId' ? Number(target.value) : target.value;
      this.refresh();
    });
    this.ui.form.addEventListener('submit', (event) => {
      event.preventDefault();
      this.saveCurrent();
    });
    this.ui.remove.addEventListener('click', () => this.deleteCurrent());
    this.ui.cancel.addEventListener('click', () => this.cancelStart());
    this.ui.addAnother.addEventListener('click', () => {
      this.editing = null;
      this.refresh();
    });
    this.ui.importButton.addEventListener('click', () => this.ui.file.click());
    this.ui.file.addEventListener('change', () => this.importFile());
    this.ui.exportButton.addEventListener('click', () => this.exportFile());
  }

  setScene(scene) {
    this.importVersion += 1;
    this.scene = scene;
    this.frame = null;
    this.draft = initialDraft();
    this.pending = null;
    this.editing = null;
    this.sceneKey = JSON.stringify([scene.clip_id, scene.segment_id]);
    this.frames = new Map(scene.frames.map((frame) => [frame.frame_idx, frame]));
    this.ui.panel.hidden = false;
    this.setFeedback('');
    let keyframes;
    try {
      keyframes = validateBallKeyframes(this.edits.get(this.sceneKey) ?? [], scene);
    } catch {
      keyframes = [];
      this.setFeedback('The saved session keyframes no longer match this scene.', true);
    }
    this.setKeyframes(keyframes);
  }

  setFrame(frame) {
    if (frame.frame_idx === this.frame?.frame_idx) return;
    this.highlightFrame(this.frame?.frame_idx, false);
    this.frame = frame;
    if (this.editing?.frame !== frame.frame_idx) this.editing = null;
    this.highlightFrame(frame.frame_idx, true);
    this.refresh();
  }

  setFeedback(message, error = false) {
    this.ui.feedback.textContent = message;
    this.ui.feedback.hidden = !message;
    this.ui.feedback.dataset.error = String(error);
  }

  highlightFrame(frame, active) {
    for (const button of this.rows.get(frame) ?? []) {
      if (active) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    }
  }

  selectMarker(marker) {
    if (!this.pending && !marker.pending) {
      this.editing = marker;
      const { entry, anchor } = marker;
      this.draft.kind = entry.kind;
      if (anchor.hoop) {
        this.draft.hoop = anchor.hoop;
      } else {
        this.draft.trackId = anchor.track_id;
        this.draft.hand = anchor.hand;
      }
    }
    this.seekFrame(marker.frame);
    this.refresh();
  }

  creationBlock(frame) {
    const action = this.keyframes.find((entry) => (
      entry.kind !== 'hold' && entry.start_frame <= frame && frame < entry.end_frame
    ));
    if (action) {
      return frame === action.start_frame
        ? `A ${action.kind} starts at this frame. Select it from the keyframe list to edit it.`
        : `A ${action.kind} spans frames ${action.start_frame}–${action.end_frame}. Choose a frame outside that action.`;
    }
    return '';
  }

  endBlock(frame) {
    const { pending } = this;
    if (frame <= pending.start_frame) return `Scrub to a later frame to end the ${pending.kind}.`;
    const next = this.keyframes.find((entry) => startFrame(entry) > pending.start_frame);
    if (next && frame > startFrame(next)) {
      return `End the ${pending.kind} at or before frame ${startFrame(next)}, the next keyframe.`;
    }
    return '';
  }

  refresh() {
    if (!this.ui.panel.open || !this.frame) return;
    const { ui, draft, pending, editing } = this;
    const frame = this.frame.frame_idx;
    const kind = pending?.kind ?? editing?.entry.kind ?? draft.kind;
    const phase = editing?.phase ?? (pending ? 'end' : 'start');
    const shotEnd = kind === 'shot' && phase === 'end';
    const blocked = pending ? this.endBlock(frame) : editing ? '' : this.creationBlock(frame);
    const existingHold = this.keyframes.find((entry) => entry.kind === 'hold' && entry.frame_idx === frame);
    ui.frame.textContent = frame;
    const options = pending || editing
      ? [[kind, actionLabel(kind, phase)]]
      : ACTIONS.map((action) => [action, actionLabel(action)]);
    fillOptions(ui.kind, options, kind);
    ui.kind.disabled = Boolean(pending || editing || blocked);
    ui.playerLabel.hidden = Boolean(shotEnd || blocked);
    ui.handLabel.hidden = Boolean(shotEnd || blocked);
    ui.hoopLabel.hidden = !shotEnd || Boolean(blocked);
    ui.hoop.value = draft.hoop;
    ui.cancel.hidden = !pending;
    ui.remove.hidden = !editing;
    ui.remove.textContent = editing?.entry.kind === 'hold' ? 'Delete hold' : 'Delete action';
    ui.addAnother.hidden = !editing;
    ui.importButton.disabled = ui.exportButton.disabled = Boolean(pending);
    if (editing) ui.save.textContent = 'Update keyframe';
    else if (kind === 'hold') ui.save.textContent = existingHold ? 'Update hold' : 'Add hold';
    else ui.save.textContent = actionLabel(kind, phase);
    if (blocked) {
      ui.context.textContent = blocked;
      ui.save.disabled = true;
      return;
    }
    const available = shotEnd || this.updateHandOptions(editing?.anchor);
    if (!available) {
      ui.context.textContent = 'No usable hand at this frame. Scrub to a frame with the player’s pose.';
    } else if (kind === 'hold') {
      ui.context.textContent = 'The ball follows this hand until the next hold or action starts.';
    } else if (pending) {
      ui.context.textContent = `Started at frame ${pending.start_frame}. ${MOTION_HELP[kind]}`;
    } else if (editing) {
      ui.context.textContent = `Editing the ${phase} keyframe. ${MOTION_HELP[kind]}`;
    } else if (existingHold) {
      ui.context.textContent = 'Saving this action will replace the hold at this frame.';
    } else {
      ui.context.textContent = `Set the release hand. Then scrub forward and choose End ${kind}.`;
    }
    ui.save.disabled = !available;
  }

  updateHandOptions(currentAnchor) {
    const { draft, frame, ui } = this;
    const available = new Map();
    for (const { id, joints } of frame.players) {
      if (!joints) continue;
      const hands = Object.keys(HANDS).filter(
        (hand) => this.resolveHand({ track_id: id, hand }, frame.t),
      );
      if (hands.length) available.set(id, hands);
    }

    const choices = [...available.keys()].map((id) => [id, `Track ${id}`]);
    if (!available.has(draft.trackId)) {
      if (currentAnchor?.track_id === draft.trackId && draft.trackId !== null) {
        choices.push([draft.trackId, `Track ${draft.trackId} (unavailable)`]);
      } else {
        draft.trackId = available.keys().next().value ?? null;
      }
    }
    fillOptions(ui.player, choices.length ? choices : [['', 'No player pose']], draft.trackId);
    ui.player.disabled = !choices.length;
    const hands = available.get(draft.trackId) ?? [];
    if (!hands.includes(draft.hand) && hands.length) draft.hand = hands[0];
    const handChoices = hands.length ? hands.map((hand) => [hand, HANDS[hand]]) : [['', 'No hand']];
    fillOptions(ui.hand, handChoices, hands.length ? draft.hand : null);
    ui.hand.disabled = !hands.length;
    return hands.length > 0;
  }

  setKeyframes(keyframes) {
    this.importVersion += 1;
    this.keyframes = keyframes;
    this.edits.set(this.sceneKey, keyframes);
    this.renderList();
    this.refresh();
    this.onChange(keyframes);
  }

  saveCurrent() {
    if (!this.frame || this.ui.save.disabled) return;
    this.pausePlayback?.();
    const { frame_idx } = this.frame;
    const { pending, editing, draft } = this;
    const kind = pending?.kind ?? editing?.entry.kind ?? draft.kind;
    const shotEnd = kind === 'shot' && (pending || editing?.phase === 'end');
    const anchor = shotEnd
      ? { hoop: draft.hoop }
      : { track_id: draft.trackId, hand: draft.hand };
    try {
      if (pending) {
        if (!this.resolveHand(pending.from, this.frames.get(pending.start_frame).t)) {
          throw new Error(`The starting hand at frame ${pending.start_frame} is unavailable.`);
        }
        const others = this.keyframes.filter((entry) => startFrame(entry) !== pending.start_frame);
        const next = [...others, { ...pending, end_frame: frame_idx, to: anchor }];
        const validated = validateBallKeyframes(next, this.scene);
        this.pending = null;
        this.setKeyframes(validated);
        this.setFeedback(`Saved ${kind} from frame ${pending.start_frame} to ${frame_idx}.`);
      } else if (editing) {
        const { entry, phase } = editing;
        const field = kind === 'hold' ? 'anchor' : phase === 'end' ? 'to' : 'from';
        const next = this.keyframes.map((saved) => (saved === entry ? { ...entry, [field]: anchor } : saved));
        const validated = validateBallKeyframes(next, this.scene);
        this.editing = null;
        this.setKeyframes(validated);
        this.setFeedback(`Updated ${actionLabel(kind, phase).toLowerCase()} at frame ${frame_idx}.`);
      } else if (kind === 'hold') {
        const others = this.keyframes.filter((entry) => startFrame(entry) !== frame_idx);
        const next = [...others, { frame_idx, kind, anchor }];
        this.setKeyframes(validateBallKeyframes(next, this.scene));
        this.setFeedback(`Saved hold at frame ${frame_idx}.`);
      } else {
        this.pending = { kind, start_frame: frame_idx, from: anchor };
        this.importVersion += 1;
        this.renderList();
        this.refresh();
        this.setFeedback(`Started ${kind} at frame ${frame_idx}. Scrub forward to finish it.`);
      }
    } catch (error) {
      this.setFeedback(error.message, true);
    }
  }

  cancelStart() {
    const { pending } = this;
    if (!pending) return;
    this.pausePlayback?.();
    this.pending = null;
    this.renderList();
    this.refresh();
    this.setFeedback(`Canceled the ${pending.kind} start at frame ${pending.start_frame}.`);
  }

  deleteCurrent() {
    const { editing } = this;
    if (!editing || this.pending) return;
    this.pausePlayback?.();
    this.editing = null;
    const { entry } = editing;
    this.setKeyframes(this.keyframes.filter((saved) => saved !== entry));
    const range = entry.kind === 'hold'
      ? `at frame ${entry.frame_idx}`
      : `from frame ${entry.start_frame} to ${entry.end_frame}`;
    this.setFeedback(`Deleted ${entry.kind} ${range}.`);
  }

  renderList() {
    const { ui, keyframes, pending } = this;
    const markers = keyframes.flatMap((entry) => (
      entry.kind === 'hold'
        ? [{ entry, frame: entry.frame_idx, phase: 'hold', anchor: entry.anchor }]
        : [
          { entry, frame: entry.start_frame, phase: 'start', anchor: entry.from },
          { entry, frame: entry.end_frame, phase: 'end', anchor: entry.to },
        ]
    ));
    if (pending) {
      markers.push({ entry: pending, frame: pending.start_frame, phase: 'start', anchor: pending.from, pending: true });
    }
    markers.sort((a, b) => a.frame - b.frame || Number(b.phase === 'end') - Number(a.phase === 'end'));
    const count = markers.length;
    ui.count.textContent = `${count} keyframe${count === 1 ? '' : 's'}`;
    ui.empty.hidden = count > 0;
    this.rows.clear();
    ui.list.replaceChildren(...markers.map((marker) => {
      const { entry, frame, phase, anchor } = marker;
      const item = document.createElement('li');
      const button = document.createElement('button');
      const number = document.createElement('span');
      const description = document.createElement('span');
      number.className = 'ball-keyframe-number';
      number.textContent = `Frame ${frame}`;
      const subject = anchor.hoop ? HOOPS[anchor.hoop] : `Track ${anchor.track_id} · ${HANDS[anchor.hand]}`;
      description.textContent = `${actionLabel(entry.kind, phase)} · ${subject}${marker.pending ? ' · pending' : ''}`;
      button.type = 'button';
      if (marker.pending) button.dataset.pending = 'true';
      button.append(number, description);
      button.addEventListener('click', () => this.selectMarker(marker));
      item.append(button);
      if (!this.rows.has(frame)) this.rows.set(frame, []);
      this.rows.get(frame).push(button);
      return item;
    }));
    this.highlightFrame(this.frame?.frame_idx, true);
  }

  async importFile() {
    const [file] = this.ui.file.files;
    this.ui.file.value = '';
    if (!file || this.pending) return;
    this.pausePlayback?.();
    const version = ++this.importVersion;
    try {
      const text = await file.text();
      if (version !== this.importVersion) return;
      const keyframes = readBallKeyframes(text, this.scene);
      this.editing = null;
      this.setKeyframes(keyframes);
      this.setFeedback(`Imported ${file.name}.`);
    } catch (error) {
      if (version === this.importVersion) this.setFeedback(error.message, true);
    }
  }

  exportFile() {
    if (this.pending) return;
    const text = writeBallKeyframes(this.keyframes, this.scene);
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'ball.json';
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    this.setFeedback('Exported ball.json.');
  }
}
