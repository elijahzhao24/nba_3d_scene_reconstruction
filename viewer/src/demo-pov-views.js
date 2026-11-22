import { CourtView } from './court-view.js';
import { playerPov } from './player-pov.js';

export class DemoPovViews {
  constructor(panels) {
    this.ball = null;
    this.panels = panels.flatMap(({ viewport, status }, index) => {
      try {
        const view = new CourtView(viewport, { reflection: false, maxPixelRatio: 1 });
        view.setMode('pov');
        return [{ view, status, index, hasPose: null }];
      } catch {
        status.textContent = 'This camera could not start.';
        status.hidden = false;
        return [];
      }
    });
  }

  setScene(court, layout, maxPlayers, tracks) {
    this.showBall(null);
    for (const panel of this.panels) {
      const { view } = panel;
      panel.trackId = tracks[panel.index];
      view.setCourt(court);
      view.clearMarkers();
      view.configurePoses(layout, maxPlayers);
    }
  }

  loadPlayerModel() {
    return Promise.allSettled(this.panels.map(({ view }) => view.loadPlayerModel()));
  }

  setFrame(frame, pose, display) {
    for (const panel of this.panels) {
      const { view, status, trackId } = panel;
      const pov = playerPov(frame, trackId, pose);
      const hasPose = Boolean(pov);
      if (hasPose !== panel.hasPose) {
        panel.hasPose = hasPose;
        status.textContent = hasPose ? '' : 'No player pose on this frame.';
        status.hidden = hasPose;
        view.renderer.domElement.hidden = !hasPose;
        view.labels.domElement.hidden = !hasPose;
      }
      if (!pov) continue;
      view.setPlayerPov(pov);
      view.showPlayers(frame.players, pose, display, trackId);
      view.needsRender = true;
    }
  }

  showBall(position) {
    if (position === this.ball || (position && this.ball
      && position.every((value, index) => value === this.ball[index]))) return;
    this.ball = position;
    for (const { view } of this.panels) {
      view.showBall(position);
      view.needsRender = true;
    }
  }

  render() {
    for (const { view, hasPose } of this.panels) {
      if (hasPose && view.needsRender) view.render();
    }
  }
}
