import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { BALL_RADIUS_M } from './ball-keyframes.js';

function makeSeams() {
  const radius = BALL_RADIUS_M * 1.012;
  const steps = 48;
  const segments = [];
  const planes = [0, Math.PI / 2, Math.PI / 4];
  for (const plane of planes) {
    function point(t) {
      const angle = 2 * Math.PI * t;
      const spread = radius * Math.sin(angle);
      return [Math.cos(plane) * spread, radius * Math.cos(angle), Math.sin(plane) * spread];
    }
    for (let i = 0; i < steps; i += 1) {
      segments.push(...point(i / steps), ...point((i + 1) / steps));
    }
  }
  return new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(segments, 3)),
    new THREE.LineBasicMaterial({ color: 0x632e16 }),
  );
}

export class BallView {
  constructor(scene) {
    this.ball = new THREE.Mesh(
      new THREE.SphereGeometry(BALL_RADIUS_M, 24, 16),
      new THREE.MeshStandardMaterial({
        color: 0xff7628, roughness: 0.85, emissive: 0x9e330e, emissiveIntensity: 0.24,
      }),
    );
    this.ball.add(makeSeams());
    const locator = document.createElement('span');
    locator.className = 'ball-locator';
    this.ball.add(new CSS2DObject(locator));
    this.shadow = new THREE.Mesh(
      new THREE.CircleGeometry(BALL_RADIUS_M * 1.45, 20),
      new THREE.MeshBasicMaterial({ color: 0x372924, transparent: true, opacity: 0.36, depthWrite: false }),
    );
    this.shadow.rotation.x = -Math.PI / 2;
    scene.add(this.ball, this.shadow);
    this.show(null);
  }

  show(position) {
    this.ball.visible = this.shadow.visible = Boolean(position);
    if (!position) return;
    const [x, y, z] = position;
    this.ball.position.set(x, y, z);
    this.shadow.position.set(x, 0.008, z);
    this.shadow.material.opacity = 0.36 / (1 + Math.max(0, y - BALL_RADIUS_M) * 0.35);
  }
}
