import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { BALL_RADIUS_M } from './ball-keyframes.js';

const BALL_LIGHT_INTENSITY = 0.34;

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
    new THREE.LineBasicMaterial({ color: 0xcb8849, transparent: true, opacity: 0.52 }),
  );
}

function makeGlowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const context = canvas.getContext('2d');
  const middle = canvas.width / 2;
  const gradient = context.createRadialGradient(middle, middle, 0, middle, middle, middle);
  gradient.addColorStop(0, 'rgba(255, 230, 159, 0.96)');
  gradient.addColorStop(0.22, 'rgba(255, 184, 72, 0.76)');
  gradient.addColorStop(0.43, 'rgba(255, 150, 33, 0.38)');
  gradient.addColorStop(0.72, 'rgba(255, 121, 16, 0.08)');
  gradient.addColorStop(1, 'rgba(255, 121, 16, 0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export class BallView {
  constructor(scene) {
    const glowTexture = makeGlowTexture();
    this.ball = new THREE.Mesh(
      new THREE.SphereGeometry(BALL_RADIUS_M, 24, 16),
      new THREE.MeshStandardMaterial({
        color: 0xffd797, roughness: 0.65, emissive: 0xffdf9d, emissiveIntensity: 0.52,
      }),
    );
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
    }));
    halo.scale.setScalar(BALL_RADIUS_M * 10.5);
    this.ball.add(makeSeams(), halo);
    const locator = document.createElement('span');
    locator.className = 'ball-locator';
    this.ball.add(new CSS2DObject(locator));
    this.floorGlow = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({
        map: glowTexture, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, toneMapped: false,
      }),
    );
    this.floorGlow.rotation.x = -Math.PI / 2;
    // Keep the light in the scene so ball visibility does not recompile player materials.
    this.light = new THREE.PointLight(0xffc476, 0, 1.45);
    scene.add(this.ball, this.floorGlow, this.light);
    this.show(null);
  }

  show(position) {
    this.ball.visible = this.floorGlow.visible = Boolean(position);
    this.light.intensity = position ? BALL_LIGHT_INTENSITY : 0;
    if (!position) return;
    const [x, y, z] = position;
    this.ball.position.set(x, y, z);
    this.light.position.copy(this.ball.position);
    this.floorGlow.position.set(x, 0.008, z);
    this.floorGlow.scale.setScalar(0.62 + Math.min(2, y) * 0.34);
    this.floorGlow.material.opacity = 0.34 / (1 + y * y * 4);
  }
}
