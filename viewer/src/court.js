import * as THREE from 'three';
import { Reflector } from 'three/addons/objects/Reflector.js';

export const DEFAULT_COURT = { length_m: 28.65, width_m: 15.24 };

const RIM_HEIGHT = 3.048;
const RIM_OFFSET = 1.6;
const ARC_CORNER = 4.24;
const ARC_INSET = 0.91;
const LANE_DEPTH = 5.79;
const LANE_HALF_WIDTH = 2.44;
const CIRCLE_RADIUS = 1.83;
const FLOOR_COLOR = 0x0a0a0b;
const REFLECTION_WIDTH = 576;
const REFLECTION_HEIGHT = 324;

// Render the reflection at a small fixed resolution and soften it as floor polish.
const reflectionShader = {
  name: 'PolishedCourt',
  uniforms: {
    color: { value: null },
    tDiffuse: { value: null },
    textureMatrix: { value: null },
    texelSize: { value: new THREE.Vector2(1 / REFLECTION_WIDTH, 1 / REFLECTION_HEIGHT) },
  },
  vertexShader: /* glsl */ `
    uniform mat4 textureMatrix;
    varying vec4 vReflection;
    varying vec3 vViewPosition;
    varying vec3 vViewNormal;
    void main() {
      vReflection = textureMatrix * vec4(position, 1.0);
      vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
      vViewPosition = viewPosition.xyz;
      vViewNormal = normalMatrix * normal;
      gl_Position = projectionMatrix * viewPosition;
    }
  `,
  fragmentShader: /* glsl */ `
    uniform vec3 color;
    uniform sampler2D tDiffuse;
    uniform vec2 texelSize;
    varying vec4 vReflection;
    varying vec3 vViewPosition;
    varying vec3 vViewNormal;
    void main() {
      vec2 uv = vReflection.xy / vReflection.w;
      vec2 spread = texelSize * vec2(1.45, 2.25);
      vec3 reflected = texture2D(tDiffuse, uv).rgb * 0.36;
      reflected += texture2D(tDiffuse, uv + vec2(spread.x, 0.0)).rgb * 0.16;
      reflected += texture2D(tDiffuse, uv - vec2(spread.x, 0.0)).rgb * 0.16;
      reflected += texture2D(tDiffuse, uv + vec2(0.0, spread.y)).rgb * 0.16;
      reflected += texture2D(tDiffuse, uv - vec2(0.0, spread.y)).rgb * 0.16;

      float facing = clamp(dot(normalize(-vViewPosition), normalize(vViewNormal)), 0.0, 1.0);
      float strength = mix(0.10, 0.031, facing);
      gl_FragColor = vec4(color + min(reflected, vec3(1.25)) * strength, 1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }
  `,
};

export function hoopCenter({ length_m: length }, hoop) {
  if (hoop !== 'left' && hoop !== 'right') return null;
  const side = hoop === 'left' ? -1 : 1;
  return [side * (length / 2 - RIM_OFFSET), RIM_HEIGHT, 0];
}

function markingTexture(length, width) {
  const canvas = document.createElement('canvas');
  canvas.width = 2048;
  canvas.height = Math.min(2048, Math.max(1, Math.round(canvas.width * (width / length))));
  const ctx = canvas.getContext('2d');
  const scale = canvas.width / length;
  ctx.scale(scale, canvas.height / width);
  ctx.translate(length / 2, width / 2);

  ctx.strokeStyle = '#aaa9a5';
  ctx.lineWidth = 0.047;
  ctx.strokeRect(-length / 2 + 0.03, -width / 2 + 0.03, length - 0.06, width - 0.06);
  line(ctx, 0, -width / 2, 0, width / 2);
  circle(ctx, 0, 0, CIRCLE_RADIUS);

  for (const side of [-1, 1]) {
    ctx.save();
    ctx.scale(side, 1);
    const baseline = -length / 2;
    const laneEnd = baseline + LANE_DEPTH;
    const cornerZ = width / 2 - ARC_INSET;
    ctx.strokeRect(baseline + 0.03, -LANE_HALF_WIDTH, LANE_DEPTH - 0.03, LANE_HALF_WIDTH * 2);
    circle(ctx, laneEnd, 0, CIRCLE_RADIUS);
    line(ctx, baseline, cornerZ, baseline + ARC_CORNER, cornerZ);
    line(ctx, baseline, -cornerZ, baseline + ARC_CORNER, -cornerZ);
    const angle = Math.atan2(cornerZ, ARC_CORNER - RIM_OFFSET);
    const radius = Math.hypot(cornerZ, ARC_CORNER - RIM_OFFSET);
    ctx.beginPath();
    ctx.arc(baseline + RIM_OFFSET, 0, radius, -angle, angle);
    ctx.stroke();
    ctx.restore();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function line(ctx, x1, z1, x2, z2) {
  ctx.beginPath();
  ctx.moveTo(x1, z1);
  ctx.lineTo(x2, z2);
  ctx.stroke();
}

function circle(ctx, x, z, radius) {
  ctx.beginPath();
  ctx.arc(x, z, radius, 0, Math.PI * 2);
  ctx.stroke();
}

function hoopNetGeometry() {
  const points = [];
  const count = 12;
  const rings = [[0, 0.216], [-0.19, 0.168], [-0.37, 0.115]];

  function point(ring, index) {
    const [y, radius] = rings[ring];
    const angle = (index + (ring % 2) * 0.5) * 2 * Math.PI / count;
    return [Math.cos(angle) * radius, y, Math.sin(angle) * radius];
  }

  for (let i = 0; i < count; i += 1) {
    points.push(...point(0, i), ...point(1, i), ...point(0, i + 1), ...point(1, i));
    points.push(...point(1, i), ...point(2, i), ...point(1, i), ...point(2, i + 1));
    points.push(...point(2, i), ...point(2, i + 1));
  }
  return new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
}

function addHoops(court, dimensions) {
  const { length_m: length } = dimensions;
  const rimGeometry = new THREE.TorusGeometry(0.2286, 0.016, 8, 32);
  const rimMaterial = new THREE.MeshBasicMaterial({ color: 0xa77545 });
  const boardGeometry = new THREE.PlaneGeometry(1.83, 1.07);
  const boardMaterial = new THREE.MeshBasicMaterial({
    color: 0x242426, transparent: true, opacity: 0.48, side: THREE.DoubleSide, depthWrite: false,
  });
  const frameGeometry = new THREE.EdgesGeometry(boardGeometry);
  const targetGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.30, -0.38, 0.002), new THREE.Vector3(0.30, -0.38, 0.002),
    new THREE.Vector3(0.30, 0.07, 0.002), new THREE.Vector3(-0.30, 0.07, 0.002),
  ]);
  const frameMaterial = new THREE.LineBasicMaterial({ color: 0x94938f });
  const netGeometry = hoopNetGeometry();
  const netMaterial = new THREE.LineBasicMaterial({ color: 0x8a8985, transparent: true, opacity: 0.43 });
  for (const hoop of ['left', 'right']) {
    const side = hoop === 'left' ? -1 : 1;
    const rim = new THREE.Mesh(rimGeometry, rimMaterial);
    rim.rotation.x = -Math.PI / 2;
    rim.position.fromArray(hoopCenter(dimensions, hoop));
    const net = new THREE.LineSegments(netGeometry, netMaterial);
    net.position.copy(rim.position);
    const board = new THREE.Mesh(boardGeometry, boardMaterial);
    board.rotation.y = Math.PI / 2;
    board.position.set(side * (length / 2 - 1.2), RIM_HEIGHT + 0.38, 0);
    board.add(new THREE.LineSegments(frameGeometry, frameMaterial));
    board.add(new THREE.LineLoop(targetGeometry, frameMaterial));
    court.add(rim, net, board);
  }
}

export function makeCourt(dimensions, enableReflection) {
  const { length_m: length, width_m: width } = dimensions;
  const court = new THREE.Group();
  const floorGeometry = new THREE.PlaneGeometry(length + 3.2, width + 3.2);
  const floor = new THREE.Mesh(
    floorGeometry,
    new THREE.MeshBasicMaterial({ color: FLOOR_COLOR }),
  );
  floor.rotation.x = -Math.PI / 2;
  court.userData.floor = floor;
  court.add(floor);
  if (enableReflection) {
    const reflector = new Reflector(floorGeometry, {
      shader: reflectionShader,
      color: FLOOR_COLOR,
      textureWidth: REFLECTION_WIDTH,
      textureHeight: REFLECTION_HEIGHT,
      multisample: 0,
    });
    reflector.rotation.x = floor.rotation.x;
    reflector.visible = false;
    court.userData.reflector = reflector;
    court.add(reflector);
  }
  const markings = new THREE.Mesh(
    new THREE.PlaneGeometry(length, width),
    new THREE.MeshBasicMaterial({
      map: markingTexture(length, width), transparent: true, depthWrite: false,
    }),
  );
  markings.rotation.x = floor.rotation.x;
  markings.position.y = 0.004;
  court.add(markings);
  addHoops(court, dimensions);
  return court;
}

export function disposeCourt(court) {
  const geometries = new Set();
  const materials = new Set();
  court.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    if (object.isReflector) object.dispose();
    else if (object.material) materials.add(object.material);
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) {
    material.map?.dispose();
    material.dispose();
  }
  court.removeFromParent();
}
