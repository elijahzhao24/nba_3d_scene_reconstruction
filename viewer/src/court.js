import * as THREE from 'three';

export const DEFAULT_COURT = { length_m: 28.65, width_m: 15.24 };

const RIM_HEIGHT = 3.048;
const RIM_OFFSET = 1.6;
const ARC_CORNER = 4.24;
const ARC_INSET = 0.91;
const LANE_DEPTH = 5.79;
const LANE_HALF_WIDTH = 2.44;
const CIRCLE_RADIUS = 1.83;

export function hoopCenter({ length_m: length }, hoop) {
  if (hoop !== 'left' && hoop !== 'right') return null;
  const side = hoop === 'left' ? -1 : 1;
  return [side * (length / 2 - RIM_OFFSET), RIM_HEIGHT, 0];
}

function courtTexture(length, width) {
  const canvas = document.createElement('canvas');
  canvas.width = 2048;
  canvas.height = Math.min(2048, Math.max(1, Math.round(canvas.width * (width / length))));
  const ctx = canvas.getContext('2d');
  const scale = canvas.width / length;
  ctx.scale(scale, canvas.height / width);
  ctx.translate(length / 2, width / 2);

  const wood = ctx.createLinearGradient(0, -width / 2, 0, width / 2);
  wood.addColorStop(0, '#b47b4d');
  wood.addColorStop(0.53, '#d5aa74');
  wood.addColorStop(1, '#b88657');
  ctx.fillStyle = wood;
  ctx.fillRect(-length / 2, -width / 2, length, width);

  const plankWidth = Math.max(0.2, width / 100);
  const plankLength = Math.max(2.2, length / 40);
  for (let row = 0; row * plankWidth < width; row += 1) {
    const z = -width / 2 + row * plankWidth;
    ctx.fillStyle = row % 4 === 0 ? '#fff1c523' : '#67401d11';
    ctx.fillRect(-length / 2, z, length, plankWidth);
    ctx.fillStyle = '#56382029';
    ctx.fillRect(-length / 2, z, length, 0.012);
    for (let x = -length / 2 + (row % 4) * plankLength / 4; x < length / 2; x += plankLength) {
      ctx.fillRect(x, z, 0.014, plankWidth);
    }
  }

  ctx.strokeStyle = '#fff1d2';
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
    ctx.fillStyle = '#253b3e';
    ctx.fillRect(baseline + 0.03, -LANE_HALF_WIDTH, LANE_DEPTH - 0.03, LANE_HALF_WIDTH * 2);
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

function addHoops(court, dimensions) {
  const { length_m: length } = dimensions;
  const rimGeometry = new THREE.TorusGeometry(0.2286, 0.016, 8, 32);
  const rimMaterial = new THREE.MeshBasicMaterial({ color: 0xff925b });
  const boardGeometry = new THREE.PlaneGeometry(1.83, 1.07);
  const boardMaterial = new THREE.MeshBasicMaterial({
    color: 0xbde5de, transparent: true, opacity: 0.38, side: THREE.DoubleSide, depthWrite: false,
  });
  const frameGeometry = new THREE.EdgesGeometry(boardGeometry);
  const frameMaterial = new THREE.LineBasicMaterial({ color: 0xcee7dc });
  for (const hoop of ['left', 'right']) {
    const side = hoop === 'left' ? -1 : 1;
    const rim = new THREE.Mesh(rimGeometry, rimMaterial);
    rim.rotation.x = -Math.PI / 2;
    rim.position.fromArray(hoopCenter(dimensions, hoop));
    const board = new THREE.Mesh(boardGeometry, boardMaterial);
    board.rotation.y = Math.PI / 2;
    board.position.set(side * (length / 2 - 1.2), RIM_HEIGHT + 0.38, 0);
    board.add(new THREE.LineSegments(frameGeometry, frameMaterial));
    court.add(rim, board);
  }
}

export function makeCourt(dimensions) {
  const { length_m: length, width_m: width } = dimensions;
  const court = new THREE.Group();
  const apron = new THREE.Mesh(
    new THREE.BoxGeometry(length + 3.2, 0.26, width + 3.2),
    new THREE.MeshBasicMaterial({ color: 0x304244 }),
  );
  apron.position.y = -0.14;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(length, width),
    new THREE.MeshBasicMaterial({ map: courtTexture(length, width) }),
  );
  floor.rotation.x = -Math.PI / 2;
  court.add(apron, floor);
  addHoops(court, dimensions);
  return court;
}

export function disposeCourt(court) {
  const geometries = new Set();
  const materials = new Set();
  court.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    if (object.material) materials.add(object.material);
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) {
    material.map?.dispose();
    material.dispose();
  }
  court.removeFromParent();
}
