import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { BallView } from './ball-view.js';
import { DEFAULT_COURT, disposeCourt, hoopCenter, makeCourt } from './court.js';
import { PlayerMeshView } from './player-mesh-view.js';
import { groundPose, poseScaleKey } from './player-pose.js';
import { SkeletonView } from './skeleton-view.js';

export class CourtView {
  constructor(viewport, { reflection = true, maxPixelRatio = 2 } = {}) {
    this.viewport = viewport;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxPixelRatio));
    this.renderer.domElement.setAttribute('aria-hidden', 'true');
    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'court-labels';
    this.labels.domElement.setAttribute('aria-hidden', 'true');
    viewport.append(this.renderer.domElement, this.labels.domElement);

    this.scene = new THREE.Scene();
    this.skeletons = new SkeletonView(this.scene);
    this.playerMeshes = new PlayerMeshView(this.scene);
    this.ball = new BallView(this.scene);
    this.scene.background = new THREE.Color(0x000000);
    const key = new THREE.DirectionalLight(0xfff9ee, 2.1);
    key.position.set(-4, 9, 5);
    const fill = new THREE.DirectionalLight(0xe4ebf5, 0.45);
    fill.position.set(5, 4, -6);
    this.scene.add(new THREE.HemisphereLight(0xf2f1ea, 0x77746e, 1.48), key, fill);
    this.topCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 300);
    this.topCamera.position.set(0, 45, 0);
    this.topCamera.up.set(0, 0, -1);
    this.topCamera.lookAt(0, 0, 0);
    this.orbitCamera = new THREE.PerspectiveCamera(40, 1, 0.1, 300);
    this.povCamera = new THREE.PerspectiveCamera(95, 1, 0.06, 150);
    this.povTarget = new THREE.Vector3();
    this.povBaseRotation = new THREE.Quaternion();
    this.povReady = false;
    this.controls = new OrbitControls(this.orbitCamera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.06;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 180;
    this.setMode('top');

    this.markers = new Map();
    this.discGeometry = new THREE.CylinderGeometry(0.22, 0.22, 0.055, 24);
    this.discMaterial = new THREE.MeshBasicMaterial({ color: 0xe8cf62 });
    this.borderGeometry = new THREE.CylinderGeometry(0.27, 0.27, 0.045, 24);
    this.borderMaterial = new THREE.MeshBasicMaterial({ color: 0x594d29 });
    this.shadowGeometry = new THREE.CircleGeometry(0.36, 24);
    this.shadowMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000, opacity: 0.36, transparent: true, depthWrite: false,
    });
    this.enableReflection = reflection && (this.renderer.extensions.has('EXT_color_buffer_float')
      || this.renderer.extensions.has('EXT_color_buffer_half_float'));
    this.hasPlayers = false;
    this.hasBall = false;
    this.setCourt(DEFAULT_COURT);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(viewport);
  }

  resize() {
    const { clientWidth: width, clientHeight: height } = this.viewport;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.labels.setSize(width, height);
    this.needsRender = true;
    const aspect = width / height;
    this.orbitCamera.aspect = aspect;
    this.orbitCamera.updateProjectionMatrix();
    this.povCamera.aspect = aspect;
    this.povCamera.updateProjectionMatrix();
    const { length_m: length, width_m: courtWidth } = this.dimensions;
    const halfHeight = Math.max((courtWidth + 5) / 2, (length + 5) / (2 * aspect));
    this.topCamera.left = -halfHeight * aspect;
    this.topCamera.right = halfHeight * aspect;
    this.topCamera.top = halfHeight;
    this.topCamera.bottom = -halfHeight;
    this.topCamera.updateProjectionMatrix();
  }

  setCourt(dimensions) {
    const court = makeCourt(dimensions, this.enableReflection);
    if (this.court) disposeCourt(this.court);
    this.dimensions = dimensions;
    this.court = court;
    this.scene.add(court);
    this.resize();
    this.frameOrbitCourt();
  }

  frameOrbitCourt() {
    const { length_m: length, width_m: width } = this.dimensions;
    const camera = this.orbitCamera;
    const direction = new THREE.Vector3(0.29, 0.46, 0.84).normalize();
    camera.position.copy(direction);
    this.controls.target.set(0, 0, 0);
    camera.lookAt(this.controls.target);

    const worldToView = camera.quaternion.clone().invert();
    const tanY = 0.88 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const tanX = tanY * camera.aspect;
    const point = new THREE.Vector3();
    let distance = 3;
    for (const x of [-length / 2 - 1.6, length / 2 + 1.6]) {
      for (const z of [-width / 2 - 1.6, width / 2 + 1.6]) {
        point.set(x, 0, z).applyQuaternion(worldToView);
        distance = Math.max(distance, point.z + Math.abs(point.x) / tanX, point.z + Math.abs(point.y) / tanY);
      }
      point.set(x, 4, 0).applyQuaternion(worldToView);
      distance = Math.max(distance, point.z + Math.abs(point.x) / tanX, point.z + Math.abs(point.y) / tanY);
    }
    camera.position.copy(direction).multiplyScalar(distance);
    this.controls.update();
  }

  setMode(mode) {
    this.mode = mode;
    this.updateActiveCamera();
  }

  setPlayerPov(pov) {
    this.povReady = Boolean(pov);
    if (pov) {
      this.povCamera.position.fromArray(pov.eye);
      this.povTarget.fromArray(pov.direction).add(this.povCamera.position);
      this.povCamera.lookAt(this.povTarget);
      if (pov.lookAt?.weight > 0) {
        this.povBaseRotation.copy(this.povCamera.quaternion);
        this.povCamera.lookAt(...pov.lookAt.target);
        this.povCamera.quaternion.slerp(this.povBaseRotation, 1 - pov.lookAt.weight);
      }
    }
    if (this.mode === 'pov') this.updateActiveCamera();
  }

  updateActiveCamera() {
    const firstPerson = this.mode === 'pov' && this.povReady;
    this.camera = firstPerson ? this.povCamera : this.mode === 'top' ? this.topCamera : this.orbitCamera;
    this.controls.enabled = this.camera === this.orbitCamera;
    this.viewport.classList.toggle('is-orbit', this.controls.enabled);
  }

  addMarker(id) {
    const group = new THREE.Group();
    const disc = new THREE.Mesh(this.discGeometry, this.discMaterial);
    disc.position.y = 0.069;
    const border = new THREE.Mesh(this.borderGeometry, this.borderMaterial);
    border.position.y = 0.029;
    const shadow = new THREE.Mesh(this.shadowGeometry, this.shadowMaterial);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = 0.004;
    shadow.scale.y = 0.76;
    shadow.visible = false;
    const text = document.createElement('div');
    text.className = 'track-label';
    text.dataset.trackId = String(id);
    text.textContent = String(id);
    const label = new CSS2DObject(text);
    label.position.y = 0.17;
    label.center.set(-0.29, 0.8);
    group.add(shadow, border, disc, label);
    this.scene.add(group);
    const marker = { group, label, border, disc, shadow };
    this.markers.set(id, marker);
    return marker;
  }

  clearMarkers() {
    for (const { group, label } of this.markers.values()) {
      label.element.remove();
      group.removeFromParent();
    }
    this.markers.clear();
    this.playerMeshes.clear();
    this.hasPlayers = false;
  }

  configurePoses(layout, maxPlayers) {
    this.skeletons.configure(layout, maxPlayers);
  }

  get playerModelReady() {
    return this.playerMeshes.ready;
  }

  loadPlayerModel() {
    return this.playerMeshes.load();
  }

  ballHandPosition(p, points, layout, hand) {
    return this.playerMeshes.handPosition(p, points, layout, hand);
  }

  hoopPosition(hoop) {
    return hoopCenter(this.dimensions, hoop);
  }

  showBall(position) {
    this.hasBall = Boolean(position);
    this.ball.show(position);
  }

  showPlayers(players, pose, display, povTrack = null) {
    this.hasPlayers = players.length > 0;
    const poses = [];
    this.playerMeshes.hidePlayers();
    for (const { group } of this.markers.values()) group.visible = false;
    for (const player of players) {
      const { id, p, joints, look_at: lookAt } = player;
      const handsOnly = id === povTrack;
      const points = (handsOnly || display !== 'markers') && pose
        && groundPose(joints, pose.layout, pose.scales.get(poseScaleKey(player)), pose.basis);
      if (handsOnly) {
        if (points) this.playerMeshes.showPlayer(id, p, points, pose.layout, { handsOnly: true });
        continue;
      }
      const { group, label, disc, border, shadow } = this.markers.get(id) ?? this.addMarker(id);
      group.position.fromArray(p);
      group.visible = true;
      const hasMesh = points && display === 'players' && this.playerMeshes.showPlayer(id, p, points, pose.layout, { lookAt });
      if (points && !hasMesh) poses.push({ p, points });
      disc.visible = border.visible = !hasMesh;
      shadow.visible = Boolean(hasMesh);
      label.element.dataset.pose = hasMesh ? 'player' : points ? 'skeleton' : 'marker';
    }
    this.skeletons.show(poses);
  }

  render() {
    if (this.controls.enabled) this.controls.update();
    const { floor, reflector } = this.court.userData;
    if (reflector) {
      reflector.visible = !this.camera.isOrthographicCamera && (this.hasPlayers || this.hasBall);
      floor.visible = !reflector.visible;
    }
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
    this.needsRender = false;
  }
}
