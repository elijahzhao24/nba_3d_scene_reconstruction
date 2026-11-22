import { clone } from 'three/addons/utils/SkeletonUtils.js';
import { createRigBallHandResolver } from './ball-hand.js';
import { createHandsGeometry } from './first-person-hands.js';
import { bindPlayerRig, placePlayerRig } from './player-rig.js';
import playerModelUrl from '../assets/player.glb?url';

export class PlayerMeshView {
  constructor(scene) {
    this.scene = scene;
    this.players = new Map();
    this.handsGeometries = new Map();
  }

  get ready() {
    return Boolean(this.template);
  }

  load() {
    if (!this.modelPromise) {
      this.modelPromise = import('three/addons/loaders/GLTFLoader.js')
        .then(({ GLTFLoader }) => new GLTFLoader().loadAsync(playerModelUrl))
        .then(({ scene }) => {
          // Keep endpoint sampling independent of the visible players and playhead.
          const handRig = bindPlayerRig(clone(scene));
          if (!scene.getObjectByProperty('isSkinnedMesh', true) || !handRig) {
            throw new Error('The bundled player rig is incompatible.');
          }
          scene.traverse((mesh) => {
            if (mesh.isSkinnedMesh && !this.handsGeometries.has(mesh.geometry)) {
              this.handsGeometries.set(mesh.geometry, createHandsGeometry(mesh));
            }
          });
          this.resolveHand = createRigBallHandResolver(handRig);
          this.template = scene;
        }).catch((error) => {
          this.modelPromise = null;
          throw error;
        });
    }
    return this.modelPromise;
  }

  hidePlayers() {
    for (const { model } of this.players.values()) model.visible = false;
  }

  clear() {
    for (const { model, skeletons } of this.players.values()) {
      model.removeFromParent();
      for (const skeleton of skeletons) skeleton.dispose();
    }
    this.players.clear();
  }

  addPlayer(id) {
    const model = clone(this.template);
    const rig = bindPlayerRig(model);
    if (!rig) return null;
    const skeletons = new Set();
    const meshes = [];
    model.traverse((object) => {
      if (!object.isSkinnedMesh) return;
      // The bind-pose bounds do not enclose every animated limb position.
      object.frustumCulled = false;
      skeletons.add(object.skeleton);
      meshes.push({
        mesh: object,
        body: object.geometry,
        hands: this.handsGeometries.get(object.geometry),
      });
    });
    model.visible = false;
    this.scene.add(model);
    const player = { model, rig, skeletons, meshes };
    this.players.set(id, player);
    return player;
  }

  showPlayer(id, p, points, layout, handsOnly = false) {
    if (!this.ready) return false;
    const player = this.players.get(id) ?? this.addPlayer(id);
    if (!player || !placePlayerRig(player.rig, p, points, layout)) return false;
    let visible = false;
    for (const { mesh, body, hands } of player.meshes) {
      mesh.geometry = handsOnly && hands ? hands : body;
      mesh.visible = !handsOnly || Boolean(hands);
      visible ||= mesh.visible;
    }
    player.model.visible = visible;
    return visible;
  }

  handPosition(p, points, layout, hand) {
    return this.resolveHand?.(p, points, layout, hand) ?? null;
  }
}
