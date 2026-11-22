import * as THREE from 'three';

const UP = new THREE.Vector3(0, 1, 0);

// All players share two draw calls, even when the frame contains a full lineup.
export class SkeletonView {
  constructor(scene) {
    this.scene = scene;
    this.boneGeometry = new THREE.CylinderGeometry(0.043, 0.043, 1, 8);
    this.boneMaterial = new THREE.MeshBasicMaterial({ color: 0xe7c957 });
    this.jointGeometry = new THREE.SphereGeometry(1, 10, 8);
    this.jointMaterial = new THREE.MeshBasicMaterial({ color: 0xfff0bd });
    this.transform = new THREE.Object3D();
    this.parent = new THREE.Vector3();
    this.child = new THREE.Vector3();
    this.direction = new THREE.Vector3();
  }

  configure(layout, maxPlayers) {
    if (this.bones) {
      this.bones.removeFromParent();
      this.joints.removeFromParent();
      this.bones.dispose();
      this.joints.dispose();
      this.bones = null;
      this.joints = null;
    }
    this.layout = layout;
    if (!layout || !maxPlayers) return;
    this.bones = this.makeInstances(this.boneGeometry, this.boneMaterial, maxPlayers * layout.bones.length);
    this.joints = this.makeInstances(this.jointGeometry, this.jointMaterial, maxPlayers * layout.names.length);
  }

  makeInstances(geometry, material, capacity) {
    const mesh = new THREE.InstancedMesh(geometry, material, capacity);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    mesh.count = 0;
    this.scene.add(mesh);
    return mesh;
  }

  show(poses) {
    if (!this.bones) return;
    const { transform, parent, child, direction } = this;
    let boneCount = 0;
    let jointCount = 0;

    for (const { p, points } of poses) {
      for (const [a, b] of this.layout.bones) {
        parent.fromArray(points[a]);
        child.fromArray(points[b]);
        direction.subVectors(child, parent);
        const length = direction.length();
        if (length < 1e-5) continue;
        transform.position.addVectors(parent, child).multiplyScalar(0.5);
        transform.position.x += p[0];
        transform.position.y += p[1];
        transform.position.z += p[2];
        transform.quaternion.setFromUnitVectors(UP, direction.divideScalar(length));
        transform.scale.set(1, length, 1);
        transform.updateMatrix();
        this.bones.setMatrixAt(boneCount++, transform.matrix);
      }

      transform.quaternion.identity();
      for (let i = 0; i < points.length; i += 1) {
        const [x, y, z] = points[i];
        transform.position.set(p[0] + x, p[1] + y, p[2] + z);
        transform.scale.setScalar(i === this.layout.index.head ? 0.115 : 0.055);
        transform.updateMatrix();
        this.joints.setMatrixAt(jointCount++, transform.matrix);
      }
    }

    this.bones.count = boneCount;
    this.joints.count = jointCount;
    this.bones.instanceMatrix.needsUpdate = true;
    this.joints.instanceMatrix.needsUpdate = true;
  }
}
