import { Matrix4, Quaternion, Vector3 } from 'three';

const PARENT_NAMES = {
  pelvis: null,
  spine: 'pelvis',
  neck: 'spine',
  head: 'neck',
  left_hip: 'pelvis',
  left_knee: 'left_hip',
  left_ankle: 'left_knee',
  right_hip: 'pelvis',
  right_knee: 'right_hip',
  right_ankle: 'right_knee',
  left_shoulder: 'neck',
  left_elbow: 'left_shoulder',
  left_wrist: 'left_elbow',
  right_shoulder: 'neck',
  right_elbow: 'right_shoulder',
  right_wrist: 'right_elbow',
};
const BONE_NAMES = Object.keys(PARENT_NAMES);
const AIM_LINKS = [
  ['spine', 'neck'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
];
const MIN_DIRECTION_SQ = 1e-8;
const UP = new Vector3(0, 1, 0);

function usableDirection(direction) {
  const lengthSquared = direction.lengthSq();
  return Number.isFinite(lengthSquared) && lengthSquared >= MIN_DIRECTION_SQ;
}

class BodyAxes {
  constructor() {
    this.x = new Vector3();
    this.y = new Vector3();
    this.z = new Vector3();
    this.matrix = new Matrix4();
  }

  frame(lateral, up, rotation) {
    if (!usableDirection(lateral) || !usableDirection(up)) return false;
    this.x.copy(lateral).normalize();
    this.z.crossVectors(this.x, this.y.copy(up).normalize());
    if (!usableDirection(this.z)) return false;
    this.z.normalize();
    this.y.crossVectors(this.z, this.x);
    rotation.setFromRotationMatrix(this.matrix.makeBasis(this.x, this.y, this.z));
    return true;
  }
}

function setWorldRotation(bone, rotation, parentRotation) {
  bone.parent.getWorldQuaternion(parentRotation);
  bone.quaternion.copy(parentRotation).invert().multiply(rotation);
}

export function bindPlayerRig(object) {
  if (!object?.isObject3D) return null;
  const bones = Object.fromEntries(BONE_NAMES.map((name) => [name, object.getObjectByName(name)]));
  for (const name of BONE_NAMES) {
    const parent = PARENT_NAMES[name] ? bones[PARENT_NAMES[name]] : object;
    if (!bones[name]?.isBone || bones[name].parent !== parent) return null;
  }

  object.updateWorldMatrix(true, true);
  const points = {};
  const rest = {};
  for (const name of BONE_NAMES) {
    const bone = bones[name];
    points[name] = object.worldToLocal(bone.getWorldPosition(new Vector3()));
    rest[name] = bone.quaternion.clone();
  }
  if (![points.left_ankle.y, points.right_ankle.y].every((value) => Number.isFinite(value) && value >= 0)) {
    return null;
  }

  const axes = new BodyAxes();
  const lateral = new Vector3();
  const up = new Vector3();
  const pelvisBasisOffset = new Quaternion();
  const neckBasisOffset = new Quaternion();
  if (!axes.frame(
    lateral.subVectors(points.left_hip, points.right_hip),
    up.subVectors(points.spine, points.pelvis),
    pelvisBasisOffset,
  ) || !axes.frame(
    lateral.subVectors(points.left_shoulder, points.right_shoulder),
    up.subVectors(points.head, points.neck),
    neckBasisOffset,
  )) {
    return null;
  }
  const rootRotation = object.getWorldQuaternion(new Quaternion()).invert();
  const bindRotation = new Quaternion();
  pelvisBasisOffset.invert().multiply(bones.pelvis.getWorldQuaternion(bindRotation).premultiply(rootRotation));
  neckBasisOffset.invert().multiply(bones.neck.getWorldQuaternion(bindRotation).premultiply(rootRotation));

  const aimLinks = AIM_LINKS.map(([name, child]) => ({
    name,
    child,
    restDirection: bones[child].position.clone().normalize().applyQuaternion(rest[name]),
  }));
  if (aimLinks.some(({ restDirection }) => !usableDirection(restDirection))) return null;

  return {
    object,
    bones,
    rest,
    aimLinks,
    axes,
    pelvisBasisOffset,
    neckBasisOffset,
    leftAnkleClearance: points.left_ankle.y,
    rightAnkleClearance: points.right_ankle.y,
    points: Object.fromEntries(BONE_NAMES.map((name) => [name, new Vector3()])),
    hips: new Vector3(),
    shoulders: new Vector3(),
    direction: new Vector3(),
    pelvisRotation: new Quaternion(),
    neckRotation: new Quaternion(),
    yaw: new Quaternion(),
    parentRotation: new Quaternion(),
    leftFoot: new Vector3(),
    rightFoot: new Vector3(),
  };
}

export function placePlayerRig(rig, p, points, layout) {
  if (!rig || !Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite) || !points || !layout?.index) {
    return false;
  }
  for (const name of BONE_NAMES) {
    const point = points[layout.index[name]];
    if (!Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite)) return false;
    rig.points[name].fromArray(point);
  }

  const {
    object, bones, rest, axes, hips, shoulders, direction, parentRotation,
    pelvisRotation, neckRotation, yaw, leftFoot, rightFoot,
  } = rig;
  const source = rig.points;
  hips.subVectors(source.left_hip, source.right_hip);
  shoulders.subVectors(source.left_shoulder, source.right_shoulder);
  direction.subVectors(source.spine, source.pelvis);
  if (!axes.frame(hips, direction, pelvisRotation) && !axes.frame(shoulders, direction, pelvisRotation)) {
    return false;
  }
  const hasNeckBasis = axes.frame(shoulders, direction.subVectors(source.head, source.neck), neckRotation);

  direction.copy(hips).setY(0);
  if (!usableDirection(direction)) direction.copy(shoulders).setY(0);
  if (!usableDirection(direction)) return false;
  yaw.setFromAxisAngle(UP, Math.atan2(-direction.z, direction.x));

  setWorldRotation(bones.pelvis, pelvisRotation.multiply(rig.pelvisBasisOffset), parentRotation);
  for (const { name, child, restDirection } of rig.aimLinks) {
    const bone = bones[name];
    direction.subVectors(source[child], source[name]);
    if (!usableDirection(direction)) {
      bone.quaternion.copy(rest[name]);
    } else {
      bone.parent.getWorldQuaternion(parentRotation);
      direction.normalize().applyQuaternion(parentRotation.invert());
      bone.quaternion.setFromUnitVectors(restDirection, direction).multiply(rest[name]);
    }
    if (name === 'spine') {
      if (hasNeckBasis) {
        setWorldRotation(bones.neck, neckRotation.multiply(rig.neckBasisOffset), parentRotation);
      } else {
        bones.neck.quaternion.copy(rest.neck);
      }
    }
  }

  // Keep the shoes flat and preserve the ankle-to-sole clearance from the rest pose.
  setWorldRotation(bones.left_ankle, neckRotation.copy(yaw).multiply(rest.left_ankle), parentRotation);
  setWorldRotation(bones.right_ankle, neckRotation.copy(yaw).multiply(rest.right_ankle), parentRotation);
  object.worldToLocal(bones.left_ankle.getWorldPosition(leftFoot));
  object.worldToLocal(bones.right_ankle.getWorldPosition(rightFoot));
  const soleY = Math.min(leftFoot.y - rig.leftAnkleClearance, rightFoot.y - rig.rightAnkleClearance);
  object.position.set(
    p[0] - (leftFoot.x + rightFoot.x) * 0.5,
    p[1] - soleY,
    p[2] - (leftFoot.z + rightFoot.z) * 0.5,
  );
  return true;
}
