"""Versioned joint, bone, side, and semantic endpoint definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str
    child: str


@dataclass(frozen=True)
class SkeletonDefinition:
    id: str
    version: str
    joints: tuple[str, ...]
    bones: tuple[Bone, ...]
    anatomical_sides: dict[str, str]
    semantic_endpoints: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


HALPE_JOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "pelvis",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)

H36M_JOINTS = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "neck",
    "nose",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

H36M_BONES = (
    Bone("right_pelvis_link", "pelvis", "right_hip"),
    Bone("right_upper_leg", "right_hip", "right_knee"),
    Bone("right_lower_leg", "right_knee", "right_ankle"),
    Bone("left_pelvis_link", "pelvis", "left_hip"),
    Bone("left_upper_leg", "left_hip", "left_knee"),
    Bone("left_lower_leg", "left_knee", "left_ankle"),
    Bone("lower_spine", "pelvis", "spine"),
    Bone("upper_spine", "spine", "neck"),
    Bone("face", "neck", "nose"),
    Bone("head", "nose", "head"),
    Bone("left_clavicle", "neck", "left_shoulder"),
    Bone("left_upper_arm", "left_shoulder", "left_elbow"),
    Bone("left_forearm", "left_elbow", "left_wrist"),
    Bone("right_clavicle", "neck", "right_shoulder"),
    Bone("right_upper_arm", "right_shoulder", "right_elbow"),
    Bone("right_forearm", "right_elbow", "right_wrist"),
)

HALPE_BONES = (
    Bone("right_pelvis_link", "pelvis", "right_hip"),
    Bone("right_upper_leg", "right_hip", "right_knee"),
    Bone("right_lower_leg", "right_knee", "right_ankle"),
    Bone("right_heel_link", "right_ankle", "right_heel"),
    Bone("right_big_toe_link", "right_ankle", "right_big_toe"),
    Bone("right_small_toe_link", "right_ankle", "right_small_toe"),
    Bone("left_pelvis_link", "pelvis", "left_hip"),
    Bone("left_upper_leg", "left_hip", "left_knee"),
    Bone("left_lower_leg", "left_knee", "left_ankle"),
    Bone("left_heel_link", "left_ankle", "left_heel"),
    Bone("left_big_toe_link", "left_ankle", "left_big_toe"),
    Bone("left_small_toe_link", "left_ankle", "left_small_toe"),
    Bone("torso", "pelvis", "neck"),
    Bone("head_link", "neck", "head"),
    Bone("nose_link", "head", "nose"),
    Bone("left_clavicle", "neck", "left_shoulder"),
    Bone("left_upper_arm", "left_shoulder", "left_elbow"),
    Bone("left_forearm", "left_elbow", "left_wrist"),
    Bone("right_clavicle", "neck", "right_shoulder"),
    Bone("right_upper_arm", "right_shoulder", "right_elbow"),
    Bone("right_forearm", "right_elbow", "right_wrist"),
)


def _sides(joints: tuple[str, ...]) -> dict[str, str]:
    return {
        joint: "left" if joint.startswith("left_") else "right"
        for joint in joints
        if joint.startswith(("left_", "right_"))
    }


ENDPOINTS = {
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}

HALPE_26 = SkeletonDefinition(
    "halpe26", "1.0", HALPE_JOINTS, HALPE_BONES, _sides(HALPE_JOINTS), ENDPOINTS
)
H36M_17 = SkeletonDefinition(
    "h36m17", "1.0", H36M_JOINTS, H36M_BONES, _sides(H36M_JOINTS), ENDPOINTS
)
