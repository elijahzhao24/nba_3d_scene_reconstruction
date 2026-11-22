"""Build the original low-poly replay player as a self-contained glTF skin."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Point = tuple[float, float, float]
Color = tuple[int, int, int, int]
Weight = tuple[str, str, float]

JERSEY = "#efd24f"
UNIFORM_ACCENT = "#9d7827"
TRIM = "#fff0b8"
SHORTS = "#d8ae39"
SKIN = "#bcc1c7"
FACE = "#b0b0b0"
HAIR = "#646464"
EYES = "#515151"
SOCK = "#e9e7df"
SOLE = "#c4c1b5"
SHOE = "#d5d1c7"
SATIN_STEEL = 0
MATTE_SURFACES = 1


def sub(a: Point, b: Point) -> Point:
    return tuple(x - y for x, y in zip(a, b))


def cross(a: Point, b: Point) -> Point:
    x, y, z = a
    u, v, w = b
    return y * w - z * v, z * u - x * w, x * v - y * u


def unit(point: Point) -> Point:
    length = math.hypot(*point)
    return tuple(component / length for component in point) if length else (0, 1, 0)


def blend(a: Point, b: Point, t: float) -> Point:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def rgba(hex_color: str) -> Color:
    srgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in srgb
    ]
    return (*[round(c * 255) for c in linear], 255)


@dataclass(frozen=True)
class Ring:
    center: Point
    half_width: float
    half_depth: float
    color: str
    bone: str | Weight


def vertical(
    x: float, z: float, profiles: list[tuple[float, float, float, str, str | Weight]]
) -> list[Ring]:
    return [Ring((x, y, z), width, depth, color, bone) for y, width, depth, color, bone in profiles]


class PlayerMesh:
    def __init__(self, joints: dict[str, Point]) -> None:
        self.joints = dict(zip(joints, range(len(joints))))
        self.positions: list[Point] = []
        self.colors: list[Color] = []
        self.joint_indices: list[tuple[int, ...]] = []
        self.joint_weights: list[tuple[int, ...]] = []
        self.indices: tuple[list[int], list[int]] = ([], [])

    def vertex(self, position: Point, color: str, binding: str | Weight) -> int:
        index = len(self.positions)
        self.positions.append(position)
        self.colors.append(rgba(color))
        if isinstance(binding, str):
            binding = (binding, binding, 1)
        first, second, amount = binding
        if first == second:
            amount = 1
        weight = round(amount * 255)
        self.joint_indices.append((
            self.joints[first] if weight else 0,
            self.joints[second] if weight < 255 else 0,
            0,
            0,
        ))
        self.joint_weights.append((weight, 255 - weight, 0, 0))
        return index

    def loft(
        self,
        rings: list[Ring],
        sides: int = 10,
        *,
        follow_axis: bool = False,
        side_color: Callable[[int, int], str | None] | None = None,
        material: int = MATTE_SURFACES,
    ) -> None:
        """Join elliptical cross-sections and cap both ends."""
        axis = unit(sub(rings[-1].center, rings[0].center)) if follow_axis else (0, 1, 0)
        right = unit(cross(axis, (0, 0, 1)))
        forward = cross(right, axis)
        start = len(self.positions)
        indices = self.indices[material]
        for row, ring in enumerate(rings):
            for side in range(sides):
                angle = 2 * math.pi * side / sides
                width = math.cos(angle) * ring.half_width
                depth = math.sin(angle) * ring.half_depth
                point = tuple(c + x * width + z * depth for c, x, z in zip(ring.center, right, forward))
                stripe = side_color(row, side) if side_color else None
                self.vertex(point, stripe or ring.color, ring.bone)

        for row in range(len(rings) - 1):
            for side in range(sides):
                base = start + row * sides
                a = base + side
                b = base + (side + 1) % sides
                indices.extend((a, a + sides, b + sides, a, b + sides, b))

        bottom = self.vertex(rings[0].center, rings[0].color, rings[0].bone)
        top = self.vertex(rings[-1].center, rings[-1].color, rings[-1].bone)
        last_ring = start + (len(rings) - 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            indices.extend((bottom, start + side, start + next_side))
            indices.extend((top, last_ring + next_side, last_ring + side))

    def normals(self) -> list[Point]:
        totals = [[0.0, 0.0, 0.0] for _ in self.positions]
        for indices in self.indices:
            for a, b, c in zip(indices[::3], indices[1::3], indices[2::3]):
                normal = cross(sub(self.positions[b], self.positions[a]), sub(self.positions[c], self.positions[a]))
                for vertex in (a, b, c):
                    for axis, value in enumerate(normal):
                        totals[vertex][axis] += value
        return [unit(tuple(normal)) for normal in totals]


def rig() -> tuple[dict[str, Point], dict[str, str]]:
    joints = {
        "pelvis": (0, 1.01, 0),
        "spine": (0, 1.22, 0),
        "neck": (0, 1.61, 0),
        "head": (0, 1.74, 0),
    }
    parents = {"spine": "pelvis", "neck": "spine", "head": "neck"}
    for side, sign in (("left", 1), ("right", -1)):
        hip, knee, ankle = (f"{side}_{name}" for name in ("hip", "knee", "ankle"))
        shoulder, elbow, wrist = (f"{side}_{name}" for name in ("shoulder", "elbow", "wrist"))
        joints.update({
            hip: (sign * 0.108, 0.969, 0),
            knee: (sign * 0.116, 0.524, 0),
            ankle: (sign * 0.120, 0.105, 0),
            shoulder: (sign * 0.225, 1.511, 0),
            elbow: (sign * 0.385, 1.257, 0),
            wrist: (sign * 0.482, 1.019, 0.019),
        })
        parents.update({hip: "pelvis", knee: hip, ankle: knee, shoulder: "neck", elbow: shoulder, wrist: elbow})
    return joints, parents


def add_torso(mesh: PlayerMesh) -> None:
    mesh.loft(vertical(0, 0, [
        (0.881, 0.065, 0.062, SHORTS, "pelvis"),
        (0.935, 0.162, 0.105, SHORTS, "pelvis"),
        (0.980, 0.194, 0.127, SHORTS, "pelvis"),
        (1.048, 0.179, 0.115, UNIFORM_ACCENT, "pelvis"),
        (1.072, 0.174, 0.110, TRIM, "pelvis"),
        (1.085, 0.170, 0.105, SHORTS, "pelvis"),
    ]), 12)
    mesh.loft(vertical(0, 0.010, [
        (1.066, 0.173, 0.108, UNIFORM_ACCENT, "pelvis"),
        (1.116, 0.163, 0.098, JERSEY, ("pelvis", "spine", 0.76)),
        (1.251, 0.174, 0.103, JERSEY, "spine"),
        (1.400, 0.203, 0.119, JERSEY, ("spine", "neck", 0.69)),
        (1.506, 0.209, 0.109, JERSEY, ("spine", "neck", 0.39)),
        (1.564, 0.152, 0.087, JERSEY, "neck"),
        (1.582, 0.085, 0.079, TRIM, "neck"),
    ]), 12, side_color=lambda row, side: UNIFORM_ACCENT if row < 5 and side in (0, 6) else None)
    mesh.loft(vertical(0, 0.001, [
        (1.546, 0.067, 0.064, FACE, "neck"),
        (1.633, 0.065, 0.062, FACE, ("neck", "head", 0.80)),
        (1.713, 0.064, 0.065, FACE, "head"),
    ]))
    mesh.loft([
        Ring((0, 1.644, 0.024), 0.049, 0.050, FACE, "head"),
        Ring((0, 1.680, 0.020), 0.067, 0.071, FACE, "head"),
        Ring((0, 1.754, 0.018), 0.084, 0.089, FACE, "head"),
        Ring((0, 1.826, 0.014), 0.090, 0.091, FACE, "head"),
        Ring((0, 1.868, 0.007), 0.080, 0.086, FACE, "head"),
        Ring((0, 1.896, 0.003), 0.061, 0.066, FACE, "head"),
    ], 12)
    mesh.loft(vertical(0, -0.005, [
        (1.806, 0.087, 0.078, HAIR, "head"),
        (1.845, 0.092, 0.092, HAIR, "head"),
        (1.892, 0.081, 0.086, HAIR, "head"),
        (1.928, 0.057, 0.068, HAIR, "head"),
        (1.947, 0.027, 0.032, HAIR, "head"),
    ]), 12)
    mesh.loft(vertical(0, 0.101, [
        (1.721, 0.013, 0.012, FACE, "head"),
        (1.736, 0.025, 0.028, FACE, "head"),
        (1.774, 0.014, 0.014, FACE, "head"),
    ]), 6)
    for sign in (-1, 1):
        mesh.loft(vertical(sign * 0.086, 0.007, [
            (1.725, 0.008, 0.010, FACE, "head"),
            (1.755, 0.018, 0.023, FACE, "head"),
            (1.785, 0.011, 0.016, FACE, "head"),
        ]), 6)
        mesh.loft(vertical(sign * 0.033, 0.102, [
            (1.779, 0.006, 0.003, EYES, "head"),
            (1.784, 0.010, 0.005, EYES, "head"),
            (1.789, 0.007, 0.003, EYES, "head"),
        ]), 6)


def add_limb(mesh: PlayerMesh, joints: dict[str, Point], side: str) -> None:
    hip, knee, ankle = (f"{side}_{name}" for name in ("hip", "knee", "ankle"))
    shoulder, elbow, wrist = (f"{side}_{name}" for name in ("shoulder", "elbow", "wrist"))

    mesh.loft([
        Ring(joints[ankle], 0.042, 0.052, SKIN, ankle),
        Ring(blend(joints[ankle], joints[knee], 0.2), 0.050, 0.052, SKIN, (ankle, knee, 0.31)),
        Ring(blend(joints[ankle], joints[knee], 0.52), 0.072, 0.073, SKIN, knee),
        Ring(blend(joints[ankle], joints[knee], 0.88), 0.069, 0.062, SKIN, (knee, hip, 0.77)),
        Ring(joints[knee], 0.067, 0.061, SKIN, (knee, hip, 0.49)),
        Ring(blend(joints[knee], joints[hip], 0.17), 0.077, 0.073, SKIN, hip),
        Ring(blend(joints[knee], joints[hip], 0.56), 0.095, 0.099, SKIN, hip),
    ], follow_axis=True, material=SATIN_STEEL)
    mesh.loft([
        Ring(blend(joints[knee], joints[hip], 0.12), 0.086, 0.089, UNIFORM_ACCENT, (hip, knee, 0.92)),
        Ring(blend(joints[knee], joints[hip], 0.15), 0.089, 0.093, JERSEY, hip),
        Ring(blend(joints[knee], joints[hip], 0.20), 0.094, 0.096, SHORTS, hip),
        Ring(blend(joints[knee], joints[hip], 0.61), 0.109, 0.115, SHORTS, hip),
        Ring(blend(joints[knee], joints[hip], 1.07), 0.108, 0.117, SHORTS, (hip, "pelvis", 0.55)),
    ], 10, follow_axis=True)
    x = joints[ankle][0]
    mesh.loft(vertical(x, 0.002, [
        (0.097, 0.046, 0.054, SOCK, ankle),
        (0.159, 0.052, 0.056, SOCK, (ankle, knee, 0.55)),
        (0.174, 0.053, 0.057, UNIFORM_ACCENT, (ankle, knee, 0.42)),
        (0.186, 0.054, 0.058, SOCK, (ankle, knee, 0.35)),
        (0.193, 0.053, 0.057, SOCK, (ankle, knee, 0.29)),
    ]))
    mesh.loft([
        Ring((x, 0.000, 0.070), 0.061, 0.130, SOLE, ankle),
        Ring((x, 0.015, 0.071), 0.067, 0.139, SOLE, ankle),
        Ring((x, 0.034, 0.073), 0.065, 0.139, TRIM, ankle),
        Ring((x, 0.050, 0.067), 0.061, 0.132, SHOE, ankle),
        Ring((x, 0.078, 0.046), 0.055, 0.111, SHOE, ankle),
        Ring((x, 0.118, 0.008), 0.046, 0.064, SHOE, ankle),
        Ring((x, 0.130, 0.000), 0.044, 0.049, UNIFORM_ACCENT, ankle),
    ], 10)

    mesh.loft([
        Ring(blend(joints[elbow], joints[wrist], 1.32), 0.028, 0.036, SKIN, wrist),
        Ring(blend(joints[elbow], joints[wrist], 1.14), 0.040, 0.033, SKIN, wrist),
        Ring(joints[wrist], 0.031, 0.035, SKIN, (wrist, elbow, 0.63)),
        Ring(blend(joints[wrist], joints[elbow], 0.53), 0.045, 0.047, SKIN, elbow),
        Ring(blend(joints[wrist], joints[elbow], 0.85), 0.047, 0.046, SKIN, (elbow, shoulder, 0.75)),
        Ring(joints[elbow], 0.049, 0.050, SKIN, (elbow, shoulder, 0.56)),
        Ring(blend(joints[elbow], joints[shoulder], 0.29), 0.057, 0.061, SKIN, shoulder),
        Ring(blend(joints[elbow], joints[shoulder], 0.71), 0.071, 0.070, SKIN, shoulder),
        Ring(blend(joints[elbow], joints[shoulder], 1.11), 0.072, 0.065, SKIN, (shoulder, "neck", 0.66)),
        Ring(blend(joints[elbow], joints[shoulder], 1.23), 0.049, 0.048, SKIN, (shoulder, "neck", 0.58)),
    ], 9, follow_axis=True, material=SATIN_STEEL)


def make_glb(mesh: PlayerMesh, joints: dict[str, Point], parents: dict[str, str]) -> bytes:
    binary = bytearray()
    views = []
    accessors = []

    def accessor(values, component, code, shape, *, target=None, normalized=False, bounds=None):
        values = list(values)
        element = struct.Struct("<" + code * len(values[0]))
        offset = len(binary)
        for item in values:
            binary.extend(element.pack(*item))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
        if target is not None:
            view["target"] = target
        views.append(view)
        binary.extend(bytes(-len(binary) % 4))
        field = {"bufferView": len(views) - 1, "componentType": component, "count": len(values), "type": shape}
        if normalized:
            field["normalized"] = True
        if bounds is not None:
            field["min"], field["max"] = bounds
        accessors.append(field)
        return len(accessors) - 1

    bounds = [[bound(coordinate) for coordinate in zip(*mesh.positions)] for bound in (min, max)]
    attributes = {
        "POSITION": accessor(mesh.positions, 5126, "f", "VEC3", target=34962, bounds=bounds),
        "NORMAL": accessor(mesh.normals(), 5126, "f", "VEC3", target=34962),
        "COLOR_0": accessor(mesh.colors, 5121, "B", "VEC4", target=34962, normalized=True),
        "JOINTS_0": accessor(mesh.joint_indices, 5121, "B", "VEC4", target=34962),
        "WEIGHTS_0": accessor(mesh.joint_weights, 5121, "B", "VEC4", target=34962, normalized=True),
    }
    primitives = [
        {
            "attributes": attributes,
            "indices": accessor(((index,) for index in indices), 5123, "H", "SCALAR", target=34963),
            "material": material,
        }
        for material, indices in enumerate(mesh.indices)
    ]
    inverse_bind = accessor(
        ((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -x, -y, -z, 1) for x, y, z in joints.values()),
        5126, "f", "MAT4",
    )

    children = defaultdict(list)
    for child, parent in parents.items():
        children[parent].append(mesh.joints[child])
    nodes = []
    for name, world_position in joints.items():
        local = sub(world_position, joints[parents[name]]) if name in parents else world_position
        node = {"name": name, "translation": local}
        if children[name]:
            node["children"] = children[name]
        nodes.append(node)
    mesh_node = len(nodes)
    nodes.append({"name": "Replay player", "mesh": 0, "skin": 0})

    model = {
        "asset": {"version": "2.0", "generator": "NBA Scene Replay player generator"},
        "scene": 0,
        "scenes": [{"nodes": [mesh.joints["pelvis"], mesh_node]}],
        "nodes": nodes,
        "meshes": [{"primitives": primitives}],
        "skins": [{"joints": list(mesh.joints.values()), "skeleton": mesh.joints["pelvis"], "inverseBindMatrices": inverse_bind}],
        "materials": [
            {"name": "Satin steel", "pbrMetallicRoughness": {"metallicFactor": 0.48, "roughnessFactor": 0.31}},
            {"name": "Matte surfaces", "pbrMetallicRoughness": {"metallicFactor": 0, "roughnessFactor": 0.88}},
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    encoded = json.dumps(model, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    length = 12 + 8 + len(encoded) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<I4s", len(encoded), b"JSON") + encoded
        + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets/player.glb")
    args = parser.parse_args()
    joints, parents = rig()
    mesh = PlayerMesh(joints)
    add_torso(mesh)
    for side in ("left", "right"):
        add_limb(mesh, joints, side)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(make_glb(mesh, joints, parents))


if __name__ == "__main__":
    main()
