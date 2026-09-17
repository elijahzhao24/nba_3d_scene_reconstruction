"""JSON artifact readers and writers for independently rerunnable stages."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

from ..tracking.schemas import ObservationSource, PlayerObservation, VideoManifest
from .schemas import Pose2D, Pose3D, PoseJoint2D, PoseJoint3D, PreparedPose2D

T = TypeVar("T")


def write_jsonl(path: str | Path, rows: tuple[object, ...]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(asdict(row), allow_nan=False) + "\n")
    return destination


def _read(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_manifest(path: str | Path) -> VideoManifest:
    return VideoManifest(**json.loads(Path(path).read_text(encoding="utf-8")))


def read_observations(path: str | Path) -> tuple[PlayerObservation, ...]:
    rows = []
    for row in _read(path):
        row["source"] = ObservationSource(row["source"])
        for field in ("bbox_xyxy", "centroid_xy", "footpoint_xy", "quality_flags"):
            if row.get(field) is not None:
                row[field] = tuple(row[field])
        rows.append(PlayerObservation(**row))
    return tuple(rows)


def read_poses_2d(
    path: str | Path, *, prepared: bool = False
) -> tuple[Pose2D | PreparedPose2D, ...]:
    cls = PreparedPose2D if prepared else Pose2D
    rows = []
    for row in _read(path):
        row["joints"] = tuple(
            PoseJoint2D(**{**joint, "image_xy": tuple(joint["image_xy"])})
            for joint in row["joints"]
        )
        row["quality_flags"] = tuple(row.get("quality_flags", ()))
        rows.append(cls(**row))
    return tuple(rows)


def read_poses_3d(path: str | Path) -> tuple[Pose3D, ...]:
    rows = []
    for row in _read(path):
        row["joints"] = tuple(
            PoseJoint3D(**{**joint, "xyz": tuple(joint["xyz"])})
            for joint in row["joints"]
        )
        row["quality_flags"] = tuple(row.get("quality_flags", ()))
        rows.append(Pose3D(**row))
    return tuple(rows)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_json(path: str | Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination
