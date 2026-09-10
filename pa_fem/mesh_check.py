"""meshio import and mesh/Physical Group audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import meshio
import numpy as np

from .config import CaseConfig
from .geometry import TAGS, PhysicalTags


@dataclass
class MeshData:
    path: Path
    points_m: np.ndarray
    triangles: np.ndarray
    triangle_tags: np.ndarray
    lines: np.ndarray
    line_tags: np.ndarray
    field_data: dict[str, list[int]]
    physical_names: dict[int, str]
    mesh_hash_sha256: str
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_points": int(self.points_m.shape[0]),
            "n_triangles": int(self.triangles.shape[0]),
            "n_lines": int(self.lines.shape[0]),
            "mesh_hash_sha256": self.mesh_hash_sha256,
            "audit": self.audit,
        }


def _cell_data(mesh: meshio.Mesh, cell_type: str) -> np.ndarray:
    try:
        data = mesh.cell_data_dict["gmsh:physical"][cell_type]
    except KeyError as exc:
        raise ValueError(f"mesh has no gmsh:physical data for {cell_type}") from exc
    return np.asarray(data, dtype=int)


def _triangle_quality(points: np.ndarray, triangles: np.ndarray) -> dict[str, float]:
    p = points[triangles]
    e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    signed = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
                    (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    area = np.abs(signed)
    # Angle opposite each edge from the cosine rule; clamp protects roundoff.
    angles = np.empty((len(p), 3), dtype=float)
    for j, (a, b, c) in enumerate(((e0, e1, e2), (e1, e2, e0), (e2, e0, e1))):
        angles[:, j] = np.arccos(np.clip((a * a + b * b - c * c) / (2.0 * np.maximum(a * b, 1e-30)), -1.0, 1.0))
    aspect = np.maximum.reduce([e0, e1, e2]) / np.maximum(np.minimum.reduce([e0, e1, e2]), 1e-30)
    return {
        "signed_area_min_m2": float(np.min(signed)),
        "area_min_m2": float(np.min(area)),
        "area_max_m2": float(np.max(area)),
        "min_angle_deg": float(np.degrees(np.min(angles))),
        "max_aspect_ratio": float(np.max(aspect)),
    }


def _line_lengths(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    if len(lines) == 0:
        return np.empty(0)
    return np.linalg.norm(points[lines[:, 1]] - points[lines[:, 0]], axis=1)


def _interface_audit(triangles: np.ndarray, tri_tags: np.ndarray) -> dict[str, Any]:
    """Check that every material-changing edge is a shared conforming edge."""

    edge_cells: dict[tuple[int, int], list[int]] = {}
    for cell, tri in enumerate(np.asarray(triangles, dtype=int)):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_cells.setdefault(tuple(sorted((int(a), int(b)))), []).append(cell)
    interface_edges = 0
    bad_edges: list[tuple[int, int, int]] = []
    for edge, cells in edge_cells.items():
        if len(cells) >= 1 and len({int(tri_tags[c]) for c in cells}) > 1:
            interface_edges += 1
            if len(cells) != 2:
                bad_edges.append((edge[0], edge[1], len(cells)))
    return {
        "material_interface_edges": interface_edges,
        "material_interface_nonconforming_or_nonmanifold": len(bad_edges),
        "material_interface_conforming": len(bad_edges) == 0,
    }


def audit_mesh(path: str | Path, cfg: CaseConfig | None = None) -> MeshData:
    """Read a Gmsh file and fail fast on missing or ambiguous labels.

    Gmsh is configured in metres, so no hidden mm→m conversion is performed
    here.  A small coordinate sanity check catches accidental millimetre
    files before they reach the PDE assembly.
    """

    path = Path(path)
    mesh = meshio.read(path)
    if "triangle" not in mesh.cells_dict or "line" not in mesh.cells_dict:
        raise ValueError("mesh must contain both triangle and line cells")
    points = np.asarray(mesh.points[:, :2], dtype=float)
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=int)
    lines = np.asarray(mesh.cells_dict["line"], dtype=int)
    tri_tags = _cell_data(mesh, "triangle")
    line_tags = _cell_data(mesh, "line")
    if len(triangles) != len(tri_tags) or len(lines) != len(line_tags):
        raise ValueError("cell connectivity and gmsh:physical lengths differ")
    if np.min(triangles) < 0 or np.max(triangles) >= len(points):
        raise ValueError("triangle contains an out-of-range node index")
    if np.min(lines) < 0 or np.max(lines) >= len(points):
        raise ValueError("line contains an out-of-range node index")

    expected_material = {TAGS.water, TAGS.tissue, TAGS.absorber}
    expected_boundary = {TAGS.sensor, TAGS.outer_absorbing}
    actual_material = set(map(int, np.unique(tri_tags)))
    actual_boundary = set(map(int, np.unique(line_tags)))
    if actual_material != expected_material:
        raise ValueError(f"triangle Physical IDs {actual_material} != {expected_material}")
    if not actual_boundary.issubset(expected_boundary) or not expected_boundary.issubset(actual_boundary):
        raise ValueError(f"line Physical IDs {actual_boundary} do not contain exactly {expected_boundary}")
    material_counts = {str(k): int(np.sum(tri_tags == k)) for k in sorted(expected_material)}
    boundary_counts = {str(k): int(np.sum(line_tags == k)) for k in sorted(expected_boundary)}
    if any(v == 0 for v in material_counts.values()) or any(v == 0 for v in boundary_counts.values()):
        raise ValueError("all required Physical Groups must be non-empty")

    quality = _triangle_quality(points, triangles)
    if quality["signed_area_min_m2"] <= 0:
        raise ValueError("triangle orientation/area audit failed: a signed area is non-positive")
    if not np.all(np.isfinite(points)):
        raise ValueError("mesh coordinates contain non-finite values")
    if np.ptp(points[:, 0]) > 1.0 or np.ptp(points[:, 1]) > 1.0:
        raise ValueError("mesh appears not to use metres (extent exceeds 1 m)")

    rounded = np.round(points / 1e-12).astype(np.int64)
    _, duplicate_counts = np.unique(rounded, axis=0, return_counts=True)
    duplicate_coordinate_nodes = int(np.sum(np.maximum(duplicate_counts - 1, 0)))
    if duplicate_coordinate_nodes:
        raise ValueError(f"mesh has {duplicate_coordinate_nodes} duplicate coordinate nodes")
    interface = _interface_audit(triangles, tri_tags)
    if not interface["material_interface_conforming"]:
        raise ValueError("material interface is not topologically conforming")

    line_lengths = _line_lengths(points, lines)
    sensor_length = float(np.sum(line_lengths[line_tags == TAGS.sensor]))
    if cfg is not None:
        expected_length = cfg.geometry.sensor_x_max_m - cfg.geometry.sensor_x_min_m
        rel = abs(sensor_length - expected_length) / expected_length
        if rel > 5e-3:
            raise ValueError(f"sensor length {sensor_length:.9e} m differs from design by {rel:.3%}")
    else:
        rel = None

    # Every tagged boundary endpoint must belong to the area mesh.  This also
    # catches accidental duplicate/disconnected line entities.
    tri_nodes = set(np.unique(triangles))
    if not set(np.unique(lines)).issubset(tri_nodes):
        raise ValueError("a tagged boundary line uses nodes not present in triangles")

    field_data = {str(k): [int(v[0]), int(v[1])] for k, v in mesh.field_data.items()}
    required_names = {"water": TAGS.water, "tissue": TAGS.tissue, "absorber": TAGS.absorber,
                      "sensor": TAGS.sensor, "outer_absorbing": TAGS.outer_absorbing}
    for name, tag in required_names.items():
        if name not in field_data or field_data[name][0] != tag:
            raise ValueError(f"field_data missing {name}={tag}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {
        "physical_name_to_id": required_names,
        "triangle_counts": material_counts,
        "line_counts": boundary_counts,
        "sensor_length_m": sensor_length,
        "sensor_length_relative_error": rel,
        "triangle_quality": quality,
        "all_boundary_nodes_shared": True,
        "duplicate_coordinate_nodes": duplicate_coordinate_nodes,
        **interface,
        "units": "m",
    }
    return MeshData(path, points, triangles, tri_tags, lines, line_tags, field_data,
                    {v[0]: k for k, v in field_data.items()}, digest, audit)
