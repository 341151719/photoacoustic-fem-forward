"""Gmsh/OpenCASCADE geometry generation for the reference 2-D tissue case."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

from ._gmsh import import_gmsh
from .config import CaseConfig


@dataclass(frozen=True)
class PhysicalTags:
    """Global-unique physical IDs, intentionally distinct across dimensions."""

    water: int = 101
    tissue: int = 102
    absorber: int = 103
    sensor: int = 201
    outer_absorbing: int = 202

    @property
    def material_names(self) -> dict[int, str]:
        return {self.water: "water", self.tissue: "tissue", self.absorber: "absorber"}

    @property
    def boundary_names(self) -> dict[int, str]:
        return {self.sensor: "sensor", self.outer_absorbing: "outer_absorbing"}


TAGS = PhysicalTags()


def _add_physical(gmsh, dim: int, entities: Iterable[int], tag: int, name: str) -> None:
    entities = sorted({int(e) for e in entities})
    if not entities:
        raise RuntimeError(f"cannot create empty Physical Group {name}")
    gmsh.model.addPhysicalGroup(dim, entities, tag)
    gmsh.model.setPhysicalName(dim, tag, name)


def _is_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _inside_rect(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float, tol: float) -> bool:
    return xmin - tol <= x <= xmax + tol and ymin - tol <= y <= ymax + tol


def generate_mesh(cfg: CaseConfig, msh_path: str | Path) -> Path:
    """Generate a conforming water/tissue/absorber mesh with Physical Groups.

    The outer rectangle is explicitly split at the finite sensor endpoints,
    while the tissue rectangle and circular absorber are fragmented with the
    water surface through OCC.  Material identification is rebuilt from
    centroids *after* the Boolean operation; no pre-fragment entity tags are
    reused.
    """

    gmsh = import_gmsh()
    g, mc = cfg.geometry, cfg.mesh
    msh_path = Path(msh_path)
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", mc.msh_file_version)
        gmsh.option.setNumber("Mesh.Algorithm", mc.algorithm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(mc.source_size_m, mc.interface_size_m))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mc.element_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.model.add(cfg.case_name)
        occ = gmsh.model.occ

        # Explicitly split the top edge so sensor is a true line entity.
        p_bl = occ.addPoint(g.domain_x_min_m, g.domain_y_min_m, 0.0, mc.element_size_m)
        p_s0 = occ.addPoint(g.sensor_x_min_m, g.domain_y_min_m, 0.0, mc.interface_size_m)
        p_s1 = occ.addPoint(g.sensor_x_max_m, g.domain_y_min_m, 0.0, mc.interface_size_m)
        p_br = occ.addPoint(g.domain_x_max_m, g.domain_y_min_m, 0.0, mc.element_size_m)
        p_tr = occ.addPoint(g.domain_x_max_m, g.domain_y_max_m, 0.0, mc.element_size_m)
        p_tl = occ.addPoint(g.domain_x_min_m, g.domain_y_max_m, 0.0, mc.element_size_m)
        l_bottom_left = occ.addLine(p_bl, p_s0)
        l_sensor = occ.addLine(p_s0, p_s1)
        l_bottom_right = occ.addLine(p_s1, p_br)
        l_right = occ.addLine(p_br, p_tr)
        l_top = occ.addLine(p_tr, p_tl)
        l_left = occ.addLine(p_tl, p_bl)
        outer_loop = occ.addCurveLoop(
            [l_bottom_left, l_sensor, l_bottom_right, l_right, l_top, l_left]
        )
        outer_surface = occ.addPlaneSurface([outer_loop])

        tissue_surface = occ.addRectangle(
            g.tissue_x_min_m,
            g.tissue_y_min_m,
            0.0,
            g.tissue_x_max_m - g.tissue_x_min_m,
            g.tissue_y_max_m - g.tissue_y_min_m,
        )
        absorber_surface = occ.addDisk(
            g.absorber_x_m, g.absorber_y_m, 0.0, g.absorber_radius_m, g.absorber_radius_m
        )

        # OCC fragment produces a single conforming arrangement.  The returned
        # map is deliberately not assumed to preserve source tags.
        object_dimtags = [(2, outer_surface)]
        tool_dimtags = [(2, tissue_surface), (2, absorber_surface)]
        _fragment_out, fragment_map = occ.fragment(object_dimtags, tool_dimtags)
        occ.synchronize()

        # Use point sizes and a distance field around all post-fragment
        # interface curves.  This keeps the circular optical source resolved
        # without forcing the far water buffer to use the smallest h.
        point_entities = gmsh.model.getEntities(0)
        for dim, tag in point_entities:
            x, y, _ = gmsh.model.getValue(dim, tag, [])
            in_tissue_box = _inside_rect(
                x,
                y,
                g.tissue_x_min_m,
                g.tissue_x_max_m,
                g.tissue_y_min_m,
                g.tissue_y_max_m,
                2.0e-12,
            )
            if in_tissue_box or math.hypot(x - g.absorber_x_m, y - g.absorber_y_m) <= 1.2 * g.absorber_radius_m:
                gmsh.model.mesh.setSize([(dim, tag)], mc.interface_size_m)

        # Curves close to the material arrangement receive a local distance
        # field.  The remaining outer boundary is allowed to grow smoothly to
        # the global element size.
        curves = gmsh.model.getEntities(1)
        interface_curves: list[int] = []
        for dim, tag in curves:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            on_tissue_box = (
                _is_close(cx, g.tissue_x_min_m, 3e-8)
                or _is_close(cx, g.tissue_x_max_m, 3e-8)
                or _is_close(cy, g.tissue_y_min_m, 3e-8)
                or _is_close(cy, g.tissue_y_max_m, 3e-8)
            )
            near_absorber = abs(math.hypot(cx - g.absorber_x_m, cy - g.absorber_y_m) - g.absorber_radius_m) < 3.0 * mc.interface_size_m
            if on_tissue_box or near_absorber:
                interface_curves.append(tag)
        if interface_curves:
            field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field, "CurvesList", interface_curves)
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", field)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", min(mc.source_size_m, mc.interface_size_m))
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", mc.element_size_m)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * mc.interface_size_m)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", 3.0 * mc.element_size_m)
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

        # Re-identify entities after fragment using the returned OCC map.  A
        # centroid test alone is wrong here: the water surface's centroid can
        # lie inside the tissue rectangle even though it surrounds tissue.
        def mapped_surface_tags(index: int) -> set[int]:
            return {int(tag) for dim, tag in fragment_map[index] if dim == 2}

        outer_parts = mapped_surface_tags(0)
        tissue_parts = mapped_surface_tags(1)
        absorber_parts = mapped_surface_tags(2)
        material_entities: dict[int, list[int]] = {
            TAGS.absorber: sorted(absorber_parts),
            TAGS.tissue: sorted(tissue_parts - absorber_parts),
            TAGS.water: sorted(outer_parts - tissue_parts),
        }
        # Ensure all post-fragment faces are classified, even if a future OCC
        # version returns an additional split face in one of the maps.
        all_faces = {int(tag) for dim, tag in gmsh.model.getEntities(2)}
        classified = set().union(*(set(v) for v in material_entities.values()))
        for face in sorted(all_faces - classified):
            x, y, _ = occ.getCenterOfMass(2, face)
            if _inside_rect(x, y, g.tissue_x_min_m, g.tissue_x_max_m,
                            g.tissue_y_min_m, g.tissue_y_max_m, 1e-10):
                material_entities[TAGS.tissue].append(face)
            else:
                material_entities[TAGS.water].append(face)
        for tag, entities in material_entities.items():
            _add_physical(gmsh, 2, entities, tag, TAGS.material_names[tag])

        boundary_entities: dict[int, list[int]] = {TAGS.sensor: [], TAGS.outer_absorbing: []}
        edge_tol = max(1e-9, 0.02 * mc.element_size_m)
        for dim, tag in gmsh.model.getEntities(1):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            on_outer = (
                _is_close(xmin, g.domain_x_min_m, edge_tol)
                or _is_close(xmax, g.domain_x_max_m, edge_tol)
                or _is_close(ymin, g.domain_y_min_m, edge_tol)
                or _is_close(ymax, g.domain_y_max_m, edge_tol)
            )
            if not on_outer:
                continue
            on_sensor = (
                _is_close(ymin, g.domain_y_min_m, edge_tol)
                and _is_close(ymax, g.domain_y_min_m, edge_tol)
                and xmin >= g.sensor_x_min_m - edge_tol
                and xmax <= g.sensor_x_max_m + edge_tol
            )
            boundary_entities[TAGS.sensor if on_sensor else TAGS.outer_absorbing].append(tag)
        _add_physical(gmsh, 1, boundary_entities[TAGS.sensor], TAGS.sensor, "sensor")
        _add_physical(gmsh, 1, boundary_entities[TAGS.outer_absorbing], TAGS.outer_absorbing, "outer_absorbing")

        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()
    return msh_path
