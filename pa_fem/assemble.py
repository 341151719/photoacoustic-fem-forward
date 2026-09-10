"""scikit-fem assembly of the heterogeneous pressure-acoustic system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import BilinearForm, Basis, ElementTriP1, FacetBasis, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad

from .config import CaseConfig
from .geometry import TAGS
from .materials import Material, material_map
from .mesh_check import MeshData
from .source import initial_pressure_from_material


@BilinearForm
def mass_form(u, v, w):
    return w.inv_bulk * u * v


@BilinearForm
def stiffness_form(u, v, w):
    return w.inv_rho * dot(grad(u), grad(v))


@BilinearForm
def geometric_mass_form(u, v, w):
    """Unweighted L2 mass used only for projecting the initial pressure."""

    return u * v


@BilinearForm
def abc_form(u, v, w):
    return w.inv_rho_c * u * v


@LinearForm
def receiver_form(v, w):
    return v


@dataclass
class AcousticSystem:
    cfg: CaseConfig
    mesh_data: MeshData
    mesh: MeshTri
    basis: Basis
    materials: dict[int, Material]
    triangle_elements_by_tag: dict[int, np.ndarray]
    boundary_facets_by_tag: dict[int, np.ndarray]
    M: csr_matrix
    geometric_mass: csr_matrix
    C: csr_matrix
    K: csr_matrix
    receiver_vector: np.ndarray
    receiver_length_m: float
    p0: np.ndarray
    p0_raw_projection: np.ndarray
    p0_consistent_projection_negative_nodes: int
    p0_rhs: np.ndarray
    matrix_metrics: dict[str, float]

    @property
    def n_dofs(self) -> int:
        return int(self.basis.N)

    def material_for_tag(self, tag: int) -> Material:
        return self.materials[int(tag)]

    def receiver_average(self, pressure: np.ndarray) -> float:
        return float(np.dot(self.receiver_vector, pressure) / self.receiver_length_m)


def _map_facets(mesh: MeshTri, mesh_data: MeshData) -> dict[int, np.ndarray]:
    facet_map = {tuple(sorted(map(int, mesh.facets[:, i]))): i
                 for i in range(mesh.facets.shape[1])}
    out: dict[int, list[int]] = {}
    for seg, tag in zip(mesh_data.lines, mesh_data.line_tags):
        key = tuple(sorted(map(int, seg)))
        if key not in facet_map:
            raise ValueError(f"Physical boundary segment {key} is not a MeshTri facet")
        out.setdefault(int(tag), []).append(facet_map[key])
    return {tag: np.asarray(sorted(set(facets)), dtype=int) for tag, facets in out.items()}


def _matrix_metrics(M: csr_matrix, C: csr_matrix, K: csr_matrix) -> dict[str, float]:
    def rel_sym(A: csr_matrix) -> float:
        d = A - A.T
        denom = max(float(np.linalg.norm(A.data)), 1e-300)
        return float(np.linalg.norm(d.data) / denom)
    return {
        "mass_symmetry_relative": rel_sym(M),
        "abc_symmetry_relative": rel_sym(C),
        "stiffness_symmetry_relative": rel_sym(K),
        "mass_diagonal_min": float(np.min(M.diagonal())),
        "stiffness_diagonal_min": float(np.min(K.diagonal())),
        "abc_diagonal_min": float(np.min(C.diagonal())),
        "abc_nonzero_diagonal_min": float(np.min(C.diagonal()[C.diagonal() > 0])) if np.any(C.diagonal() > 0) else 0.0,
        "mass_nnz": int(M.nnz),
        "stiffness_nnz": int(K.nnz),
        "abc_nnz": int(C.nnz),
    }


def build_system(cfg: CaseConfig, mesh_data: MeshData) -> AcousticSystem:
    """Create M, K, first-order ABC C, receiver vector, and projected p0."""

    # Gmsh generated P1 nodes and triangles use the same metres coordinates.
    mesh = MeshTri(mesh_data.points_m.T, mesh_data.triangles.T)
    basis = Basis(mesh, ElementTriP1(), intorder=4)
    materials = material_map(cfg)
    by_tag = {
        tag: np.flatnonzero(mesh_data.triangle_tags == tag).astype(int)
        for tag in materials
    }
    if any(len(indices) == 0 for indices in by_tag.values()):
        raise ValueError("one or more acoustic material subdomains are empty")
    M = csr_matrix((basis.N, basis.N), dtype=float)
    geometric_mass = csr_matrix((basis.N, basis.N), dtype=float)
    K = csr_matrix((basis.N, basis.N), dtype=float)
    p0_rhs = np.zeros(basis.N, dtype=float)
    for tag, mat in materials.items():
        subbasis = basis.with_elements(by_tag[tag])
        M = M + asm(mass_form, subbasis, inv_bulk=mat.inv_bulk_pa_inv).tocsr()
        geometric_mass = geometric_mass + asm(geometric_mass_form, subbasis).tocsr()
        K = K + asm(stiffness_form, subbasis, inv_rho=mat.inv_rho_m3_kg).tocsr()

        # Assemble p0 from the same quadrature/subdomain.  This is an L2
        # projection, not nearest-node assignment, and preserves the stated
        # Pa/J/m²/m^-1 unit chain.
        def make_p0_load(_mat: Material):
            @LinearForm
            def p0_load(v, w):
                return initial_pressure_from_material(w.x[0], w.x[1], _mat, cfg) * v
            return p0_load
        p0_rhs += asm(make_p0_load(mat), subbasis)
    # p0 is an L2 projection in the geometric inner product.  The acoustic
    # mass M contains 1/K and must not be used here (doing so would scale the
    # pressure by a bulk modulus and violate the Pa unit).
    p0_raw = np.asarray(spsolve(geometric_mass.tocsc(), p0_rhs), dtype=float)
    # A discontinuous absorber/tissue optical coefficient can create a
    # Gibbs-like negative overshoot in a consistent continuous P1 projection.
    # Use the positive, conservative mass-lumped L2 projection for propagation:
    # p0_i = (∫ p0 N_i)/(∫ N_i).  Since P1 basis functions are non-negative,
    # this preserves non-negativity and sum_i ∫p0 N_i exactly.  Keep the
    # consistent projection only as a diagnostic field for the metadata.
    lumped = np.asarray(geometric_mass @ np.ones(basis.N), dtype=float)
    if np.any(lumped <= 0):
        raise ValueError("geometric mass lumping has a non-positive diagonal")
    p0 = p0_rhs / lumped
    negative = p0_raw < 0.0

    facets = _map_facets(mesh, mesh_data)
    if TAGS.sensor not in facets or TAGS.outer_absorbing not in facets:
        raise ValueError("sensor and outer_absorbing facets are required")
    abc_facets = np.unique(np.concatenate([facets[TAGS.sensor], facets[TAGS.outer_absorbing]])).astype(int)
    fb_abc = FacetBasis(mesh, basis.elem, facets=abc_facets, intorder=4)
    # The exterior water buffer is deliberately used on every outer facet.
    # This is valid because the generated domain places all tissue interfaces
    # strictly inside the water rectangle.
    water = materials[TAGS.water]
    C = asm(abc_form, fb_abc, inv_rho_c=1.0 / (water.rho_kg_m3 * water.c_m_s)).tocsr()
    fb_sensor = FacetBasis(mesh, basis.elem, facets=facets[TAGS.sensor], intorder=4)
    receiver = np.asarray(asm(receiver_form, fb_sensor), dtype=float)
    receiver_length = float(np.sum(receiver))
    if receiver_length <= 0:
        raise ValueError("receiver integral has non-positive length")
    expected_length = cfg.geometry.sensor_x_max_m - cfg.geometry.sensor_x_min_m
    if abs(receiver_length - expected_length) / expected_length > 5e-3:
        raise ValueError("FacetBasis receiver length does not match design length")
    metrics = _matrix_metrics(M, C, K)
    if metrics["mass_diagonal_min"] <= 0 or metrics["stiffness_diagonal_min"] <= 0:
        raise ValueError("assembled M/K has non-positive diagonal")
    return AcousticSystem(cfg, mesh_data, mesh, basis, materials, by_tag, facets,
                          M, geometric_mass, C, K, receiver, receiver_length, p0,
                          p0_raw, int(np.sum(negative)), p0_rhs, metrics)
