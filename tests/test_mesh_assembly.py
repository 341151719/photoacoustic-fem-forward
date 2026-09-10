from __future__ import annotations

import numpy as np
import pytest

from pa_fem.assemble import build_system
from pa_fem.geometry import TAGS


def test_mesh_labels_and_conforming_interfaces(smoke_mesh):
    audit = smoke_mesh.audit
    assert audit["triangle_counts"][str(TAGS.water)] > 0
    assert audit["triangle_counts"][str(TAGS.tissue)] > 0
    assert audit["triangle_counts"][str(TAGS.absorber)] > 0
    assert audit["line_counts"][str(TAGS.sensor)] > 0
    assert audit["sensor_length_relative_error"] < 5e-3
    assert audit["duplicate_coordinate_nodes"] == 0
    assert audit["material_interface_conforming"]


def test_conservative_assembly_and_positive_lumped_source(smoke_case, smoke_mesh):
    system = build_system(smoke_case, smoke_mesh)
    assert system.n_dofs == len(smoke_mesh.points_m)
    assert system.receiver_length_m == pytest.approx(3.0e-3, rel=1e-12)
    assert system.matrix_metrics["mass_symmetry_relative"] < 1e-12
    assert system.matrix_metrics["stiffness_symmetry_relative"] < 1e-12
    assert np.min(system.p0) >= 0.0
    assert np.max(system.p0) > 7.0e3
    # The lumped L2 source conserves the assembled integral.
    weights = system.geometric_mass @ np.ones(system.n_dofs)
    assert np.sum(weights * system.p0) == pytest.approx(np.sum(system.p0_rhs), rel=1e-13)
