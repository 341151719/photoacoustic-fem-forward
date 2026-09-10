from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pa_fem.config import CaseConfig
from pa_fem.geometry import generate_mesh
from pa_fem.mesh_check import audit_mesh


@pytest.fixture(scope="session")
def smoke_case() -> CaseConfig:
    base = CaseConfig()
    return replace(
        base,
        case_name="pytest_smoke",
        mesh=replace(base.mesh, element_size_m=0.50e-3,
                     interface_size_m=0.25e-3, source_size_m=0.20e-3),
        time=replace(base.time, dt_s=50e-9, t_end_s=5.0e-6, save_every=10),
    )


@pytest.fixture(scope="session")
def smoke_mesh(tmp_path_factory: pytest.TempPathFactory, smoke_case: CaseConfig):
    out = tmp_path_factory.mktemp("pa_mesh") / "domain.msh"
    return audit_mesh(generate_mesh(smoke_case, out), smoke_case)

