"""Single-case orchestration for mesh, assembly, transient solve and output."""

from __future__ import annotations

import shutil
from pathlib import Path
import sys
from typing import Any

from .assemble import build_system
from .config import CaseConfig
from .geometry import generate_mesh
from .mesh_check import audit_mesh
from .output import (write_json, write_metadata, write_plots, write_snapshots,
                     write_waveform)
from .receiver import apply_receiver_response
from .source import source_summary
from .time_integrator import run_newmark
from .validation import validate_case


def solve_case(cfg: CaseConfig, outdir: str | Path, *, make_plots: bool = True,
               validate_linearity: bool = True,
               mesh_path: str | Path | None = None,
               command: list[str] | None = None) -> dict[str, Any]:
    """Run one independent physical forward simulation."""

    cfg.validate()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    generated_mesh = Path(mesh_path) if mesh_path is not None else outdir / "mesh.msh"
    if mesh_path is None or not generated_mesh.exists():
        generate_mesh(cfg, generated_mesh)
    elif Path(generated_mesh).resolve() != (outdir / "mesh.msh").resolve():
        shutil.copy2(generated_mesh, outdir / "mesh.msh")
        generated_mesh = outdir / "mesh.msh"
    mesh_data = audit_mesh(generated_mesh, cfg)
    system = build_system(cfg, mesh_data)
    result = run_newmark(system, cfg.time.dt_s, cfg.time.t_end_s, cfg.time.save_every)
    linearity_result = None
    if validate_linearity:
        # The second run is intentionally real time integration, not the
        # tautological post-hoc multiplication of one waveform.  It verifies
        # the linear M/C/K update and initial-value path at the configured
        # resolution.
        linearity_result = run_newmark(system, cfg.time.dt_s, cfg.time.t_end_s,
                                       cfg.time.save_every, p0_override=2.0 * system.p0)
    measured, filter_meta = apply_receiver_response(result.raw_signal_pa, cfg.time.dt_s, cfg.sensor)
    validation = validate_case(system, result, measured, linearity_result)
    adc_meta = write_waveform(outdir, result, measured, cfg)
    write_snapshots(outdir, system, result)
    if make_plots:
        write_plots(outdir, system, result, measured)
    else:
        # Keep output manifest stable while allowing CI/headless smoke runs.
        for name in ("mesh.png", "initial_pressure.png", "waveform.png"):
            (outdir / name).unlink(missing_ok=True)
    write_json(outdir / "validation.json", validation)
    write_metadata(outdir, cfg, mesh_data, system, result, validation, filter_meta,
                   adc_meta, command or list(sys.argv))
    # A compact stdout-friendly summary is useful for scripted sweeps.
    return {
        "outdir": str(outdir),
        "mesh": mesh_data.to_dict(),
        "n_dofs": system.n_dofs,
        "runtime_s": result.runtime_s,
        "p0_peak_pa": validation["source"]["p0_projected_peak_pa"],
        "raw_peak_pa": validation["receiver"]["raw_peak_pa"],
        "arrival_s": validation["receiver"]["arrival_estimate_s"],
        "validation_status": validation["status"],
        "resolution_warning": not validation["mesh"]["reference_resolution"],
        "source": source_summary(cfg, system.materials),
    }
