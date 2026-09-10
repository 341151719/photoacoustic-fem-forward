"""Machine-readable and quick-look output products."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np

from . import __version__
from .assemble import AcousticSystem
from .config import CaseConfig
from .mesh_check import MeshData
from .receiver import resample_signal
from .time_integrator import TransientResult


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def write_waveform(outdir: Path, result: TransientResult, measured_signal_pa: np.ndarray,
                   cfg: CaseConfig) -> dict[str, Any]:
    adc_time, adc_signal, adc_meta = resample_signal(result.time_s, measured_signal_pa, cfg.sensor.adc_rate_hz)
    np.savez_compressed(
        outdir / "waveform.npz",
        time_s=result.time_s,
        s_raw_pa=result.raw_signal_pa,
        s_meas_pa=np.asarray(measured_signal_pa),
        time_adc_s=adc_time,
        s_adc_pa=adc_signal,
        energy_j_like=result.energy,
    )
    return adc_meta


def write_snapshots(outdir: Path, system: AcousticSystem, result: TransientResult) -> None:
    points = np.column_stack([system.mesh_data.points_m, np.zeros(len(system.mesh_data.points_m))])
    cells = [("triangle", system.mesh_data.triangles)]
    cell_data = {"physical_id": [system.mesh_data.triangle_tags.astype(np.int32)]}
    point_data = {"p0_pa": system.p0.astype(float)}
    for label, values in result.snapshots.items():
        point_data[f"pressure_{label}_pa"] = np.asarray(values, dtype=float)
    mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data,
                       field_data={name: np.asarray(value, dtype=int)
                                   for name, value in system.mesh_data.field_data.items()})
    meshio.write(outdir / "snapshots.vtu", mesh)


def write_plots(outdir: Path, system: AcousticSystem, result: TransientResult,
                measured_signal_pa: np.ndarray) -> None:
    x_mm = system.mesh_data.points_m[:, 0] * 1e3
    y_mm = system.mesh_data.points_m[:, 1] * 1e3
    tris = system.mesh_data.triangles
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.triplot(x_mm, y_mm, tris, color="0.65", lw=0.25)
    ax.scatter([system.cfg.optical.source_x_m * 1e3], [system.cfg.optical.source_y_m * 1e3],
               c="crimson", s=25, label="laser source")
    ax.plot([system.cfg.geometry.sensor_x_min_m * 1e3, system.cfg.geometry.sensor_x_max_m * 1e3],
            [system.cfg.geometry.sensor_y_m * 1e3] * 2, color="navy", lw=3, label="sensor aperture")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Gmsh 2-D water/tissue/absorber mesh")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.savefig(outdir / "mesh.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    tpc = ax.tripcolor(x_mm, y_mm, tris, system.p0, shading="gouraud", cmap="magma")
    fig.colorbar(tpc, ax=ax, label="p0 (Pa)")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Projected initial photoacoustic pressure")
    ax.set_aspect("equal")
    fig.savefig(outdir / "initial_pressure.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    t_us = result.time_s * 1e6
    ax.plot(t_us, result.raw_signal_pa, label="finite-aperture raw", lw=1.0)
    ax.plot(t_us, measured_signal_pa, label="causal bandwidth output", lw=1.0)
    ax.set(xlabel="time (µs)", ylabel="pressure (Pa)", title="Transducer waveform")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "waveform.png", dpi=180)
    plt.close(fig)


def write_metadata(outdir: Path, cfg: CaseConfig, mesh_data: MeshData,
                   system: AcousticSystem, result: TransientResult,
                   validation: dict[str, Any], filter_metadata: dict[str, Any],
                   adc_metadata: dict[str, Any], command: list[str]) -> None:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "meshio", "scikit-fem", "gmsh", "matplotlib"):
        try:
            versions[package] = package_version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    metadata = {
        "schema_version": "pa_fem.forward.v1",
        "software": {"pa_fem": __version__, "python": sys.version,
                      "platform": platform.platform(), "packages": versions},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "case_name": cfg.case_name,
        "units": {"length": "m", "time": "s", "pressure": "Pa", "fluence": "J/m^2",
                  "density": "kg/m^3", "sound_speed": "m/s", "absorption": "1/m"},
        "model": {
            "optical": "analytic finite-width Gaussian fluence; no optical transport solve",
            "photoacoustic_conversion": "p0 = Gamma * eta_th * mu_a * Phi under stress/thermal confinement",
            "acoustics": "heterogeneous conservative inviscid pressure equation",
            "boundary": "first-order Sommerfeld/Engquist-Majda ABC on outer_absorbing and sensor",
            "receiver": "finite line-aperture pressure average followed by causal filter",
            "algorithm_scope": "physical forward model only; no compressed sensing/reconstruction",
        },
        "configuration": cfg.to_dict(),
        "mesh": mesh_data.to_dict(),
        "assembly": {"n_dofs": system.n_dofs, "matrix_metrics": system.matrix_metrics,
                      "p0_consistent_projection_negative_nodes": system.p0_consistent_projection_negative_nodes,
                      "p0_projection": "positive conservative mass-lumped L2 projection; consistent L2 overshoot count recorded"},
        "transient": {"dt_s": result.dt_s, "n_steps": len(result.time_s) - 1,
                       "runtime_s": result.runtime_s, "newmark_beta": 0.25,
                       "newmark_gamma": 0.5, "snapshot_indices": result.snapshot_indices},
        "receiver_filter": filter_metadata,
        "adc": adc_metadata,
        "validation_summary": validation,
        "outputs": [
            "mesh.msh", "snapshots.vtu", "waveform.npz", "metadata.json", "validation.json",
            *[name for name in ("mesh.png", "initial_pressure.png", "waveform.png")
              if (outdir / name).exists()],
        ],
    }
    write_json(outdir / "metadata.json", metadata)
