"""Independent physical benchmarks for the Stage A/B acceptance contract.

The production case validates its own mesh and algebra.  This module adds
problems with an analytic answer or an independently varied computational
domain, so a self-consistent implementation error cannot pass unnoticed.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from skfem import FacetBasis, asm

from .assemble import AcousticSystem, abc_form, build_system
from .config import (
    CaseConfig,
    GeometryConfig,
    MaterialConfig,
    MeshConfig,
    OpticalConfig,
    SensorConfig,
    TimeConfig,
)
from .geometry import TAGS, generate_mesh
from .mesh_check import audit_mesh
from .receiver import apply_receiver_response
from .time_integrator import TransientResult, run_newmark


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _db(ratio: float) -> float:
    return float(20.0 * np.log10(max(float(ratio), 1e-300)))


def _case(
    name: str,
    *,
    width_m: float,
    height_m: float,
    interface_y_m: float,
    source_y_m: float,
    source_sigma_m: float,
    h_m: float,
    dt_s: float,
    t_end_s: float,
    center_frequency_hz: float,
    tissue_rho: float = 1000.0,
    tissue_c: float = 1480.0,
) -> CaseConfig:
    """Create a small, fully labelled benchmark geometry in SI units."""

    half = 0.5 * width_m
    margin = max(2.0 * h_m, 0.02 * width_m)
    tissue_top = height_m - margin
    if not interface_y_m < tissue_top:
        raise ValueError("benchmark tissue layer has no positive height")
    absorber_radius = min(0.06 * width_m, 0.12 * (tissue_top - interface_y_m))
    absorber_y = interface_y_m + 0.45 * (tissue_top - interface_y_m)
    geometry = GeometryConfig(
        domain_x_min_m=-half,
        domain_x_max_m=half,
        domain_y_min_m=0.0,
        domain_y_max_m=height_m,
        tissue_x_min_m=-half + margin,
        tissue_x_max_m=half - margin,
        tissue_y_min_m=interface_y_m,
        tissue_y_max_m=tissue_top,
        absorber_x_m=0.0,
        absorber_y_m=absorber_y,
        absorber_radius_m=absorber_radius,
        sensor_x_min_m=-half + margin,
        sensor_x_max_m=half - margin,
        sensor_y_m=0.0,
    )
    materials = MaterialConfig(
        rho_water_kg_m3=1000.0,
        c_water_m_s=1480.0,
        rho_tissue_kg_m3=tissue_rho,
        c_tissue_m_s=tissue_c,
        rho_absorber_kg_m3=tissue_rho,
        c_absorber_m_s=tissue_c,
        mu_a_water_m_inv=0.0,
        mu_a_tissue_m_inv=0.0,
        mu_a_absorber_m_inv=1.0,
        gamma_water=0.1,
        gamma_tissue=0.1,
        gamma_absorber=0.1,
    )
    return CaseConfig(
        case_name=name,
        geometry=geometry,
        mesh=MeshConfig(element_size_m=h_m, interface_size_m=h_m,
                        source_size_m=h_m, algorithm=6, msh_file_version=4.1),
        materials=materials,
        optical=OpticalConfig(phi_peak_j_m2=1.0, sigma_m=source_sigma_m,
                              source_x_m=0.0, source_y_m=source_y_m,
                              laser_pulse_width_s=min(1e-9, 0.02 * source_sigma_m / 1480.0),
                              require_initial_pressure_approximation=True),
        sensor=SensorConfig(center_frequency_hz=center_frequency_hz,
                            fractional_bandwidth_fwhm=0.8, filter_order=4,
                            adc_rate_hz=None, noise_rms_pa=0.0, noise_seed=20260910),
        time=TimeConfig(t_end_s=t_end_s, dt_s=dt_s, save_every=max(1, round(t_end_s / dt_s))),
        random_seed=20260910,
    )


def _build(cfg: CaseConfig, directory: Path, mesh_path: Path | None = None) -> AcousticSystem:
    directory.mkdir(parents=True, exist_ok=True)
    path = mesh_path or directory / "mesh.msh"
    if mesh_path is None:
        generate_mesh(cfg, path)
    return build_system(cfg, audit_mesh(path, cfg))


def _with_abc_on_horizontal(system: AcousticSystem, *, bottom: bool, top: bool) -> AcousticSystem:
    """Use ABC only on selected horizontal edges; side walls stay symmetric.

    This preserves the x-invariant plane-wave benchmark.  Applying a local ABC
    on a side parallel to propagation would damp that exact solution.
    """

    mesh = system.mesh
    mids = np.mean(mesh.p[:, mesh.facets], axis=1)
    y0 = system.cfg.geometry.domain_y_min_m
    y1 = system.cfg.geometry.domain_y_max_m
    tol = max(1e-12, 0.05 * system.cfg.mesh.element_size_m)
    chosen: list[np.ndarray] = []
    if bottom:
        chosen.append(np.flatnonzero(np.abs(mids[1] - y0) <= tol))
    if top:
        chosen.append(np.flatnonzero(np.abs(mids[1] - y1) <= tol))
    if not chosen:
        C = csr_matrix(system.C.shape, dtype=float)
    else:
        facets = np.unique(np.concatenate(chosen)).astype(int)
        fb = FacetBasis(mesh, system.basis.elem, facets=facets, intorder=4)
        water = system.materials[TAGS.water]
        C = asm(abc_form, fb, inv_rho_c=1.0 / (water.rho_kg_m3 * water.c_m_s)).tocsr()
    return replace(system, C=C)


def _plane_gaussian(system: AcousticSystem, y0_m: float, sigma_m: float,
                    amplitude_pa: float = 1.0) -> np.ndarray:
    y = system.mesh.p[1]
    return amplitude_pa * np.exp(-0.5 * ((y - y0_m) / sigma_m) ** 2)


def _run_plane(system: AcousticSystem, y0_m: float, sigma_m: float) -> TransientResult:
    p0 = _plane_gaussian(system, y0_m, sigma_m)
    return run_newmark(replace(system, p0=p0), system.cfg.time.dt_s,
                       system.cfg.time.t_end_s, system.cfg.time.save_every)


def _analytic_sensor(times: np.ndarray, y0_m: float, sigma_m: float,
                     c_m_s: float = 1480.0) -> np.ndarray:
    """Whole-space d'Alembert solution evaluated at y=0 for zero initial velocity."""

    return 0.5 * np.exp(-0.5 * ((c_m_s * np.asarray(times) - y0_m) / sigma_m) ** 2)


def _window(times: np.ndarray, centre: float, half_width: float) -> np.ndarray:
    return np.abs(np.asarray(times) - centre) <= half_width


def _uniform_analytic_and_energy(root: Path) -> tuple[dict[str, Any], AcousticSystem]:
    cfg = _case("strict_uniform", width_m=1.2e-3, height_m=3.0e-3,
                interface_y_m=1.8e-3, source_y_m=1.0e-3, source_sigma_m=0.10e-3,
                h_m=25e-6, dt_s=4e-9, t_end_s=1.248e-6,
                center_frequency_hz=1e6)
    system = _with_abc_on_horizontal(_build(cfg, root / "uniform"), bottom=True, top=False)
    result = _run_plane(system, cfg.optical.source_y_m, cfg.optical.sigma_m)
    exact = _analytic_sensor(result.time_s, cfg.optical.source_y_m, cfg.optical.sigma_m)
    centre = cfg.optical.source_y_m / cfg.materials.c_water_m_s
    mask = _window(result.time_s, centre, 4.0 * cfg.optical.sigma_m / cfg.materials.c_water_m_s)
    waveform_error = _relative_l2(result.raw_signal_pa[mask], exact[mask])
    peak_num = float(result.time_s[mask][np.argmax(result.raw_signal_pa[mask])])
    peak_error = abs(peak_num - centre) / centre

    reflecting = _with_abc_on_horizontal(system, bottom=False, top=False)
    energy_result = _run_plane(reflecting, cfg.optical.source_y_m, cfg.optical.sigma_m)
    energy_drift = float(np.max(np.abs(energy_result.energy / energy_result.energy[0] - 1.0)))
    checks = {
        "analytic_waveform_relative_l2": waveform_error,
        "peak_time_s": peak_num,
        "expected_peak_time_s": centre,
        "peak_time_relative_error": peak_error,
        "closed_boundary_energy_max_relative_drift": energy_drift,
        "thresholds": {"waveform_relative_l2_max": 0.03,
                       "peak_time_relative_error_max": 0.01,
                       "energy_drift_max": 1e-10},
    }
    checks["passed"] = bool(waveform_error <= 0.03 and peak_error <= 0.01 and energy_drift <= 1e-10)
    return checks, system


def _abc_large_domain(root: Path) -> dict[str, Any]:
    common = dict(width_m=1.2e-3, interface_y_m=1.65e-3, source_y_m=1.0e-3,
                  source_sigma_m=0.10e-3, h_m=30e-6, dt_s=5e-9,
                  t_end_s=3.0e-6, center_frequency_hz=1e6)
    small_cfg = _case("strict_abc_small", height_m=2.4e-3, **common)
    large_cfg = _case("strict_abc_large", height_m=4.2e-3,
                      **{**common, "interface_y_m": 3.2e-3})
    small = _with_abc_on_horizontal(_build(small_cfg, root / "abc_small"), bottom=True, top=True)
    large = _with_abc_on_horizontal(_build(large_cfg, root / "abc_large"), bottom=True, top=True)
    a = _run_plane(small, common["source_y_m"], common["source_sigma_m"])
    b = _run_plane(large, common["source_y_m"], common["source_sigma_m"])
    direct_t = common["source_y_m"] / 1480.0
    reflected_t = (2.0 * small_cfg.geometry.domain_y_max_m - common["source_y_m"]) / 1480.0
    direct = _window(a.time_s, direct_t, 4.0 * common["source_sigma_m"] / 1480.0)
    late = _window(a.time_s, reflected_t, 4.0 * common["source_sigma_m"] / 1480.0)
    main_peak = float(np.max(np.abs(b.raw_signal_pa[direct])))
    reflection = float(np.max(np.abs(a.raw_signal_pa[late] - b.raw_signal_pa[late])) / main_peak)
    out = {
        "small_domain_top_m": small_cfg.geometry.domain_y_max_m,
        "large_domain_top_m": large_cfg.geometry.domain_y_max_m,
        "expected_small_domain_return_s": reflected_t,
        "reflection_amplitude_ratio": reflection,
        "reflection_db": _db(reflection),
        "threshold_db_max": -30.0,
    }
    out["passed"] = bool(out["reflection_db"] <= -30.0)
    return out


def _interface_reflection(root: Path) -> dict[str, Any]:
    cfg = _case("strict_interface", width_m=1.6e-3, height_m=3.0e-3,
                interface_y_m=1.2e-3, source_y_m=0.60e-3, source_sigma_m=0.060e-3,
                h_m=20e-6, dt_s=4e-9, t_end_s=1.548e-6,
                center_frequency_hz=1e6, tissue_rho=1050.0, tissue_c=1540.0)
    mesh_path = root / "interface" / "mesh.msh"
    hetero = _with_abc_on_horizontal(_build(cfg, root / "interface", mesh_path=None), bottom=True, top=False)
    homogeneous_cfg = replace(cfg, case_name="strict_interface_homogeneous",
                              materials=replace(cfg.materials,
                                                rho_tissue_kg_m3=1000.0, c_tissue_m_s=1480.0,
                                                rho_absorber_kg_m3=1000.0, c_absorber_m_s=1480.0))
    homogeneous = _with_abc_on_horizontal(_build(homogeneous_cfg, root / "interface_homogeneous",
                                                  mesh_path=mesh_path), bottom=True, top=False)
    rh = _run_plane(hetero, cfg.optical.source_y_m, cfg.optical.sigma_m)
    r0 = _run_plane(homogeneous, cfg.optical.source_y_m, cfg.optical.sigma_m)
    incident_t = cfg.optical.source_y_m / cfg.materials.c_water_m_s
    reflected_t = ((cfg.geometry.tissue_y_min_m - cfg.optical.source_y_m)
                   + cfg.geometry.tissue_y_min_m) / cfg.materials.c_water_m_s
    half_window = 3.5 * cfg.optical.sigma_m / cfg.materials.c_water_m_s
    incident = _window(r0.time_s, incident_t, half_window)
    reflected = _window(rh.time_s, reflected_t, half_window)
    incident_peak = float(np.max(np.abs(r0.raw_signal_pa[incident])))
    reflected_peak = float(np.max(rh.raw_signal_pa[reflected] - r0.raw_signal_pa[reflected]))
    observed = reflected_peak / incident_peak
    z1 = cfg.materials.rho_water_kg_m3 * cfg.materials.c_water_m_s
    z2 = cfg.materials.rho_tissue_kg_m3 * cfg.materials.c_tissue_m_s
    expected = (z2 - z1) / (z2 + z1)
    rel_error = abs(observed - expected) / abs(expected)
    out = {
        "expected_pressure_reflection_coefficient": expected,
        "observed_pressure_reflection_coefficient": observed,
        "relative_error": rel_error,
        "threshold_relative_error_max": 0.15,
    }
    out["passed"] = bool(rel_error <= 0.15 and np.sign(observed) == np.sign(expected))
    return out


def _aperture_quadrature(root: Path) -> dict[str, Any]:
    def evaluate(h: float, name: str) -> tuple[float, int]:
        cfg = _case(name, width_m=1.2e-3, height_m=1.5e-3,
                    interface_y_m=0.8e-3, source_y_m=0.45e-3, source_sigma_m=0.1e-3,
                    h_m=h, dt_s=5e-9, t_end_s=0.1e-6, center_frequency_hz=1e6)
        system = _build(cfg, root / name)
        wavelength = 0.40e-3
        field = np.cos(2.0 * np.pi * system.mesh.p[0] / wavelength)
        value = system.receiver_average(field)
        half_aperture = 0.5 * (cfg.geometry.sensor_x_max_m - cfg.geometry.sensor_x_min_m)
        exact = math.sin(2.0 * math.pi * half_aperture / wavelength) / (2.0 * math.pi * half_aperture / wavelength)
        return abs(value - exact) / max(abs(exact), 1e-30), len(system.boundary_facets_by_tag[TAGS.sensor])

    coarse, nc = evaluate(60e-6, "aperture_coarse")
    fine, nf = evaluate(20e-6, "aperture_fine")
    out = {
        "coarse_relative_error": coarse,
        "fine_relative_error": fine,
        "coarse_sensor_facets": nc,
        "fine_sensor_facets": nf,
        "threshold_fine_relative_error_max": 0.01,
    }
    out["passed"] = bool(fine <= 0.01 and fine < coarse)
    return out


def _ten_mhz(root: Path) -> dict[str, Any]:
    common = dict(width_m=0.40e-3, height_m=0.80e-3, interface_y_m=0.50e-3,
                  source_y_m=0.32e-3, source_sigma_m=0.030e-3,
                  t_end_s=0.55e-6, center_frequency_hz=10e6)
    fine_cfg = _case("10MHz_h2p5um_dt0p5ns", h_m=2.5e-6, dt_s=0.5e-9, **common)
    coarse_cfg = _case("10MHz_h5um_dt0p5ns", h_m=5e-6, dt_s=0.5e-9, **common)
    time_cfg = _case("10MHz_h2p5um_dt1ns", h_m=2.5e-6, dt_s=1e-9, **common)

    fine = _with_abc_on_horizontal(_build(fine_cfg, root / "10MHz_fine"), bottom=True, top=True)
    coarse = _with_abc_on_horizontal(_build(coarse_cfg, root / "10MHz_coarse"), bottom=True, top=True)
    # Reuse the exact fine mesh and matrices for a controlled time-step-only comparison.
    time_system = replace(fine, cfg=time_cfg)
    rf = _run_plane(fine, common["source_y_m"], common["source_sigma_m"])
    rc = _run_plane(coarse, common["source_y_m"], common["source_sigma_m"])
    rt = _run_plane(time_system, common["source_y_m"], common["source_sigma_m"])
    mf, _ = apply_receiver_response(rf.raw_signal_pa, fine_cfg.time.dt_s, fine_cfg.sensor)
    mc, _ = apply_receiver_response(rc.raw_signal_pa, coarse_cfg.time.dt_s, coarse_cfg.sensor)
    mt, _ = apply_receiver_response(rt.raw_signal_pa, time_cfg.time.dt_s, time_cfg.sensor)
    mt_on_fine = np.interp(rf.time_s, rt.time_s, mt)
    exact = _analytic_sensor(rf.time_s, common["source_y_m"], common["source_sigma_m"])
    centre = common["source_y_m"] / 1480.0
    mask = _window(rf.time_s, centre, 5.0 * common["source_sigma_m"] / 1480.0)
    analytic_error = _relative_l2(rf.raw_signal_pa[mask], exact[mask])
    spatial = _relative_l2(mc, mf)
    temporal = _relative_l2(mt_on_fine, mf)
    peak_t = float(rf.time_s[mask][np.argmax(rf.raw_signal_pa[mask])])
    peak_error = abs(peak_t - centre) / centre
    out = {
        "scope": "scaled homogeneous 10 MHz canonical benchmark; not unreported paper geometry",
        "fine_dofs": fine.n_dofs,
        "coarse_dofs": coarse.n_dofs,
        "fine_h_m": fine_cfg.mesh.element_size_m,
        "coarse_h_m": coarse_cfg.mesh.element_size_m,
        "fine_dt_s": fine_cfg.time.dt_s,
        "coarse_dt_s": time_cfg.time.dt_s,
        "analytic_raw_waveform_relative_l2": analytic_error,
        "peak_time_relative_error": peak_error,
        "spatial_10MHz_bandlimited_relative_l2_h5_vs_h2p5": spatial,
        "temporal_10MHz_bandlimited_relative_l2_dt1ns_vs_dt0p5ns": temporal,
        "thresholds": {"analytic_l2_max": 0.05, "peak_time_error_max": 0.01,
                       "spatial_l2_max": 0.02, "temporal_l2_max": 0.01},
    }
    out["passed"] = bool(analytic_error <= 0.05 and peak_error <= 0.01
                         and spatial <= 0.02 and temporal <= 0.01)
    return out


def run_strict_validation(outdir: str | Path) -> dict[str, Any]:
    """Run all independent Stage A/B checks and the scaled 10 MHz suite."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    uniform, _ = _uniform_analytic_and_energy(outdir)
    report: dict[str, Any] = {
        "schema_version": "pa_fem.strict_validation.v1",
        "basis": {
            "analytic": "1-D d'Alembert plane-Gaussian solution embedded in the 2-D FEM domain",
            "abc": "small-domain versus extended-domain waveform comparison",
            "interface": "normal-incidence pressure coefficient (Z2-Z1)/(Z2+Z1)",
            "receiver": "analytic integral of a cosine field over the finite line aperture",
            "10MHz": "spatial/time refinement plus analytic plane-wave reference",
        },
        "uniform_analytic_and_closed_energy": uniform,
        "abc_large_domain": _abc_large_domain(outdir),
        "interface_reflection": _interface_reflection(outdir),
        "aperture_quadrature": _aperture_quadrature(outdir),
        "ten_mhz_parameterized": _ten_mhz(outdir),
    }
    names = [k for k in report if k not in {"schema_version", "basis"}]
    report["hard_checks"] = {name: bool(report[name]["passed"]) for name in names}
    report["hard_checks_passed"] = int(sum(report["hard_checks"].values()))
    report["hard_checks_total"] = len(names)
    report["status"] = "pass" if report["hard_checks_passed"] == report["hard_checks_total"] else "fail"
    report["runtime_s"] = float(time.perf_counter() - started)
    (outdir / "strict_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
