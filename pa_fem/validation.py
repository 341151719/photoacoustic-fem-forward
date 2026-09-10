"""Numerical/physical acceptance metrics for the minimum forward loop."""

from __future__ import annotations

from typing import Any

import numpy as np

from .assemble import AcousticSystem
from .config import CaseConfig
from .source import source_summary
from .time_integrator import TransientResult


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(np.asarray(b)), 1e-30))


def estimate_arrival(time_s: np.ndarray, signal_pa: np.ndarray, fraction: float = 0.05) -> float | None:
    """Estimate a causal first crossing from ``abs(signal)``.

    A Hilbert envelope over a finite record leaks information backwards from
    the end of the record, so it is intentionally not used for first-arrival
    timing.  This simple threshold remains a diagnostic because a finite
    Gaussian source has no mathematically sharp front.
    """

    signal_pa = np.asarray(signal_pa, dtype=float)
    if len(signal_pa) < 4 or not np.any(np.abs(signal_pa) > 0):
        return None
    magnitude = np.abs(signal_pa)
    threshold = fraction * float(np.max(magnitude))
    indices = np.flatnonzero(magnitude >= threshold)
    return None if len(indices) == 0 else float(np.asarray(time_s)[indices[0]])


def expected_center_arrival_s(cfg: CaseConfig) -> float:
    """Straight-path estimate through tissue then water to sensor centre."""

    g, m, o = cfg.geometry, cfg.materials, cfg.optical
    # The default source and sensor are vertically aligned.  Include a simple
    # water/tissue segment estimate; this is diagnostic, not a Green function.
    tissue_distance = max(o.source_y_m - g.tissue_y_min_m, 0.0)
    water_distance = max(g.tissue_y_min_m - g.sensor_y_m, 0.0)
    return tissue_distance / m.c_tissue_m_s + water_distance / m.c_water_m_s


def _energy_metrics(result: TransientResult) -> dict[str, float | bool]:
    e = np.asarray(result.energy, dtype=float)
    e0 = max(abs(float(e[0])), 1e-30)
    growth = np.diff(e)
    max_growth = max(float(np.max(growth)), 0.0) if len(growth) else 0.0
    # Damping/ABC should not produce systematic energy growth; the small
    # tolerance allows floating point and sparse-factorization roundoff.
    return {
        "initial_energy": float(e[0]),
        "final_energy": float(e[-1]),
        "final_to_initial_energy": float(e[-1] / e0),
        "max_positive_energy_increment_relative": float(max_growth / e0),
        "abc_energy_nonincreasing": bool(max_growth / e0 <= 2e-7),
    }


def validate_case(system: AcousticSystem, result: TransientResult,
                  measured_signal_pa: np.ndarray | None = None,
                  linearity_result: TransientResult | None = None) -> dict[str, Any]:
    """Build a JSON-serializable validation report.

    The report distinguishes hard assembly checks from diagnostic accuracy
    criteria.  A coarse debug mesh is allowed to run but is explicitly marked
    ``reference_resolution=False``.
    """

    cfg = system.cfg
    quality = system.mesh_data.audit["triangle_quality"]
    edge_lengths = _all_edge_lengths(system.mesh_data.points_m, system.mesh_data.triangles)
    mesh_p95 = float(np.percentile(edge_lengths, 95.0))
    h_rec = cfg.recommended_mesh_size_m()
    dt_f_limit = 1.0 / (25.0 * max(cfg.source_frequency_hz(),
                                    cfg.sensor.center_frequency_hz * (1.0 + cfg.sensor.fractional_bandwidth_fwhm / 2.0)))
    c_max = max(cfg.materials.c_water_m_s, cfg.materials.c_tissue_m_s, cfg.materials.c_absorber_m_s)
    min_edge = _minimum_edge_length(system.mesh_data.points_m, system.mesh_data.triangles)
    cfl = c_max * cfg.time.dt_s / max(min_edge, 1e-30)
    p0 = system.p0
    p0_peak = float(np.max(np.abs(p0)))
    # For the mass-lumped projection, sum of nodal lumped weights equals the
    # assembled source RHS integral exactly.
    p0_integral = float(np.sum(system.p0_rhs))
    p0_consistent_integral = float(np.sum(system.geometric_mass @ system.p0_raw_projection))
    p0_rhs_integral = float(np.sum(system.p0_rhs))
    raw = result.raw_signal_pa
    arrival = estimate_arrival(result.time_s, raw)
    expected = expected_center_arrival_s(cfg)
    arrival_rel_error = None if arrival is None else abs(arrival - expected) / expected
    linearity_residual = None if linearity_result is None else _relative_l2(linearity_result.raw_signal_pa, 2.0 * raw)
    report = {
        "status": "pass",
        "units": {"pressure": "Pa", "time": "s", "length": "m", "fluence": "J/m^2"},
        "mesh": {
            "n_points": int(len(system.mesh_data.points_m)),
            "n_triangles": int(len(system.mesh_data.triangles)),
            "minimum_edge_m": min_edge,
            "p95_edge_m": mesh_p95,
            "recommendation_h_m": h_rec,
            "resolution_ratio_p95_over_recommended": float(mesh_p95 / h_rec),
            "reference_resolution": bool(mesh_p95 <= 1.25 * h_rec and cfg.time.dt_s <= dt_f_limit),
            "quality": quality,
        },
        "time": {
            "dt_s": float(cfg.time.dt_s),
            "t_end_s": float(cfg.time.t_end_s),
            "steps": int(len(result.time_s) - 1),
            "recommended_dt_upper_bound_s": dt_f_limit,
            "cfl_cdt_over_hmin": float(cfl),
            "time_resolution_ok": bool(cfg.time.dt_s <= dt_f_limit and cfl <= 0.4),
        },
        "source": {
            **source_summary(cfg, system.materials),
            "p0_projected_peak_pa": p0_peak,
            "p0_projected_integral_pa_m2": p0_integral,
            "p0_raw_projection_integral_pa_m2": p0_rhs_integral,
            "p0_consistent_projection_integral_pa_m2": p0_consistent_integral,
            "p0_consistent_projection_negative_nodes": int(system.p0_consistent_projection_negative_nodes),
            "p0_nonnegative": bool(np.min(p0) >= -1e-9 * max(p0_peak, 1.0)),
        },
        "matrices": system.matrix_metrics,
        "receiver": {
            "finite_aperture_length_m": float(system.receiver_length_m),
            "design_length_m": float(cfg.geometry.sensor_x_max_m - cfg.geometry.sensor_x_min_m),
            "length_relative_error": float(abs(system.receiver_length_m - (cfg.geometry.sensor_x_max_m - cfg.geometry.sensor_x_min_m)) /
                                             (cfg.geometry.sensor_x_max_m - cfg.geometry.sensor_x_min_m)),
            "raw_peak_pa": float(np.max(np.abs(raw))),
            "arrival_estimate_s": arrival,
            "expected_center_arrival_s": expected,
            "arrival_relative_error": arrival_rel_error,
            "linearity_scaled_waveform_residual": linearity_residual,
            "linearity_check": "passed" if linearity_result is not None and linearity_residual <= 1e-10
            else ("not_run" if linearity_result is None else "failed"),
        },
        "energy": _energy_metrics(result),
        "limitations": [
            "2-D pressure acoustics represent an out-of-plane infinite line source and line receiver.",
            "Tissue is treated as an inviscid fluid; no shear, thermal, or power-law attenuation is included.",
            "The finite-aperture receiver uses a diagnostic causal Butterworth response unless measured h_tr is supplied.",
            "No encoded acoustic aperture or compressed/reconstruction algorithm is implemented.",
        ],
    }
    hard_checks = [
        report["receiver"]["length_relative_error"] <= 5e-3,
        report["source"]["p0_nonnegative"],
        report["energy"]["abc_energy_nonincreasing"],
        system.matrix_metrics["mass_symmetry_relative"] <= 1e-12,
        system.matrix_metrics["stiffness_symmetry_relative"] <= 1e-12,
    ]
    if linearity_result is not None:
        hard_checks.append(bool(linearity_residual is not None and linearity_residual <= 1e-10))
    report["status"] = "pass" if all(hard_checks) else "fail"
    report["hard_checks_passed"] = int(sum(bool(x) for x in hard_checks))
    report["hard_checks_total"] = len(hard_checks)
    return report


def _minimum_edge_length(points: np.ndarray, triangles: np.ndarray) -> float:
    return float(np.min(_all_edge_lengths(points, triangles)))


def _all_edge_lengths(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p = np.asarray(points)[np.asarray(triangles)]
    return np.concatenate([
        np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
        np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
        np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
    ])
