"""Regularized optical fluence and initial photoacoustic pressure."""

from __future__ import annotations

import numpy as np

from .config import CaseConfig
from .materials import Material


def gaussian_fluence(x: np.ndarray, y: np.ndarray, cfg: CaseConfig) -> np.ndarray:
    """Return a finite-width Gaussian fluence in J/m² at arbitrary points."""

    o = cfg.optical
    r2 = (x - o.source_x_m) ** 2 + (y - o.source_y_m) ** 2
    return o.phi_peak_j_m2 * np.exp(-0.5 * r2 / o.sigma_m**2)


def initial_pressure_from_material(x: np.ndarray, y: np.ndarray, material: Material,
                                   cfg: CaseConfig) -> np.ndarray:
    """Compute p0 = Gamma eta_th mu_a Phi (Pa) in one material subdomain."""

    return material.gamma * material.eta_th * material.mu_a_m_inv * gaussian_fluence(x, y, cfg)


def source_summary(cfg: CaseConfig, materials: dict[int, Material]) -> dict[str, float | bool]:
    """Return unit-aware source estimates for metadata and startup checks."""

    peak = max((m.gamma * m.eta_th * m.mu_a_m_inv for m in materials.values()), default=0.0) * cfg.optical.phi_peak_j_m2
    tau_s, tau_th = cfg.constraint_times()
    limit = 0.1 * min(tau_s, tau_th)
    return {
        "fluence_peak_j_m2": cfg.optical.phi_peak_j_m2,
        "sigma_m": cfg.optical.sigma_m,
        "characteristic_length_m": min(cfg.optical.sigma_m, cfg.geometry.absorber_radius_m),
        "fwhm_m": 2.0 * np.sqrt(2.0 * np.log(2.0)) * cfg.optical.sigma_m,
        "source_bandlimit_hz": cfg.source_frequency_hz(),
        "source_bandlimit_interpretation": (
            "three-sigma characteristic estimate; not a strict cutoff for discontinuous absorption"
        ),
        "p0_peak_estimate_pa": peak,
        "p0_peak_estimate_kind": "upper_bound_if_source_is_in_maximum_absorption_region",
        "laser_pulse_width_s": cfg.optical.laser_pulse_width_s,
        "stress_confinement_time_s": tau_s,
        "thermal_confinement_time_s": tau_th,
        "initial_pressure_limit_s": limit,
        "initial_pressure_approximation_valid": cfg.optical.laser_pulse_width_s < limit,
    }
