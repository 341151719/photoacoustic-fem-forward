"""Dual-encoded photoacoustic forward prototype.

Random non-negative optical patterns excite an FEM initial-pressure basis. An
internal aperture is sampled in spatial channels; independent causal delays are
then applied to those channels before coherent bucket integration. The
post-aperture channels are an explicit idealization, not a manufactured mask.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.sparse.linalg import factorized
from skfem import LinearForm, asm

from .assemble import AcousticSystem, build_system
from .config import CaseConfig, load_config
from .geometry import generate_mesh
from .mesh_check import audit_mesh
from .receiver import apply_receiver_response


@dataclass(frozen=True)
class DualSettings:
    source_basis_count: int = 6
    pattern_count: int = 16
    aperture_channels: int = 8
    aperture_y_m: float | None = None
    common_delay_fraction_of_tof: float = 0.15
    max_extra_delay_fraction_of_tof: float = 0.80
    quadrature_order: int = 8

    @classmethod
    def from_file(cls, path: str | Path) -> "DualSettings":
        raw = json.loads(Path(path).read_text(encoding="utf-8")).get("dual", {})
        if not isinstance(raw, dict):
            raise TypeError("config section 'dual' must be an object")
        defaults = cls()
        values = {name: raw.get(name, getattr(defaults, name))
                  for name in defaults.__dataclass_fields__}
        return cls(**values)

    def validate(self, cfg: CaseConfig) -> None:
        if self.source_basis_count < 1 or self.pattern_count < 1 or self.aperture_channels < 1:
            raise ValueError("dual basis, pattern, and aperture-channel counts must be positive")
        if self.quadrature_order < 2:
            raise ValueError("dual quadrature_order must be >= 2")
        if self.common_delay_fraction_of_tof < 0 or self.max_extra_delay_fraction_of_tof < 0:
            raise ValueError("dual delay fractions cannot be negative")
        y = aperture_y(cfg, self)
        if not cfg.geometry.domain_y_min_m < y < cfg.geometry.tissue_y_min_m:
            raise ValueError("dual aperture_y_m must lie inside the water layer")
        if cfg.sensor.noise_rms_pa != 0:
            raise ValueError("set noise_rms_pa=0 for the linear basis; add measurement noise afterwards")


def aperture_y(cfg: CaseConfig, settings: DualSettings) -> float:
    if settings.aperture_y_m is not None:
        return float(settings.aperture_y_m)
    g = cfg.geometry
    return g.domain_y_min_m + 0.5 * (g.tissue_y_min_m - g.domain_y_min_m)


def make_patterns(rng: np.random.Generator, count: int, basis_count: int) -> np.ndarray:
    """Return reproducible non-negative binary patterns with no dark rows."""

    patterns = rng.integers(0, 2, size=(count, basis_count)).astype(float)
    for row in patterns:
        if not row.any():
            row[rng.integers(basis_count)] = 1.0
    return patterns


def _source_centres(cfg: CaseConfig, count: int) -> np.ndarray:
    g = cfg.geometry
    return np.column_stack([
        g.absorber_x_m + np.linspace(-0.65, 0.65, count) * g.absorber_radius_m,
        np.full(count, g.absorber_y_m),
    ])


def assemble_initial_pressure_basis(system: AcousticSystem, centres: np.ndarray) -> np.ndarray:
    """Mass-lumped conservative projection of each illuminated source basis."""

    cfg = system.cfg
    lumped = np.asarray(system.geometric_mass @ np.ones(system.n_dofs), dtype=float)
    initial = np.zeros((system.n_dofs, len(centres)), dtype=float)
    for j, centre in enumerate(np.asarray(centres, dtype=float)):
        rhs = np.zeros(system.n_dofs, dtype=float)
        for tag, material in system.materials.items():
            def make_load(mat, xy):
                @LinearForm
                def source_load(v, w):
                    radius2 = (w.x[0] - xy[0]) ** 2 + (w.x[1] - xy[1]) ** 2
                    fluence = cfg.optical.phi_peak_j_m2 * np.exp(
                        -radius2 / (2.0 * cfg.optical.sigma_m ** 2))
                    return mat.gamma * mat.eta_th * mat.mu_a_m_inv * fluence * v
                return source_load

            rhs += asm(make_load(material, centre),
                       system.basis.with_elements(system.triangle_elements_by_tag[tag]))
        initial[:, j] = rhs / lumped
    return initial


def propagate_to_aperture(system: AcousticSystem, initial: np.ndarray,
                          pattern_for_linearity: np.ndarray, settings: DualSettings
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Propagate all basis columns and one direct composite pattern together."""

    cfg = system.cfg
    g = cfg.geometry
    nchannel = settings.aperture_channels
    edges = np.linspace(g.sensor_x_min_m, g.sensor_x_max_m, nchannel + 1)
    gx, gw = np.polynomial.legendre.leggauss(settings.quadrature_order)
    rx = np.concatenate([
        (edges[k] + edges[k + 1]) / 2.0 + gx * (edges[k + 1] - edges[k]) / 2.0
        for k in range(nchannel)
    ])
    y_aperture = aperture_y(cfg, settings)
    probe = system.basis.probes(np.vstack([rx, np.full_like(rx, y_aperture)]))

    p = np.column_stack([initial, initial @ pattern_for_linearity])
    velocity = np.zeros_like(p)
    acceleration = factorized(system.M.tocsc())(-system.K @ p)
    dt = cfg.time.dt_s
    solve = factorized(
        (system.M + 0.5 * dt * system.C + 0.25 * dt ** 2 * system.K).tocsc())
    count = round(cfg.time.t_end_s / dt) + 1
    if not np.isclose((count - 1) * dt, cfg.time.t_end_s, rtol=1e-9, atol=1e-18):
        raise ValueError("dual t_end_s must be an integer multiple of dt_s")
    fem_time = np.arange(count, dtype=float) * dt
    fields = np.empty((count, nchannel, initial.shape[1] + 1), dtype=float)

    for step in range(count):
        sample = (probe @ p).reshape(nchannel, len(gx), initial.shape[1] + 1)
        fields[step] = np.einsum("q,cqj->cj", gw / 2.0, sample)
        if step == count - 1:
            break
        predicted = p + dt * velocity + 0.25 * dt ** 2 * acceleration
        velocity_predicted = velocity + 0.5 * dt * acceleration
        acceleration_new = solve(-system.C @ velocity_predicted - system.K @ predicted)
        p = predicted + 0.25 * dt ** 2 * acceleration_new
        velocity = velocity_predicted + 0.5 * dt * acceleration_new
        acceleration = acceleration_new
    return fields, fem_time, edges, y_aperture


def bucket_delayed(aperture_signals: np.ndarray, fem_time: np.ndarray,
                   output_time: np.ndarray, delays_s: np.ndarray,
                   channel_weights: np.ndarray) -> np.ndarray:
    """Apply per-channel causal delays before the coherent spatial sum."""

    signals = np.asarray(aperture_signals, dtype=float)
    if signals.ndim != 3:
        raise ValueError("aperture_signals must have shape [time, channel, basis]")
    if signals.shape[1] != len(delays_s) or signals.shape[1] != len(channel_weights):
        raise ValueError("delay/weight count must equal aperture channel count")
    output = np.zeros((len(output_time), signals.shape[2]), dtype=float)
    for channel in range(signals.shape[1]):
        shifted_time = output_time - delays_s[channel]
        for basis in range(signals.shape[2]):
            output[:, basis] += channel_weights[channel] * np.interp(
                shifted_time, fem_time, signals[:, channel, basis], left=0.0, right=0.0)
    return output


def _operator_metrics(operator: np.ndarray) -> dict[str, Any]:
    columns = np.asarray(operator, dtype=float).T
    norms = np.linalg.norm(columns, axis=1)
    normalized = columns / np.maximum(norms[:, None], 1e-30)
    gram = normalized @ normalized.T
    off_diagonal = np.abs(gram - np.eye(len(columns)))
    singular = np.linalg.svd(columns, compute_uv=False)
    return {
        "maximum_absolute_column_coherence": float(np.max(off_diagonal)),
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / max(singular[-1], 1e-30)),
        "rank": int(np.linalg.matrix_rank(columns)),
    }


def run(config: str | Path, outdir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config)
    settings = DualSettings.from_file(config)
    settings.validate(cfg)
    system = build_system(cfg, audit_mesh(generate_mesh(cfg, outdir / "mesh.msh"), cfg))
    rng = np.random.default_rng(cfg.random_seed)
    patterns = make_patterns(rng, settings.pattern_count, settings.source_basis_count)
    centres = _source_centres(cfg, settings.source_basis_count)
    initial = assemble_initial_pressure_basis(system, centres)
    fields, fem_time, edges, y_aperture = propagate_to_aperture(
        system, initial, patterns[0], settings)

    g = cfg.geometry
    water_distance = max(g.tissue_y_min_m - y_aperture, 0.0)
    tissue_distance = max(g.absorber_y_m - g.tissue_y_min_m, 0.0)
    tof = (water_distance / cfg.materials.c_water_m_s
           + tissue_distance / cfg.materials.c_tissue_m_s)
    common_delay = settings.common_delay_fraction_of_tof * tof
    extra_delays = rng.uniform(0.0, settings.max_extra_delay_fraction_of_tof * tof,
                               size=settings.aperture_channels)
    end = (fem_time[-1] + common_delay + float(np.max(extra_delays))
           + 8.0 / cfg.sensor.center_frequency_hz)
    output_time = np.arange(int(np.ceil(end / cfg.time.dt_s)) + 1) * cfg.time.dt_s
    weights = np.diff(edges) / (edges[-1] - edges[0])
    basis_fields = fields[:, :, :settings.source_basis_count]
    uncoded_raw = bucket_delayed(
        basis_fields, fem_time, output_time,
        np.full(settings.aperture_channels, common_delay), weights)
    coded_raw = bucket_delayed(
        basis_fields, fem_time, output_time, common_delay + extra_delays, weights)
    uncoded_basis = np.column_stack([
        apply_receiver_response(uncoded_raw[:, j], cfg.time.dt_s, cfg.sensor)[0]
        for j in range(settings.source_basis_count)
    ])
    coded_basis = np.column_stack([
        apply_receiver_response(coded_raw[:, j], cfg.time.dt_s, cfg.sensor)[0]
        for j in range(settings.source_basis_count)
    ])
    measurements_uncoded = uncoded_basis @ patterns.T
    measurements_coded = coded_basis @ patterns.T

    direct_coded = bucket_delayed(
        fields[:, :, -1:], fem_time, output_time, common_delay + extra_delays, weights)[:, 0]
    expected_direct = coded_raw @ patterns[0]
    direct_residual = np.linalg.norm(direct_coded - expected_direct) / max(
        np.linalg.norm(expected_direct), 1e-30)
    zero = bucket_delayed(basis_fields, fem_time, output_time,
                          np.zeros(settings.aperture_channels), weights)
    zero_expected = np.column_stack([
        np.interp(output_time, fem_time,
                  np.einsum("c,tc->t", weights, basis_fields[:, :, j]), left=0.0, right=0.0)
        for j in range(settings.source_basis_count)
    ])
    zero_error = np.linalg.norm(zero - zero_expected) / max(np.linalg.norm(zero_expected), 1e-30)
    finite = bool(np.isfinite(measurements_coded).all()
                  and np.isfinite(measurements_uncoded).all())
    tri_points = system.mesh_data.points_m[system.mesh_data.triangles]
    edge_lengths = np.concatenate([
        np.linalg.norm(tri_points[:, 1] - tri_points[:, 0], axis=1),
        np.linalg.norm(tri_points[:, 2] - tri_points[:, 1], axis=1),
        np.linalg.norm(tri_points[:, 0] - tri_points[:, 2], axis=1),
    ])
    p95_edge = float(np.percentile(edge_lengths, 95.0))
    recommended_h = cfg.recommended_mesh_size_m()

    summary: dict[str, Any] = {
        "schema_version": "pa_fem.dual_forward.v2",
        "status_scope": "dataflow, finiteness, and linear-superposition consistency; not accuracy convergence",
        "model": "2D FEM plus ideal unloaded independent delay-channel aperture",
        "limitations": [
            "Not a manufactured mask FEM or paper geometry reproduction",
            "No optical scattering, shear, channel loading, losses, or post-pickup cross-talk",
            "No reconstruction or claimed subwavelength resolution",
            "Fractional delays use linear interpolation; channel signals are zero-extended",
            "Receiver bandwidth is Butterworth power-FWHM (-3 dB edges)",
        ],
        "config": cfg.to_dict(),
        "dual_settings": settings.__dict__,
        "dofs": system.n_dofs,
        "source_centers_m": centres.tolist(),
        "aperture_y_m": y_aperture,
        "estimated_source_to_aperture_tof_s": tof,
        "common_delay_s": common_delay,
        "additional_channel_delays_s": extra_delays.tolist(),
        "fem_samples": len(fem_time),
        "measurement_samples": len(output_time),
        "numerical_resolution": {
            "p95_edge_m": p95_edge,
            "characteristic_recommendation_h_m": recommended_h,
            "p95_over_recommendation": p95_edge / recommended_h,
            "characteristic_resolution_met": bool(p95_edge <= 1.25 * recommended_h),
            "interpretation": (
                "source estimate is not a strict cutoff for the discontinuous absorption field; "
                "the compact case has no convergence claim"
            ),
        },
        "coded_peak_pa": float(np.max(np.abs(measurements_coded))),
        "uncoded_peak_pa": float(np.max(np.abs(measurements_uncoded))),
        "coded_vs_uncoded_relative_l2": float(
            np.linalg.norm(measurements_coded - measurements_uncoded)
            / max(np.linalg.norm(measurements_uncoded), 1e-30)),
        "coded_operator": _operator_metrics(coded_basis),
        "uncoded_operator": _operator_metrics(uncoded_basis),
        "pattern_matrix_rank": int(np.linalg.matrix_rank(patterns)),
        "direct_pattern_vs_basis_relative_l2": float(direct_residual),
        "zero_delay_bucket_relative_l2": float(zero_error),
        "finite": finite,
        "status": "pass" if finite and direct_residual < 1e-9 and zero_error < 1e-12 else "fail",
        "runtime_s": float(time.perf_counter() - started),
    }
    np.savez_compressed(
        outdir / "dual_data.npz", time_s=output_time, fem_time_s=fem_time,
        source_centers_m=centres, optical_patterns=patterns,
        initial_pressure_basis_pa=initial, mesh_points_m=system.mesh.p.T,
        mesh_triangles=system.mesh.t.T, aperture_edges_m=edges,
        aperture_y_m=y_aperture, aperture_basis_pa=basis_fields,
        channel_delays_s=common_delay + extra_delays,
        uncoded_basis_pa=uncoded_basis, coded_basis_pa=coded_basis,
        uncoded_measurements_pa=measurements_uncoded,
        coded_measurements_pa=measurements_coded,
    )
    _write_plot(outdir, patterns, extra_delays, output_time,
                measurements_uncoded, measurements_coded)
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _write_plot(outdir: Path, patterns: np.ndarray, extra_delays: np.ndarray,
                time_s: np.ndarray, uncoded: np.ndarray, coded: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    axes[0].imshow(patterns, aspect="auto", cmap="gray", vmin=0, vmax=1)
    axes[0].set(xlabel="Optical basis index", ylabel="Pattern index",
                title="Known random non-negative optical patterns")
    axes[1].bar(np.arange(len(extra_delays)), extra_delays * 1e9)
    axes[1].set(xlabel="Aperture channel", ylabel="Additional delay (ns)",
                title="Ideal channel delays before bucket sum")
    axes[2].plot(time_s * 1e6, uncoded[:, 0], label="No differential channel coding")
    axes[2].plot(time_s * 1e6, coded[:, 0], label="Coded aperture")
    axes[2].set(xlabel="Time (µs)", ylabel="Pressure (Pa)",
                title="Same optical pattern, 10 MHz receiver")
    axes[2].legend()
    fig.savefig(outdir / "dual_comparison.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dual_10MHz_compact.json")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    result = run(args.config, args.outdir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
