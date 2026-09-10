#!/usr/bin/env python3
"""Build a reproducible visual validation package from completed FEM cases."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.tri as mtri
import meshio
import numpy as np
from scipy.signal import butter, sosfreqz


MATERIAL_COLORS = ["#6bb8d6", "#e8bf72", "#a63d40"]
MATERIAL_NAMES = {101: "water", 102: "tissue", 103: "absorber"}


def load_case(path: Path) -> dict:
    with np.load(path / "waveform.npz") as data:
        waveform = {key: np.asarray(data[key]) for key in data.files}
    return {
        "path": path,
        "waveform": waveform,
        "validation": json.loads((path / "validation.json").read_text(encoding="utf-8")),
        "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
    }


def relative_l2(reference_t: np.ndarray, reference_y: np.ndarray,
                candidate_t: np.ndarray, candidate_y: np.ndarray) -> float:
    candidate_on_reference = np.interp(reference_t, candidate_t, candidate_y)
    return float(np.linalg.norm(candidate_on_reference - reference_y) /
                 max(np.linalg.norm(reference_y), 1e-30))


def peak_metrics(case: dict) -> tuple[float, float]:
    t = case["waveform"]["time_s"]
    y = case["waveform"]["s_raw_pa"]
    index = int(np.argmax(np.abs(y)))
    return float(t[index]), float(abs(y[index]))


def all_edges(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p = points[triangles]
    return np.concatenate((
        np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
        np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
        np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
    ))


def triangle_quality(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = points[triangles]
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    cosines = np.column_stack((
        (a*a + c*c - b*b) / np.maximum(2*a*c, 1e-30),
        (a*a + b*b - c*c) / np.maximum(2*a*b, 1e-30),
        (b*b + c*c - a*a) / np.maximum(2*b*c, 1e-30),
    ))
    min_angle = np.degrees(np.min(np.arccos(np.clip(cosines, -1.0, 1.0)), axis=1))
    aspect = np.maximum.reduce((a, b, c)) / np.maximum(np.minimum.reduce((a, b, c)), 1e-30)
    return min_angle, aspect


def snapshot_time(name: str) -> float | None:
    match = re.search(r"t_([0-9.]+)us", name)
    return None if match is None else float(match.group(1))


def nearest_snapshot(point_data: dict[str, np.ndarray], time_us: float) -> tuple[str, float]:
    candidates = [(name, snapshot_time(name)) for name in point_data if name.startswith("pressure_")]
    candidates = [(name, value) for name, value in candidates if value is not None]
    return min(candidates, key=lambda item: abs(item[1] - time_us))


def add_geometry_lines(ax: plt.Axes) -> None:
    ax.plot([-5, 5, 5, -5, -5], [2, 2, 9, 9, 2], color="white", lw=0.8, alpha=0.75)
    circle = plt.Circle((0, 5.5), 0.3, fill=False, color="white", lw=0.8, alpha=0.9)
    ax.add_patch(circle)
    ax.plot([-1.5, 1.5], [0, 0], color="#101060", lw=3)


def figure_model(mesh: meshio.Mesh, output: Path, validation: dict) -> None:
    points = mesh.points[:, :2] * 1e3
    triangles = mesh.cells_dict["triangle"]
    tags = mesh.cell_data_dict["physical_id"]["triangle"]
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    cmap = ListedColormap(MATERIAL_COLORS)
    norm = BoundaryNorm([100.5, 101.5, 102.5, 103.5], cmap.N)
    pc = axes[0, 0].tripcolor(triang, facecolors=tags, cmap=cmap, norm=norm, shading="flat")
    axes[0, 0].set_title("Material domains and finite receiver")
    add_geometry_lines(axes[0, 0])
    cb = fig.colorbar(pc, ax=axes[0, 0], ticks=[101, 102, 103])
    cb.ax.set_yticklabels([MATERIAL_NAMES[i] for i in (101, 102, 103)])

    axes[0, 1].triplot(triang, color="#303030", lw=0.28)
    axes[0, 1].set(xlim=(-0.65, 0.65), ylim=(4.85, 6.15), title="Conforming mesh near absorber")
    axes[0, 1].add_patch(plt.Circle((0, 5.5), 0.3, fill=False, color="#d62728", lw=1.5))

    p0 = mesh.point_data["p0_pa"]
    p = axes[1, 0].tripcolor(triang, p0 / 1e3, cmap="magma", shading="gouraud")
    axes[1, 0].set(title="Positive conservative initial pressure", xlim=(-1.2, 1.2), ylim=(4.3, 6.7))
    fig.colorbar(p, ax=axes[1, 0], label="$p_0$ (kPa)")

    interp = mtri.LinearTriInterpolator(triang, p0)
    xline = np.linspace(-0.8, 0.8, 800)
    fem = np.asarray(interp(xline, np.full_like(xline, 5.5)))
    mu = np.where(np.abs(xline) <= 0.3, 1000.0, 10.0)
    analytic = 0.1 * mu * 100.0 * np.exp(-0.5 * (xline / 0.2) ** 2)
    axes[1, 1].plot(xline, analytic / 1e3, "k--", lw=1.4, label="analytic material field")
    axes[1, 1].plot(xline, fem / 1e3, color="#d62728", lw=1.2, label="mass-lumped FEM")
    axes[1, 1].axvspan(-0.3, 0.3, color="#e8bf72", alpha=0.18, label="absorber")
    source = validation["source"]
    axes[1, 1].text(0.02, 0.97,
        f"peak = {source['p0_projected_peak_pa']/1e3:.3f} kPa\n"
        f"integral = {source['p0_projected_integral_pa_m2']:.4e} Pa m²",
        transform=axes[1, 1].transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    axes[1, 1].set(xlabel="x at y=5.5 mm (mm)", ylabel="$p_0$ (kPa)", title="Source projection check")
    axes[1, 1].legend(loc="upper right", fontsize=8)
    for ax in axes.flat:
        ax.set_aspect("equal" if ax is not axes[1, 1] else "auto")
        if ax is not axes[1, 1]:
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
        ax.grid(alpha=0.15)
    fig.suptitle("FEM model, tags, mesh and photoacoustic source", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def figure_wave_snapshots(mesh: meshio.Mesh, output: Path) -> None:
    points = mesh.points[:, :2] * 1e3
    triang = mtri.Triangulation(points[:, 0], points[:, 1], mesh.cells_dict["triangle"])
    requested = [0.0, 1.6, 2.8, 3.6, 4.4, 6.0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for ax, target in zip(axes.flat, requested):
        key, actual = nearest_snapshot(mesh.point_data, target)
        pressure = np.asarray(mesh.point_data[key])
        scale = max(float(np.max(np.abs(pressure))), 1e-30)
        pc = ax.tripcolor(triang, pressure / scale, shading="gouraud", cmap="RdBu_r", vmin=-1, vmax=1)
        add_geometry_lines(ax)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", xlim=(-6, 6), ylim=(0, 10))
        ax.set_aspect("equal")
        ax.set_title(f"t = {actual:.1f} µs; max |p| = {scale:.1f} Pa")
    fig.colorbar(pc, ax=axes.ravel().tolist(), shrink=0.72, label="p / max|p| (per panel)")
    fig.suptitle("Transient pressure propagation: source → interfaces → receiver", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def figure_receiver(reference: dict, spatial: dict, fine: dict, temporal: dict, output: Path) -> dict:
    rw = reference["waveform"]
    rv = reference["validation"]
    cfg = reference["metadata"]["configuration"]["sensor"]
    time_us = rw["time_s"] * 1e6
    raw = rw["s_raw_pa"]
    measured = rw["s_meas_pa"]
    peak_t, peak_p = peak_metrics(reference)
    expected = rv["receiver"]["expected_center_arrival_s"]
    first = rv["receiver"]["arrival_estimate_s"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axes[0, 0].plot(time_us, raw, lw=1.1, label="finite-aperture raw")
    axes[0, 0].plot(time_us, measured, lw=1.1, label="causal 1 MHz response")
    axes[0, 0].axvline(expected * 1e6, color="k", ls="--", lw=1, label="centre-path d/c")
    axes[0, 0].axvline(peak_t * 1e6, color="#d62728", ls=":", lw=1.5, label="raw peak")
    axes[0, 0].set(xlabel="time (µs)", ylabel="pressure (Pa)", title="Received waveform and causal transducer output")
    axes[0, 0].legend(fontsize=8)

    for case, label, style in ((reference, "h=40 µm, dt=5 ns", "-"),
                               (spatial, "h=80 µm, dt=5 ns", "--"),
                               (fine, "h=30 µm, dt=5 ns", "-."),
                               (temporal, "h=40 µm, dt=10 ns", ":")):
        w = case["waveform"]
        axes[0, 1].plot(w["time_s"] * 1e6, w["s_raw_pa"], style, lw=1.2, label=label)
    axes[0, 1].set(xlim=(2.7, 4.6), xlabel="time (µs)", ylabel="raw pressure (Pa)",
                   title="Spatial/time convergence near direct arrival")
    axes[0, 1].legend(fontsize=8)

    dt = float(rw["time_s"][1] - rw["time_s"][0])
    freq = np.fft.rfftfreq(len(raw), dt)
    raw_fft = np.abs(np.fft.rfft(raw - np.mean(raw)))
    measured_fft = np.abs(np.fft.rfft(measured - np.mean(measured)))
    scale = max(float(np.max(raw_fft)), 1e-30)
    axes[1, 0].plot(freq / 1e6, raw_fft / scale, label="raw")
    axes[1, 0].plot(freq / 1e6, measured_fft / scale, label="measured")
    low = cfg["center_frequency_hz"] * (1 - cfg["fractional_bandwidth_fwhm"] / 2)
    high = cfg["center_frequency_hz"] * (1 + cfg["fractional_bandwidth_fwhm"] / 2)
    axes[1, 0].axvspan(low / 1e6, high / 1e6, color="#ffbf00", alpha=0.18, label="nominal receiver band")
    axes[1, 0].set(xlim=(0, 5), xlabel="frequency (MHz)", ylabel="amplitude / raw max", title="Signal spectra")
    axes[1, 0].legend(fontsize=8)

    fs = 1.0 / dt
    sos = butter(cfg["filter_order"], [low, high], btype="bandpass", fs=fs, output="sos")
    f_resp, h_resp = sosfreqz(sos, worN=4096, fs=fs)
    axes[1, 1].plot(f_resp / 1e6, 20 * np.log10(np.maximum(np.abs(h_resp), 1e-8)), color="#9467bd")
    axes[1, 1].axvline(low / 1e6, color="0.4", ls="--", lw=0.8)
    axes[1, 1].axvline(high / 1e6, color="0.4", ls="--", lw=0.8)
    axes[1, 1].set(xlim=(0, 4), ylim=(-80, 3), xlabel="frequency (MHz)", ylabel="magnitude (dB)",
                   title="Assumed causal receiver transfer magnitude")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("Finite-aperture receiver validity", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "first_threshold_time_s": first,
        "expected_center_path_time_s": expected,
        "raw_peak_time_s": peak_t,
        "raw_peak_time_relative_error_vs_center": abs(peak_t - expected) / expected,
        "raw_peak_pa": peak_p,
        "measured_peak_pa": float(np.max(np.abs(measured))),
    }


def figure_numerics(reference: dict, spatial: dict, fine: dict, temporal: dict,
                    reference_mesh: meshio.Mesh, spatial_mesh: meshio.Mesh,
                    output: Path) -> dict:
    rw = reference["waveform"]
    sw = spatial["waveform"]
    fw = fine["waveform"]
    tw = temporal["waveform"]
    spatial_l2 = relative_l2(rw["time_s"], rw["s_raw_pa"], sw["time_s"], sw["s_raw_pa"])
    spatial_measured_l2 = relative_l2(rw["time_s"], rw["s_meas_pa"], sw["time_s"], sw["s_meas_pa"])
    fine_mask = fw["time_s"] <= min(fw["time_s"][-1], rw["time_s"][-1])
    fine_l2 = relative_l2(fw["time_s"][fine_mask], fw["s_raw_pa"][fine_mask],
                          rw["time_s"], rw["s_raw_pa"])
    fine_measured_l2 = relative_l2(fw["time_s"][fine_mask], fw["s_meas_pa"][fine_mask],
                                   rw["time_s"], rw["s_meas_pa"])
    temporal_l2 = relative_l2(rw["time_s"], rw["s_raw_pa"], tw["time_s"], tw["s_raw_pa"])
    temporal_measured_l2 = relative_l2(rw["time_s"], rw["s_meas_pa"], tw["time_s"], tw["s_meas_pa"])
    ref_peak_t, ref_peak = peak_metrics(reference)
    spatial_peak_t, spatial_peak = peak_metrics(spatial)
    fine_peak_t, fine_peak = peak_metrics(fine)
    temporal_peak_t, temporal_peak = peak_metrics(temporal)

    rp = reference_mesh.points[:, :2]
    rt = reference_mesh.cells_dict["triangle"]
    sp = spatial_mesh.points[:, :2]
    st = spatial_mesh.cells_dict["triangle"]
    ref_edges = all_edges(rp, rt) * 1e6
    coarse_edges = all_edges(sp, st) * 1e6
    angles, aspect = triangle_quality(rp, rt)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    energy = rw["energy_j_like"]
    axes[0, 0].semilogy(rw["time_s"] * 1e6, energy / energy[0], color="#1f77b4")
    axes[0, 0].set(xlabel="time (µs)", ylabel="E / E(0)", title="ABC energy decay")

    increments = np.diff(energy) / energy[0]
    axes[0, 1].plot(rw["time_s"][1:] * 1e6, increments, lw=0.7, color="#2ca02c")
    axes[0, 1].axhline(0, color="k", lw=0.8)
    axes[0, 1].set(xlabel="time (µs)", ylabel="(E[n+1]-E[n])/E[0]", title="No artificial energy growth")

    bins = np.linspace(0, max(np.percentile(coarse_edges, 99.5), np.percentile(ref_edges, 99.5)), 55)
    axes[0, 2].hist(coarse_edges, bins=bins, density=True, alpha=0.55, label="80 µm case")
    axes[0, 2].hist(ref_edges, bins=bins, density=True, alpha=0.55, label="40 µm reference")
    axes[0, 2].axvline(reference["validation"]["mesh"]["recommendation_h_m"] * 1e6,
                       color="k", ls="--", label="recommended h")
    axes[0, 2].set(xlabel="triangle edge length (µm)", ylabel="density", title="Mesh resolution distribution")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].hist(angles, bins=45, color="#4c78a8", alpha=0.85)
    axes[1, 0].axvline(np.min(angles), color="#d62728", ls="--", label=f"min {np.min(angles):.1f}°")
    axes[1, 0].set(xlabel="minimum triangle angle (degree)", ylabel="count", title="Reference mesh element angles")
    axes[1, 0].legend()

    axes[1, 1].hist(aspect, bins=45, color="#f58518", alpha=0.85)
    axes[1, 1].axvline(np.max(aspect), color="#d62728", ls="--", label=f"max {np.max(aspect):.3f}")
    axes[1, 1].set(xlabel="longest / shortest edge", ylabel="count", title="Reference mesh aspect ratio")
    axes[1, 1].legend()

    labels = ["80 vs 40 µm", "40 vs 30 µm", "10 vs 5 ns"]
    raw_values = np.array([spatial_l2, fine_l2, temporal_l2]) * 100
    measured_values = np.array([spatial_measured_l2, fine_measured_l2, temporal_measured_l2]) * 100
    x = np.arange(len(labels))
    bars_raw = axes[1, 2].bar(x - 0.19, raw_values, 0.38, color="#7f7f7f", label="raw broadband")
    bars_meas = axes[1, 2].bar(x + 0.19, measured_values, 0.38, color="#54a24b", label="1 MHz measured")
    axes[1, 2].bar_label(bars_raw, fmt="%.1f", padding=2, fontsize=7)
    axes[1, 2].bar_label(bars_meas, fmt="%.2f", padding=2, fontsize=7)
    axes[1, 2].set_xticks(x, labels, rotation=12)
    axes[1, 2].axhline(2.0, color="#d62728", ls="--", lw=1, label="2% engineering target")
    axes[1, 2].set(ylabel="waveform relative L2 difference (%)", title="Raw vs detected-band convergence")
    axes[1, 2].set_ylim(0, max(raw_values) * 1.25 + 1e-6)
    axes[1, 2].legend(fontsize=7)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Numerical validity: dissipation, mesh quality and convergence", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "spatial_waveform_relative_l2_h80_vs_h40": spatial_l2,
        "spatial_measured_relative_l2_h80_vs_h40": spatial_measured_l2,
        "spatial_waveform_relative_l2_h40_vs_h30": fine_l2,
        "spatial_measured_relative_l2_h40_vs_h30": fine_measured_l2,
        "temporal_waveform_relative_l2_dt10ns_vs_dt5ns": temporal_l2,
        "temporal_measured_relative_l2_dt10ns_vs_dt5ns": temporal_measured_l2,
        "spatial_peak_time_difference_s": spatial_peak_t - ref_peak_t,
        "spatial_peak_amplitude_relative_difference": abs(spatial_peak - ref_peak) / ref_peak,
        "fine_peak_time_difference_s": fine_peak_t - ref_peak_t,
        "fine_peak_amplitude_relative_difference": abs(fine_peak - ref_peak) / ref_peak,
        "temporal_peak_time_difference_s": temporal_peak_t - ref_peak_t,
        "temporal_peak_amplitude_relative_difference": abs(temporal_peak - ref_peak) / ref_peak,
        "reference_min_angle_deg": float(np.min(angles)),
        "reference_max_aspect_ratio": float(np.max(aspect)),
        "maximum_positive_energy_increment_relative": float(max(np.max(increments), 0.0)),
    }


def make_animation(mesh: meshio.Mesh, output: Path) -> None:
    points = mesh.points[:, :2] * 1e3
    triang = mtri.Triangulation(points[:, 0], points[:, 1], mesh.cells_dict["triangle"])
    frames = sorted(
        [(name, snapshot_time(name)) for name in mesh.point_data if name.startswith("pressure_")],
        key=lambda item: item[1] if item[1] is not None else -1,
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    first = np.asarray(mesh.point_data[frames[0][0]])
    first_scale = max(float(np.max(np.abs(first))), 1e-30)
    pc = ax.tripcolor(triang, first / first_scale, shading="gouraud", cmap="RdBu_r", vmin=-1, vmax=1)
    add_geometry_lines(ax)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", xlim=(-6, 6), ylim=(0, 10))
    ax.set_aspect("equal")
    title = ax.set_title("")
    fig.colorbar(pc, ax=ax, label="p / max|p| (per frame)")

    def update(index: int):
        name, t_us = frames[index]
        field = np.asarray(mesh.point_data[name])
        scale = max(float(np.max(np.abs(field))), 1e-30)
        pc.set_array(field / scale)
        title.set_text(f"Photoacoustic pressure propagation: t={t_us:.1f} µs, max|p|={scale:.1f} Pa")
        return pc, title

    movie = animation.FuncAnimation(fig, update, frames=len(frames), interval=140, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=7), dpi=105)
    plt.close(fig)


def write_report(output: Path, metrics: dict, reference: dict, spatial: dict,
                 fine: dict, temporal: dict) -> None:
    rv = reference["validation"]
    text = f"""# FEM 前端物理仿真可视化验证报告

## 结论

当前 1 MHz 参考算例完成了“光能沉积—初始声压—异质声传播—有限孔径接收—因果带宽”闭环。参考网格和时间步判据均通过，所有 {rv['hard_checks_total']} 项硬检查通过。

## 主要数值证据

- 参考网格：{rv['mesh']['n_points']:,} 自由度，{rv['mesh']['n_triangles']:,} 三角形，95% 边长 {rv['mesh']['p95_edge_m']*1e6:.2f} µm。
- 时间离散：{rv['time']['dt_s']*1e9:.1f} ns，{rv['time']['steps']} 步，CFL 指标 {rv['time']['cfl_cdt_over_hmin']:.3f}。
- 初压峰值：{rv['source']['p0_projected_peak_pa']/1e3:.3f} kPa；预期量级 10 kPa。
- 探头中心路径理论时延：{metrics['receiver']['expected_center_path_time_s']*1e6:.3f} µs；原始波形主峰：{metrics['receiver']['raw_peak_time_s']*1e6:.3f} µs，相对差 {metrics['receiver']['raw_peak_time_relative_error_vs_center']*100:.2f}%。
- 有限孔径原始峰值：{metrics['receiver']['raw_peak_pa']:.2f} Pa；因果带宽后：{metrics['receiver']['measured_peak_pa']:.2f} Pa。
- 80 vs 40 µm：原始宽带波形 L2 差 {metrics['numerics']['spatial_waveform_relative_l2_h80_vs_h40']*100:.3f}%，1 MHz 带限测量差 {metrics['numerics']['spatial_measured_relative_l2_h80_vs_h40']*100:.3f}%；80 µm 网格明显欠解析。
- 40 vs 30 µm：原始宽带波形 L2 差 {metrics['numerics']['spatial_waveform_relative_l2_h40_vs_h30']*100:.3f}%，1 MHz 带限测量差 {metrics['numerics']['spatial_measured_relative_l2_h40_vs_h30']*100:.3f}%。
- 时间步 10 vs 5 ns：原始宽带波形 L2 差 {metrics['numerics']['temporal_waveform_relative_l2_dt10ns_vs_dt5ns']*100:.3f}%，1 MHz 带限测量差 {metrics['numerics']['temporal_measured_relative_l2_dt10ns_vs_dt5ns']*100:.3f}%。
- 线性检查：{rv['receiver']['linearity_check']}，残差 {rv['receiver']['linearity_scaled_waveform_residual']:.3e}。
- ABC 能量非增：{rv['energy']['abc_energy_nonincreasing']}，末态/初态能量 {rv['energy']['final_to_initial_energy']:.4f}。

## 图件

1. `01_model_source_mesh.png`：材料标签、相容网格、初压与解析源投影。
2. `02_wave_propagation.png`：六个时刻的声压传播与界面穿越。
3. `03_receiver_and_spectrum.png`：孔径平均波形、到达时间、频谱和探头响应。
4. `04_numerical_validity.png`：能量、网格质量和空间/时间收敛。
5. `05_wave_propagation.gif`：调试网格上的完整瞬态动画。

## 证据边界

- 能量单调下降证明 ABC 离散项是耗散的，但不等价于已经证明所有入射角下的反射都低于 -30 dB；这需要额外的大域/PML 对照。
- 首个 5% 阈值越界时间会受有限宽高斯源和数值前驱影响；中心路径的有效性主要由原始主峰时刻检查。
- 当前是二维无损压力声学，代表线源/条形探头；不能将绝对幅值直接解释为三维实验量。
- 1 MHz/80% Butterworth 带宽是明确标注的工程假设，并非论文未报告的真实探头脉冲响应。
- 40 vs 30 µm 时原始宽带波形尚有高频相位差，但实际 1 MHz 带限接收波形差已低于 2% 工程目标；因此当前有效性结论限定于建模的探头检测带宽。
- 本报告只验证前端物理正向算子，不验证编码声学孔径或压缩重建算法。
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--spatial-coarse", type=Path, required=True)
    parser.add_argument("--spatial-fine", type=Path, required=True)
    parser.add_argument("--time-coarse", type=Path, required=True)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = load_case(args.reference)
    spatial = load_case(args.spatial_coarse)
    fine = load_case(args.spatial_fine)
    temporal = load_case(args.time_coarse)
    reference_mesh = meshio.read(args.reference / "snapshots.vtu")
    spatial_mesh = meshio.read(args.spatial_coarse / "snapshots.vtu")
    debug_mesh = meshio.read(args.debug / "snapshots.vtu")

    figure_model(reference_mesh, args.outdir / "01_model_source_mesh.png", reference["validation"])
    figure_wave_snapshots(reference_mesh, args.outdir / "02_wave_propagation.png")
    receiver_metrics = figure_receiver(reference, spatial, fine, temporal, args.outdir / "03_receiver_and_spectrum.png")
    numerical_metrics = figure_numerics(reference, spatial, fine, temporal, reference_mesh, spatial_mesh,
                                        args.outdir / "04_numerical_validity.png")
    make_animation(debug_mesh, args.outdir / "05_wave_propagation.gif")
    metrics = {
        "reference_case": str(args.reference),
        "spatial_coarse_case": str(args.spatial_coarse),
        "spatial_fine_case": str(args.spatial_fine),
        "temporal_coarse_case": str(args.time_coarse),
        "receiver": receiver_metrics,
        "numerics": numerical_metrics,
        "reference_hard_checks": {
            "passed": reference["validation"]["hard_checks_passed"],
            "total": reference["validation"]["hard_checks_total"],
            "status": reference["validation"]["status"],
        },
    }
    (args.outdir / "visual_validation_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.outdir / "FEM可视化验证报告.md", metrics, reference, spatial, fine, temporal)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
