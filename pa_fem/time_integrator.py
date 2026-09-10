"""Newmark average-acceleration transient pressure integrator."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse.linalg import factorized, spsolve

from .assemble import AcousticSystem


@dataclass
class TransientResult:
    time_s: np.ndarray
    raw_signal_pa: np.ndarray
    energy: np.ndarray
    snapshots: dict[str, np.ndarray]
    snapshot_indices: dict[str, int]
    dt_s: float
    runtime_s: float
    initial_acceleration: np.ndarray


def _time_grid(t_end_s: float, dt_s: float) -> np.ndarray:
    n_steps = int(round(t_end_s / dt_s))
    if n_steps < 1:
        raise ValueError("t_end/dt must produce at least one time step")
    if not np.isclose(n_steps * dt_s, t_end_s, rtol=1e-9, atol=1e-18):
        raise ValueError("t_end_s must be an integer multiple of dt_s")
    return np.arange(n_steps + 1, dtype=float) * dt_s


def run_newmark(system: AcousticSystem, dt_s: float, t_end_s: float,
                save_every: int = 20, p0_override: np.ndarray | None = None) -> TransientResult:
    """Propagate ``M p¨ + C p˙ + K p = 0`` from a pressure initial value.

    The average-acceleration parameters beta=1/4 and gamma=1/2 are fixed by
    the reference method.  The effective sparse matrix is factorized once.
    ``p0_override`` is useful for a direct linearity check and is never used as
    a second time-dependent source.
    """

    beta, gamma = 0.25, 0.5
    times = _time_grid(t_end_s, dt_s)
    p = np.asarray(system.p0 if p0_override is None else p0_override, dtype=float).copy()
    if p.shape != system.p0.shape:
        raise ValueError("p0_override has the wrong number of DOFs")
    v = np.zeros_like(p)
    # Consistent initial acceleration: M a0 = -K p0 (f0=0, v0=0).
    a = np.asarray(spsolve(system.M.tocsc(), -system.K @ p), dtype=float)
    initial_acceleration = a.copy()
    a_eff = (system.M + gamma * dt_s * system.C + beta * dt_s**2 * system.K).tocsc()
    solve_eff = factorized(a_eff)
    raw = np.empty(len(times), dtype=float)
    energy = np.empty(len(times), dtype=float)
    snapshots: dict[str, np.ndarray] = {}
    snapshot_indices: dict[str, int] = {}
    requested = {0, len(times) - 1, len(times) // 2}
    requested.update(range(0, len(times), max(int(save_every), 1)))
    started = time.perf_counter()

    def save_state(index: int) -> None:
        if index in requested:
            label = f"t_{times[index]*1e6:09.3f}us"
            snapshots[label] = p.copy()
            snapshot_indices[label] = int(index)

    for n, _ in enumerate(times):
        raw[n] = system.receiver_average(p)
        energy[n] = 0.5 * (float(v @ (system.M @ v)) + float(p @ (system.K @ p)))
        save_state(n)
        if n == len(times) - 1:
            break
        p_pred = p + dt_s * v + dt_s**2 * (0.5 - beta) * a
        v_pred = v + dt_s * (1.0 - gamma) * a
        rhs = -system.C @ v_pred - system.K @ p_pred
        a_new = np.asarray(solve_eff(rhs), dtype=float)
        p = p_pred + beta * dt_s**2 * a_new
        v = v_pred + gamma * dt_s * a_new
        a = a_new
    return TransientResult(times, raw, energy, snapshots, snapshot_indices,
                           dt_s, time.perf_counter() - started, initial_acceleration)
