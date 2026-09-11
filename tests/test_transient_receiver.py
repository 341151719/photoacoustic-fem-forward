from __future__ import annotations

import numpy as np
import pytest

from pa_fem.assemble import build_system
from pa_fem.receiver import apply_receiver_response, resample_signal
from pa_fem.time_integrator import run_newmark
from pa_fem.validation import estimate_arrival, validate_case


def test_newmark_abc_and_actual_linearity(smoke_case, smoke_mesh):
    system = build_system(smoke_case, smoke_mesh)
    one = run_newmark(system, smoke_case.time.dt_s, smoke_case.time.t_end_s, smoke_case.time.save_every)
    two = run_newmark(system, smoke_case.time.dt_s, smoke_case.time.t_end_s, smoke_case.time.save_every,
                      p0_override=2.0 * system.p0)
    assert np.max(np.diff(one.energy)) <= 1e-8 * one.energy[0]
    assert np.linalg.norm(two.raw_signal_pa - 2.0 * one.raw_signal_pa) / np.linalg.norm(2.0 * one.raw_signal_pa) < 1e-10
    measured, meta = apply_receiver_response(one.raw_signal_pa, smoke_case.time.dt_s, smoke_case.sensor)
    assert meta["causal"] is True
    assert np.isfinite(measured).all()
    report = validate_case(system, one, measured, two)
    assert report["status"] == "pass"
    assert report["receiver"]["linearity_check"] == "passed"


def test_arrival_returns_none_for_zero_trace():
    assert estimate_arrival(np.linspace(0.0, 1.0, 8), np.zeros(8)) is None


def test_adc_resampling_is_antialiased_and_checks_nyquist():
    time = np.arange(1001) / 100e6
    signal = np.sin(2.0 * np.pi * 1e6 * time)
    with pytest.raises(ValueError, match="Nyquist"):
        resample_signal(time, signal, 3e6, required_band_hz=2e6)
    target, sampled, meta = resample_signal(time, signal, 10e6, required_band_hz=2e6)
    assert len(target) == len(sampled)
    assert meta["method"] == "polyphase_FIR_antialias"
    assert np.isfinite(sampled).all()
