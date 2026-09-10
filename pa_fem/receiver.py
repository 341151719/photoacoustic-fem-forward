"""Finite-aperture post-processing and a causal receiver bandwidth model."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from .config import SensorConfig


def apply_receiver_response(raw_signal_pa: np.ndarray, dt_s: float,
                            cfg: SensorConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply an explicitly causal Butterworth band-pass response.

    The default bandwidth is an engineering baseline (80% FWHM around 1 MHz),
    not a claim about the paper's unreported transducer response.  ``sosfilt``
    is causal; no zero-phase filtering is used for arrival-time data.
    """

    x = np.asarray(raw_signal_pa, dtype=float)
    fs = 1.0 / dt_s
    fc = float(cfg.center_frequency_hz)
    frac = float(cfg.fractional_bandwidth_fwhm)
    low = fc * (1.0 - frac / 2.0)
    high = fc * (1.0 + frac / 2.0)
    if low <= 0 or high >= 0.5 * fs:
        raise ValueError(
            f"receiver band [{low:.4g}, {high:.4g}] Hz exceeds causal sampling Nyquist {0.5*fs:.4g} Hz"
        )
    sos = butter(int(cfg.filter_order), [low, high], btype="bandpass", fs=fs, output="sos")
    measured = sosfilt(sos, x)
    metadata: dict[str, Any] = {
        "model": "causal_butterworth_bandpass",
        "causal": True,
        "zero_phase": False,
        "center_frequency_hz": fc,
        "fractional_bandwidth_fwhm_assumption": frac,
        "nominal_low_cut_hz": low,
        "nominal_high_cut_hz": high,
        "filter_order": int(cfg.filter_order),
        "sampling_rate_hz": fs,
    }
    if cfg.noise_rms_pa > 0:
        rng = np.random.default_rng(cfg.noise_seed)
        measured = measured + rng.normal(0.0, cfg.noise_rms_pa, size=len(measured))
        metadata.update({"noise_rms_pa": float(cfg.noise_rms_pa), "noise_seed": int(cfg.noise_seed),
                         "noise_model": "independent_white_gaussian"})
    else:
        metadata.update({"noise_rms_pa": 0.0, "noise_seed": int(cfg.noise_seed), "noise_model": "none"})
    return measured, metadata


def resample_signal(time_s: np.ndarray, signal: np.ndarray,
                    adc_rate_hz: float | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return an optional uniformly sampled ADC waveform by interpolation."""

    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if adc_rate_hz is None:
        return time_s.copy(), signal.copy(), {"adc_rate_hz": None, "method": "native_fem_sampling"}
    if adc_rate_hz <= 0:
        raise ValueError("adc_rate_hz must be positive")
    n = int(np.floor((time_s[-1] - time_s[0]) * adc_rate_hz)) + 1
    target = time_s[0] + np.arange(max(n, 2), dtype=float) / adc_rate_hz
    target = target[target <= time_s[-1] + 0.5 / adc_rate_hz]
    out = np.interp(target, time_s, signal)
    return target, out, {"adc_rate_hz": float(adc_rate_hz), "method": "linear_interpolation"}

