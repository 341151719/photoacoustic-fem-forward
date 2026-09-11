"""Finite-aperture post-processing and a causal receiver bandwidth model."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .config import SensorConfig


def apply_receiver_response(raw_signal_pa: np.ndarray, dt_s: float,
                            cfg: SensorConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply an explicitly causal Butterworth band-pass response.

    The configured bandwidth is the total power-FWHM: its Butterworth edges
    are -3 dB points (amplitude 1/sqrt(2)).  It is an engineering baseline,
    not a claim about the paper's unreported transducer response. ``sosfilt``
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
        "fractional_bandwidth_power_fwhm_assumption": frac,
        "edge_definition": "Butterworth -3 dB; power half maximum; amplitude 1/sqrt(2)",
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
                    adc_rate_hz: float | None, *,
                    required_band_hz: float | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return an optional anti-aliased uniformly sampled ADC waveform."""

    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if adc_rate_hz is None:
        return time_s.copy(), signal.copy(), {"adc_rate_hz": None, "method": "native_fem_sampling"}
    if adc_rate_hz <= 0:
        raise ValueError("adc_rate_hz must be positive")
    if required_band_hz is not None and adc_rate_hz < 2.0 * required_band_hz:
        raise ValueError(
            f"adc_rate_hz={adc_rate_hz:.4g} cannot Nyquist-sample the required "
            f"band edge {required_band_hz:.4g} Hz"
        )
    native_rate = 1.0 / float(time_s[1] - time_s[0])
    ratio = Fraction(float(adc_rate_hz / native_rate)).limit_denominator(100_000)
    out = resample_poly(signal, ratio.numerator, ratio.denominator)
    target = time_s[0] + np.arange(len(out), dtype=float) / float(adc_rate_hz)
    keep = target <= time_s[-1] + 0.5 / float(adc_rate_hz)
    out = np.asarray(out[keep], dtype=float)
    target = target[keep]
    return target, out, {
        "adc_rate_hz": float(adc_rate_hz),
        "method": "polyphase_FIR_antialias",
        "up": int(ratio.numerator),
        "down": int(ratio.denominator),
        "required_band_hz": None if required_band_hz is None else float(required_band_hz),
    }
