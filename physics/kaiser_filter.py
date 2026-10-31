"""Kaiser windowing and single-sided spectral diagnostics.

A taper reduces finite-record spectral leakage; it is not a denoiser. Outputs
include explicit metrics so downstream agents do not confuse windowing with a
physical improvement in sensor signal-to-noise ratio.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def kaiser(length: int, beta: float, sym: bool = True) -> np.ndarray:
    """Return a validated NumPy Kaiser window.

    ``sym`` is accepted to keep the interface explicit and compatible with
    common DSP libraries. NumPy's symmetric Kaiser window is used for finite
    records; ``sym=False`` is not needed by the present analysis pipeline.
    """
    if not isinstance(length, int) or length < 0:
        raise ValueError("length must be a non-negative integer")
    if not math.isfinite(float(beta)) or not 0.0 <= float(beta) <= 20.0:
        raise ValueError("beta must be finite and in [0, 20]")
    if length == 0:
        return np.asarray([], dtype=np.float64)
    if length == 1:
        return np.ones(1, dtype=np.float64)
    if sym:
        return np.asarray(np.kaiser(length, float(beta)), dtype=np.float64)
    # Periodic form for spectral analysis: generate one extra point and omit
    # the repeated endpoint, matching the common DSP convention.
    return np.asarray(np.kaiser(length + 1, float(beta))[:-1], dtype=np.float64)


def _validate_samples(samples: list[float] | np.ndarray) -> np.ndarray:
    """Convert samples to a finite, one-dimensional float64 vector."""
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if data.size < 2:
        raise ValueError("at least two samples are required for spectral diagnostics")
    if not np.isfinite(data).all():
        raise ValueError("samples must contain only finite values")
    return data


def apply_kaiser(
    samples: list[float] | np.ndarray,
    beta: float,
    sampling_rate_hz: float,
) -> dict[str, Any]:
    """Apply the Kaiser taper and calculate basic power-spectrum diagnostics."""
    data = _validate_samples(samples)
    fs = float(sampling_rate_hz)
    beta_value = float(beta)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")

    window = kaiser(data.size, beta_value, sym=True)
    window_sum = float(window.sum())
    coherent_gain = float(window.mean())
    enbw_denominator = max(window_sum * window_sum, np.finfo(np.float64).tiny)
    enbw_bins = float(data.size * np.dot(window, window) / enbw_denominator)
    tapered = data * window
    spectrum = np.abs(np.fft.rfft(tapered)) ** 2

    # Ignore DC when searching for an oscillatory carrier, but handle a DC-only
    # signal without returning NaN or indexing an empty spectrum.
    if spectrum.size > 1 and float(spectrum[1:].sum()) > np.finfo(np.float64).tiny:
        peak_bin = int(np.argmax(spectrum[1:]) + 1)
        total_power = float(spectrum.sum())
        peak_band = np.zeros(spectrum.size, dtype=bool)
        peak_band[max(0, peak_bin - 1): min(spectrum.size, peak_bin + 2)] = True
        leakage_ratio = float(spectrum[~peak_band].sum() / max(total_power, 1e-30))
        dominant_frequency_hz = float(np.fft.rfftfreq(data.size, d=1.0 / fs)[peak_bin])
    else:
        peak_bin = 0
        leakage_ratio = 0.0
        dominant_frequency_hz = 0.0

    return {
        "samples": tapered.tolist(),
        "window": window.tolist(),
        "metrics": {
            "sample_count": int(data.size),
            "beta": beta_value,
            "coherent_gain": coherent_gain,
            "equivalent_noise_bandwidth_bins": enbw_bins,
            "dominant_frequency_hz": dominant_frequency_hz,
            "dominant_frequency_bin": peak_bin,
            "spectral_leakage_ratio_outside_peak_bins": leakage_ratio,
            "input_rms": float(np.sqrt(np.mean(np.square(data)))),
            "tapered_rms": float(np.sqrt(np.mean(np.square(tapered)))),
            "window_min": float(window.min()),
            "window_max": float(window.max()),
        },
        "note": "Kaiser taper reduces finite-window spectral leakage; it does not remove noise by itself.",
    }
