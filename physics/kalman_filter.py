"""Wrapped-phase/rate extended Kalman estimator with uncertainty diagnostics."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def kaiser(length: int, beta: float, sym: bool = True) -> np.ndarray:
    """Compatibility wrapper for the shared, real Kaiser-window implementation."""
    from physics.kaiser_filter import kaiser as make_kaiser

    return make_kaiser(length, beta, sym=sym)


def wrap_phase(value: float) -> float:
    """Wrap an angle to [-pi, pi] without discontinuity outside the interval."""
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def phase_observation(
    samples: np.ndarray,
    fs: float,
    frequency: float,
    start_index: int,
    beta: float,
    window_weights: np.ndarray | None = None,
    noise_sigma_floor: float = 0.0,
) -> tuple[float, float, float]:
    """Fit a local sinusoid and estimate phase, phase variance and residual RMS."""
    data = np.asarray(samples, dtype=np.float64)
    n = data.size
    if data.ndim != 1 or n < 8:
        raise ValueError("a phase observation needs at least 8 one-dimensional samples")
    if not np.isfinite(data).all():
        raise ValueError("phase samples must be finite")
    if not math.isfinite(fs) or fs <= 0 or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("sample rate and frequency must be finite and positive")

    if not math.isfinite(float(noise_sigma_floor)) or noise_sigma_floor < 0:
        raise ValueError("noise_sigma_floor must be finite and non-negative")
    t = (int(start_index) + np.arange(n, dtype=np.float64)) / fs
    window = kaiser(n, beta, sym=True) if window_weights is None else np.asarray(window_weights, dtype=np.float64)
    if window.shape != (n,) or not np.isfinite(window).all():
        raise ValueError("segment window weights are invalid or have the wrong length")
    omega_t = 2.0 * math.pi * frequency * t
    design = np.column_stack((window * np.cos(omega_t), window * np.sin(omega_t), window))
    y = data * window
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    dof = max(1, n - design.shape[1])
    empirical_noise_variance = max(0.0, float(residual @ residual / dof))
    noise_variance = max(empirical_noise_variance, float(noise_sigma_floor) ** 2)
    covariance = noise_variance * np.linalg.pinv(design.T @ design)

    a, b = float(coefficients[0]), float(coefficients[1])
    amplitude_squared = a * a + b * b
    if amplitude_squared <= 1e-24:
        return 0.0, math.pi**2, math.sqrt(empirical_noise_variance)
    phase = math.atan2(-b, a)
    gradient = np.asarray([b / amplitude_squared, -a / amplitude_squared], dtype=np.float64)
    phase_variance = max(1e-12, float(gradient @ covariance[:2, :2] @ gradient))
    return phase, phase_variance, math.sqrt(empirical_noise_variance)


def estimate_phase(
    samples: list[float] | np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    beta: float,
    reference_phase_rad: float,
    process_noise: float = 1e-4,
    window_weights: list[float] | np.ndarray | None = None,
    noise_sigma_floor: float = 0.0,
) -> dict[str, Any]:
    """Estimate wrapped phase and phase rate from non-overlapping sample blocks.

    The process model is constant phase rate. This is a software estimator, not
    a validated model of a particular atom interferometer or sensor instrument.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size < 16 or not np.isfinite(x).all():
        raise ValueError("samples must be a finite one-dimensional trace of at least 16 values")
    if not math.isfinite(float(process_noise)) or process_noise < 0:
        raise ValueError("process_noise must be finite and non-negative")
    if not math.isfinite(float(reference_phase_rad)):
        raise ValueError("reference phase must be finite")

    full_window: np.ndarray | None = None
    if window_weights is not None:
        full_window = np.asarray(window_weights, dtype=np.float64)
        if full_window.shape != x.shape or not np.isfinite(full_window).all():
            raise ValueError("window_weights must be finite and match the sample count")

    segment_count = max(1, min(8, x.size // 16))
    chunks = [chunk for chunk in np.array_split(np.arange(x.size), segment_count) if chunk.size >= 8]
    observations: list[tuple[float, float, float, int]] = []
    for chunk in chunks:
        weights = full_window[chunk] if full_window is not None else None
        observation = phase_observation(
            x[chunk], float(sampling_rate_hz), float(frequency_hz), int(chunk[0]),
            float(beta), weights, noise_sigma_floor=float(noise_sigma_floor),
        )
        observations.append((observation[0], observation[1], observation[2], int(chunk[0])))
    if not observations:
        raise ValueError("trace is too short for a phase observation")

    phase0, variance0, _, index0 = observations[0]
    state = np.asarray([phase0, 0.0], dtype=np.float64)  # phase, phase-rate
    covariance = np.diag([max(variance0, 1e-8), 1.0])
    measurement_matrix = np.asarray([[1.0, 0.0]], dtype=np.float64)
    previous_index = index0
    innovations: list[float] = []
    residual_rms_values: list[float] = [item[2] for item in observations]

    for phase_measurement, measurement_variance, _, index in observations[1:]:
        dt = max((index - previous_index) / float(sampling_rate_hz), 1.0 / float(sampling_rate_hz))
        transition = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = float(process_noise) * np.asarray(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=np.float64
        )
        state = transition @ state
        covariance = transition @ covariance @ transition.T + q
        innovation = wrap_phase(phase_measurement - float(state[0]))
        measurement_noise = max(float(measurement_variance), 1e-8)
        innovation_variance = float((measurement_matrix @ covariance @ measurement_matrix.T)[0, 0]) + measurement_noise
        gain = (covariance @ measurement_matrix.T) / max(innovation_variance, 1e-12)
        state = state + gain[:, 0] * innovation
        identity = np.eye(2, dtype=np.float64)
        residual_projection = identity - gain @ measurement_matrix
        covariance = (
            residual_projection @ covariance @ residual_projection.T
            + gain * measurement_noise @ gain.T
        )
        covariance = (covariance + covariance.T) / 2.0
        innovations.append(float(innovation))
        previous_index = index

    phase = wrap_phase(float(state[0]))
    phase_error = wrap_phase(phase - float(reference_phase_rad))
    phase_variance = max(0.0, float(covariance[0, 0]))
    phase_uncertainty = math.sqrt(phase_variance)
    return {
        "status": "success",
        "phase_estimate_rad": phase,
        "phase_rate_rad_s": float(state[1]),
        "phase_error_rad": phase_error,
        "phase_uncertainty_rad": phase_uncertainty,
        "phase_variance_rad2": phase_variance,
        "measurement_count": len(observations),
        "innovation_rms_rad": float(np.sqrt(np.mean(np.square(innovations)))) if innovations else 0.0,
        "residual_rms": float(np.sqrt(np.mean(np.square(residual_rms_values)))),
        "measurement_noise_floor_sigma": float(noise_sigma_floor),
        "failure_indicator": False,
        "estimator": "wrapped phase/rate EKF; constant-rate process model",
    }
