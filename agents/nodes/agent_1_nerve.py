"""Agent 1 — deterministic telemetry integrity and anomaly screening."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from agents.models import ask_model
from app.core.config import MAX_WAVEFORM_SAMPLES


def _check_basic_shape(samples: np.ndarray) -> list[str]:
    """Validate array shape, length, finiteness and configured resource bounds."""
    issues: list[str] = []
    if samples.ndim != 1:
        issues.append("SAMPLES_NOT_1D")
    if samples.size < 16:
        issues.append("INSUFFICIENT_SAMPLES")
    if samples.size > MAX_WAVEFORM_SAMPLES:
        issues.append("SAMPLE_LIMIT_EXCEEDED")
    if not np.isfinite(samples).all():
        issues.append("NONFINITE_SAMPLE")
    return issues


def _check_metadata(request: dict[str, Any]) -> list[str]:
    """Check acquisition metadata independently of Pydantic's request checks."""
    issues: list[str] = []
    if not request.get("calibration_valid", False):
        issues.append("CALIBRATION_INVALID")
    required_fields = ("shot_id", "sensor_trace_id", "calibration_version")
    if not all(str(request.get(field, "")).strip() for field in required_fields):
        issues.append("SHOT_LINKAGE_INCOMPLETE")

    interval = request.get("population_ratio_interval_s", ())
    try:
        if len(interval) != 2 or not all(math.isfinite(float(v)) for v in interval):
            raise ValueError
        if float(interval[0]) >= float(interval[1]):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        issues.append("INVALID_MEASUREMENT_INTERVAL")

    pulses = request.get("laser_pulse_times_s", ())
    try:
        if len(pulses) != 3 or not all(math.isfinite(float(v)) for v in pulses):
            raise ValueError
        if not (float(pulses[0]) < float(pulses[1]) < float(pulses[2])):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        issues.append("INVALID_LASER_PULSE_TIMES")
    return issues


def _anomaly_features(samples: np.ndarray) -> dict[str, Any]:
    """Return robust aggregate statistics without retaining or transmitting raw data."""
    if samples.size == 0 or not np.isfinite(samples).all():
        return {
            "median": 0.0,
            "robust_scale": 0.0,
            "spike_ratio": 1.0,
            "flatline_ratio": 1.0,
            "anomaly_flags": ["NO_VALID_DATA"],
        }
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    scale = max(1.4826 * mad, float(np.std(samples)) * 1e-6, 1e-12)
    spike_ratio = float(np.mean(np.abs(samples - median) > 8.0 * scale))
    flatline_ratio = float(np.mean(np.abs(np.diff(samples)) <= 1e-12)) if samples.size > 1 else 1.0
    flags: list[str] = []
    if spike_ratio > 0.01:
        flags.append("ROBUST_SPIKE_RATE_HIGH")
    if flatline_ratio > 0.95:
        flags.append("POSSIBLE_FLATLINE")
    return {
        "median": median,
        "robust_scale": scale,
        "spike_ratio": spike_ratio,
        "flatline_ratio": flatline_ratio,
        "anomaly_flags": flags,
    }


async def agent_1_nerve(state: dict[str, Any]) -> dict[str, Any]:
    """Gate entry into numerical processing using deterministic integrity checks."""
    request = state.get("request", {})
    try:
        samples = np.asarray(request.get("samples", []), dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        samples = np.asarray([], dtype=np.float64)
    issues = _check_basic_shape(samples) + _check_metadata(request)
    features = _anomaly_features(samples)
    # A near-flat trace is not usable for phase extraction. Spike anomalies are
    # surfaced but not automatically rejected unless their rate is extreme.
    if "POSSIBLE_FLATLINE" in features["anomaly_flags"]:
        issues.append("ACQUISITION_FLATLINE")
    if features["spike_ratio"] > 0.20:
        issues.append("ACQUISITION_SPIKE_RATE_CRITICAL")
    issues = list(dict.fromkeys(issues))
    integrity = {
        "status": "valid" if not issues else "invalid",
        "valid": not issues,
        "issues": issues,
        "anomaly_flags": features["anomaly_flags"],
        "anomaly_detected": bool(features["anomaly_flags"]),
        "sample_count": int(samples.size),
        "median": features["median"],
        "robust_scale": features["robust_scale"],
        "spike_ratio": features["spike_ratio"],
        "flatline_ratio": features["flatline_ratio"],
    }

    notes = dict(state.get("llm_notes", {}))
    note = await ask_model(
        "agent1",
        "You summarize acquisition diagnostics. Never decide validity or invent causes.",
        (
            f"valid={integrity['valid']}; issues={issues}; flags={features['anomaly_flags']}; "
            f"sample_count={samples.size}; spike_ratio={features['spike_ratio']:.6g}; "
            f"flatline_ratio={features['flatline_ratio']:.6g}"
        ),
        max_tokens=100,
    )
    if note:
        notes["agent1"] = note.strip()
    return {"integrity": integrity, "llm_notes": notes}
