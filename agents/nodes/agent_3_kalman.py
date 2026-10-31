"""Agent 3 — wrapped phase and phase-rate estimation."""
from __future__ import annotations

from typing import Any

from physics.kalman_filter import estimate_phase


async def agent_3_kalman(state: dict[str, Any]) -> dict[str, Any]:
    """Estimate phase from raw samples using Agent 2's taper as fit weights."""
    request = state.get("request", {})
    signal = state.get("signal", {})
    if signal.get("status") != "success":
        return {"estimate": {"status": "skipped", "reason": "SIGNAL_STAGE_NOT_SUCCESSFUL"}}
    adjusted_reference = float(request.get("phase_reference_rad", 0.0)) + float(
        request.get("calibration_offset_rad", 0.0)
    )
    estimate = estimate_phase(
        samples=request.get("samples", []),
        sampling_rate_hz=float(request["sampling_rate_hz"]),
        frequency_hz=float(request["carrier_frequency_hz"]),
        beta=float(request.get("kaiser_beta", 6.0)),
        reference_phase_rad=adjusted_reference,
        window_weights=signal.get("window"),
        noise_sigma_floor=float(request.get("noise_sigma", 0.0)),
    )
    phase_error = abs(float(estimate.get("phase_error_rad", 0.0)))
    uncertainty = float(estimate.get("phase_uncertainty_rad", float("inf")))
    phase_tolerance = float(request.get("phase_tolerance_rad", 0.10))
    max_uncertainty = float(request.get("max_phase_uncertainty_rad", 0.15))
    uncertainty_exceeded = not uncertainty <= max_uncertainty
    phase_exceeded = not phase_error <= phase_tolerance
    estimate.update(
        {
            "status": "success",
            "variance": estimate.get("phase_variance_rad2"),
            "max_innovation_rad": estimate.get("innovation_rms_rad"),
            "phase_error_exceeded": phase_exceeded,
            "uncertainty_exceeded": uncertainty_exceeded,
            "failure_indicator": phase_exceeded or uncertainty_exceeded,
            "phase_tolerance_rad": phase_tolerance,
            "max_phase_uncertainty_rad": max_uncertainty,
            "calibration_offset_applied_rad": float(request.get("calibration_offset_rad", 0.0)),
        }
    )
    return {"estimate": estimate}
