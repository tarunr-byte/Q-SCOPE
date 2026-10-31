"""Agent 2 — real Kaiser taper and transparent spectral metrics."""
from __future__ import annotations

from physics.kaiser_filter import apply_kaiser


async def agent_2_kaiser(state: dict[str, Any]) -> dict[str, Any]:
    """Window valid raw telemetry; the stage deliberately does not claim denoising."""
    request = state.get("request", {})
    if not state.get("integrity", {}).get("valid", False):
        return {"signal": {"status": "skipped_invalid_input", "issues": ["INTEGRITY_GATE_FAILED"]}}
    try:
        result = apply_kaiser(
            request.get("samples", []),
            beta=float(request.get("kaiser_beta", 6.0)),
            sampling_rate_hz=float(request.get("sampling_rate_hz", 1000.0)),
        )
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        return {"signal": {"status": "error", "issues": ["KAISER_PROCESSING_FAILED"], "detail": str(exc)}}
    return {
        "signal": {
            "status": "success",
            "filtered_samples": result["samples"],
            "window": result["window"],
            "metrics": result["metrics"],
            "note": result["note"],
        }
    }
