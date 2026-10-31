"""Agent 7 — final four-dimensional software health certificate."""
from __future__ import annotations

import math
from typing import Any

from agents.models import ask_model


def _clamp01(value: float) -> float:
    """Clamp finite numeric scores to the inclusive unit interval."""
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))



def _graph_diagnostics(request: dict[str, Any]) -> dict[str, Any]:
    """Summarize graph and sensitivity inputs without claiming trained inference."""
    nodes = request.get("pulse_graph_nodes", [])
    edges = request.get("pulse_graph_edges", [])
    sensitivity = [float(item) for item in request.get("sensitivity_function", [])]
    node_count = len(nodes)
    edge_count = len(edges)
    possible_edges = max(node_count * max(node_count - 1, 1), 1)
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "directed_edge_density": _clamp01(edge_count / possible_edges),
        "sensitivity_sample_count": len(sensitivity),
        "sensitivity_min": min(sensitivity) if sensitivity else None,
        "sensitivity_max": max(sensitivity) if sensitivity else None,
        "sensitivity_mean": sum(sensitivity) / len(sensitivity) if sensitivity else None,
        "sensitivity_range": (max(sensitivity) - min(sensitivity)) if sensitivity else None,
        "learned_graph_model_used": False,
        "interpretation": "Input structure diagnostics only; not a learned machine-condition prediction.",
    }


async def agent_7_output(state: dict[str, Any]) -> dict[str, Any]:
    """Build stable certificate dimensions and a scope-aware overall summary."""
    request = state.get("request", {})
    integrity = state.get("integrity", {})
    signal = state.get("signal", {})
    estimate = state.get("estimate", {})
    decision = state.get("decision", {})
    quantum = state.get("quantum", {})
    signal_metrics = signal.get("metrics", {})

    acquisition_score = 1.0 if integrity.get("valid", False) else 0.0
    leakage = float(signal_metrics.get("spectral_leakage_ratio_outside_peak_bins", 1.0))
    signal_score = _clamp01(1.0 - leakage) if signal.get("status") == "success" else 0.0

    phase_error = abs(float(estimate.get("phase_error_rad", math.pi)))
    phase_tolerance = max(float(request.get("phase_tolerance_rad", 0.1)), 1e-12)
    uncertainty = float(estimate.get("phase_uncertainty_rad", math.pi))
    uncertainty_limit = max(float(request.get("max_phase_uncertainty_rad", 0.15)), 1e-12)
    phase_score = _clamp01(1.0 - max(phase_error / phase_tolerance, uncertainty / uncertainty_limit))
    if estimate.get("status") != "success":
        phase_score = 0.0

    qfi_score = quantum.get("normalized_qfi")
    qfi_target = float(request.get("target_normalized_qfi", 0.95))
    quantum_evaluated = quantum.get("verification_status") == "passed" and isinstance(qfi_score, (int, float))
    quantum_score = _clamp01(float(qfi_score)) if quantum_evaluated else 0.0

    dimensions = {
        "acquisition_integrity": {
            "status": "PASS" if acquisition_score == 1.0 else "FAIL",
            "score": acquisition_score,
            "metrics": {
                "sample_count": integrity.get("sample_count", 0),
                "spike_ratio": integrity.get("spike_ratio", 1.0),
                "flatline_ratio": integrity.get("flatline_ratio", 1.0),
                "anomaly_flags": integrity.get("anomaly_flags", []),
            },
        },
        "signal_taper_diagnostics": {
            "status": (
                "PASS" if leakage <= 0.20 else "CHECK"
            ) if signal.get("status") == "success" else "NOT_EVALUATED",
            "score": signal_score,
            "metric_definition": "1 - spectral leakage ratio outside the dominant peak band; not an SNR estimate.",
            "metrics": signal_metrics,
        },
        "phase_stability": {
            "status": "PASS" if estimate.get("status") == "success" and not estimate.get("failure_indicator", True) else "FAIL",
            "score": phase_score,
            "metrics": {
                "phase_estimate_rad": estimate.get("phase_estimate_rad"),
                "phase_error_rad": estimate.get("phase_error_rad"),
                "phase_uncertainty_rad": estimate.get("phase_uncertainty_rad"),
                "phase_tolerance_rad": phase_tolerance,
                "max_phase_uncertainty_rad": uncertainty_limit,
            },
        },
        "quantum_readiness": {
            "status": "PASS" if quantum_evaluated and quantum_score + 1e-8 >= qfi_target else (
                "FAIL" if quantum.get("verification_status") == "rejected" else "NOT_EVALUATED"
            ),
            "score": quantum_score,
            "target_normalized_qfi": qfi_target,
            "simulation_verified": bool(quantum_evaluated),
            "hardware_status": quantum.get("hardware_status", "NOT_CONNECTED"),
            "quantum_advantage_claimed": False,
        },
    }
    final = {
        "overall_status": decision.get("decision", "ABSTAIN"),
        "reason_codes": list(decision.get("reason_codes", ["DECISION_MISSING"])),
        "dimensions": dimensions,
        "graph_diagnostics": _graph_diagnostics(request),
        "scope": "software diagnostics only; neither hardware health nor physical quantum advantage is certified",
    }

    notes = dict(state.get("llm_notes", {}))
    note = await ask_model(
        "agent7",
        "Summarize this software diagnostic certificate in two sentences. Do not change metrics or decisions.",
        (
            f"decision={final['overall_status']}; reasons={final['reason_codes']}; "
            f"integrity={dimensions['acquisition_integrity']['status']}; "
            f"phase={dimensions['phase_stability']['status']}; "
            f"quantum={dimensions['quantum_readiness']['status']}"
        ),
        max_tokens=120,
    )
    if note:
        notes["agent7"] = note.strip()
    return {"final_payload": final, "llm_notes": notes}
