"""Agent 4 — deterministic decision policy and fail-closed routing."""
from __future__ import annotations

from typing import Any

from agents.models import ask_model


async def agent_4_orchestrator(state: dict[str, Any]) -> dict[str, Any]:
    """Decide whether numerical processing may proceed to the circuit stage.

    Advisory model text is intentionally unable to change the decision, reason
    codes, measured phase, uncertainty threshold or graph route.
    """
    integrity = state.get("integrity", {})
    signal = state.get("signal", {})
    estimate = state.get("estimate", {})
    reasons: list[str] = []
    decision = "CONTINUE"

    if not integrity.get("valid", False):
        decision = "ABSTAIN"
        reasons.append("TELEMETRY_INVALID")
        reasons.extend(str(item) for item in integrity.get("issues", []))
    elif signal.get("status") != "success":
        decision = "ABSTAIN"
        reasons.append("SIGNAL_PROCESSING_FAILED")
    elif estimate.get("status") != "success":
        decision = "ABSTAIN"
        reasons.append("PHASE_ESTIMATION_FAILED")
    elif estimate.get("uncertainty_exceeded", False):
        decision = "ABSTAIN"
        reasons.append("PHASE_UNCERTAINTY_EXCEEDED")
    elif estimate.get("phase_error_exceeded", False):
        decision = "ESCALATE_NONLINEAR"
        reasons.append("PHASE_ERROR_EXCEEDED")

    if not reasons:
        reasons.append("DETERMINISTIC_CHECKS_PASSED")
    result = {
        "decision": decision,
        "reason_codes": list(dict.fromkeys(reasons)),
        "fail_closed": decision != "CONTINUE",
        "rule_version": "qscope-policy-1.0.0",
    }
    notes = dict(state.get("llm_notes", {}))
    note = await ask_model(
        "agent4",
        "Summarize this already-determined policy decision. Do not change it.",
        f"decision={decision}; reason_codes={result['reason_codes']}",
        max_tokens=100,
    )
    if note:
        notes["agent4"] = note.strip()
    return {"decision": result, "llm_notes": notes}
