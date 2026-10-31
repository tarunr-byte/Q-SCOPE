"""Agent 5 — exact state-vector circuit verification and fail-closed tripwire."""
from __future__ import annotations

import math
from typing import Any

from physics.quantum_engine import evaluate_sequence


def _decision_after_verification(
    decision: dict[str, Any], reason_code: str
) -> dict[str, Any]:
    """Promote a verifier failure to an explicit fail-closed decision."""
    result = dict(decision)
    reasons = list(result.get("reason_codes", []))
    if reason_code not in reasons:
        reasons.append(reason_code)
    result.update({"decision": "ABSTAIN", "reason_codes": reasons, "fail_closed": True})
    return result


async def agent_5_verifier(state: dict[str, Any]) -> dict[str, Any]:
    """Recompute circuit probabilities/QFI and verify mathematical invariants.

    This agent reports software simulation results only. It has no connection to
    a QPU, and it never assigns a fabricated hardware fidelity or PASS label.
    """
    decision = dict(state.get("decision", {}))
    if decision.get("decision") != "CONTINUE" or decision.get("fail_closed", True):
        return {"quantum": {"status": "skipped", "hardware_status": "NOT_CONNECTED"}}

    circuit = state.get("circuit", {})
    request = state.get("request", {})
    if circuit.get("status") != "success":
        return {
            "quantum": {
                "status": "fail_closed",
                "verification_status": "rejected",
                "hardware_status": "NOT_CONNECTED",
                "reason": "CIRCUIT_TARGET_NOT_MET",
            },
            "decision": _decision_after_verification(decision, "CIRCUIT_TARGET_NOT_MET"),
        }

    try:
        evaluation = evaluate_sequence(
            phase_rad=float(state.get("estimate", {}).get("phase_estimate_rad", 0.0)),
            coefficients=tuple(request.get("signal_coefficients", (1.0, 1.0))),
            sequence=list(circuit.get("sequence", [])),
        )
        probabilities = [float(value) for value in evaluation["probabilities"]]
        probability_sum = float(sum(probabilities))
        normalized_qfi = float(evaluation["normalized_qfi"])
        target = float(request.get("target_normalized_qfi", 0.95))
        finite = all(math.isfinite(value) for value in probabilities + [normalized_qfi, probability_sum])
        probabilities_valid = all(value >= -1e-10 for value in probabilities) and abs(probability_sum - 1.0) <= 1e-8
        qfi_valid = -1e-10 <= normalized_qfi <= 1.0 + 1e-8
        target_met = normalized_qfi + 1e-8 >= target
        passed = finite and probabilities_valid and qfi_valid and target_met
        quantum = {
            "status": "success" if passed else "fail_closed",
            "verification_status": "passed" if passed else "rejected",
            "hardware_status": "NOT_CONNECTED",
            "probabilities": probabilities,
            "probability_sum": probability_sum,
            "probabilities_valid": probabilities_valid,
            "qfi": float(evaluation["qfi"]),
            "qfi_limit": float(evaluation["qfi_limit"]),
            "normalized_qfi": normalized_qfi,
            "target_normalized_qfi": target,
            "target_met": target_met,
            "measurement_classical_fisher_information": float(
                evaluation["measurement_classical_fisher_information"]
            ),
            "sequence": list(evaluation["sequence"]),
            "quantum_advantage_claimed": False,
            "model_scope": evaluation["model_scope"],
        }
        if not passed:
            quantum["reason"] = "SIMULATION_INVARIANT_OR_QFI_TARGET_FAILED"
            return {
                "quantum": quantum,
                "decision": _decision_after_verification(
                    decision, "QUANTUM_VERIFICATION_FAILED"
                ),
            }
        return {"quantum": quantum}
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError):
        return {
            "quantum": {
                "status": "fail_closed",
                "verification_status": "error",
                "hardware_status": "NOT_CONNECTED",
                "reason": "SIMULATION_VERIFICATION_ERROR",
            },
            "decision": _decision_after_verification(
                decision, "QUANTUM_VERIFICATION_FAILED"
            ),
        }
