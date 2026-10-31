"""Agent 6 — bounded beam search for the two-qubit HCQA demonstrator."""
from __future__ import annotations

from typing import Any

import numpy as np

from physics.quantum_engine import ACTIONS, apply_action, normalized_qfi

BEAM_WIDTH = 32


def _state_key(state: np.ndarray) -> tuple[float, ...]:
    """Round amplitudes for deterministic state de-duplication."""
    flattened = np.concatenate((state.real, state.imag))
    return tuple(float(value) for value in np.round(flattened, 8))


def _score(state: np.ndarray, coefficients: tuple[float, float]) -> float:
    """Return normalized QFI, clamped against numerical roundoff."""
    _, _, score = normalized_qfi(state, coefficients)
    return float(min(1.0, max(0.0, score)))


def search_best_sequence(
    coefficients: tuple[float, float],
    max_depth: int,
    target_normalized_qfi: float,
) -> dict[str, Any]:
    """Search a bounded, deterministic set of candidate circuits.

    This is deliberately independent of LLMs and hardware. Each state vector
    is evaluated exactly by the local two-qubit simulator. The beam is capped
    at a fixed width, so worst-case work is bounded by depth * beam * actions.
    """
    coefficient_pair = (float(coefficients[0]), float(coefficients[1]))
    if not all(np.isfinite(coefficient_pair)):
        raise ValueError("signal coefficients must be finite")
    if max_depth < 1 or max_depth > 8:
        raise ValueError("max_depth must be in [1, 8]")
    if not 0.0 < target_normalized_qfi <= 1.0:
        raise ValueError("target_normalized_qfi must be in (0, 1]")

    initial = np.asarray([1.0 + 0j, 0j, 0j, 0j], dtype=np.complex128)
    beam: list[tuple[np.ndarray, list[str], float]] = [(initial, [], _score(initial, coefficient_pair))]
    global_best = beam[0]
    explored_count = 1
    target_hit_at_depth: int | None = None

    for depth in range(1, max_depth + 1):
        expanded: dict[tuple[float, ...], tuple[np.ndarray, list[str], float]] = {}
        for current_state, sequence, _ in beam:
            for action in ACTIONS:
                candidate_state = apply_action(current_state, action)
                candidate_sequence = [*sequence, action]
                candidate_score = _score(candidate_state, coefficient_pair)
                explored_count += 1
                key = _state_key(candidate_state)
                candidate = (candidate_state, candidate_sequence, candidate_score)
                previous = expanded.get(key)
                if previous is None or candidate_score > previous[2] + 1e-12:
                    expanded[key] = candidate

        candidates = sorted(
            expanded.values(),
            key=lambda item: (item[2], -len(item[1]), tuple(item[1])),
            reverse=True,
        )
        if candidates and candidates[0][2] > global_best[2] + 1e-12:
            global_best = candidates[0]
        target_candidates = [item for item in candidates if item[2] + 1e-12 >= target_normalized_qfi]
        if target_candidates:
            global_best = sorted(
                target_candidates,
                key=lambda item: (len(item[1]), -item[2], tuple(item[1])),
            )[0]
            target_hit_at_depth = depth
            break
        beam = candidates[:BEAM_WIDTH]
        if not beam:
            break

    sequence = list(global_best[1])
    score = float(global_best[2])
    return {
        "status": "success" if score + 1e-12 >= target_normalized_qfi else "target_not_met",
        "sequence": sequence,
        "predicted_normalized_qfi": score,
        "target_normalized_qfi": float(target_normalized_qfi),
        "depth": len(sequence),
        "max_depth": int(max_depth),
        "search_algorithm": "bounded deterministic beam search",
        "beam_width": BEAM_WIDTH,
        "states_explored": explored_count,
        "target_hit_at_depth": target_hit_at_depth,
        "checkpoint_used": False,
        "model_scope": "exact two-qubit state-vector search; not a physical hardware circuit",
    }


async def agent_6_hcqa(state: dict[str, Any]) -> dict[str, Any]:
    """Build a candidate preparation circuit only after Agent 4 allows it."""
    decision = state.get("decision", {})
    if decision.get("decision") != "CONTINUE" or decision.get("fail_closed", True):
        return {"circuit": {"status": "skipped", "reason": "ORCHESTRATOR_DID_NOT_CONTINUE"}}
    request = state.get("request", {})
    try:
        circuit = search_best_sequence(
            coefficients=tuple(request.get("signal_coefficients", (1.0, 1.0))),
            max_depth=int(request.get("max_gate_depth", 4)),
            target_normalized_qfi=float(request.get("target_normalized_qfi", 0.95)),
        )
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        circuit = {
            "status": "error",
            "reason": "CIRCUIT_SEARCH_FAILED",
            "model_scope": "no circuit accepted",
        }
    notes = dict(state.get("llm_notes", {}))
    notes["agent6"] = (
        f"Circuit search {circuit.get('status')}; depth={circuit.get('depth', 0)}; "
        f"normalized_qfi={circuit.get('predicted_normalized_qfi', 0.0):.6g}."
    )
    return {"circuit": circuit, "llm_notes": notes}
