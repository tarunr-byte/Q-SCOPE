"""Typed state contract shared by the seven connected Q-SCOPE agents."""
from __future__ import annotations

from typing import Any, TypedDict


class QScopeState(TypedDict, total=False):
    """Mutable pipeline state; every node returns only the fields it updates."""

    run_id: str
    request: dict[str, Any]
    integrity: dict[str, Any]
    signal: dict[str, Any]
    estimate: dict[str, Any]
    decision: dict[str, Any]
    quantum: dict[str, Any]
    circuit: dict[str, Any]
    final_payload: dict[str, Any] | None
    api_status: int | None
    llm_notes: dict[str, str]
    capability_limits: list[str]
    cusum_score: float


def initialize_state(raw_request: dict[str, Any], run_id: str = "") -> QScopeState:
    """Create a clean state for one request without sharing mutable objects."""
    return {
        "run_id": run_id,
        "request": dict(raw_request),
        "integrity": {"status": "pending", "valid": False, "issues": []},
        "signal": {"status": "pending"},
        "estimate": {"status": "pending"},
        "decision": {"decision": "ABSTAIN", "reason_codes": ["PIPELINE_NOT_RUN"], "fail_closed": True},
        "quantum": {"status": "pending"},
        "circuit": {"status": "pending"},
        "final_payload": None,
        "api_status": None,
        "llm_notes": {},
        "capability_limits": [
            "The quantum component is an exact two-qubit software demonstrator, not a hardware certification."
        ],
        "cusum_score": 0.0,
    }
