"""Single source of truth for the responsibilities of the seven agents."""
from __future__ import annotations

from typing import TypedDict


class AgentDefinition(TypedDict):
    agent_id: str
    name: str
    role: str
    description: str
    llm_optional: bool


AGENT_REGISTRY: tuple[AgentDefinition, ...] = (
    {
        "agent_id": "agent1",
        "name": "Nerve / Integrity Gate",
        "role": "telemetry_integrity",
        "description": "Checks identifiers, calibration flags, sample integrity and robust anomalies.",
        "llm_optional": True,
    },
    {
        "agent_id": "agent2",
        "name": "Kaiser Receiver",
        "role": "signal_windowing",
        "description": "Applies a real Kaiser taper and calculates spectral diagnostics.",
        "llm_optional": False,
    },
    {
        "agent_id": "agent3",
        "name": "Phase / Kalman Estimator",
        "role": "phase_estimation",
        "description": "Estimates wrapped phase, rate, uncertainty and innovations.",
        "llm_optional": False,
    },
    {
        "agent_id": "agent4",
        "name": "Fail-Closed Orchestrator",
        "role": "decision_policy",
        "description": "Makes deterministic routing decisions from integrity and estimator outputs.",
        "llm_optional": True,
    },
    {
        "agent_id": "agent5",
        "name": "Quantum Verifier",
        "role": "simulation_verification",
        "description": "Checks state-vector probabilities and QFI against deterministic invariants.",
        "llm_optional": False,
    },
    {
        "agent_id": "agent6",
        "name": "HCQA Circuit Search",
        "role": "circuit_search",
        "description": "Runs bounded, deterministic two-qubit gate search for the configured generator.",
        "llm_optional": False,
    },
    {
        "agent_id": "agent7",
        "name": "Certificate Builder",
        "role": "certificate_output",
        "description": "Builds the four software-diagnostic dimensions and graph statistics.",
        "llm_optional": True,
    },
)


def agent_definition(agent_id: str) -> AgentDefinition | None:
    """Get a defensive copy of the static definition for one agent ID."""
    for definition in AGENT_REGISTRY:
        if definition["agent_id"] == agent_id:
            return dict(definition)
    return None
