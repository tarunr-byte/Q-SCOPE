"""Connection layer for the seven Q-SCOPE agents.

This small explicit runner is intentionally deterministic and observable. It
keeps the ``ainvoke`` interface used by the API, but does not require a separate
workflow runtime to be installed. Every transition and every fail-closed
fallback is implemented here, so the service can still return a certificate if
an optional note-generation call or a non-critical stage fails.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agents.nodes.agent_1_nerve import agent_1_nerve
from agents.nodes.agent_2_kaiser import agent_2_kaiser
from agents.nodes.agent_3_kalman import agent_3_kalman
from agents.nodes.agent_4_orchestrator import agent_4_orchestrator
from agents.nodes.agent_5_verifier import agent_5_verifier
from agents.nodes.agent_6_hcqa import agent_6_hcqa
from agents.nodes.agent_7_output import agent_7_output
from agents.state import QScopeState, initialize_state

logger = logging.getLogger(__name__)
Node = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


def route_after_agent_1(state: dict[str, Any]) -> str:
    """Route invalid telemetry directly to the orchestrator for abstention."""
    return "agent_2_kaiser" if state.get("integrity", {}).get("valid", False) else "agent_4_orchestrator"


def route_after_agent_4(state: dict[str, Any]) -> str:
    """Enter quantum circuit generation only for a CONTINUE decision."""
    decision = state.get("decision", {}).get("decision", "ABSTAIN")
    return "agent_6_hcqa" if decision == "CONTINUE" else "agent_7_output"


def route_after_agent_5(state: dict[str, Any]) -> str:
    """Agent 7 always builds the final response after verification."""
    _ = state
    return "agent_7_output"


class QScopeEngine:
    """Run the seven agents in order with fail-closed stage boundaries."""

    async def _notify_progress(
        self,
        callback: ProgressCallback | None,
        name: str,
        state: dict[str, Any],
    ) -> None:
        """Emit a read-only snapshot hint without letting UI streaming affect analysis."""
        if callback is None:
            return
        try:
            await callback(name, dict(state))
        except Exception as exc:  # progress is observability, never a decision input
            logger.warning("Progress observer failed for %s (%s)", name, type(exc).__name__)

    async def _run_node(
        self,
        name: str,
        node: Node,
        state: dict[str, Any],
        fallback: dict[str, Any],
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Merge a node's state patch, substituting a safe patch on exceptions."""
        try:
            patch = await node(state)
            if not isinstance(patch, dict):
                raise TypeError("agent returned a non-dictionary state patch")
            state.update(patch)
        except Exception as exc:  # last-resort boundary around every node
            logger.exception("Agent %s failed (%s)", name, type(exc).__name__)
            state.update(fallback)
            capability_limits = list(state.get("capability_limits", []))
            capability_limits.append(f"{name} failed; a fail-closed fallback was used.")
            state["capability_limits"] = list(dict.fromkeys(capability_limits))
        await self._notify_progress(progress_callback, name, state)

    async def ainvoke(
        self,
        initial_state: dict[str, Any] | QScopeState,
        progress_callback: ProgressCallback | None = None,
    ) -> QScopeState:
        """Run the connected workflow and return its fully merged state.

        The method matches the common async graph invocation shape, allowing
        the HTTP route and unit tests to invoke the engine without knowing its
        internal orchestration details.
        """
        incoming = dict(initial_state)
        request = incoming.get("request", {})
        working: dict[str, Any] = dict(initialize_state(request, incoming.get("run_id", "")))
        # Preserve explicitly supplied non-request metadata while rebuilding
        # safe, fresh stage containers to avoid accidental state sharing.
        for key in ("run_id", "capability_limits", "cusum_score"):
            if key in incoming:
                working[key] = incoming[key]

        await self._run_node(
            "agent1_nerve",
            agent_1_nerve,
            working,
            {
                "integrity": {
                    "status": "invalid", "valid": False,
                    "issues": ["AGENT1_RUNTIME_ERROR"], "anomaly_flags": [],
                    "anomaly_detected": True, "sample_count": 0,
                    "median": 0.0, "robust_scale": 0.0,
                    "spike_ratio": 1.0, "flatline_ratio": 1.0,
                }
            },
            progress_callback,
        )

        await self._run_node(
            "agent2_kaiser",
            agent_2_kaiser,
            working,
            {"signal": {"status": "error", "issues": ["AGENT2_RUNTIME_ERROR"]}},
            progress_callback,
        )
        await self._run_node(
            "agent3_kalman",
            agent_3_kalman,
            working,
            {"estimate": {"status": "error", "failure_indicator": True, "reason": "AGENT3_RUNTIME_ERROR"}},
            progress_callback,
        )
        await self._run_node(
            "agent4_orchestrator",
            agent_4_orchestrator,
            working,
            {
                "decision": {
                    "decision": "ABSTAIN",
                    "reason_codes": ["ORCHESTRATOR_RUNTIME_ERROR"],
                    "fail_closed": True,
                    "rule_version": "qscope-policy-1.0.0",
                }
            },
            progress_callback,
        )

        decision = working.get("decision", {}).get("decision", "ABSTAIN")
        if decision == "CONTINUE":
            await self._run_node(
                "agent6_hcqa",
                agent_6_hcqa,
                working,
                {"circuit": {"status": "error", "reason": "AGENT6_RUNTIME_ERROR"}},
                progress_callback,
            )
            await self._run_node(
                "agent5_verifier",
                agent_5_verifier,
                working,
                {
                    "quantum": {
                        "status": "fail_closed", "verification_status": "error",
                        "hardware_status": "NOT_CONNECTED", "reason": "AGENT5_RUNTIME_ERROR",
                    },
                    "decision": {
                        "decision": "ABSTAIN",
                        "reason_codes": ["QUANTUM_VERIFICATION_FAILED"],
                        "fail_closed": True,
                        "rule_version": "qscope-policy-1.0.0",
                    },
                },
                progress_callback,
            )
        else:
            working["circuit"] = {
                "status": "skipped",
                "reason": "ORCHESTRATOR_DID_NOT_CONTINUE",
            }
            working["quantum"] = {
                "status": "skipped",
                "verification_status": "not_evaluated",
                "hardware_status": "NOT_CONNECTED",
            }
            await self._notify_progress(progress_callback, "agent6_hcqa", working)
            await self._notify_progress(progress_callback, "agent5_verifier", working)

        await self._run_node(
            "agent7_output",
            agent_7_output,
            working,
            {
                "final_payload": {
                    "overall_status": "ABSTAIN",
                    "reason_codes": ["CERTIFICATE_BUILD_FAILED"],
                    "dimensions": {},
                    "scope": "software diagnostics only; certificate generation failed",
                }
            },
            progress_callback,
        )
        working["api_status"] = 200
        return working  # type: ignore[return-value]


def build_qscope_graph() -> QScopeEngine:
    """Construct a fresh runner, suitable for an application lifespan."""
    logger.info("Q-SCOPE seven-agent pipeline initialized")
    return QScopeEngine()


# Compatibility singleton for scripts that historically imported this symbol.
qscope_engine = build_qscope_graph()
