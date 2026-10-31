"""Unit tests for the numerical components used by the agent chain."""
from __future__ import annotations

import math

import numpy as np
import pytest

from agents.nodes.agent_6_hcqa import search_best_sequence
from physics.kaiser_filter import apply_kaiser, kaiser
from physics.quantum_engine import evaluate_sequence


def test_kaiser_window_is_not_a_placeholder() -> None:
    window = kaiser(32, 6.0)
    assert window.shape == (32,)
    assert window[0] < window[16]
    assert np.all(window >= 0.0)
    assert np.all(window <= 1.0)


def test_kaiser_rejects_nonfinite_samples() -> None:
    with pytest.raises(ValueError, match="finite"):
        apply_kaiser([0.0, float("nan"), 1.0], 6.0, 1000.0)


def test_circuit_search_finds_target_and_verifier_invariants_hold() -> None:
    result = search_best_sequence((1.0, 1.0), max_depth=4, target_normalized_qfi=0.95)
    assert result["status"] == "success"
    evaluation = evaluate_sequence(phase_rad=0.2, coefficients=(1.0, 1.0), sequence=result["sequence"])
    assert math.isclose(evaluation["probability_sum"], 1.0, abs_tol=1e-8)
    assert evaluation["normalized_qfi"] >= 0.95
    assert evaluation["quantum_advantage_claimed"] is False
