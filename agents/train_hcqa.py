"""
train_hcqa.py
=============
Offline training script for Agent-6's NumpyDQN.

This script generates randomized signal coefficients (simulating different
HGV flight regimes), runs the DQN environment loop, calculates the reward
based on QFI improvement, and saves the trained weights to HCQA_CHECKPOINT.
"""
from __future__ import annotations

import os
import logging
import argparse
from pathlib import Path
import numpy as np

from agents.dqn import NumpyDQN, train_dqn
from app.core.config import HCQA_CHECKPOINT
from physics.quantum_engine import ACTIONS, apply_action, normalized_qfi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_dqn(model: NumpyDQN, episodes: int = 50) -> float:
    """Evaluates the trained DQN without exploration (epsilon=0)."""
    total_qfi = 0.0
    rng = np.random.default_rng(42)

    for _ in range(episodes):
        coefficients = tuple(float(v) for v in rng.uniform(-1.0, 1.0, 2))
        if max(abs(coefficients[0]), abs(coefficients[1])) < 0.1:
            coefficients = (1.0, 1.0)

        state = np.array([1, 0, 0, 0], dtype=np.complex128)
        _, _, max_qfi = normalized_qfi(state, coefficients)

        for depth in range(6):
            obs = model.features(state, depth, 6, coefficients)
            action = int(np.argmax(model.q_values(obs)[0]))
            state = apply_action(state, ACTIONS[action])
            _, _, qfi = normalized_qfi(state, coefficients)
            max_qfi = max(max_qfi, qfi)

        total_qfi += max_qfi

    return total_qfi / episodes


def main():
    parser = argparse.ArgumentParser(description="Train the HCQA DQN for Agent-6")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum circuit depth")
    parser.add_argument("--output", type=str, default=str(HCQA_CHECKPOINT), help="Path to save checkpoint")
    args = parser.parse_args()

    logger.info("🚀 Starting HCQA DQN Training...")
    logger.info(f"Episodes: {args.episodes}, Max Depth: {args.max_depth}")

    # Train the model
    policy = train_dqn(episodes=args.episodes, max_depth=args.max_depth)

    # Evaluate the model
    logger.info("📊 Evaluating trained policy...")
    avg_qfi = evaluate_dqn(policy, episodes=100)
    logger.info(f"✅ Average QFI after training: {avg_qfi:.4f}")

    if avg_qfi < 0.90:
        logger.warning("⚠️ Average QFI is low. Consider increasing training episodes or tuning hyperparameters.")

    # Save the model
    output_path = Path(args.output)
    policy.save(output_path)
    logger.info(f"💾 Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()