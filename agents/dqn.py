"""
agents/dqn.py
=============
Small NumPy DQN for discrete two-qubit preparation circuits.

This module provides the Deep Q-Network used by Agent-6 (HCQA Integrator).
It is a lightweight, purely NumPy-based implementation designed for fast
inference during real-time HGV flight operations.

The objective of the DQN is to select a sequence of quantum gates that
maximizes the normalized pure-state QFI (Quantum Fisher Information) while
minimizing the circuit depth (gate count).
"""
from __future__ import annotations

import json
import random
import logging
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np

# Internal imports
from physics.quantum_engine import ACTIONS, apply_action, normalized_qfi

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
# INPUT_SIZE = 4 (state.real) + 4 (state.imag) + 4 (metadata) = 12
INPUT_SIZE = 12
HIDDEN_SIZE = 48
ACTION_COUNT = len(ACTIONS)


class NumpyDQN:
    """
    A 2-layer Multi-Layer Perceptron (MLP) acting as the Q-function approximator.
    Contains the policy network, target network, and experience replay buffer.
    """

    def __init__(self, seed: int = 7):
        rng = np.random.default_rng(seed)

        # Policy Network Weights (He initialization)
        self.w1 = rng.normal(0, np.sqrt(2.0 / INPUT_SIZE), (INPUT_SIZE, HIDDEN_SIZE))
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.w2 = rng.normal(0, np.sqrt(2.0 / HIDDEN_SIZE), (HIDDEN_SIZE, ACTION_COUNT))
        self.b2 = np.zeros(ACTION_COUNT)

        # Target Network Weights
        self.t_w1, self.t_b1 = self.w1.copy(), self.b1.copy()
        self.t_w2, self.t_b2 = self.w2.copy(), self.b2.copy()

        # Experience Replay
        self.memory = deque(maxlen=12000)

        # Hyperparameters
        self.gamma = 0.97
        self.learning_rate = 0.001

    @staticmethod
    def features(
            state: np.ndarray,
            depth: int,
            max_depth: int,
            coefficients: Tuple[float, ...],
            uncertainty: float = 0.0
    ) -> np.ndarray:
        """
        Constructs the 12-dimensional observation vector for the DQN.

        Args:
            state: The 4-dimensional complex quantum state vector.
            depth: Current circuit depth (number of gates applied).
            max_depth: Maximum allowed circuit depth.
            coefficients: Signal coefficients from Agent-3.
            uncertainty: Phase variance from Agent-3.

        Returns:
            A 1D NumPy array of shape (12,).
        """
        coeff = np.asarray(coefficients, dtype=np.float64)
        scale = max(float(np.max(np.abs(coeff))), 1e-12)

        # Normalize inputs to [-1, 1] or [0, 1] for stable neural network training
        return np.concatenate((
            state.real,  # 4 dims
            state.imag,  # 4 dims
            [
                depth / max(max_depth, 1),
                coeff[0] / scale,
                coeff[1] / scale,
                min(max(float(uncertainty), 0.0), np.pi) / np.pi
            ]  # 4 dims
        )).astype(np.float64)

    def q_values(self, observations: np.ndarray, target: bool = False) -> np.ndarray:
        """
        Forward pass to calculate Q-values for a given observation.
        Agent-6 calls this with `model.q_values(obs)[0]` to pick the best action.
        """
        x = np.atleast_2d(np.asarray(observations, dtype=np.float64))

        if target:
            w1, b1, w2, b2 = self.t_w1, self.t_b1, self.t_w2, self.t_b2
        else:
            w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2

        hidden = np.maximum(0.0, x @ w1 + b1)  # ReLU activation
        return hidden @ w2 + b2

    def choose(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection (used during training)."""
        if random.random() < epsilon:
            return random.randrange(ACTION_COUNT)
        return int(np.argmax(self.q_values(observation)[0]))

    def remember(self, transition: Tuple[np.ndarray, int, float, np.ndarray, float]) -> None:
        """Store a transition in the replay buffer."""
        self.memory.append(transition)

    def train_batch(self, batch_size: int = 64) -> float:
        """
        Sample a batch from memory and perform one step of gradient descent.
        Uses manual backpropagation for the 2-layer MLP.
        """
        if len(self.memory) < batch_size:
            return 0.0

        batch = random.sample(self.memory, batch_size)

        # Unpack batch
        states = np.stack([item[0] for item in batch])
        actions = np.asarray([item[1] for item in batch], dtype=int)
        rewards = np.asarray([item[2] for item in batch], dtype=np.float64)
        next_states = np.stack([item[3] for item in batch])
        dones = np.asarray([item[4] for item in batch], dtype=np.float64)

        # Forward pass (Policy Network)
        h_pre = states @ self.w1 + self.b1
        hidden = np.maximum(0.0, h_pre)
        predictions = hidden @ self.w2 + self.b2

        # Calculate target Q-values using Target Network
        next_q = self.q_values(next_states, target=True).max(axis=1)
        targets = rewards + self.gamma * (1.0 - dones) * next_q

        # Calculate loss (MSE) and gradients
        errors = predictions[np.arange(batch_size), actions] - targets
        grad_q = np.zeros_like(predictions)
        grad_q[np.arange(batch_size), actions] = 2.0 * errors / batch_size

        # Backpropagation
        grad_w2 = hidden.T @ grad_q
        grad_b2 = grad_q.sum(axis=0)
        grad_hidden = grad_q @ self.w2.T
        grad_pre = grad_hidden * (h_pre > 0)  # ReLU derivative
        grad_w1 = states.T @ grad_pre
        grad_b1 = grad_pre.sum(axis=0)

        # Gradient clipping (to prevent exploding gradients)
        for grad in (grad_w1, grad_b1, grad_w2, grad_b2):
            np.clip(grad, -5.0, 5.0, out=grad)

        # Update weights
        lr = self.learning_rate
        self.w1 -= lr * grad_w1;
        self.b1 -= lr * grad_b1
        self.w2 -= lr * grad_w2;
        self.b2 -= lr * grad_b2

        return float(np.mean(errors ** 2))

    def sync_target(self) -> None:
        """Copy policy network weights to the target network."""
        self.t_w1, self.t_b1 = self.w1.copy(), self.b1.copy()
        self.t_w2, self.t_b2 = self.w2.copy(), self.b2.copy()

    def save(self, path: Path) -> None:
        """Save the model weights to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
            actions=np.asarray(ACTIONS),
            input_size=np.asarray([INPUT_SIZE]),
            hidden_size=np.asarray([HIDDEN_SIZE])
        )
        logger.info(f"✅ DQN Checkpoint saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "NumpyDQN":
        """
        Load model weights from disk.
        Agent-6 calls this method: `model = NumpyDQN.load(HCQA_CHECKPOINT)`
        """
        with np.load(path, allow_pickle=False) as data:
            # Validate architecture and action space compatibility
            if tuple(data["actions"].tolist()) != ACTIONS:
                raise ValueError("DQN checkpoint action vocabulary does not match this build")
            if int(data["input_size"][0]) != INPUT_SIZE or int(data["hidden_size"][0]) != HIDDEN_SIZE:
                raise ValueError("DQN checkpoint architecture does not match this build")

            model = cls()
            for name in ("w1", "b1", "w2", "b2"):
                value = np.asarray(data[name], dtype=np.float64)
                expected = getattr(model, name).shape
                if value.shape != expected or not np.isfinite(value).all():
                    raise ValueError(f"Invalid checkpoint tensor: {name}")
                setattr(model, name, value.copy())

            model.sync_target()
        logger.info(f"✅ DQN Checkpoint loaded from {path}")
        return model


# ----------------------------------------------------------------------
# TRAINING LOOP (Run this separately to generate the HCQA_CHECKPOINT)
# ----------------------------------------------------------------------
def train_dqn(episodes: int = 2500, max_depth: int = 4, seed: int = 7) -> NumpyDQN:
    """
    Trains the NumpyDQN agent from scratch.
    This should be run offline (e.g., via train_hcqa.py) to generate the checkpoint
    that Agent-6 will load during live inference.
    """
    logger.info(f"🚀 Starting DQN Training for {episodes} episodes...")
    random.seed(seed)
    rng = np.random.default_rng(seed)
    policy = NumpyDQN(seed)

    for episode in range(episodes):
        # Generate random coefficients for this episode to prevent overfitting
        coefficients = tuple(float(v) for v in rng.uniform(-1.0, 1.0, 2))
        if max(abs(coefficients[0]), abs(coefficients[1])) < 0.1:
            coefficients = (1.0, 1.0)

        state = np.array([1, 0, 0, 0], dtype=np.complex128)
        qfi, _, _ = normalized_qfi(state, coefficients)

        # Decay epsilon over time
        epsilon = max(0.03, 1.0 - episode / max(episodes * 0.8, 1))

        for depth in range(max_depth):
            obs = policy.features(state, depth, max_depth, coefficients)
            action_index = policy.choose(obs, epsilon)

            next_state = apply_action(state, ACTIONS[action_index])
            next_qfi, _, normalized = normalized_qfi(next_state, coefficients)

            # Episode ends if we hit max QFI or max depth
            done = normalized >= 0.995 or depth == max_depth - 1

            # Reward: Increase in QFI minus a penalty for adding a gate
            reward = (next_qfi - qfi) - 0.015
            if done:
                reward += 0.5 * normalized  # Bonus for finishing the circuit

            next_obs = policy.features(next_state, depth + 1, max_depth, coefficients)
            policy.remember((obs, action_index, reward, next_obs, float(done)))

            # Train the network
            policy.train_batch()

            state, qfi = next_state, next_qfi
            if done:
                break

        # Update target network periodically
        if episode % 25 == 0:
            policy.sync_target()

    logger.info("✅ DQN Training Complete.")
    return policy