"""Exact two-qubit state-vector engine for the configured P1 demonstration model.

This simulator is not an atom-interferometer or hardware adapter. The signal is
encoded as exp(-i * phi * G), with G=(c0 Z0+c1 Z1)/2.
"""
from __future__ import annotations
import math
import numpy as np

I2 = np.eye(2, dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
S = np.diag([1, 1j]).astype(np.complex128)
ACTIONS = ("H0", "H1", "X0", "X1", "RX0_PI2", "RY0_PI2", "RX1_PI2", "RY1_PI2", "S0", "CNOT01", "CNOT10")
CNOT01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
CNOT10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=np.complex128)

def rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle/2), math.sin(angle/2)
    if axis == "RX": return np.array([[c,-1j*s],[-1j*s,c]], dtype=np.complex128)
    if axis == "RY": return np.array([[c,-s],[s,c]], dtype=np.complex128)
    if axis == "RZ": return np.diag([np.exp(-1j*angle/2), np.exp(1j*angle/2)]).astype(np.complex128)
    raise ValueError(f"unsupported rotation axis: {axis}")

def apply_action(state: np.ndarray, action: str) -> np.ndarray:
    if action == "CNOT01": return CNOT01 @ state
    if action == "CNOT10": return CNOT10 @ state
    if action == "H0": gate, q = H, 0
    elif action == "H1": gate, q = H, 1
    elif action == "X0": gate, q = X, 0
    elif action == "X1": gate, q = X, 1
    elif action == "S0": gate, q = S, 0
    elif action == "RX0_PI2": gate, q = rotation("RX", math.pi/2), 0
    elif action == "RY0_PI2": gate, q = rotation("RY", math.pi/2), 0
    elif action == "RX1_PI2": gate, q = rotation("RX", math.pi/2), 1
    elif action == "RY1_PI2": gate, q = rotation("RY", math.pi/2), 1
    else: raise ValueError(f"unsupported action: {action}")
    return (np.kron(gate, I2) if q == 0 else np.kron(I2, gate)) @ state

def prepare_state(sequence: list[str]) -> np.ndarray:
    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    for action in sequence:
        state = apply_action(state, action)
    return state

def generator_eigenvalues(coefficients: tuple[float, float] | list[float]) -> np.ndarray:
    c0, c1 = map(float, coefficients)
    # Computational basis order |00>, |01>, |10>, |11>.
    return 0.5 * np.array([c0+c1, c0-c1, -c0+c1, -c0-c1], dtype=np.float64)

def normalized_qfi(state: np.ndarray, coefficients: tuple[float, float] | list[float]) -> tuple[float, float, float]:
    lam = generator_eigenvalues(coefficients)
    p = np.abs(state) ** 2
    mean = float(np.dot(p, lam))
    variance = max(0.0, float(np.dot(p, lam*lam) - mean*mean))
    qfi = 4.0 * variance
    qfi_limit = float((lam.max() - lam.min()) ** 2)
    return qfi, qfi_limit, (qfi / qfi_limit if qfi_limit > 0 else 0.0)

def evaluate_sequence(phase_rad: float, coefficients: tuple[float, float] | list[float],
                      sequence: list[str]) -> dict:
    if not math.isfinite(phase_rad): raise ValueError("phase must be finite")
    state = prepare_state(sequence)
    lam = generator_eigenvalues(coefficients)
    encoded = np.exp(-1j * phase_rad * lam) * state
    derivative = -1j * lam * encoded
    # Local Hadamard readout maps phase information into computational-basis counts.
    readout = np.kron(H, H)
    measured = readout @ encoded
    measured_derivative = readout @ derivative
    probabilities = np.abs(measured) ** 2
    dp = 2.0 * np.real(np.conj(measured) * measured_derivative)
    clfi = float(sum(float(dp[i]**2 / probabilities[i]) for i in range(4) if probabilities[i] > 1e-12))
    qfi, qfi_limit, norm_qfi = normalized_qfi(state, coefficients)
    return {"probabilities": [float(v) for v in probabilities],
            "probability_sum": float(probabilities.sum()),
            "qfi": qfi, "qfi_limit": qfi_limit, "normalized_qfi": norm_qfi,
            "measurement_classical_fisher_information": clfi,
            "sequence": list(sequence), "qubits": 2,
            "generator_eigenvalues": [float(v) for v in lam],
            "model_scope": "exact two-qubit simulator; project-specific sensor coupling is not validated",
            "helstrom_comparison": "not_evaluated_no_error_distributions_supplied",
            "quantum_advantage_claimed": False}

def to_qiskit_circuit(sequence: list[str], phase_rad: float, coefficients: tuple[float, float]):
    """Build a Qiskit circuit if Qiskit is installed; returns an unexecuted circuit."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    for action in sequence:
        if action.startswith("H"): qc.h(1-int(action[-1]))
        elif action.startswith("X"): qc.x(1-int(action[-1]))
        elif action == "S0": qc.s(1)
        elif action.startswith("RX"): qc.rx(math.pi/2, 1-int(action[2]))
        elif action.startswith("RY"): qc.ry(math.pi/2, 1-int(action[2]))
        elif action == "CNOT01": qc.cx(1, 0)
        elif action == "CNOT10": qc.cx(0, 1)
        else: raise ValueError(f"unsupported action: {action}")
    # q0 is the most-significant simulator wire, hence it maps to Qiskit wire 1.
    qc.rz(float(coefficients[0])*phase_rad, 1)
    qc.rz(float(coefficients[1])*phase_rad, 0)
    qc.h(1); qc.h(0)
    qc.measure([1, 0], [1, 0])
    return qc

def to_pennylane_qnode(sequence: list[str], coefficients: tuple[float, float]):
    """Create an analytic PennyLane probability QNode, without executing hardware."""
    import pennylane as qml
    dev = qml.device("default.qubit", wires=2, shots=None)
    @qml.qnode(dev)
    def circuit(phase_rad):
        for action in sequence:
            if action.startswith("H"): qml.Hadamard(wires=int(action[-1]))
            elif action.startswith("X"): qml.PauliX(wires=int(action[-1]))
            elif action == "S0": qml.S(wires=0)
            elif action.startswith("RX"): qml.RX(math.pi/2, wires=int(action[2]))
            elif action.startswith("RY"): qml.RY(math.pi/2, wires=int(action[2]))
            elif action == "CNOT01": qml.CNOT(wires=[0, 1])
            elif action == "CNOT10": qml.CNOT(wires=[1, 0])
            else: raise ValueError(f"unsupported action: {action}")
        qml.RZ(float(coefficients[0])*phase_rad, wires=0)
        qml.RZ(float(coefficients[1])*phase_rad, wires=1)
        qml.Hadamard(wires=0); qml.Hadamard(wires=1)
        return qml.probs(wires=[0, 1])
    return circuit
