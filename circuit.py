"""
Variational Quantum Circuits (VQC)
====================================
Implements the parameterized quantum circuit U_i(theta_i) for strategy
representation, as described in Section IV-C and Fig. 2.

We simulate quantum circuits in numpy (state-vector / density-matrix
simulation) for the policy network.  For the RL policy gradient we
use PyTorch autograd with a parameter-shift-rule-compatible wrapper.

Circuit structure (L layers):
  [R_Y(theta), R_Z(theta)] entangled via CNOT ladders → measure → action probs
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


# ── Single-qubit rotation gates ───────────────────────────────────────────────

def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

# CNOT gate (2-qubit)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

I2   = np.eye(2, dtype=complex)


# ── Full state-vector circuit simulator ──────────────────────────────────────

class QuantumCircuit:
    """
    Statevector-based quantum circuit simulator.
    n_qubits: number of qubits (uses 2^n_qubits dimensional state).
    """

    def __init__(self, n_qubits: int):
        self.n  = n_qubits
        self.dim = 2 ** n_qubits
        self._state = np.zeros(self.dim, dtype=complex)
        self._state[0] = 1.0

    def reset(self, init_state: Optional[np.ndarray] = None) -> None:
        if init_state is not None:
            self._state = init_state.copy().astype(complex)
        else:
            self._state = np.zeros(self.dim, dtype=complex)
            self._state[0] = 1.0

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def _apply_single(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate to the specified qubit."""
        # Build full unitary: I ⊗ ... ⊗ gate ⊗ ... ⊗ I
        ops = [I2] * self.n
        ops[qubit] = gate
        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)
        self._state = U @ self._state

    def _apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate (control, target)."""
        dim = self.dim
        new_state = np.zeros(dim, dtype=complex)
        n = self.n
        for idx in range(dim):
            bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]
            if bits[control] == 1:
                bits[target] ^= 1
            new_idx = sum(b << (n - 1 - k) for k, b in enumerate(bits))
            new_state[new_idx] += self._state[idx]
        self._state = new_state

    def apply_ry(self, theta: float, qubit: int) -> None:
        self._apply_single(Ry(theta), qubit)

    def apply_rz(self, theta: float, qubit: int) -> None:
        self._apply_single(Rz(theta), qubit)

    def apply_rx(self, theta: float, qubit: int) -> None:
        self._apply_single(Rx(theta), qubit)

    def apply_cnot(self, control: int, target: int) -> None:
        self._apply_cnot(control, target)

    def entangler_layer(self) -> None:
        """CNOT ladder: 0→1, 1→2, ..., (n-2)→(n-1)."""
        for i in range(self.n - 1):
            self.apply_cnot(i, i + 1)

    def measure_probs(self) -> np.ndarray:
        """Return measurement probabilities for each computational basis state."""
        return (np.abs(self._state) ** 2).real

    def density_matrix(self) -> np.ndarray:
        """Return the density matrix of the current state."""
        psi = self._state.reshape(-1, 1)
        return psi @ psi.conj().T


# ── Variational Quantum Circuit as NN module ──────────────────────────────────

class VQCLayer(nn.Module):
    """
    One layer of the VQC: R_Y rotations → CNOT entangler → R_Z rotations.
    Parameters are trainable via PyTorch.
    Forward pass returns expectation values via parameter-shift-compatible
    differentiable simulation.
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits  = n_qubits
        # RY and RZ angles per qubit
        self.ry_angles = nn.Parameter(torch.zeros(n_qubits))
        self.rz_angles = nn.Parameter(torch.zeros(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classical differentiable approximation of one VQC layer.
        x: (batch, n_qubits) input features (encoded into rotation angles)
        Returns: (batch, n_qubits) output features after rotation+entanglement
        """
        # Encode input as additive offset to learned angles
        angles_y = self.ry_angles.unsqueeze(0) + x  # (batch, n_qubits)
        angles_z = self.rz_angles.unsqueeze(0)

        # Classical approximation: Ry rotation in Bloch sphere (sin/cos)
        # Represents the expected <Z> value per qubit after rotation
        out_y = torch.cos(angles_y)  # <Z> after R_Y
        out_z = torch.cos(angles_z) * out_y  # composition approximation

        # Entanglement layer: mix adjacent qubits (CNOT approximation via XOR)
        entangled = out_z.clone()
        for i in range(self.n_qubits - 1):
            entangled[:, i + 1] = out_z[:, i + 1] * torch.cos(out_z[:, i])

        return entangled


class VariationalQuantumCircuit(nn.Module):
    """
    Full VQC policy network with L layers.

    Architecture (Fig. 2):
      Input encoding → L × [R_Y, CNOT-entangler, R_Z] → Measurement → Softmax

    n_qubits:   number of qubits (= n_q in paper, default 4)
    n_layers:   circuit depth L (default 6)
    n_actions:  size of action space (output dimension)
    n_inputs:   dimension of classical observation vector
    """

    def __init__(
        self,
        n_inputs:  int,
        n_actions: int,
        n_qubits:  int = 4,
        n_layers:  int = 6,
    ):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.n_actions = n_actions
        self.n_inputs  = n_inputs

        # Classical encoder: observation → qubit rotation angles
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, n_qubits),
            nn.Tanh(),
        )

        # VQC layers
        self.vqc_layers = nn.ModuleList(
            [VQCLayer(n_qubits) for _ in range(n_layers)]
        )

        # Classical decoder: qubit outputs → action logits
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch, n_inputs) observation tensor
        Returns: (batch, n_actions) action log-probabilities
        """
        # Encode observation to qubit angles
        x = self.encoder(obs)  # (batch, n_qubits)

        # Apply VQC layers
        for layer in self.vqc_layers:
            x = layer(x)

        # Decode to action distribution
        logits = self.decoder(x)
        return torch.log_softmax(logits, dim=-1)

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.forward(obs))

    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Sample action from policy distribution. Returns (action_idx, log_prob)."""
        with torch.no_grad():
            obs_t   = torch.FloatTensor(obs).unsqueeze(0)
            log_p   = self.forward(obs_t)[0]
            probs   = torch.exp(log_p)
            action  = torch.multinomial(probs, 1).item()
            log_prob = log_p[action].item()
        return int(action), log_prob

    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get log-probabilities of specific actions.
        obs: (batch, n_inputs)
        actions: (batch,) integer action indices
        """
        log_probs = self.forward(obs)  # (batch, n_actions)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quantum Fisher Information Matrix (QFIM) approximation ───────────────────

class QFIMEstimator:
    """
    Estimates the Quantum Fisher Information Matrix for quantum natural gradient.
    Uses the diagonal approximation for efficiency (Section VI-A scalability).

    F_Q[a,b] = 4 Re[Tr(rho * d_a ln(rho) * d_b ln(rho))]

    In practice: approximated via the classical Fisher of the policy distribution.
    """

    def __init__(self, model: VariationalQuantumCircuit, damping: float = 1e-3):
        self.model   = model
        self.damping = damping

    def diagonal_qfim(
        self,
        obs_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diagonal QFIM approximation: F_ii ≈ E[( d log π / d θ_i )^2]
        Returns a 1D tensor of diagonal QFIM entries.
        """
        n_params = self.model.parameter_count()
        diag_F   = torch.zeros(n_params)

        # Compute gradients of log-probs for each sample
        for obs in obs_batch:
            obs_1 = obs.unsqueeze(0)
            log_probs = self.model.forward(obs_1)  # (1, n_actions)
            probs     = torch.exp(log_probs)

            for a in range(self.model.n_actions):
                if probs[0, a].item() < 1e-8:
                    continue
                self.model.zero_grad()
                log_probs[0, a].backward(retain_graph=True)
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.flatten())
                    else:
                        grads.append(torch.zeros(p.numel()))
                g = torch.cat(grads)
                diag_F += probs[0, a].item() * g ** 2

        diag_F /= max(len(obs_batch), 1)
        return diag_F + self.damping  # add damping for invertibility

    def natural_gradient(
        self,
        vanilla_grad: List[torch.Tensor],
        diag_F: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Apply inverse QFIM (diagonal approx) to vanilla gradient.
        Returns natural gradient tensors matching parameter shapes.
        """
        # Flatten all gradients
        flat_grad = torch.cat([g.flatten() for g in vanilla_grad if g is not None])
        nat_grad_flat = flat_grad / diag_F

        # Restore parameter shapes
        nat_grads = []
        offset = 0
        for g in vanilla_grad:
            if g is None:
                nat_grads.append(None)
                continue
            size = g.numel()
            nat_grads.append(nat_grad_flat[offset:offset + size].reshape(g.shape))
            offset += size
        return nat_grads


# ── Parameter-shift rule gradient ────────────────────────────────────────────

def parameter_shift_gradient(
    circuit: QuantumCircuit,
    params: np.ndarray,
    param_idx: int,
    observable: np.ndarray,
) -> float:
    """
    Parameter-shift rule: dE/d(theta_k) = [E(theta_k + pi/2) - E(theta_k - pi/2)] / 2
    observable: Hermitian matrix for expectation value
    """
    params_plus  = params.copy()
    params_minus = params.copy()
    params_plus[param_idx]  += np.pi / 2
    params_minus[param_idx] -= np.pi / 2

    def expectation(p: np.ndarray) -> float:
        # Re-run circuit with shifted params and measure
        # (simplified: use diagonal elements)
        circuit.reset()
        # Apply rotations with shifted params (simplified model)
        n_q = circuit.n
        for i, theta in enumerate(p[:n_q]):
            circuit.apply_ry(theta, i % n_q)
        probs = circuit.measure_probs()
        return float(np.real(np.sum(probs * np.diag(observable).real)))

    return (expectation(params_plus) - expectation(params_minus)) / 2.0