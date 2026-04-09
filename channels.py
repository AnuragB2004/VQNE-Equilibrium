"""
Quantum Channel Operations
===========================
Density-matrix representation of quantum states and CPTP channels.
Implements:
  - Depolarizing channel  (Eq. 2)
  - Memory fidelity decay (Eq. 3)
  - Entanglement swapping (Eq. 4)
  - Entanglement purification BBPSSW (Eq. 5)
  - Uhlmann fidelity

All states are represented as 2x2 or 4x4 density matrices (numpy arrays).
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


# ── Pauli matrices ────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]

# Bell states
PHI_PLUS_VEC = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_PLUS     = np.outer(PHI_PLUS_VEC, PHI_PLUS_VEC.conj())

PHI_MINUS_VEC = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_PLUS_VEC  = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_MINUS_VEC = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
BELL_STATES   = [
    np.outer(PHI_PLUS_VEC,  PHI_PLUS_VEC.conj()),
    np.outer(PHI_MINUS_VEC, PHI_MINUS_VEC.conj()),
    np.outer(PSI_PLUS_VEC,  PSI_PLUS_VEC.conj()),
    np.outer(PSI_MINUS_VEC, PSI_MINUS_VEC.conj()),
]


# ── Basic density-matrix utilities ───────────────────────────────────────────

def tensor(*rhos: np.ndarray) -> np.ndarray:
    """Tensor product of density matrices."""
    result = rhos[0]
    for r in rhos[1:]:
        result = np.kron(result, r)
    return result


def partial_trace(rho: np.ndarray, keep: int, dims: Tuple[int, int]) -> np.ndarray:
    """
    Partial trace over one subsystem.
    keep=0 → keep first subsystem, trace out second.
    keep=1 → keep second subsystem, trace out first.
    dims = (d0, d1)
    """
    d0, d1 = dims
    rho_r = rho.reshape(d0, d1, d0, d1)
    if keep == 0:
        return np.einsum("iaja->ij", rho_r)
    else:
        return np.einsum("aibj->ij", rho_r)


def uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Uhlmann fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    For sigma = |psi><psi| this simplifies to Tr(sigma rho).
    """
    # Use simplified form when sigma is pure
    eigenvalues = np.linalg.eigvalsh(sigma)
    if np.allclose(eigenvalues[:-1], 0, atol=1e-8):
        # sigma is pure
        return float(np.real(np.trace(sigma @ rho)))
    # General case
    sqrt_rho = _matrix_sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    sqrt_M_eig = np.sqrt(np.maximum(np.linalg.eigvalsh(M), 0))
    return float(np.real(np.sum(sqrt_M_eig) ** 2))


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(A)
    sqrt_vals = np.sqrt(np.maximum(vals, 0))
    return (vecs * sqrt_vals) @ vecs.conj().T


def bell_fidelity(rho: np.ndarray) -> float:
    """Fidelity of rho with respect to |Phi+><Phi+|."""
    return uhlmann_fidelity(rho, PHI_PLUS)


def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-6) -> bool:
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    if abs(np.trace(rho) - 1.0) > tol:
        return False
    eigs = np.linalg.eigvalsh(rho)
    return bool(np.all(eigs >= -tol))


# ── CPTP Channels ─────────────────────────────────────────────────────────────

def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Eq. (2): E_dep(rho) = (1-p)*rho + p/4 * sum_k (sigma_k⊗I) rho (sigma_k⊗I)†
    Single-qubit depolarizing: maps rho → (1 - 4p/3)*rho + (4p/3)*(I/2)
    Two-qubit version: applied independently to each qubit.
    """
    d = rho.shape[0]
    if d == 2:
        return (1 - p) * rho + (p / 4) * sum(
            P @ rho @ P.conj().T for P in PAULIS
        )
    elif d == 4:
        # Two-qubit depolarizing (tensor product of single-qubit channels)
        ops = [np.kron(P1, P2) for P1 in PAULIS for P2 in PAULIS]
        return (1 - p) * rho + (p / 16) * sum(O @ rho @ O.conj().T for O in ops)
    else:
        raise ValueError(f"Unsupported dimension {d} for depolarizing channel")


def amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Single-qubit amplitude damping with parameter gamma."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def dephasing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Single-qubit phase-flip channel."""
    return (1 - p) * rho + p * (Z @ rho @ Z.conj().T)


def memory_decoherence(rho: np.ndarray, storage_time: float, T2: float) -> np.ndarray:
    """
    Eq. (3): Apply memory decoherence over storage_time with coherence time T2.
    Modeled as dephasing: p = 1/2 * (1 - exp(-t/T2))
    """
    p = 0.5 * (1.0 - np.exp(-storage_time / max(T2, 1e-12)))
    d = rho.shape[0]
    if d == 2:
        return dephasing_channel(rho, p)
    elif d == 4:
        # Apply to each qubit independently
        p_eff = 1.0 - (1.0 - 2 * p) ** 2  # combined effect approximation
        return depolarizing_channel(rho, p_eff)
    return rho


def channel_transmission(rho: np.ndarray, channel_params) -> np.ndarray:
    """
    Apply transmission channel effects (depolarizing + loss).
    Returns the transmitted Bell pair density matrix.
    """
    F0 = channel_params.raw_bell_fidelity()
    # Map fidelity to depolarizing parameter: F = (1 + 3*(1-4p/3))/4 => p = (1-F)*4/3
    p_dep = max(0.0, (1.0 - F0) * 4.0 / 3.0)
    p_dep = min(p_dep, 0.75)  # physical bound
    return depolarizing_channel(rho, p_dep / 4.0)


# ── Entanglement Generation ────────────────────────────────────────────────────

def generate_bell_pair(channel_params) -> Tuple[np.ndarray, bool]:
    """
    Attempt to generate a Bell pair through a quantum channel.
    Returns (rho, success) where rho is the density matrix.
    """
    p_trans = channel_params.p_trans ** (channel_params.distance / 22.0)
    success = np.random.random() < p_trans * channel_params.eta
    if not success:
        return PHI_PLUS.copy(), False
    rho = channel_transmission(PHI_PLUS.copy(), channel_params)
    return rho, True


# ── Entanglement Swapping ──────────────────────────────────────────────────────

def entanglement_swapping(
    rho_AB: np.ndarray,
    rho_BC: np.ndarray,
    p_bsm: float,
    gate_fidelity: float = 0.998,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Eq. (4): Entanglement swapping at node B.
    Given Bell pairs AB and BC, perform BSM on B's qubits to yield AC pair.

    Input states are 4x4 density matrices.
    Returns (rho_AC, success).
    """
    success = np.random.random() < p_bsm
    if not success:
        return None, False

    # Full 4-qubit state rho_ABBC
    rho_full = tensor(rho_AB, rho_BC)  # 16x16

    # BSM projects onto one of 4 Bell states on qubits B (index 1) and B' (index 2)
    # In the ideal case, this yields |Phi+>_AC perfectly.
    # With gate infidelity, we model a depolarizing error.
    rho_AC = _ideal_swap_result(rho_full)

    # Apply gate error
    p_gate_error = 1.0 - gate_fidelity ** 2
    rho_AC = depolarizing_channel(rho_AC, p_gate_error)

    return rho_AC, True


def _ideal_swap_result(rho_ABBC: np.ndarray) -> np.ndarray:
    """
    Compute the reduced state on qubits A,C after BSM on qubits B,B'.
    For a perfect Bell state input, yields |Phi+>_AC.
    Uses partial trace after BSM CPTP map.
    """
    d = 16  # 4 qubits: A B B' C

    # BSM measurement operators on qubits 1,2 (B, B') - indices in 4-qubit system
    # We project and sum over all 4 outcomes (tracing out measurement)
    rho_AC = np.zeros((4, 4), dtype=complex)

    for bell_state in BELL_STATES:
        # Measurement operator: I_A ⊗ |phi><phi|_{BB'} ⊗ I_C
        M = np.kron(np.kron(I2, bell_state), I2)
        # Post-measurement unnormalized state
        rho_post = M @ rho_ABBC @ M.conj().T
        # Partial trace over B, B' (dimensions 2,3 in 4-qubit system)
        rho_AC += _partial_trace_middle(rho_post)

    # Normalize
    tr = np.real(np.trace(rho_AC))
    if tr > 1e-10:
        rho_AC /= tr
    return rho_AC


def _partial_trace_middle(rho: np.ndarray) -> np.ndarray:
    """Partial trace over qubits 1,2 of a 4-qubit (16x16) state, keeping 0,3."""
    rho_r = rho.reshape(2, 2, 2, 2, 2, 2, 2, 2)
    # i,j,k,l are ket indices for A, B, B', C.
    # m,j,k,n are bra indices for A, B, B', C.
    # We trace over j (B) and k (B').
    result = np.einsum("ijklmjkn->ilmn", rho_r)
    return result.reshape(4, 4)


# ── Entanglement Purification (BBPSSW) ────────────────────────────────────────

def bbpssw_fidelity_out(F_in: float) -> float:
    """
    Eq. (5): Output fidelity of BBPSSW purification from input fidelity F_in.
    """
    f = F_in
    e = (1 - f) / 3.0
    numerator   = f**2 + e**2
    denominator = f**2 + 2 * f * e * 2 + 5 * e**2
    # Correct BBPSSW formula:
    numerator   = f**2 + ((1 - f) / 3) ** 2
    denominator = (f**2
                   + (2 * f * (1 - f)) / 3
                   + 5 * ((1 - f) / 3) ** 2)
    return numerator / max(denominator, 1e-12)


def bbpssw_success_prob(F_in: float) -> float:
    """Success probability of one round of BBPSSW purification."""
    f = F_in
    return (f**2
            + (2 * f * (1 - f)) / 3
            + 5 * ((1 - f) / 3) ** 2)


def entanglement_purification(
    rho1: np.ndarray,
    rho2: np.ndarray,
    p_bsm: float = 0.75,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    BBPSSW purification of two noisy Bell pairs rho1, rho2.
    Returns (purified_rho, success).
    Consumes both input pairs; on success returns one higher-fidelity pair.
    """
    F_in = bell_fidelity(rho1)
    p_success = bbpssw_success_prob(F_in) * p_bsm

    if np.random.random() > p_success:
        return None, False

    F_out = bbpssw_fidelity_out(F_in)
    # Construct Werner state with output fidelity
    rho_out = werner_state(F_out)
    return rho_out, True


def werner_state(F: float) -> np.ndarray:
    """
    Werner state with fidelity F to |Phi+>.
    rho_W = F * |Phi+><Phi+| + (1-F)/3 * (I_4 - |Phi+><Phi+|)
    """
    F = float(np.clip(F, 0.0, 1.0))
    return F * PHI_PLUS + (1.0 - F) / 3.0 * (np.eye(4, dtype=complex) - PHI_PLUS)


# ── Multi-hop path fidelity ───────────────────────────────────────────────────

def path_fidelity(
    path: list,
    qnet,
    storage_times: Optional[dict] = None,
) -> float:
    """
    Compute approximate end-to-end fidelity over a path using
    successive entanglement swapping (no purification).
    """
    if len(path) < 2:
        return 1.0

    storage_times = storage_times or {}

    # Start with first link
    rho = channel_transmission(PHI_PLUS.copy(), qnet.edge_params(path[0], path[1]))

    for i in range(1, len(path) - 1):
        # Next link
        rho_next = channel_transmission(
            PHI_PLUS.copy(), qnet.edge_params(path[i], path[i + 1])
        )
        # Memory decoherence
        tau = storage_times.get((path[i], i), 0.001)
        node_p = qnet.node_params(path[i])
        rho = memory_decoherence(rho, tau, node_p.T2)

        # Swap
        p_bsm = node_p.p_bsm
        rho_swapped, success = entanglement_swapping(
            rho, rho_next, p_bsm, node_p.gate_fidelity
        )
        if not success:
            return 0.0
        rho = rho_swapped

    return bell_fidelity(rho)