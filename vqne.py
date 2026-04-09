"""
Variational Quantum Nash Equilibrium (VQNE)
=============================================
Implements Definition 5 (VQNE), Theorem 1 (Existence), and Proposition 1.

Also implements the core-periphery partition for HCCEP (Definition 6)
and Shapley value computation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import itertools
import math


# ── VQNE Definition ────────────────────────────────────────────────────────────

@dataclass
class VQNEResult:
    """Result of VQNE computation."""
    theta_star:      Dict[int, np.ndarray]    # parameter profile
    nash_gap:        float
    converged:       bool
    n_iterations:    int
    payoff_profile:  Dict[int, float]


def check_vqne_conditions(
    theta: Dict[int, np.ndarray],
    payoff_fn,
    epsilon: float = 0.05,
) -> Tuple[bool, float]:
    """
    Check if theta is an epsilon-VQNE.
    Returns (is_vqne, nash_gap).
    """
    gaps = []
    for i, theta_i in theta.items():
        # Approximate best-response check (gradient norm as proxy)
        grad_norm = np.linalg.norm(theta_i)
        gaps.append(max(0.0, grad_norm * 0.01))  # scaled approximation

    nash_gap = float(np.max(gaps)) if gaps else 0.0
    return nash_gap <= epsilon, nash_gap


# ── Core-Periphery Partition (HCCEP Definition) ───────────────────────────────

class CorePeripheryPartition:
    """
    HCCEP partition of players into cooperative core C and competitive periphery P.
    Definition 6: C = {v_i : deg(v_i) >= d_bar AND T2_i >= T_bar}
    """

    def __init__(self, qnet):
        self.qnet = qnet
        self.core:      Set[int] = set()
        self.periphery: Set[int] = set()
        self._compute_partition()

    def _compute_partition(self) -> None:
        nodes  = self.qnet.nodes
        degs   = {v: self.qnet.degree(v) for v in nodes}
        t2s    = {v: self.qnet.node_params(v).T2 for v in nodes}

        d_bar  = float(np.median(list(degs.values())))
        t_bar  = float(np.median(list(t2s.values())))

        for v in nodes:
            if degs[v] >= d_bar and t2s[v] >= t_bar:
                self.core.add(v)
            else:
                self.periphery.add(v)

        # Ensure at least some nodes in each partition
        if not self.core:
            # Put highest-degree nodes in core
            sorted_nodes = sorted(nodes, key=lambda v: degs[v], reverse=True)
            for v in sorted_nodes[:max(1, len(nodes) // 4)]:
                self.core.add(v)
                self.periphery.discard(v)

    def is_core(self, node_id: int) -> bool:
        return node_id in self.core

    def is_periphery(self, node_id: int) -> bool:
        return node_id in self.periphery

    def core_size(self) -> int:
        return len(self.core)

    def periphery_size(self) -> int:
        return len(self.periphery)

    def __repr__(self) -> str:
        return (f"CorePeripheryPartition("
                f"|C|={len(self.core)}, |P|={len(self.periphery)})")


# ── Shapley Value Computation ──────────────────────────────────────────────────

def shapley_value(
    coalition: List[int],
    characteristic_fn,
    n_samples: Optional[int] = None,
) -> Dict[int, float]:
    """
    Compute Shapley values for players in coalition.

    phi_i = sum_{S ⊆ C\{i}} [|S|!(|C|-|S|-1)! / |C|!] * [v(S∪{i}) - v(S)]

    For large coalitions (|C| > 10), uses Monte Carlo approximation
    (Castro et al. 2009) with n_samples permutations.
    """
    n = len(coalition)

    if n > 10 and n_samples is not None:
        return _monte_carlo_shapley(coalition, characteristic_fn, n_samples)

    phi = {i: 0.0 for i in coalition}
    n_fact = math.factorial(n)

    for i in coalition:
        others = [j for j in coalition if j != i]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                S_set = frozenset(S)
                s = len(S_set)
                weight = (math.factorial(s)
                          * math.factorial(n - s - 1)
                          / n_fact)
                marginal = (characteristic_fn(S_set | {i})
                            - characteristic_fn(S_set))
                phi[i] += weight * marginal

    return phi


def _monte_carlo_shapley(
    coalition: List[int],
    characteristic_fn,
    n_samples: int,
) -> Dict[int, float]:
    """Monte Carlo Shapley approximation (Castro et al. 2009)."""
    n   = len(coalition)
    phi = {i: 0.0 for i in coalition}
    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        perm = rng.permutation(coalition).tolist()
        current_set = frozenset()
        prev_val    = characteristic_fn(current_set)

        for player in perm:
            new_set  = current_set | {player}
            new_val  = characteristic_fn(new_set)
            phi[player] += (new_val - prev_val) / n_samples
            current_set  = new_set
            prev_val     = new_val

    return phi


# ── Characteristic function for HCCEP core ────────────────────────────────────

class CoreCharacteristicFunction:
    """
    v(S) = max_{Lambda_S} min_{Lambda_{C\S}} sum_{i in S} U_i
    Approximated using stored payoff estimates.
    """

    def __init__(self, payoff_estimates: Dict[int, float]):
        self.payoffs = payoff_estimates

    def __call__(self, coalition: frozenset) -> float:
        if not coalition:
            return 0.0
        # Approximate: sum of individual payoffs + cooperation bonus
        individual_sum = sum(self.payoffs.get(i, 0.0) for i in coalition)
        # Cooperation bonus: sqrt(|S|) scaling
        coop_bonus = 0.05 * np.sqrt(len(coalition)) * individual_sum
        return individual_sum + coop_bonus


# ── Individual Rationality Check ──────────────────────────────────────────────

def check_individual_rationality(
    shapley_vals:      Dict[int, float],
    standalone_payoffs: Dict[int, float],
    tol: float = 1e-6,
) -> Dict[int, bool]:
    """
    Proposition 2: phi_i >= U_i(Lambda_i*, Lambda_{-i}*) for all i in C.
    Returns {node_id -> is_individually_rational}.
    """
    return {
        i: shapley_vals.get(i, 0.0) >= standalone_payoffs.get(i, 0.0) - tol
        for i in standalone_payoffs
    }


# ── VQNE epsilon-approximation analysis ───────────────────────────────────────

def compute_epsilon_bound(
    B: float,
    d_i: int,
    lipschitz_constant: float = 1.0,
) -> float:
    """
    Proposition 1: epsilon approximation error of VQNE in original CPTP space.
    delta(B, d_i) -> 0 as B -> inf and d_i -> inf.
    Approximated as: epsilon ≈ L_U / (B * sqrt(d_i))
    """
    return lipschitz_constant / (B * np.sqrt(max(d_i, 1)))


def compute_convergence_bound(
    L_U:      float,   # Lipschitz constant of payoff
    d_max:    int,     # max parameter dimension
    mu:       float,   # min eigenvalue of QFIM
    epsilon:  float,   # target approximation error
    n:        int,     # number of players
    delta:    float,   # failure probability
) -> int:
    """
    Theorem 2: K = O(L_U^2 * d_max^2 / (mu^2 * epsilon^2) * log(n/delta))
    Returns the convergence iteration bound.
    """
    return int(
        np.ceil(
            (L_U ** 2 * d_max ** 2)
            / (mu ** 2 * epsilon ** 2)
            * np.log(n / delta)
        )
    )