"""
Quantum Network Entanglement Game (QNEG)
=========================================
Implements Definitions 2–5 from the paper:
  - Quantum Strategy (CPTP maps, Definition 2)
  - Quantum Payoff (Definition 3)
  - QNEG (Definition 4)

Strategy space: three primitive action classes
  - ES  : Entanglement Swapping
  - EP  : Entanglement Purification
  - RD  : Routing Decision
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

from channels import (
    entanglement_swapping,
    entanglement_purification,
    generate_bell_pair,
    bell_fidelity,
    uhlmann_fidelity,
    PHI_PLUS,
    memory_decoherence,
)
from graph import QuantumNetworkGraph


# ── Action types ──────────────────────────────────────────────────────────────

class ActionType(Enum):
    ENTANGLEMENT_SWAP   = auto()  # ES
    ENTANGLEMENT_PURIFY = auto()  # EP
    ROUTE               = auto()  # RD
    IDLE                = auto()  # do nothing (wait)


@dataclass
class Action:
    action_type: ActionType
    target_neighbor: Optional[int] = None   # for RD: next-hop
    swap_pair: Optional[Tuple[int, int]] = None  # for ES: (nb1, nb2)

    def __repr__(self):
        return f"Action({self.action_type.name}, nb={self.target_neighbor})"


# ── Local quantum state of a node ─────────────────────────────────────────────

@dataclass
class NodeState:
    """Local state of a quantum repeater node."""
    node_id:       int
    memory_pairs:  Dict[int, np.ndarray] = field(default_factory=dict)
    # memory_pairs: {neighbor_id -> Bell pair density matrix stored in memory}
    storage_times: Dict[int, float] = field(default_factory=dict)
    # storage_times: {neighbor_id -> time (s) the pair has been stored}
    successful_deliveries: int = 0
    total_attempts:        int = 0

    def has_pair(self, neighbor: int) -> bool:
        return neighbor in self.memory_pairs

    def store_pair(self, neighbor: int, rho: np.ndarray, time: float = 0.0) -> None:
        self.memory_pairs[neighbor] = rho
        self.storage_times[neighbor] = time

    def consume_pair(self, neighbor: int) -> Optional[np.ndarray]:
        rho = self.memory_pairs.pop(neighbor, None)
        self.storage_times.pop(neighbor, None)
        return rho

    def memory_usage(self) -> int:
        return len(self.memory_pairs)

    def age_memories(self, dt: float) -> None:
        for nb in list(self.storage_times.keys()):
            self.storage_times[nb] += dt


# ── Payoff weights ─────────────────────────────────────────────────────────────

@dataclass
class PayoffWeights:
    """Trade-off weights in Eq. (6): alpha + beta + gamma = 1."""
    alpha: float = 0.5   # fidelity weight
    beta:  float = 0.3   # rate weight
    gamma: float = 0.2   # cost weight

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta  /= total
        self.gamma /= total


# ── Payoff computation ─────────────────────────────────────────────────────────

class PayoffFunction:
    """
    Eq. (6): U_i(Lambda) = alpha * F_i + beta * R_i - gamma * C_i

    F_i: Uhlmann fidelity of delivered state w.r.t. |Phi+>
    R_i: entanglement generation rate (ebit/s)
    C_i: resource cost (memory + purification + classical bits)
    """

    MEMORY_COST_WEIGHT = 0.1
    PURIF_COST_WEIGHT  = 0.3
    CLASSICAL_BIT_COST = 0.01

    def __init__(self, weights: Optional[PayoffWeights] = None):
        self.weights = weights or PayoffWeights()

    def fidelity_term(
        self,
        delivered_rho: Optional[np.ndarray],
        target_rho: np.ndarray = None,
    ) -> float:
        if delivered_rho is None:
            return 0.0
        target = target_rho if target_rho is not None else PHI_PLUS
        return uhlmann_fidelity(delivered_rho, target)

    def rate_term(self, n_delivered: int, total_slots: int, slot_duration: float = 1e-3) -> float:
        """ebit/s"""
        if total_slots == 0:
            return 0.0
        return n_delivered / (total_slots * slot_duration)

    def cost_term(
        self,
        memory_used: int,
        n_purifications: int,
        classical_bits: int,
    ) -> float:
        return (self.MEMORY_COST_WEIGHT * memory_used
                + self.PURIF_COST_WEIGHT * n_purifications
                + self.CLASSICAL_BIT_COST * classical_bits)

    def compute(
        self,
        fidelity: float,
        rate: float,
        cost: float,
    ) -> float:
        w = self.weights
        return w.alpha * fidelity + w.beta * rate - w.gamma * cost

    def compute_from_episode(
        self,
        episode_stats: "EpisodeStats",
        total_slots: int,
    ) -> float:
        f = episode_stats.mean_fidelity
        r = self.rate_term(episode_stats.n_delivered, total_slots)
        c = self.cost_term(
            episode_stats.mean_memory_used,
            episode_stats.n_purifications,
            episode_stats.classical_bits_used,
        )
        return self.compute(f, r, c)


@dataclass
class EpisodeStats:
    """Accumulated statistics for one agent over an episode."""
    n_delivered:       int   = 0
    n_purifications:   int   = 0
    classical_bits_used: int = 0
    mean_memory_used:  float = 0.0
    mean_fidelity:     float = 0.0
    fidelities:        list  = field(default_factory=list)

    def update_fidelity(self, f: float) -> None:
        self.fidelities.append(f)
        self.mean_fidelity = float(np.mean(self.fidelities))

    def record_delivery(self, rho: np.ndarray) -> None:
        self.n_delivered += 1
        self.update_fidelity(bell_fidelity(rho))

    def record_purification(self) -> None:
        self.n_purifications += 1
        self.classical_bits_used += 2  # 2 classical bits per BSM outcome


# ── QNEG definition ───────────────────────────────────────────────────────────

class QNEG:
    """
    Quantum Network Entanglement Game (Definition 4).

    G = (n, H_net, rho_0, {S_i}, {U_i})

    The game is played over T time slots; at each slot each agent
    chooses an action from their strategy set.
    """

    def __init__(
        self,
        qnet: QuantumNetworkGraph,
        src: int,
        dst: int,
        weights: Optional[PayoffWeights] = None,
        slot_duration: float = 1e-3,  # seconds per time slot
        seed: int = 42,
    ):
        self.qnet = qnet
        self.src  = src
        self.dst  = dst
        self.n    = len(qnet)
        self.payoff_fn = PayoffFunction(weights)
        self.slot_duration = slot_duration
        self.rng  = np.random.default_rng(seed)

        # Initialize node states
        self.node_states: Dict[int, NodeState] = {
            v: NodeState(node_id=v) for v in qnet.nodes
        }
        self.episode_stats: Dict[int, EpisodeStats] = {
            v: EpisodeStats() for v in qnet.nodes
        }
        self.global_step = 0

    # ── Action space for each node ─────────────────────────────────────────────

    def available_actions(self, node_id: int) -> List[Action]:
        """Return the list of valid actions for node_id given current state."""
        actions = [Action(ActionType.IDLE)]
        ns     = self.node_states[node_id]
        params = self.qnet.node_params(node_id)
        nbs    = self.qnet.neighbors(node_id)

        # Routing decisions: forward to each neighbor
        for nb in nbs:
            actions.append(Action(ActionType.ROUTE, target_neighbor=nb))

        # Entanglement swap: if two memory slots occupied by different neighbors
        stored_nbs = list(ns.memory_pairs.keys())
        if len(stored_nbs) >= 2:
            for i in range(len(stored_nbs)):
                for j in range(i + 1, len(stored_nbs)):
                    actions.append(Action(
                        ActionType.ENTANGLEMENT_SWAP,
                        swap_pair=(stored_nbs[i], stored_nbs[j]),
                    ))

        # Purification: if two pairs with same neighbor
        for nb in stored_nbs:
            if ns.memory_pairs.get(nb) is not None:
                # Simplified: can purify if we'd have a second incoming pair
                actions.append(Action(ActionType.ENTANGLEMENT_PURIFY, target_neighbor=nb))

        return actions

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(
        self,
        joint_actions: Dict[int, Action],
    ) -> Dict[int, float]:
        """
        Execute one time slot with the given joint action profile.
        Returns per-node immediate payoff (reward).
        """
        self.global_step += 1
        rewards: Dict[int, float] = {v: 0.0 for v in self.qnet.nodes}

        # 1. Age all memories
        for ns in self.node_states.values():
            ns.age_memories(self.slot_duration)

        # 2. Attempt Bell pair generation on all links
        self._attempt_generation()

        # 3. Execute actions
        for node_id, action in joint_actions.items():
            reward = self._execute_action(node_id, action)
            rewards[node_id] = reward

        # 4. Check end-to-end delivery
        self._check_delivery(rewards)

        return rewards

    def _attempt_generation(self) -> None:
        """Probabilistically generate Bell pairs on each edge."""
        seen = set()
        for u, v in self.qnet.edges:
            if (min(u, v), max(u, v)) in seen:
                continue
            seen.add((min(u, v), max(u, v)))

            ch = self.qnet.edge_params(u, v)
            rho, success = generate_bell_pair(ch)
            if success:
                ns_u = self.node_states[u]
                ns_v = self.node_states[v]
                max_mem_u = self.qnet.node_params(u).n_memories
                max_mem_v = self.qnet.node_params(v).n_memories
                if ns_u.memory_usage() < max_mem_u and ns_v.memory_usage() < max_mem_v:
                    ns_u.store_pair(v, rho.copy())
                    ns_v.store_pair(u, rho.copy())

    def _execute_action(self, node_id: int, action: Action) -> float:
        ns     = self.node_states[node_id]
        params = self.qnet.node_params(node_id)
        stats  = self.episode_stats[node_id]

        if action.action_type == ActionType.IDLE:
            return 0.0

        elif action.action_type == ActionType.ENTANGLEMENT_SWAP:
            if action.swap_pair is None:
                return 0.0
            nb1, nb2 = action.swap_pair
            if not (ns.has_pair(nb1) and ns.has_pair(nb2)):
                return -0.01  # failed action penalty

            rho1 = ns.consume_pair(nb1)
            rho2 = ns.consume_pair(nb2)

            # Apply memory decoherence before swap
            t1 = ns.storage_times.pop(nb1, 0.0)
            t2 = ns.storage_times.pop(nb2, 0.0)
            rho1 = memory_decoherence(rho1, t1, params.T2)
            rho2 = memory_decoherence(rho2, t2, params.T2)

            rho_out, success = entanglement_swapping(
                rho1, rho2, params.p_bsm, params.gate_fidelity
            )
            if success and rho_out is not None:
                # Store the swapped pair at both ends
                ns_nb1 = self.node_states.get(nb1)
                ns_nb2 = self.node_states.get(nb2)
                if ns_nb1:
                    ns_nb1.store_pair(nb2, rho_out.copy())
                if ns_nb2:
                    ns_nb2.store_pair(nb1, rho_out.copy())
                f = bell_fidelity(rho_out)
                stats.update_fidelity(f)
                return self.payoff_fn.weights.alpha * f
            return -0.02

        elif action.action_type == ActionType.ENTANGLEMENT_PURIFY:
            nb = action.target_neighbor
            if nb is None or not ns.has_pair(nb):
                return 0.0
            # Need two pairs; if only one, skip
            rho1 = ns.memory_pairs.get(nb)
            rho2 = ns.memory_pairs.get(nb)  # simplified: same pair as approximation
            if rho1 is None:
                return 0.0

            rho_out, success = entanglement_purification(rho1, rho2, params.p_bsm)
            if success and rho_out is not None:
                ns.store_pair(nb, rho_out)
                stats.record_purification()
                f = bell_fidelity(rho_out)
                return self.payoff_fn.weights.alpha * (f - bell_fidelity(rho1))
            return -0.01

        elif action.action_type == ActionType.ROUTE:
            nb = action.target_neighbor
            if nb is None or not ns.has_pair(nb):
                return 0.0
            # Move pair toward destination
            rho = ns.consume_pair(nb)
            stats.update_fidelity(bell_fidelity(rho))
            return 0.01

        return 0.0

    def _check_delivery(self, rewards: Dict[int, float]) -> None:
        """Check if end-to-end entanglement has been delivered."""
        ns_src = self.node_states[self.src]
        ns_dst = self.node_states[self.dst]
        if ns_src.has_pair(self.dst) or ns_dst.has_pair(self.src):
            # Delivery!
            rho = ns_src.consume_pair(self.dst)
            if rho is None:
                rho = ns_dst.consume_pair(self.src)
            if rho is not None:
                f = bell_fidelity(rho)
                self.episode_stats[self.src].record_delivery(rho)
                self.episode_stats[self.dst].record_delivery(rho)
                bonus = self.payoff_fn.weights.alpha * f + self.payoff_fn.weights.beta
                rewards[self.src] += bonus
                rewards[self.dst] += bonus

    # ── Nash gap ───────────────────────────────────────────────────────────────

    def nash_gap(
        self,
        current_payoffs: Dict[int, float],
        strategies: Dict[int, Any],
        n_samples: int = 10,
    ) -> float:
        """
        Approximate Nash gap: max_i max_{a_i} [U_i(a_i, Lambda_{-i}) - U_i(Lambda)]
        """
        gaps = []
        for node_id in self.qnet.nodes:
            best_alt = current_payoffs.get(node_id, 0.0)
            for action in self.available_actions(node_id)[:n_samples]:
                # Approximate by current payoff + action-value heuristic
                alt_payoff = current_payoffs.get(node_id, 0.0)
                if action.action_type == ActionType.ENTANGLEMENT_SWAP:
                    alt_payoff += 0.05  # heuristic improvement estimate
                gaps.append(max(0.0, alt_payoff - current_payoffs.get(node_id, 0.0)))
        return float(np.max(gaps)) if gaps else 0.0

    def reset(self) -> None:
        """Reset game state."""
        for v in self.qnet.nodes:
            self.node_states[v] = NodeState(node_id=v)
            self.episode_stats[v] = EpisodeStats()
        self.global_step = 0

    def get_observation(self, node_id: int) -> np.ndarray:
        """
        Local observation for node_id:
        [fidelities_per_nb (padded), memory_occupancy, storage_times_per_nb (padded)]
        Fixed-size vector for RL.
        """
        ns    = self.node_states[node_id]
        nbs   = self.qnet.neighbors(node_id)
        max_nb = 8  # pad to this many neighbors

        fidelities   = np.zeros(max_nb)
        occupancy    = np.zeros(max_nb)
        storage_t    = np.zeros(max_nb)
        node_params  = self.qnet.node_params(node_id)

        for i, nb in enumerate(nbs[:max_nb]):
            if ns.has_pair(nb):
                fidelities[i] = bell_fidelity(ns.memory_pairs[nb])
                occupancy[i]  = 1.0
                storage_t[i]  = ns.storage_times.get(nb, 0.0) / node_params.T2

        obs = np.concatenate([
            fidelities,
            occupancy,
            storage_t,
            [ns.memory_usage() / max(node_params.n_memories, 1)],
        ])
        return obs.astype(np.float32)