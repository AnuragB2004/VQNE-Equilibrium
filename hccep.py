"""
Protocols: HCCEP and Baselines
================================
Implements:
  - HCCEP: Hybrid Cooperative-Competitive Entanglement Protocol
  - SP-Routing: Shortest-path (Dijkstra) baseline
  - Greedy-ED: Greedy entanglement distribution
  - Random: uniform random strategy
  - DQN-Single: single-agent DQN (classical policy, no VQC)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

from qneg import QNEG, Action, ActionType, EpisodeStats
from vqne import CorePeripheryPartition
from channels import bell_fidelity, PHI_PLUS
from graph import QuantumNetworkGraph


# ── Protocol base class ───────────────────────────────────────────────────────

class EntanglementProtocol:
    """Base class for all protocols."""

    name: str = "BaseProtocol"

    def select_actions(self, game: QNEG) -> Dict[int, Action]:
        raise NotImplementedError

    def run_episode(
        self,
        game:       QNEG,
        n_slots:    int = 10_000,
        n_trials:   int = 50,
        seed:       int = 42,
    ) -> Dict[str, float]:
        """Run protocol for n_slots time slots, averaged over n_trials."""
        rng = np.random.default_rng(seed)
        all_fidelities  = []
        all_rates       = []
        all_poas        = []

        for trial in range(n_trials):
            game.reset()
            np.random.seed(int(rng.integers(0, 10_000)))

            trial_fidelities = []
            trial_deliveries = 0

            for _slot in range(n_slots):
                actions = self.select_actions(game)
                rewards = game.step(actions)

                # Collect fidelity from deliveries
                stats_src = game.episode_stats[game.src]
                if stats_src.n_delivered > trial_deliveries:
                    trial_deliveries = stats_src.n_delivered
                    if stats_src.fidelities:
                        trial_fidelities.append(stats_src.fidelities[-1])

            all_fidelities.extend(trial_fidelities)
            rate = trial_deliveries / (n_slots * game.slot_duration)
            all_rates.append(rate)

        mean_f  = float(np.mean(all_fidelities)) if all_fidelities else 0.0
        std_f   = float(np.std(all_fidelities))  if all_fidelities else 0.0
        mean_r  = float(np.mean(all_rates))
        std_r   = float(np.std(all_rates))

        return {
            "protocol":       self.name,
            "mean_fidelity":  mean_f,
            "std_fidelity":   std_f,
            "mean_rate":      mean_r,
            "std_rate":       std_r,
        }


# ── HCCEP ─────────────────────────────────────────────────────────────────────

class HCCEP(EntanglementProtocol):
    """
    Hybrid Cooperative-Competitive Entanglement Protocol.

    Core nodes: cooperatively maximize joint welfare (prioritize swap + purify)
    Periphery:  competitive best-response greedy
    """

    name = "HCCEP"

    def __init__(self, game: QNEG, trained_agents: Optional[Dict] = None):
        self.game      = game
        self.partition = CorePeripheryPartition(game.qnet)
        self.agents    = trained_agents  # optional trained MARL agents

    def select_actions(self, game: QNEG) -> Dict[int, Action]:
        actions = {}

        for v in game.qnet.nodes:
            ns    = game.node_states[v]
            avail = game.available_actions(v)

            if v in self.partition.core:
                actions[v] = self._core_strategy(v, ns, avail, game)
            else:
                actions[v] = self._periphery_strategy(v, ns, avail, game)

        return actions

    def _core_strategy(self, v: int, ns, avail, game: QNEG) -> Action:
        """
        Core cooperative strategy:
        Priority: EP > ES > RD toward dst > IDLE
        """
        # Priority 1: purification (improves fidelity)
        for a in avail:
            if a.action_type == ActionType.ENTANGLEMENT_PURIFY:
                nb = a.target_neighbor
                if nb and ns.has_pair(nb):
                    rho = ns.memory_pairs[nb]
                    if bell_fidelity(rho) < 0.85:  # only purify if below threshold
                        return a

        # Priority 2: entanglement swap toward destination
        dst = game.dst
        path = game.qnet.shortest_path(v, dst)
        if len(path) > 1:
            next_hop = path[1]
            for a in avail:
                if (a.action_type == ActionType.ENTANGLEMENT_SWAP
                        and a.swap_pair
                        and next_hop in a.swap_pair):
                    return a

        # Priority 3: best available swap by fidelity
        best_swap = None
        best_f    = -1.0
        for a in avail:
            if a.action_type == ActionType.ENTANGLEMENT_SWAP and a.swap_pair:
                nb1, nb2 = a.swap_pair
                if ns.has_pair(nb1) and ns.has_pair(nb2):
                    f = min(bell_fidelity(ns.memory_pairs[nb1]),
                            bell_fidelity(ns.memory_pairs[nb2]))
                    if f > best_f:
                        best_f = f
                        best_swap = a
        if best_swap:
            return best_swap

        # Priority 4: route toward destination
        for a in avail:
            if a.action_type == ActionType.ROUTE:
                return a

        return Action(ActionType.IDLE)

    def _periphery_strategy(self, v: int, ns, avail, game: QNEG) -> Action:
        """
        Periphery greedy strategy: best immediate action.
        """
        # If near destination, try to deliver
        if v == game.src or v == game.dst:
            for a in avail:
                if a.action_type == ActionType.ENTANGLEMENT_SWAP:
                    return a
            for a in avail:
                if a.action_type == ActionType.ROUTE:
                    return a

        # Otherwise greedy swap
        for a in avail:
            if a.action_type == ActionType.ENTANGLEMENT_SWAP:
                return a

        return Action(ActionType.IDLE)


# ── SP-Routing (Dijkstra) baseline ───────────────────────────────────────────

class SPRouting(EntanglementProtocol):
    """
    Shortest-path routing: always swap along the shortest path, no purification.
    """

    name = "SP-Routing"

    def __init__(self, game: QNEG):
        self.game = game
        self._path_cache: Dict[Tuple, List] = {}

    def select_actions(self, game: QNEG) -> Dict[int, Action]:
        path = game.qnet.shortest_path(game.src, game.dst)
        path_set = set(path)
        actions = {}

        for v in game.qnet.nodes:
            ns    = game.node_states[v]
            avail = game.available_actions(v)

            if v in path_set:
                # On shortest path: try to swap toward destination
                best = Action(ActionType.IDLE)
                for a in avail:
                    if a.action_type == ActionType.ENTANGLEMENT_SWAP:
                        best = a
                        break
                actions[v] = best
            else:
                actions[v] = Action(ActionType.IDLE)

        return actions


# ── Greedy-ED baseline ────────────────────────────────────────────────────────

class GreedyED(EntanglementProtocol):
    """
    Greedy Entanglement Distribution: always select the highest-fidelity link action.
    """

    name = "Greedy-ED"

    def select_actions(self, game: QNEG) -> Dict[int, Action]:
        actions = {}

        for v in game.qnet.nodes:
            ns    = game.node_states[v]
            avail = game.available_actions(v)

            best_action = Action(ActionType.IDLE)
            best_score  = -1.0

            for a in avail:
                score = self._score_action(a, ns, game)
                if score > best_score:
                    best_score  = score
                    best_action = a

            actions[v] = best_action

        return actions

    def _score_action(self, action: Action, ns, game: QNEG) -> float:
        if action.action_type == ActionType.ENTANGLEMENT_SWAP and action.swap_pair:
            nb1, nb2 = action.swap_pair
            if ns.has_pair(nb1) and ns.has_pair(nb2):
                return min(bell_fidelity(ns.memory_pairs[nb1]),
                           bell_fidelity(ns.memory_pairs[nb2]))
        elif action.action_type == ActionType.ENTANGLEMENT_PURIFY:
            nb = action.target_neighbor
            if nb and ns.has_pair(nb):
                return bell_fidelity(ns.memory_pairs[nb]) + 0.1
        elif action.action_type == ActionType.ROUTE:
            return 0.01
        return 0.0


# ── Random baseline ───────────────────────────────────────────────────────────

class RandomProtocol(EntanglementProtocol):
    """Uniform random strategy selection."""

    name = "Random"

    def select_actions(self, game: QNEG) -> Dict[int, Action]:
        return {
            v: random.choice(game.available_actions(v))
            for v in game.qnet.nodes
        }


# ── DQN-Single (classical, no VQC) ───────────────────────────────────────────

class ClassicalDQN(nn.Module):
    """Classical DQN network (no quantum circuits)."""

    def __init__(self, n_inputs: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNSingle(EntanglementProtocol):
    """
    Single-agent DQN with classical policy network.
    Centralized controller for all nodes.
    """

    name = "DQN-Single"

    def __init__(
        self,
        game:      QNEG,
        lr:        float = 1e-3,
        gamma:     float = 0.99,
        epsilon:   float = 1.0,
        eps_decay: float = 0.995,
        eps_min:   float = 0.05,
        buffer_cap: int  = 50_000,
        batch_size: int  = 256,
        n_actions:  int  = 20,
    ):
        self.game       = game
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.eps_decay  = eps_decay
        self.eps_min    = eps_min
        self.batch_size = batch_size
        self.n_actions  = n_actions

        sample_obs = game.get_observation(game.qnet.nodes[0])
        n_inputs   = sample_obs.shape[0]

        self.q_net     = ClassicalDQN(n_inputs, n_actions)
        self.target_net = ClassicalDQN(n_inputs, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer     = deque(maxlen=buffer_cap)

    def select_actions(self, game: QNEG) -> Dict[int, Action]:
        actions = {}
        for v in game.qnet.nodes:
            obs   = game.get_observation(v)
            avail = game.available_actions(v)
            n_a   = min(len(avail), self.n_actions)

            if np.random.random() < self.epsilon:
                a_idx = np.random.randint(n_a)
            else:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    q_vals = self.q_net(obs_t)[0, :n_a]
                a_idx = int(q_vals.argmax().item())

            actions[v] = avail[min(a_idx, n_a - 1)]

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        return actions

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        obs_t     = torch.FloatTensor(np.stack([b[0] for b in batch]))
        acts_t    = torch.LongTensor([b[1] for b in batch])
        rew_t     = torch.FloatTensor([b[2] for b in batch])
        next_obs_t = torch.FloatTensor(np.stack([b[3] for b in batch]))
        done_t    = torch.FloatTensor([b[4] for b in batch])

        # Current Q values
        q_curr = self.q_net(obs_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            q_next = self.target_net(next_obs_t).max(1)[0]
            q_target = rew_t + self.gamma * q_next * (1 - done_t)

        loss = nn.functional.mse_loss(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def pretrain(self, n_steps: int = 2000) -> None:
        """Pretrain DQN on random exploration."""
        game = self.game
        game.reset()

        for step in range(n_steps):
            obs = {v: game.get_observation(v) for v in game.qnet.nodes}
            actions = self.select_actions(game)
            rewards = game.step(actions)
            next_obs = {v: game.get_observation(v) for v in game.qnet.nodes}

            for v in game.qnet.nodes:
                avail  = game.available_actions(v)
                a_idx  = min(0, len(avail) - 1)
                self.buffer.append((
                    obs[v], a_idx,
                    rewards.get(v, 0.0),
                    next_obs[v], False,
                ))

            self.train_step()

            if (step + 1) % 500 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())