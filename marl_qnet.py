"""
MARL-QNet: Multi-Agent RL with Quantum Circuits
================================================
Implements Algorithm 1 from the paper.

Training phases:
  Phase 1: Initialization
  Phase 2: Cooperative Core Pre-training
  Phase 3: Main MARL Loop (quantum natural gradient policy updates)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

from circuit import VariationalQuantumCircuit, QFIMEstimator
from qneg import QNEG, Action, ActionType
from vqne import CorePeripheryPartition, CoreCharacteristicFunction, shapley_value
from graph import QuantumNetworkGraph


# ── Replay Buffer ─────────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs:      np.ndarray
    action:   int
    reward:   float
    next_obs: np.ndarray
    done:     bool
    log_prob: float


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ── Per-agent learner ─────────────────────────────────────────────────────────

class AgentLearner:
    """One agent's VQC policy + optimizer + replay buffer."""

    def __init__(
        self,
        node_id:    int,
        n_inputs:   int,
        n_actions:  int,
        n_qubits:   int = 4,
        n_layers:   int = 6,
        lr:         float = 5e-3,
        gamma:      float = 0.99,
        buffer_cap: int   = 50_000,
        use_qng:    bool  = True,
    ):
        self.node_id  = node_id
        self.gamma    = gamma
        self.use_qng  = use_qng
        self.n_actions = n_actions

        # Policy network (VQC)
        self.policy = VariationalQuantumCircuit(n_inputs, n_actions, n_qubits, n_layers)
        # Target network (soft-updated)
        self.target = VariationalQuantumCircuit(n_inputs, n_actions, n_qubits, n_layers)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_cap)
        self.qfim_est  = QFIMEstimator(self.policy, damping=1e-3)

        # Trajectory buffer for REINFORCE
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards:   List[float]        = []

        # Statistics
        self.total_reward = 0.0
        self.update_count = 0

    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        return self.policy.select_action(obs)

    def store_transition(self, t: Transition) -> None:
        self.buffer.push(t)
        self.episode_log_probs.append(torch.tensor(t.log_prob))
        self.episode_rewards.append(t.reward)
        self.total_reward += t.reward

    def update(
        self,
        batch_size:    int   = 256,
        shapley_bonus: float = 0.0,
    ) -> float:
        """
        Policy gradient update (REINFORCE + quantum natural gradient).
        Returns the loss value.
        """
        if len(self.buffer) < batch_size:
            return 0.0

        batch = self.buffer.sample(batch_size)

        obs_t     = torch.FloatTensor(np.stack([t.obs for t in batch]))
        actions_t = torch.LongTensor([t.action for t in batch])
        rewards_t = torch.FloatTensor([t.reward for t in batch])
        next_obs_t = torch.FloatTensor(np.stack([t.next_obs for t in batch]))
        dones_t   = torch.FloatTensor([float(t.done) for t in batch])

        # Compute discounted returns
        returns = self._compute_returns(rewards_t.numpy())
        returns_t = torch.FloatTensor(returns)
        # Normalize returns
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy gradient loss
        log_probs = self.policy.get_log_prob(obs_t, actions_t)
        loss      = -(log_probs * returns_t).mean()

        # Add Shapley bonus for core nodes
        if shapley_bonus != 0.0:
            loss = loss - shapley_bonus * log_probs.mean()

        self.optimizer.zero_grad()

        if self.use_qng:
            # Quantum Natural Gradient
            loss.backward()
            raw_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                         for p in self.policy.parameters()]

            # Compute diagonal QFIM
            diag_F = self.qfim_est.diagonal_qfim(obs_t[:32])  # subsample for efficiency

            # Apply natural gradient
            nat_grads = self.qfim_est.natural_gradient(raw_grads, diag_F)

            # Manually set gradients
            self.optimizer.zero_grad()
            for p, ng in zip(self.policy.parameters(), nat_grads):
                if ng is not None:
                    p.grad = ng
        else:
            loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_count += 1
        return loss.item()

    def _compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}."""
        T       = len(rewards)
        returns = np.zeros(T)
        G       = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def soft_update_target(self, tau: float = 0.005) -> None:
        """Polyak averaging: theta_target = tau*theta + (1-tau)*theta_target."""
        for tp, sp in zip(self.target.parameters(), self.policy.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    def get_param_vector(self) -> np.ndarray:
        return np.concatenate([p.detach().numpy().flatten()
                               for p in self.policy.parameters()])

    def param_change(self, prev_params: np.ndarray) -> float:
        current = self.get_param_vector()
        return float(np.linalg.norm(current - prev_params))


# ── MARL-QNet ─────────────────────────────────────────────────────────────────

class MARLQNet:
    """
    Algorithm 1: MARL-QNet Multi-Agent RL for QNEG.

    Phases:
      1. Initialization
      2. Cooperative Core Pre-training (T_coop iterations)
      3. Main MARL loop (T_max iterations)
    """

    def __init__(
        self,
        game:        QNEG,
        n_qubits:    int   = 4,
        n_layers:    int   = 6,
        lr:          float = 5e-3,
        gamma:       float = 0.99,
        batch_size:  int   = 256,
        T_coop:      int   = 500,
        T_max:       int   = 5_000,
        T_ep:        int   = 20,
        target_freq: int   = 50,
        eps_conv:    float = 1e-4,
        use_qng:     bool  = True,
        seed:        int   = 42,
    ):
        self.game       = game
        self.batch_size = batch_size
        self.T_coop     = T_coop
        self.T_max      = T_max
        self.T_ep       = T_ep
        self.target_freq = target_freq
        self.eps_conv   = eps_conv

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ── Phase 1: Partition ─────────────────────────────────────────────────
        self.partition = CorePeripheryPartition(game.qnet)
        print(f"[MARL-QNet] {self.partition}")

        # ── Determine observation and action dimensions ─────────────────────────
        sample_node = game.qnet.nodes[0]
        obs_dim     = game.get_observation(sample_node).shape[0]
        # Action space: ES pairs + EP + RD per neighbor + IDLE
        max_actions = 20  # fixed upper bound; masked at runtime

        # ── Initialize agents ──────────────────────────────────────────────────
        self.agents: Dict[int, AgentLearner] = {}
        for v in game.qnet.nodes:
            self.agents[v] = AgentLearner(
                node_id=v,
                n_inputs=obs_dim,
                n_actions=max_actions,
                n_qubits=n_qubits,
                n_layers=n_layers,
                lr=lr,
                gamma=gamma,
                use_qng=use_qng,
            )

        # Training history
        self.nash_gap_history:  List[float] = []
        self.reward_history:    List[float] = []
        self.param_change_hist: List[float] = []
        self.loss_history:      List[float] = []

    # ── Phase 2: Cooperative core pre-training ────────────────────────────────

    def _pretrain_core(self) -> None:
        """
        Jointly optimize sum_{i in C} U_i via centralized gradient.
        Core nodes share experiences and update together.
        """
        print(f"[Phase 2] Pre-training cooperative core ({self.T_coop} iters)...")
        core_nodes = list(self.partition.core)

        for t in range(self.T_coop):
            self.game.reset()
            obs = {v: self.game.get_observation(v) for v in core_nodes}

            for _ in range(self.T_ep):
                actions_dict = {}
                log_probs    = {}

                for v in core_nodes:
                    avail  = self.game.available_actions(v)
                    n_avail = min(len(avail), self.agents[v].n_actions)
                    obs_t  = torch.FloatTensor(obs[v]).unsqueeze(0)
                    probs  = self.agents[v].policy.action_probs(obs_t)[0, :n_avail]
                    # Renormalize
                    probs = probs / (probs.sum() + 1e-8)
                    a_idx = torch.multinomial(probs.detach(), 1).item()
                    a_idx = min(int(a_idx), n_avail - 1)
                    actions_dict[v] = avail[a_idx]
                    log_probs[v]    = float(torch.log(probs[a_idx] + 1e-8))

                # Add idle actions for periphery
                for v in self.partition.periphery:
                    actions_dict[v] = Action(ActionType.IDLE)

                rewards  = self.game.step(actions_dict)
                next_obs = {v: self.game.get_observation(v) for v in core_nodes}

                for v in core_nodes:
                    self.agents[v].store_transition(Transition(
                        obs=obs[v], action=0,
                        reward=rewards.get(v, 0.0) + sum(rewards.values()) * 0.1,
                        next_obs=next_obs[v], done=False,
                        log_prob=log_probs[v],
                    ))
                obs = next_obs

            # Update core agents jointly
            for v in core_nodes:
                self.agents[v].update(self.batch_size)

            if (t + 1) % 100 == 0:
                print(f"  [Pretrain] iter {t+1}/{self.T_coop}")

        # Compute Shapley values
        payoff_ests = {v: self.agents[v].total_reward for v in core_nodes}
        char_fn     = CoreCharacteristicFunction(payoff_ests)
        self.shapley_vals = shapley_value(
            core_nodes, char_fn, n_samples=min(200, 10 * len(core_nodes))
        )
        print(f"[Phase 2] Shapley values computed for {len(self.shapley_vals)} core nodes.")

    # ── Phase 3: Main MARL loop ───────────────────────────────────────────────

    def train(self, verbose: bool = True) -> Dict:
        """
        Full MARL-QNet training (Algorithm 1).
        Returns training history dict.
        """
        # Phase 2
        self._pretrain_core()

        # Phase 3
        print(f"[Phase 3] Main MARL loop ({self.T_max} iters)...")
        all_nodes = self.game.qnet.nodes

        prev_params = {v: self.agents[v].get_param_vector() for v in all_nodes}
        delta = float("inf")
        k     = 0

        while k < self.T_max and delta > self.eps_conv:
            self.game.reset()
            obs = {v: self.game.get_observation(v) for v in all_nodes}

            ep_rewards = {v: 0.0 for v in all_nodes}

            # ── Episode ──────────────────────────────────────────────────────
            for _t in range(self.T_ep):
                actions_dict = {}
                log_probs_dict = {}

                for v in all_nodes:
                    avail  = self.game.available_actions(v)
                    n_avail = min(len(avail), self.agents[v].n_actions)
                    obs_t  = torch.FloatTensor(obs[v]).unsqueeze(0)
                    with torch.no_grad():
                        probs = self.agents[v].policy.action_probs(obs_t)[0, :n_avail]
                    probs = probs / (probs.sum() + 1e-8)
                    a_idx = torch.multinomial(probs, 1).item()
                    a_idx = min(int(a_idx), n_avail - 1)
                    actions_dict[v]    = avail[a_idx]
                    log_probs_dict[v]  = float(torch.log(probs[a_idx] + 1e-8))

                rewards  = self.game.step(actions_dict)
                next_obs = {v: self.game.get_observation(v) for v in all_nodes}

                for v in all_nodes:
                    ep_rewards[v] += rewards.get(v, 0.0)
                    self.agents[v].store_transition(Transition(
                        obs=obs[v], action=0,
                        reward=rewards.get(v, 0.0),
                        next_obs=next_obs[v], done=False,
                        log_prob=log_probs_dict[v],
                    ))
                obs = next_obs

            # ── Strategy Updates (parallel) ───────────────────────────────────
            total_loss = 0.0
            for v in all_nodes:
                shapley_bonus = self.shapley_vals.get(v, 0.0) * 0.01 \
                    if v in self.partition.core else 0.0
                loss = self.agents[v].update(self.batch_size, shapley_bonus)
                total_loss += loss

            # ── Convergence check ─────────────────────────────────────────────
            delta = max(
                self.agents[v].param_change(prev_params[v])
                for v in all_nodes
            )
            prev_params = {v: self.agents[v].get_param_vector() for v in all_nodes}

            mean_reward = float(np.mean(list(ep_rewards.values())))
            self.reward_history.append(mean_reward)
            self.param_change_hist.append(delta)
            self.loss_history.append(total_loss / len(all_nodes))

            # ── Periodic logging ──────────────────────────────────────────────
            if k % self.target_freq == 0:
                # Approximate Nash gap
                current_payoffs = ep_rewards
                ng = self.game.nash_gap(current_payoffs, {})
                self.nash_gap_history.append(ng)

                # Soft-update target networks
                for v in all_nodes:
                    self.agents[v].soft_update_target(tau=0.005)

                if verbose:
                    print(f"  [k={k:4d}] Nash gap={ng:.4f}  "
                          f"delta={delta:.2e}  "
                          f"mean_reward={mean_reward:.4f}  "
                          f"loss={total_loss/len(all_nodes):.4f}")

            k += 1

        print(f"[Phase 3] Converged at k={k}, delta={delta:.2e}")
        return {
            "nash_gap_history":  self.nash_gap_history,
            "reward_history":    self.reward_history,
            "param_change":      self.param_change_hist,
            "loss_history":      self.loss_history,
            "n_iterations":      k,
            "converged":         delta <= self.eps_conv,
        }

    def get_vqne(self) -> Dict[int, np.ndarray]:
        """Return the VQNE parameter profile theta*."""
        return {v: self.agents[v].get_param_vector()
                for v in self.game.qnet.nodes}

    def evaluate(self, n_episodes: int = 50) -> Dict[str, float]:
        """Evaluate trained policies over n_episodes."""
        all_fidelities  = []
        all_rates       = []
        all_deliveries  = []

        for ep in range(n_episodes):
            self.game.reset()
            obs = {v: self.game.get_observation(v) for v in self.game.qnet.nodes}
            ep_deliveries = 0

            for _t in range(self.T_ep * 5):
                actions_dict = {}
                for v in self.game.qnet.nodes:
                    avail  = self.game.available_actions(v)
                    n_avail = min(len(avail), self.agents[v].n_actions)
                    a_idx, _ = self.agents[v].select_action(obs[v])
                    a_idx = min(a_idx, n_avail - 1)
                    actions_dict[v] = avail[a_idx]

                self.game.step(actions_dict)
                obs = {v: self.game.get_observation(v) for v in self.game.qnet.nodes}

            # Collect per-episode stats
            for v in self.game.qnet.nodes:
                stats = self.game.episode_stats[v]
                if stats.fidelities:
                    all_fidelities.extend(stats.fidelities)
                all_deliveries.append(stats.n_delivered)

        return {
            "mean_fidelity":  float(np.mean(all_fidelities)) if all_fidelities else 0.0,
            "std_fidelity":   float(np.std(all_fidelities))  if all_fidelities else 0.0,
            "mean_deliveries": float(np.mean(all_deliveries)),
            "mean_rate_ebit_s": float(np.mean(all_deliveries)) / (self.T_ep * 5 * 1e-3),
        }


# ── Iterative Best Response for Periphery (Algorithm 2) ───────────────────────

class IterativeBestResponse:
    """
    Algorithm 2: IBR for competitive periphery nodes.
    Periphery nodes best-respond against fixed core strategy.
    """

    def __init__(
        self,
        game:      QNEG,
        partition: CorePeripheryPartition,
        epsilon:   float = 1e-3,
        max_iter:  int   = 1000,
    ):
        self.game      = game
        self.partition = partition
        self.epsilon   = epsilon
        self.max_iter  = max_iter

    def run(self, core_strategies: Dict[int, np.ndarray]) -> Dict[str, any]:
        """
        Run IBR for periphery nodes given fixed core strategies.
        Returns convergence info.
        """
        periph_nodes = list(self.partition.periphery)
        if not periph_nodes:
            return {"converged": True, "n_iterations": 0}

        prev_payoffs = {v: 0.0 for v in periph_nodes}
        t = 0
        converged = False

        while t < self.max_iter:
            # Shuffle periphery order (random order per iteration)
            rng = np.random.default_rng(t)
            order = rng.permutation(periph_nodes).tolist()

            curr_payoffs = {}
            for v in order:
                # Best-response: greedily pick best available action
                best_r = -float("inf")
                best_a = Action(ActionType.IDLE)

                for action in self.game.available_actions(v):
                    # Simulate action and get reward
                    reward = self._simulate_action(v, action)
                    if reward > best_r:
                        best_r = reward
                        best_a = action

                curr_payoffs[v] = best_r

            # Check convergence
            max_change = max(
                abs(curr_payoffs.get(v, 0) - prev_payoffs.get(v, 0))
                for v in periph_nodes
            )
            if max_change <= self.epsilon:
                converged = True
                break

            prev_payoffs = curr_payoffs
            t += 1

        return {"converged": converged, "n_iterations": t, "payoffs": prev_payoffs}

    def _simulate_action(self, node_id: int, action: Action) -> float:
        """Estimate reward for a single action (heuristic)."""
        ns = self.game.node_states[node_id]

        if action.action_type == ActionType.IDLE:
            return 0.0
        elif action.action_type == ActionType.ENTANGLEMENT_SWAP:
            if action.swap_pair:
                nb1, nb2 = action.swap_pair
                if ns.has_pair(nb1) and ns.has_pair(nb2):
                    f1 = float(np.real(np.trace(
                        ns.memory_pairs[nb1]
                    ))) * 0.5
                    return f1 * self.game.payoff_fn.weights.alpha
            return -0.01
        elif action.action_type == ActionType.ROUTE:
            return 0.005
        return 0.0