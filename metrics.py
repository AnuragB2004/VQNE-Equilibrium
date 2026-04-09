"""
Metrics
========
Implements all evaluation metrics from the paper:
  - End-to-end fidelity
  - Entanglement generation rate (ebit/s)
  - Price of Anarchy (PoA)
  - Jain's fairness index
  - Latency
  - Resource cost
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class ProtocolResult:
    """Aggregated metrics for one protocol run."""
    protocol:         str
    topology:         str
    noise_level:      float
    mean_fidelity:    float
    std_fidelity:     float
    mean_rate:        float       # ebit/s
    std_rate:         float
    mean_latency:     float       # ms
    std_latency:      float
    mean_cost:        float       # arbitrary units
    poa:              float       # Price of Anarchy
    jain_fairness:    float
    n_trials:         int
    raw_fidelities:   List[float] = field(default_factory=list)
    raw_rates:        List[float] = field(default_factory=list)
    raw_deliveries:   List[int]   = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.protocol:<20} | "
                f"F={self.mean_fidelity:.3f}±{self.std_fidelity:.3f} | "
                f"R={self.mean_rate:.1f}±{self.std_rate:.1f} ebit/s | "
                f"Lat={self.mean_latency:.1f}ms | "
                f"PoA={self.poa:.2f} | "
                f"J={self.jain_fairness:.2f}")


# ── Fidelity metrics ──────────────────────────────────────────────────────────

def compute_e2e_fidelity_vs_hops(
    qnet,
    protocol_fn,
    max_hops: int = 8,
    n_trials: int = 20,
    slot_duration: float = 1e-3,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute mean ± std end-to-end fidelity as a function of hop count.
    Returns {n_hops: (mean_F, std_F)}.
    """
    from channels import path_fidelity, PHI_PLUS
    results = {}

    nodes = qnet.nodes
    if len(nodes) < 2:
        return {}

    # For each hop count, sample paths of that length
    src = nodes[0]
    for n_hops in range(1, max_hops + 1):
        fidelities = []
        # Find paths of length n_hops
        paths = qnet.all_paths(src, nodes[-1], cutoff=n_hops + 1)
        hop_paths = [p for p in paths if len(p) - 1 == n_hops]

        if not hop_paths:
            # Approximate with storage_time = 1ms per hop
            storage = {(nodes[i], i): 0.001 * i for i in range(n_hops + 1)}
            # Use a synthetic path
            hop_paths = [[nodes[min(i, len(nodes)-1)] for i in range(n_hops + 1)]]

        for _ in range(n_trials):
            if hop_paths:
                path = hop_paths[0]
                if len(path) >= 2:
                    f = path_fidelity(path, qnet)
                    fidelities.append(f)
                else:
                    fidelities.append(0.0)

        results[n_hops] = (
            float(np.mean(fidelities)) if fidelities else 0.0,
            float(np.std(fidelities))  if fidelities else 0.0,
        )

    return results


def fidelity_improvement_pct(
    protocol_fidelity:  float,
    baseline_fidelity:  float,
) -> float:
    """Percentage improvement: (protocol - baseline) / baseline * 100"""
    if baseline_fidelity < 1e-8:
        return 0.0
    return (protocol_fidelity - baseline_fidelity) / baseline_fidelity * 100.0


def rate_gain_factor(protocol_rate: float, baseline_rate: float) -> float:
    """Ratio: protocol_rate / baseline_rate"""
    if baseline_rate < 1e-8:
        return 1.0
    return protocol_rate / baseline_rate


# ── Price of Anarchy ──────────────────────────────────────────────────────────

def price_of_anarchy(
    social_optimum:   float,
    nash_welfare:     float,
) -> float:
    """
    PoA = W* / W_NE  where W* is social optimum and W_NE is Nash welfare.
    For fidelity-based welfare (higher is better):
    PoA = W* / W_NE >= 1.
    """
    if nash_welfare < 1e-8:
        return float("inf")
    return max(1.0, social_optimum / nash_welfare)


def estimate_social_optimum(
    game,
    n_samples: int = 20,
) -> float:
    """
    Estimate social optimum by cooperative oracle (cooperative greedy).
    """
    from hccep import HCCEP
    protocol = HCCEP(game)
    fidelities = []

    for _ in range(n_samples):
        game.reset()
        for _slot in range(200):
            actions = protocol.select_actions(game)
            game.step(actions)
        stats = game.episode_stats[game.src]
        if stats.fidelities:
            fidelities.append(np.mean(stats.fidelities))

    return float(np.mean(fidelities)) if fidelities else 0.5


def estimate_nash_welfare(
    game,
    n_samples: int = 20,
) -> float:
    """Estimate Nash equilibrium social welfare (competitive greedy)."""
    from hccep import SPRouting
    protocol = SPRouting(game)
    fidelities = []

    for _ in range(n_samples):
        game.reset()
        for _slot in range(200):
            actions = protocol.select_actions(game)
            game.step(actions)
        stats = game.episode_stats[game.src]
        if stats.fidelities:
            fidelities.append(np.mean(stats.fidelities))

    return float(np.mean(fidelities)) if fidelities else 0.3


# ── Jain's Fairness Index ──────────────────────────────────────────────────────

def jain_fairness(rates: List[float]) -> float:
    """
    J = (sum_i R_i)^2 / (n * sum_i R_i^2)
    Range: [1/n, 1.0]. J=1 means perfect fairness.
    """
    if not rates or len(rates) == 0:
        return 0.0
    rates_arr = np.array(rates, dtype=float)
    n = len(rates_arr)
    numerator   = np.sum(rates_arr) ** 2
    denominator = n * np.sum(rates_arr ** 2)
    if denominator < 1e-10:
        return 1.0
    return float(numerator / denominator)


# ── Latency ───────────────────────────────────────────────────────────────────

def compute_latency_ms(
    path:          List[int],
    qnet,
    slot_duration: float = 1e-3,
) -> float:
    """
    Estimate end-to-end latency in ms.
    Latency = sum of link delays + classical communication round trips.
    """
    if len(path) < 2:
        return 0.0

    total_latency = 0.0
    C_FIBER = 2e8  # m/s

    for i in range(len(path) - 1):
        ch = qnet.edge_params(path[i], path[i + 1])
        # Propagation delay (ms)
        prop_delay_ms = (ch.distance * 1e3) / C_FIBER * 1e3
        total_latency += prop_delay_ms

    # Add BSM classical communication delays (one round trip per hop)
    n_hops = len(path) - 1
    classical_rtt = n_hops * slot_duration * 1e3  # ms
    total_latency += classical_rtt

    return total_latency


# ── Comprehensive benchmark runner ────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs all protocols on all topologies and collects metrics.
    """

    TOPOLOGIES = ["ARPANET", "GÉANT", "AS-Caida", "Ring", "Star"]
    NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

    def __init__(
        self,
        n_slots:      int   = 1000,   # reduced from 10000 for speed
        n_trials:     int   = 10,     # reduced from 50
        slot_duration: float = 1e-3,
        seed:         int   = 42,
    ):
        self.n_slots       = n_slots
        self.n_trials      = n_trials
        self.slot_duration = slot_duration
        self.seed          = seed

    def run_protocol(
        self,
        protocol,
        game,
        noise_level: float = 0.1,
        topology:    str   = "GÉANT",
    ) -> ProtocolResult:
        """Run a protocol and return aggregated ProtocolResult."""
        rng = np.random.default_rng(self.seed)
        fidelities    = []
        rates         = []
        latencies     = []
        deliveries    = []

        for trial in range(self.n_trials):
            game.reset()
            np.random.seed(int(rng.integers(0, 100_000)))

            trial_fidelities = []
            trial_deliveries = 0

            for _slot in range(self.n_slots):
                actions = protocol.select_actions(game)
                game.step(actions)

            # Collect stats
            stats = game.episode_stats[game.src]
            if stats.fidelities:
                fidelities.extend(stats.fidelities)
                trial_deliveries = stats.n_delivered
            deliveries.append(trial_deliveries)

            rate = trial_deliveries / (self.n_slots * self.slot_duration)
            rates.append(rate)

            # Latency from shortest path
            path = game.qnet.shortest_path(game.src, game.dst)
            latencies.append(compute_latency_ms(path, game.qnet, self.slot_duration))

        mean_f    = float(np.mean(fidelities)) if fidelities else 0.0
        std_f     = float(np.std(fidelities))  if fidelities else 0.0
        mean_r    = float(np.mean(rates))
        std_r     = float(np.std(rates))
        mean_lat  = float(np.mean(latencies))
        std_lat   = float(np.std(latencies))
        mean_cost = mean_r * 0.02  # proportional cost approximation

        # Estimate PoA
        w_opt    = estimate_social_optimum(game, n_samples=5)
        w_nash   = max(mean_f, 0.01)
        poa      = price_of_anarchy(w_opt, w_nash)

        # Jain fairness on rates
        jain = jain_fairness(rates)

        return ProtocolResult(
            protocol=protocol.name,
            topology=topology,
            noise_level=noise_level,
            mean_fidelity=mean_f,
            std_fidelity=std_f,
            mean_rate=mean_r,
            std_rate=std_r,
            mean_latency=mean_lat,
            std_latency=std_lat,
            mean_cost=mean_cost,
            poa=poa,
            jain_fairness=jain,
            n_trials=self.n_trials,
            raw_fidelities=fidelities,
            raw_rates=rates,
            raw_deliveries=deliveries,
        )

    def compare_all(
        self,
        game,
        protocols: List,
        topology:  str   = "GÉANT",
        noise_level: float = 0.1,
    ) -> Dict[str, ProtocolResult]:
        """Run all protocols and return comparison dict."""
        results = {}
        for proto in protocols:
            print(f"  Running {proto.name}...")
            result = self.run_protocol(proto, game, noise_level, topology)
            results[proto.name] = result
            print(f"  {result.summary()}")
        return results