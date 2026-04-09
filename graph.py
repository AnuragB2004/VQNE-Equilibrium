"""
Quantum Network Graph and Channel Model
========================================
Implements Definition 1 (Quantum Network Graph) from the paper.

Each node is a quantum repeater/end-user.
Each edge carries physical channel parameters:
  (p_ij, gamma_ij, eta_ij, T2_ij)
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Physical constants ────────────────────────────────────────────────────────
L_ATT_KM = 22.0          # Attenuation length in standard SMF-28 fiber (km)
C_FIBER   = 2e8          # Speed of light in fiber (m/s)
LOSS_DB_PER_KM = 0.1     # SMF-28 fiber loss (dB/km)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ChannelParams:
    """Physical parameters for a quantum channel (edge)."""
    p_trans:   float  # photon transmission probability per km hop
    gamma:     float  # depolarizing rate (1/km)
    eta:       float  # detector efficiency
    T2:        float  # memory coherence time (s) at destination node
    distance:  float  # link distance (km)

    def raw_bell_fidelity(self) -> float:
        """
        Eq. (1): F0_ij = 1/4 * (1 + 3*exp(-gamma*d/c)) * p^(d/L_att)
        Returns the initial Bell-pair fidelity after transmission.
        """
        decay   = np.exp(-self.gamma * self.distance)
        loss    = (self.p_trans ** (self.distance / L_ATT_KM))
        return 0.25 * (1.0 + 3.0 * decay) * loss


@dataclass
class NodeParams:
    """Physical parameters for a quantum repeater node."""
    n_memories:    int    # number of memory registers
    T2:            float  # coherence time (s)
    p_bsm:         float  # BSM success probability
    gate_fidelity: float  # single-qubit gate fidelity
    cnot_fidelity: float  # two-qubit CNOT fidelity
    is_end_user:   bool = False

    def memory_fidelity(self, storage_time: float) -> float:
        """
        Eq. (3): F_mem(tau) = 1/4 * (1 + 3*exp(-tau/T2))
        """
        return 0.25 * (1.0 + 3.0 * np.exp(-storage_time / max(self.T2, 1e-9)))


# ── IBM Quantum-derived default noise parameters ──────────────────────────────

IBM_DEFAULT_NODE = NodeParams(
    n_memories=4,
    T2=50e-3,          # 50 ms NV-center
    p_bsm=0.75,
    gate_fidelity=1.0 - 0.002,
    cnot_fidelity=1.0 - 0.008,
    is_end_user=False,
)

IBM_DEFAULT_CHANNEL = ChannelParams(
    p_trans=np.exp(-LOSS_DB_PER_KM / 10 * np.log(10)),  # per-km
    gamma=0.01,
    eta=0.90,
    T2=50e-3,
    distance=50.0,
)


# ── QuantumNetworkGraph ───────────────────────────────────────────────────────

class QuantumNetworkGraph:
    """
    Weighted directed graph G = (V, E, W) as defined in Definition 1.

    Node attributes: NodeParams
    Edge attributes: ChannelParams
    """

    def __init__(self, seed: int = 42):
        self.G: nx.DiGraph = nx.DiGraph()
        self.rng = np.random.default_rng(seed)

    # ── Construction helpers ──────────────────────────────────────────────────

    def add_node(self, node_id: int, params: Optional[NodeParams] = None) -> None:
        p = params or IBM_DEFAULT_NODE
        self.G.add_node(node_id, params=p)

    def add_channel(
        self,
        src: int,
        dst: int,
        params: Optional[ChannelParams] = None,
        bidirectional: bool = True,
    ) -> None:
        p = params or IBM_DEFAULT_CHANNEL
        self.G.add_edge(src, dst, params=p)
        if bidirectional:
            self.G.add_edge(dst, src, params=p)

    # ── Named topology builders ───────────────────────────────────────────────

    @classmethod
    def arpanet(cls, seed: int = 42) -> "QuantumNetworkGraph":
        """20-node ARPANET-style topology (Table II)."""
        g = cls(seed=seed)
        n = 20
        for i in range(n):
            is_leaf = (i >= 14)
            g.add_node(i, NodeParams(
                n_memories=4 if not is_leaf else 2,
                T2=g.rng.uniform(10e-3, 100e-3),
                p_bsm=0.75,
                gate_fidelity=0.998,
                cnot_fidelity=0.992,
                is_end_user=is_leaf,
            ))
        # backbone ring + cross-links (32 edges total)
        backbone = list(range(14))
        for i in range(len(backbone)):
            j = backbone[(i + 1) % len(backbone)]
            g.add_channel(i, j, _random_channel(g.rng, 30, 80))
        cross = [(0,5),(1,7),(2,9),(3,11),(4,12)]
        for a, b in cross:
            g.add_channel(a, b, _random_channel(g.rng, 50, 150))
        # leaf attachments
        for leaf in range(14, 20):
            hub = leaf % 14
            g.add_channel(hub, leaf, _random_channel(g.rng, 10, 40))
        return g

    @classmethod
    def geant(cls, seed: int = 42) -> "QuantumNetworkGraph":
        """40-node GÉANT-style European topology (Table II)."""
        g = cls(seed=seed)
        n = 40
        for i in range(n):
            g.add_node(i, NodeParams(
                n_memories=g.rng.integers(2, 6),
                T2=g.rng.uniform(10e-3, 100e-3),
                p_bsm=0.75,
                gate_fidelity=0.998,
                cnot_fidelity=0.992,
                is_end_user=(i >= 35),
            ))
        # Erdos-Renyi backbone with p=0.08 → ~62 edges
        er = nx.erdos_renyi_graph(n, 0.08, seed=seed, directed=False)
        for u, v in er.edges():
            g.add_channel(u, v, _random_channel(g.rng, 30, 200))
        # Ensure connectivity
        _ensure_connected(g, g.rng)
        return g

    @classmethod
    def as_caida(cls, seed: int = 42) -> "QuantumNetworkGraph":
        """100-node AS-Caida-style topology (Table II)."""
        g = cls(seed=seed)
        n = 100
        # Power-law (Barabasi-Albert) degree distribution matching AS topology
        ba = nx.barabasi_albert_graph(n, 3, seed=seed)
        for i in range(n):
            deg = ba.degree(i)
            g.add_node(i, NodeParams(
                n_memories=max(2, int(deg * 0.5)),
                T2=g.rng.uniform(1e-3, 100e-3),
                p_bsm=g.rng.uniform(0.6, 0.85),
                gate_fidelity=0.998,
                cnot_fidelity=0.992,
                is_end_user=(deg == 1),
            ))
        for u, v in ba.edges():
            g.add_channel(u, v, _random_channel(g.rng, 10, 200))
        return g

    @classmethod
    def ring(cls, n: int = 16, seed: int = 42) -> "QuantumNetworkGraph":
        """Ring topology."""
        g = cls(seed=seed)
        for i in range(n):
            g.add_node(i, NodeParams(
                n_memories=2, T2=50e-3, p_bsm=0.75,
                gate_fidelity=0.998, cnot_fidelity=0.992,
                is_end_user=(i in [0, n // 2]),
            ))
        for i in range(n):
            g.add_channel(i, (i + 1) % n, _random_channel(g.rng, 20, 60))
        return g

    @classmethod
    def star(cls, k: int = 12, seed: int = 42) -> "QuantumNetworkGraph":
        """Star topology with k leaves + 1 hub."""
        g = cls(seed=seed)
        g.add_node(0, NodeParams(
            n_memories=k, T2=100e-3, p_bsm=0.80,
            gate_fidelity=0.998, cnot_fidelity=0.992,
            is_end_user=False,
        ))
        for i in range(1, k + 1):
            g.add_node(i, NodeParams(
                n_memories=2, T2=20e-3, p_bsm=0.75,
                gate_fidelity=0.998, cnot_fidelity=0.992,
                is_end_user=True,
            ))
            g.add_channel(0, i, _random_channel(g.rng, 10, 50))
        return g

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def nodes(self) -> List[int]:
        return list(self.G.nodes())

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return list(self.G.edges())

    def node_params(self, v: int) -> NodeParams:
        return self.G.nodes[v]["params"]

    def edge_params(self, u: int, v: int) -> ChannelParams:
        return self.G[u][v]["params"]

    def neighbors(self, v: int) -> List[int]:
        return list(self.G.successors(v))

    def degree(self, v: int) -> int:
        return self.G.degree(v)

    def shortest_path(self, src: int, dst: int) -> List[int]:
        try:
            return nx.shortest_path(self.G, src, dst)
        except nx.NetworkXNoPath:
            return []

    def all_paths(self, src: int, dst: int, cutoff: int = 6) -> List[List[int]]:
        return list(nx.all_simple_paths(self.G.to_undirected(), src, dst, cutoff=cutoff))

    def __len__(self) -> int:
        return len(self.G.nodes())

    def __repr__(self) -> str:
        return (f"QuantumNetworkGraph(n={len(self.G.nodes())}, "
                f"m={len(self.G.edges())})")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_channel(rng: np.random.Generator, d_min: float, d_max: float) -> ChannelParams:
    d = rng.uniform(d_min, d_max)
    return ChannelParams(
        p_trans=np.exp(-LOSS_DB_PER_KM / 10 * np.log(10)),
        gamma=rng.uniform(0.005, 0.02),
        eta=rng.uniform(0.80, 0.95),
        T2=rng.uniform(10e-3, 100e-3),
        distance=d,
    )


def _ensure_connected(g: QuantumNetworkGraph, rng: np.random.Generator) -> None:
    """Add edges to make graph strongly connected if needed."""
    ug = g.G.to_undirected()
    comps = list(nx.connected_components(ug))
    if len(comps) == 1:
        return
    for i in range(len(comps) - 1):
        u = rng.choice(list(comps[i]))
        v = rng.choice(list(comps[i + 1]))
        g.add_channel(int(u), int(v), _random_channel(rng, 50, 150))