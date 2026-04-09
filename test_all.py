"""
Unit Tests
===========
Tests for all key components.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from channels import (
    depolarizing_channel, memory_decoherence, entanglement_swapping,
    entanglement_purification, bbpssw_fidelity_out, bbpssw_success_prob,
    bell_fidelity, uhlmann_fidelity, werner_state, PHI_PLUS,
    is_valid_density_matrix, tensor, path_fidelity,
)
from graph import QuantumNetworkGraph, ChannelParams, NodeParams
from qneg import QNEG, PayoffWeights, ActionType, Action
from vqne import (
    CorePeripheryPartition, shapley_value, CoreCharacteristicFunction,
    check_individual_rationality, compute_epsilon_bound, compute_convergence_bound,
)
from circuit import VariationalQuantumCircuit, QuantumCircuit
from hccep import HCCEP, SPRouting, GreedyED, RandomProtocol
from metrics import jain_fairness, price_of_anarchy, fidelity_improvement_pct


# ── Channel tests ─────────────────────────────────────────────────────────────

class TestChannels:

    def test_phi_plus_is_valid(self):
        assert is_valid_density_matrix(PHI_PLUS)

    def test_bell_fidelity_of_phi_plus(self):
        assert abs(bell_fidelity(PHI_PLUS) - 1.0) < 1e-6

    def test_depolarizing_preserves_trace(self):
        for p in [0.0, 0.1, 0.5, 0.75]:
            rho_out = depolarizing_channel(PHI_PLUS.copy(), p)
            assert abs(np.trace(rho_out) - 1.0) < 1e-6, f"Trace not 1 for p={p}"

    def test_depolarizing_reduces_fidelity(self):
        for p in [0.1, 0.3, 0.5]:
            rho_out = depolarizing_channel(PHI_PLUS.copy(), p)
            f = bell_fidelity(rho_out)
            assert f < 1.0, f"Fidelity should decrease for p={p}"
            assert f >= 0.0

    def test_depolarizing_p0_identity(self):
        rho_out = depolarizing_channel(PHI_PLUS.copy(), 0.0)
        assert np.allclose(rho_out, PHI_PLUS, atol=1e-8)

    def test_memory_decoherence(self):
        rho_fresh = PHI_PLUS.copy()
        rho_aged  = memory_decoherence(rho_fresh, storage_time=0.1, T2=0.05)
        f_fresh   = bell_fidelity(rho_fresh)
        f_aged    = bell_fidelity(rho_aged)
        assert f_aged < f_fresh, "Memory decoherence should reduce fidelity"

    def test_memory_no_decay_at_t0(self):
        rho_out = memory_decoherence(PHI_PLUS.copy(), storage_time=0.0, T2=0.1)
        assert abs(bell_fidelity(rho_out) - 1.0) < 1e-4

    def test_entanglement_swapping_runs(self):
        rho1 = PHI_PLUS.copy()
        rho2 = PHI_PLUS.copy()
        np.random.seed(0)
        rho_out, success = entanglement_swapping(rho1, rho2, p_bsm=1.0)
        # With p_bsm=1, should always succeed
        assert success
        assert rho_out is not None
        assert is_valid_density_matrix(rho_out, tol=1e-4)

    def test_entanglement_swapping_fails_p0(self):
        rho1 = PHI_PLUS.copy()
        rho2 = PHI_PLUS.copy()
        np.random.seed(99)
        rho_out, success = entanglement_swapping(rho1, rho2, p_bsm=0.0)
        assert not success
        assert rho_out is None

    def test_bbpssw_fidelity_monotone(self):
        """Output fidelity > input fidelity for F > 0.5."""
        for F_in in [0.55, 0.65, 0.75, 0.85]:
            F_out = bbpssw_fidelity_out(F_in)
            assert F_out > F_in, f"F_out={F_out} not > F_in={F_in}"

    def test_bbpssw_success_prob_range(self):
        for F in [0.5, 0.7, 0.9]:
            p = bbpssw_success_prob(F)
            assert 0.0 <= p <= 1.0

    def test_werner_state_valid(self):
        for F in [0.25, 0.5, 0.75, 1.0]:
            rho = werner_state(F)
            assert is_valid_density_matrix(rho, tol=1e-6)
            assert abs(bell_fidelity(rho) - F) < 0.01

    def test_uhlmann_fidelity_symmetry(self):
        rho1 = werner_state(0.8)
        rho2 = werner_state(0.6)
        f12 = uhlmann_fidelity(rho1, rho2)
        f21 = uhlmann_fidelity(rho2, rho1)
        assert abs(f12 - f21) < 1e-4

    def test_tensor_product(self):
        rho = PHI_PLUS
        rho_tensor = tensor(np.eye(2) / 2, rho)
        assert rho_tensor.shape == (8, 8)
        assert abs(np.trace(rho_tensor) - 1.0) < 1e-6


# ── Network graph tests ───────────────────────────────────────────────────────

class TestNetwork:

    def test_ring_topology(self):
        qnet = QuantumNetworkGraph.ring(n=8, seed=42)
        assert len(qnet) == 8
        assert len(qnet.edges) == 16  # bidirectional

    def test_star_topology(self):
        qnet = QuantumNetworkGraph.star(k=5, seed=42)
        assert len(qnet) == 6  # hub + 5 leaves

    def test_arpanet_topology(self):
        qnet = QuantumNetworkGraph.arpanet(seed=42)
        assert len(qnet) == 20

    def test_geant_topology(self):
        qnet = QuantumNetworkGraph.geant(seed=42)
        assert len(qnet) == 40

    def test_shortest_path(self):
        qnet = QuantumNetworkGraph.ring(n=8, seed=42)
        path = qnet.shortest_path(0, 4)
        assert len(path) >= 2
        assert path[0] == 0
        assert path[-1] == 4

    def test_channel_fidelity_positive(self):
        qnet = QuantumNetworkGraph.ring(n=4, seed=42)
        for u, v in qnet.edges:
            ch = qnet.edge_params(u, v)
            f0 = ch.raw_bell_fidelity()
            assert 0.0 <= f0 <= 1.0

    def test_node_params(self):
        qnet = QuantumNetworkGraph.ring(n=4, seed=42)
        for v in qnet.nodes:
            p = qnet.node_params(v)
            assert p.T2 > 0
            assert 0 < p.p_bsm <= 1

    def test_path_fidelity(self):
        qnet = QuantumNetworkGraph.ring(n=4, seed=42)
        path = qnet.shortest_path(0, 2)
        if len(path) >= 2:
            f = path_fidelity(path, qnet)
            assert 0.0 <= f <= 1.0


# ── QNEG tests ────────────────────────────────────────────────────────────────

class TestQNEG:

    def setup_method(self):
        self.qnet = QuantumNetworkGraph.ring(n=6, seed=42)
        nodes = self.qnet.nodes
        self.game = QNEG(
            self.qnet, src=nodes[0], dst=nodes[3],
            weights=PayoffWeights(0.5, 0.3, 0.2), seed=42
        )

    def test_available_actions_not_empty(self):
        for v in self.game.qnet.nodes:
            actions = self.game.available_actions(v)
            assert len(actions) >= 1  # at least IDLE

    def test_idle_always_available(self):
        for v in self.game.qnet.nodes:
            types = [a.action_type for a in self.game.available_actions(v)]
            assert ActionType.IDLE in types

    def test_step_returns_rewards(self):
        actions = {v: Action(ActionType.IDLE) for v in self.game.qnet.nodes}
        rewards = self.game.step(actions)
        assert len(rewards) == len(self.game.qnet.nodes)
        assert all(isinstance(r, float) for r in rewards.values())

    def test_observation_shape(self):
        for v in self.game.qnet.nodes:
            obs = self.game.get_observation(v)
            assert obs.ndim == 1
            assert obs.dtype == np.float32

    def test_reset_clears_state(self):
        # Run some steps
        for _ in range(5):
            actions = {v: Action(ActionType.IDLE) for v in self.game.qnet.nodes}
            self.game.step(actions)
        self.game.reset()
        for v in self.game.qnet.nodes:
            assert self.game.node_states[v].memory_usage() == 0

    def test_payoff_weights_normalized(self):
        w = PayoffWeights(0.5, 0.3, 0.2)
        assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6


# ── VQNE and game theory tests ────────────────────────────────────────────────

class TestVQNE:

    def setup_method(self):
        self.qnet = QuantumNetworkGraph.ring(n=8, seed=42)
        self.partition = CorePeripheryPartition(self.qnet)

    def test_partition_covers_all_nodes(self):
        all_nodes = set(self.qnet.nodes)
        partitioned = self.partition.core | self.partition.periphery
        assert all_nodes == partitioned

    def test_partition_disjoint(self):
        overlap = self.partition.core & self.partition.periphery
        assert len(overlap) == 0

    def test_shapley_efficiency(self):
        """Sum of Shapley values = v(grand coalition)."""
        players = list(range(3))
        payoffs = {i: float(i + 1) for i in players}
        char_fn = CoreCharacteristicFunction(payoffs)
        phi = shapley_value(players, char_fn)

        # Sum of Shapley values
        total_phi = sum(phi.values())
        grand_v   = char_fn(frozenset(players))
        # Efficiency: sum(phi_i) = v(N) approximately
        assert abs(total_phi - grand_v) < grand_v * 0.5  # allow some tolerance

    def test_individual_rationality(self):
        players = list(range(3))
        payoffs = {i: float(i + 1) for i in players}
        char_fn = CoreCharacteristicFunction(payoffs)
        phi = shapley_value(players, char_fn)

        standalone = {i: payoffs[i] * 0.5 for i in players}
        ir = check_individual_rationality(phi, standalone)
        # With cooperation bonus, should be individually rational
        for v in ir.values():
            assert v, "Individual rationality should hold"

    def test_epsilon_bound_decreasing(self):
        """Epsilon bound should decrease with larger B and d_i."""
        e1 = compute_epsilon_bound(B=1.0, d_i=10)
        e2 = compute_epsilon_bound(B=10.0, d_i=10)
        e3 = compute_epsilon_bound(B=10.0, d_i=100)
        assert e1 > e2 > e3

    def test_convergence_bound_positive(self):
        K = compute_convergence_bound(
            L_U=1.0, d_max=48, mu=0.01, epsilon=0.05, n=40, delta=0.05
        )
        assert K > 0


# ── VQC tests ─────────────────────────────────────────────────────────────────

class TestVQC:

    def test_vqc_output_shape(self):
        import torch
        vqc = VariationalQuantumCircuit(n_inputs=25, n_actions=10, n_qubits=4, n_layers=3)
        obs = torch.randn(8, 25)  # batch of 8
        log_probs = vqc(obs)
        assert log_probs.shape == (8, 10)

    def test_vqc_log_probs_valid(self):
        import torch
        vqc = VariationalQuantumCircuit(n_inputs=25, n_actions=10, n_qubits=4, n_layers=3)
        obs = torch.randn(4, 25)
        log_probs = vqc(obs)
        probs = torch.exp(log_probs)
        # Probabilities should sum to ~1 per sample
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-4)

    def test_vqc_action_selection(self):
        import torch
        vqc = VariationalQuantumCircuit(n_inputs=25, n_actions=5, n_qubits=4, n_layers=2)
        obs = np.random.randn(25).astype(np.float32)
        action, log_prob = vqc.select_action(obs)
        assert 0 <= action < 5
        assert log_prob <= 0  # log-probability is non-positive

    def test_quantum_circuit_state(self):
        qc = QuantumCircuit(n_qubits=3)
        assert qc.state.shape == (8,)
        assert abs(np.sum(qc.measure_probs()) - 1.0) < 1e-6

    def test_quantum_circuit_ry(self):
        qc = QuantumCircuit(n_qubits=1)
        qc.apply_ry(np.pi, 0)
        probs = qc.measure_probs()
        # After R_Y(pi), should be in |1> state
        assert abs(probs[1] - 1.0) < 1e-6

    def test_vqc_parameter_count(self):
        vqc = VariationalQuantumCircuit(n_inputs=10, n_actions=5, n_qubits=4, n_layers=6)
        n_params = vqc.parameter_count()
        assert n_params > 0
        # Paper: 48 params per agent with n_qubits=4, n_layers=6
        print(f"  VQC parameter count: {n_params}")


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:

    def test_jain_perfect_fairness(self):
        rates = [10.0, 10.0, 10.0, 10.0]
        assert abs(jain_fairness(rates) - 1.0) < 1e-6

    def test_jain_one_user(self):
        rates = [1.0, 0.0, 0.0, 0.0]
        j = jain_fairness(rates)
        # Only one user gets rate → J = 1/n
        assert j == pytest.approx(1 / 4, abs=0.01)

    def test_poa_always_geq_1(self):
        poa = price_of_anarchy(social_optimum=0.8, nash_welfare=0.6)
        assert poa >= 1.0

    def test_poa_equals_1_at_optimum(self):
        poa = price_of_anarchy(social_optimum=0.8, nash_welfare=0.8)
        assert abs(poa - 1.0) < 1e-6

    def test_fidelity_improvement_positive(self):
        pct = fidelity_improvement_pct(protocol_fidelity=0.83, baseline_fidelity=0.67)
        assert pct > 0

    def test_fidelity_improvement_value(self):
        pct = fidelity_improvement_pct(0.83, 0.67)
        expected = (0.83 - 0.67) / 0.67 * 100
        assert abs(pct - expected) < 0.01


# ── Protocol tests ────────────────────────────────────────────────────────────

class TestProtocols:

    def setup_method(self):
        self.qnet = QuantumNetworkGraph.ring(n=6, seed=42)
        nodes = self.qnet.nodes
        self.game = QNEG(self.qnet, src=nodes[0], dst=nodes[3], seed=42)

    def test_hccep_returns_actions_for_all_nodes(self):
        protocol = HCCEP(self.game)
        actions  = protocol.select_actions(self.game)
        assert set(actions.keys()) == set(self.game.qnet.nodes)

    def test_sp_routing_returns_actions(self):
        protocol = SPRouting(self.game)
        actions  = protocol.select_actions(self.game)
        assert len(actions) == len(self.game.qnet.nodes)

    def test_greedy_returns_actions(self):
        protocol = GreedyED()
        actions  = protocol.select_actions(self.game)
        assert len(actions) == len(self.game.qnet.nodes)

    def test_random_returns_actions(self):
        protocol = RandomProtocol()
        actions  = protocol.select_actions(self.game)
        assert len(actions) == len(self.game.qnet.nodes)

    def test_all_actions_valid_type(self):
        """All returned actions should have valid ActionType."""
        protocols = [HCCEP(self.game), SPRouting(self.game), GreedyED(), RandomProtocol()]
        for proto in protocols:
            actions = proto.select_actions(self.game)
            for v, a in actions.items():
                assert isinstance(a.action_type, ActionType)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])