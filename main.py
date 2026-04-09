"""
Main entry point for Quantum Game-Theoretic Entanglement Distribution
======================================================================
Usage:
    python main.py                         # Run all experiments (quick mode)
    python main.py --experiment demo       # Just the MARL training demo
    python main.py --full                  # Full runs (slow)
    python main.py --test                  # Run unit tests
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch


def print_header():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   Quantum Game-Theoretic Models for Optimal Entanglement             ║
║   Distribution in Quantum Networks                                   ║
║                                                                      ║
║   Implements: VQNE · HCCEP · MARL-QNet                              ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def run_tests():
    """Run all unit tests."""
    import pytest
    ret = pytest.main(["tests/test_all.py", "-v", "--tb=short"])
    return ret


def main():
    print_header()

    parser = argparse.ArgumentParser(
        description="Quantum Game-Theoretic Entanglement Distribution"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--full", action="store_true",
                        help="Full experiments (slow, many trials)")
    parser.add_argument("--test", action="store_true",
                        help="Run unit tests")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "fidelity", "rate", "convergence",
                                 "scalability", "poa", "table", "ablation", "demo"],
                        help="Which experiment to run")
    args = parser.parse_args()

    # Set global seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.test:
        sys.exit(run_tests())

    # Import and run experiments
    from run_experiments import (
        run_all_experiments, run_marl_demo, run_fidelity_vs_hops,
        run_rate_vs_noise, run_nash_convergence, run_scalability,
        run_poa_analysis, run_main_results_table, run_ablation,
    )

    exp_map = {
        "all":         lambda: run_all_experiments(seed=args.seed, quick=not args.full),
        "fidelity":    lambda: run_fidelity_vs_hops(seed=args.seed),
        "rate":        lambda: run_rate_vs_noise(seed=args.seed),
        "convergence": lambda: run_nash_convergence(seed=args.seed),
        "scalability": lambda: run_scalability(seed=args.seed),
        "poa":         run_poa_analysis,
        "table":       lambda: run_main_results_table(seed=args.seed),
        "ablation":    run_ablation,
        "demo":        lambda: run_marl_demo(seed=args.seed),
    }

    exp_map[args.experiment]()


if __name__ == "__main__":
    main()