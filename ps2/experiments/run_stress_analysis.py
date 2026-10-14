"""
Full Stress Analysis & Ablation Runner for PS-2.
Executes:
1. Crowd Density Scaling Sweep (5 to 50 simultaneous pedestrians).
2. Tracker Architecture Ablations (SORT vs ByteTrack vs DeepSORT vs Proposed OAR-Tracker).
3. Generates comparative degradation curves and markdown reports.
"""

import os
import sys
import argparse

# Enable direct script execution from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(parent_dir, ".."))
for p in [parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from evaluation.stress_tester import CrowdStressAnalyzer
except ImportError:
    from ps2.evaluation.stress_tester import CrowdStressAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Run PS-2 Stress Analysis")
    parser.add_argument("--output_dir", type=str, default="./results/stress_tests")
    parser.add_argument("--densities", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30, 40, 50])
    parser.add_argument("--seq_length", type=int, default=200)
    args = parser.parse_args()

    print("=" * 70)
    print("  PS-2: CROWD DENSITY & OCCLUSION STRESS ANALYSIS SUITE")
    print("=" * 70)

    analyzer = CrowdStressAnalyzer(output_dir=args.output_dir)

    # 1. Density Scaling Stress Test
    density_results = analyzer.run_density_stress_test(
        densities=args.densities,
        seq_length=args.seq_length
    )

    # 2. Architectural Ablations under Heavy Occlusion (25 pedestrians in narrow gate)
    ablation_results = analyzer.run_tracker_ablations(
        num_people=25,
        seq_length=250
    )

    print("\n[StressAnalysis] Complete! All plots and markdown tables saved to:")
    print(f"  --> {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
