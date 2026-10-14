"""
Real-Time Latency & Bounded Execution Profiler.
Measures per-frame tracking execution latency across varying track volumes
to guarantee causal bounded latency compliance (< 15 ms, > 66 FPS).
"""

import time
import argparse
import numpy as np
from tabulate import tabulate

import os
import sys

# Enable direct script execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(parent_dir, ".."))
for p in [parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ..tracking.tracker import OcclusionAwareReIDTracker
except (ImportError, ValueError):
    try:
        from tracking.tracker import OcclusionAwareReIDTracker
    except ImportError:
        from ps2.tracking.tracker import OcclusionAwareReIDTracker


def profile_tracker_latency(track_counts=[5, 10, 20, 30, 50, 75, 100], iterations=100):
    """
    Profiles latency across scaling numbers of concurrent tracked targets.
    """
    tracker = OcclusionAwareReIDTracker()
    results = []

    print(f"\n[Profiler] Profiling Causal Tracking Latency ({iterations} iterations per setting)...")

    for n_tracks in track_counts:
        # Pre-seed tracker with n_tracks active tracks
        tracker.reset()
        dummy_boxes = []
        for i in range(n_tracks):
            x = 200.0 + (i % 10) * 80.0
            y = 200.0 + (i // 10) * 120.0
            dummy_boxes.append([x, y, x + 50.0, y + 140.0])
        dummy_boxes = np.array(dummy_boxes, dtype=np.float32)
        dummy_scores = np.full(n_tracks, 0.95, dtype=np.float32)
        dummy_feats = np.random.randn(n_tracks, 128).astype(np.float32)

        # Warmup and confirm tracks
        for _ in range(4):
            _ = tracker.update(dummy_boxes, dummy_scores, dummy_feats)

        # Measure latency
        latencies = []
        for _ in range(iterations):
            # Perturb boxes slightly
            jitter = dummy_boxes + np.random.uniform(-2, 2, dummy_boxes.shape).astype(np.float32)
            t0 = time.perf_counter()
            _ = tracker.update(jitter, dummy_scores, dummy_feats)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        latencies = np.array(latencies)
        mean_ms = float(np.mean(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        p99_ms = float(np.percentile(latencies, 99))
        fps = 1000.0 / max(1e-4, mean_ms)

        results.append({
            "Track Count": n_tracks,
            "Mean Latency (ms)": f"{mean_ms:.3f} ms",
            "p95 Latency (ms)": f"{p95_ms:.3f} ms",
            "p99 Latency (ms)": f"{p99_ms:.3f} ms",
            "Throughput (FPS)": f"{fps:.1f} FPS",
            "Bounded Latency Check": "PASSED (< 15 ms)" if p99_ms < 15.0 else "WARNING"
        })

    print("=" * 80)
    print("           CAUSAL ONLINE TRACKING LATENCY PROFILE")
    print("=" * 80)
    print(tabulate(results, headers="keys", tablefmt="github"))
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    profile_tracker_latency(iterations=args.iters)
