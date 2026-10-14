"""
Automated Stress Analysis Engine for Multi-Object Tracking.
Evaluates tracking degradation and ID stability as:
1. Crowd density scales from low to extreme congestion (5 to 50 persons/frame).
2. Mutual occlusion frequency and duration increase.
3. Compares Proposed OAR-Tracker against Baseline SORT, ByteTrack, and DeepSORT.
"""

import os
import time
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

try:
    from ..simulation.crowded_gate_generator import CrowdedGateBenchmarkGenerator
    from ..tracking.tracker import OcclusionAwareReIDTracker
    from .mot_metrics import MOTMetricsEvaluator
except (ImportError, ValueError):
    try:
        from simulation.crowded_gate_generator import CrowdedGateBenchmarkGenerator
        from tracking.tracker import OcclusionAwareReIDTracker
        from evaluation.mot_metrics import MOTMetricsEvaluator
    except ImportError:
        from ps2.simulation.crowded_gate_generator import CrowdedGateBenchmarkGenerator
        from ps2.tracking.tracker import OcclusionAwareReIDTracker
        from ps2.evaluation.mot_metrics import MOTMetricsEvaluator


class CrowdStressAnalyzer:
    """
    Executes stress testing sweeps and plots performance degradation curves.
    """
    def __init__(self, output_dir: str = "./results/stress_tests"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.evaluator = MOTMetricsEvaluator(iou_threshold=0.50)

    def run_density_stress_test(
        self,
        densities: List[int] = [5, 10, 15, 20, 25, 30, 40, 50],
        seq_length: int = 200
    ) -> List[Dict]:
        """
        Sweeps crowd density and records MOT metrics and per-frame latency.
        """
        print(f"\n[StressTester] Starting Crowd Density Stress Sweep: {densities} people...")
        results = []

        for num_p in densities:
            gen = CrowdedGateBenchmarkGenerator(
                num_pedestrians=num_p,
                seq_length=seq_length
            )
            sim_data = gen.generate()

            gt_by_frame = {}
            for rec in sim_data["gt_records"]:
                f = rec["frame"]
                if f not in gt_by_frame:
                    gt_by_frame[f] = []
                gt_by_frame[f].append({"id": rec["id"], "bbox_xyxy": rec["bbox_xyxy"]})

            tracker = OcclusionAwareReIDTracker()
            pred_by_frame = {}
            latencies = []

            for det_rec in sim_data["det_records"]:
                f = det_rec["frame"]
                boxes = det_rec["bboxes"]
                scores = det_rec["scores"]
                feats = det_rec["features"]

                t0 = time.perf_counter()
                tracks = tracker.update(boxes, scores, feats)
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000.0)  # ms

                pred_by_frame[f] = [{"id": t.track_id, "bbox_xyxy": t.to_xyxy} for t in tracks]

            metrics = self.evaluator.evaluate(gt_by_frame, pred_by_frame)
            mean_latency = float(np.mean(latencies))
            fps = 1000.0 / max(1e-3, mean_latency)

            summary = {
                "density": num_p,
                "MOTA": metrics["MOTA"],
                "IDF1": metrics["IDF1"],
                "HOTA": metrics["HOTA"],
                "IDSW": metrics["IDSW"],
                "MT_pct": metrics["MT_pct"],
                "ML_pct": metrics["ML_pct"],
                "latency_ms": mean_latency,
                "fps": fps
            }
            results.append(summary)
            print(f"  Density: {num_p:02d} | IDF1: {metrics['IDF1']:.2%} | MOTA: {metrics['MOTA']:.2%} | IDSW: {metrics['IDSW']:02d} | Latency: {mean_latency:.2f}ms")

        # Plot stress degradation curves
        self._plot_density_curves(results)
        return results

    def _plot_density_curves(self, results: List[Dict]):
        densities = [r["density"] for r in results]
        idf1_scores = [r["IDF1"] * 100 for r in results]
        mota_scores = [r["MOTA"] * 100 for r in results]
        idsw_counts = [r["IDSW"] for r in results]
        latencies = [r["latency_ms"] for r in results]

        # 1. IDF1 & MOTA vs Crowd Density
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(densities, idf1_scores, marker="o", color="#1f77b4", linewidth=2.5, label="IDF1 (Identity F1 %)")
        plt.plot(densities, mota_scores, marker="s", color="#2ca02c", linewidth=2.2, label="MOTA (Tracking Accuracy %)")
        plt.title("OAR-Tracker: Accuracy Degradation vs. Escalating Crowd Density\n[Narrow Gate Choke-Point]", fontsize=12, fontweight="bold")
        plt.xlabel("Crowd Density (Simultaneous Pedestrians in Gate)", fontsize=11)
        plt.ylabel("Tracking Performance (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim([0, 105])
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        plot_path_acc = os.path.join(self.output_dir, "stress_density_vs_accuracy.png")
        plt.savefig(plot_path_acc)
        plt.close()

        # 2. ID-Switches vs Crowd Density
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(densities, idsw_counts, marker="^", color="#d62728", linewidth=2.5, label="ID Switches (Lower is Better)")
        plt.title("Identity Switches vs. Escalating Crowd Density\n[Choke-Point Mutual Occlusions]", fontsize=12, fontweight="bold")
        plt.xlabel("Crowd Density (Simultaneous Pedestrians)", fontsize=11)
        plt.ylabel("Total ID Switches", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        plot_path_idsw = os.path.join(self.output_dir, "stress_density_vs_idsw.png")
        plt.savefig(plot_path_idsw)
        plt.close()

        # 3. Latency vs Crowd Density (Bounded Latency verification)
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(densities, latencies, marker="d", color="#9467bd", linewidth=2.5, label="Per-Frame Latency (ms)")
        plt.axhline(y=15.0, color="r", linestyle="--", label="Real-time Budget (15 ms = 66 FPS)")
        plt.title("Bounded Execution Latency vs. Crowd Density", fontsize=12, fontweight="bold")
        plt.xlabel("Crowd Density (Pedestrians)", fontsize=11)
        plt.ylabel("Tracking Latency (ms)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        plot_path_lat = os.path.join(self.output_dir, "stress_latency_vs_density.png")
        plt.savefig(plot_path_lat)
        plt.close()

        print(f"[StressTester] Saved degradation plots to {self.output_dir}")

    def run_tracker_ablations(self, num_people: int = 25, seq_length: int = 250) -> List[Dict]:
        """
        Compares 4 tracker architectures under dense crowd conditions:
        1. SORT Baseline (IoU only, single-stage)
        2. ByteTrack (Two-stage score gating, motion only)
        3. DeepSORT (Unweighted appearance, no occlusion freezing)
        4. Proposed OAR-Tracker (Full: Two-stage + Re-Association Bank + Quality-weighted EMA)
        """
        print(f"\n[StressTester] Running Tracker Ablations under Dense Crowd ({num_people} people)...")
        gen = CrowdedGateBenchmarkGenerator(num_pedestrians=num_people, seq_length=seq_length)
        sim_data = gen.generate()

        gt_by_frame = {}
        for rec in sim_data["gt_records"]:
            f = rec["frame"]
            if f not in gt_by_frame:
                gt_by_frame[f] = []
            gt_by_frame[f].append({"id": rec["id"], "bbox_xyxy": rec["bbox_xyxy"]})

        configs = [
            {
                "name": "1. Standard SORT (IoU Only)",
                "cfg": {"tracker": {"use_appearance": False, "track_low_thresh": 0.60, "max_lost_time": 1, "freeze_appearance_on_occlusion": False}}
            },
            {
                "name": "2. ByteTrack (Two-Stage Motion)",
                "cfg": {"tracker": {"use_appearance": False, "track_low_thresh": 0.15, "max_lost_time": 30, "freeze_appearance_on_occlusion": False}}
            },
            {
                "name": "3. DeepSORT (Unweighted Appearance)",
                "cfg": {"tracker": {"use_appearance": True, "lambda_appearance": 0.50, "track_low_thresh": 0.60, "max_lost_time": 30, "freeze_appearance_on_occlusion": False}}
            },
            {
                "name": "4. Proposed OAR-Tracker (Full)",
                "cfg": {"tracker": {"use_appearance": True, "lambda_appearance": 0.40, "track_low_thresh": 0.15, "max_lost_time": 60, "freeze_appearance_on_occlusion": True}}
            }
        ]

        ablation_results = []
        for item in configs:
            tracker = OcclusionAwareReIDTracker(config=item["cfg"])
            pred_by_frame = {}
            latencies = []

            for det_rec in sim_data["det_records"]:
                f = det_rec["frame"]
                boxes = det_rec["bboxes"]
                scores = det_rec["scores"]
                feats = det_rec["features"]

                t0 = time.perf_counter()
                tracks = tracker.update(boxes, scores, feats)
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000.0)
                pred_by_frame[f] = [{"id": t.track_id, "bbox_xyxy": t.to_xyxy} for t in tracks]

            metrics = self.evaluator.evaluate(gt_by_frame, pred_by_frame)
            mean_latency = float(np.mean(latencies))

            res = {
                "Architecture": item["name"],
                "IDF1 (%)": f"{metrics['IDF1'] * 100:.1f}%",
                "MOTA (%)": f"{metrics['MOTA'] * 100:.1f}%",
                "HOTA (%)": f"{metrics['HOTA'] * 100:.1f}%",
                "ID-Switches": metrics["IDSW"],
                "MT (%)": f"{metrics['MT_pct'] * 100:.1f}%",
                "ML (%)": f"{metrics['ML_pct'] * 100:.1f}%",
                "Latency (ms)": f"{mean_latency:.2f} ms",
                "FPS": f"{1000.0 / max(1e-3, mean_latency):.1f}"
            }
            ablation_results.append(res)

        table_str = tabulate(ablation_results, headers="keys", tablefmt="github")
        print("\n" + "=" * 75)
        print("     TRACKER ARCHITECTURE ABLATION UNDER DENSE MUTUAL OCCLUSION")
        print("=" * 75)
        print(table_str)
        print("=" * 75 + "\n")

        # Save summary report to markdown
        report_path = os.path.join(self.output_dir, "ablation_summary.md")
        with open(report_path, "w") as f:
            f.write("# PS-2: Tracker Architecture Ablations under Dense Occlusion\n\n")
            f.write(f"Evaluated on Narrow Gate Choke-Point ({num_people} simultaneous pedestrians, {seq_length} frames):\n\n")
            f.write(table_str + "\n\n")
            f.write("![Density vs Accuracy](./stress_density_vs_accuracy.png)\n\n")
            f.write("![Density vs ID Switches](./stress_density_vs_idsw.png)\n\n")
            f.write("![Density vs Latency](./stress_latency_vs_density.png)\n")

        return ablation_results
