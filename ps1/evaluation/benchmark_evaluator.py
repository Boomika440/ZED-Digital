"""
End-to-End Open-Set Benchmark Evaluator.
Executes complete evaluation protocol:
1. Feature extraction over Gallery, Known Probes, and Unknown Impostor Probes.
2. Dynamic Gallery Enrollment & Score Calibration (EVT / S-Norm / Platt).
3. Full ROC / DIR @ FAR trade-off analysis and plot generation.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

try:
    from .metrics import compute_roc_and_dir, compute_cmc_and_map
    from ..calibration.open_set_matcher import OpenSetReIDMatcher
except (ImportError, ValueError):
    try:
        from evaluation.metrics import compute_roc_and_dir, compute_cmc_and_map
        from calibration.open_set_matcher import OpenSetReIDMatcher
    except ImportError:
        from ps1.evaluation.metrics import compute_roc_and_dir, compute_cmc_and_map
        from ps1.calibration.open_set_matcher import OpenSetReIDMatcher


def extract_features_from_loader(model: torch.nn.Module, dataloader, device: str = "cpu") -> Tuple:
    """
    Extracts L2-normalized feature embeddings from a DataLoader.
    """
    model.eval()
    features_list = []
    pids_list = []
    camids_list = []

    with torch.no_grad():
        for imgs, pids, cams, _ in dataloader:
            imgs = imgs.to(device)
            # Use inference fast forward
            feats = model.extract_features(imgs)
            features_list.append(feats.cpu().numpy())
            pids_list.append(pids.numpy())
            camids_list.append(cams.numpy())

    features = np.concatenate(features_list, axis=0)
    pids = np.concatenate(pids_list, axis=0)
    camids = np.concatenate(camids_list, axis=0)

    # Normalize to unit sphere
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-8, None)

    return features, pids, camids


class OpenSetBenchmarkEvaluator:
    """
    Evaluates Re-ID performance under open-set conditions and low inter-class variance.
    """
    def __init__(self, config: dict, matcher: Optional[OpenSetReIDMatcher] = None):
        self.config = config
        eval_cfg = config.get("evaluation", {})
        self.target_fars = eval_cfg.get("target_fars", [0.001, 0.01, 0.05, 0.10])
        self.save_roc_curve = eval_cfg.get("save_roc_curve", True)
        self.save_dir = eval_cfg.get("save_dir", "./results")
        os.makedirs(self.save_dir, exist_ok=True)

        calib_cfg = config.get("calibration", {})
        self.matcher = matcher or OpenSetReIDMatcher(
            calibration_method=calib_cfg.get("method", "evt_weibull"),
            threshold=calib_cfg.get("rejection_threshold_far01", 0.72),
            ambiguity_margin=calib_cfg.get("ambiguity_margin", 0.08),
            evt_tail_size=calib_cfg.get("tail_size", 20)
        )

    def evaluate(
        self,
        model: torch.nn.Module,
        gallery_loader,
        probe_known_loader,
        probe_unknown_loader,
        device: str = "cpu",
        experiment_name: str = "Proposed_Framework"
    ) -> Dict:
        """
        Runs the complete open-set evaluation protocol.
        """
        print(f"\n[Evaluator] Starting Open-Set Evaluation for '{experiment_name}'...")
        start_time = time.time()

        # 1. Feature Extraction
        print("[Evaluator] Extracting Gallery features...")
        g_feats, g_pids, g_cams = extract_features_from_loader(model, gallery_loader, device=device)

        print("[Evaluator] Extracting Known Probe features...")
        q_k_feats, q_k_pids, _ = extract_features_from_loader(model, probe_known_loader, device=device)

        print("[Evaluator] Extracting Unknown Impostor features...")
        q_u_feats, _, _ = extract_features_from_loader(model, probe_unknown_loader, device=device)

        print(f"[Evaluator] Extracted: {len(g_feats)} gallery, {len(q_k_feats)} known probes, {len(q_u_feats)} unknown probes.")

        # 2. Dynamic Gallery Enrollment & Calibration
        self.matcher.enroll_gallery(g_feats, g_pids, g_cams)
        self.matcher.calibrate()

        # 3. Closed-Set Verification Baseline on Known Probes
        raw_sim_matrix = np.dot(q_k_feats, g_feats.T)
        dist_matrix = 1.0 - raw_sim_matrix
        cmc_metrics = compute_cmc_and_map(dist_matrix, q_k_pids, g_pids)

        # 4. Open-Set Match Predictions
        print("[Evaluator] Computing open-set matches & calibration...")
        known_results = self.matcher.match_batch(q_k_feats)
        unknown_results = self.matcher.match_batch(q_u_feats)

        # Extract top-1 confidences and ground-truth match correctness
        known_confidences = np.array([r["confidence"] for r in known_results])
        known_top1_pids = np.array([r["matched_pid"] if r["is_accepted"] else -1 for r in known_results])
        known_correct_mask = (known_top1_pids == q_k_pids)

        unknown_confidences = np.array([r["confidence"] for r in unknown_results])

        # 5. ROC & DIR @ FAR Analysis
        roc_results = compute_roc_and_dir(
            known_scores=known_confidences,
            known_correct_mask=known_correct_mask,
            unknown_scores=unknown_confidences,
            target_fars=self.target_fars
        )

        elapsed = time.time() - start_time
        print(f"[Evaluator] Evaluation complete in {elapsed:.2f}s")

        # 6. Save Plot
        if self.save_roc_curve:
            plot_path = os.path.join(self.save_dir, f"roc_curve_{experiment_name.lower().replace(' ', '_')}.png")
            self._plot_roc_curve(roc_results, experiment_name, plot_path)
            roc_results["plot_path"] = plot_path

        report = {
            "experiment_name": experiment_name,
            "rank1": cmc_metrics["Rank-1"],
            "rank5": cmc_metrics["Rank-5"],
            "map": cmc_metrics["mAP"],
            "auc": roc_results["auc"],
            "dir_at_far": roc_results["dir_at_far"],
            "far_at_tar": roc_results["far_at_tar"],
            "threshold_at_far": roc_results["threshold_at_far"],
            "roc_data": roc_results
        }

        self._print_summary_table(report)
        return report

    def _plot_roc_curve(self, roc_results: dict, exp_name: str, save_path: str):
        """
        Generates publication-quality ROC curve: Detection and Identification Rate (DIR) vs False Accept Rate (FAR).
        """
        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot(roc_results["far"], roc_results["dir"], label=f"{exp_name} (AUC = {roc_results['auc']:.3f})", color="#1f77b4", linewidth=2.5)

        # Plot target FAR points
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
        for i, target_far in enumerate(self.target_fars):
            key = f"DIR@FAR={target_far:.1%}"
            if key in roc_results["dir_at_far"]:
                dir_val = roc_results["dir_at_far"][key]
                c = colors[i % len(colors)]
                plt.scatter([target_far], [dir_val], color=c, s=60, zorder=5)
                plt.annotate(
                    f"{dir_val:.1%}",
                    (target_far, dir_val),
                    textcoords="offset points",
                    xytext=(-15, 10),
                    fontsize=9,
                    fontweight="bold",
                    color=c
                )

        plt.xscale("log")
        plt.xlim([1e-4, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Accept Rate (FAR, log scale)", fontsize=12, fontweight="bold")
        plt.ylabel("Detection and Identification Rate (DIR @ Rank-1)", fontsize=12, fontweight="bold")
        plt.title(f"Open-Set Person Re-ID ROC Trade-Off ({exp_name})\n[Low Inter-Class Variance Benchmark]", fontsize=13, fontweight="bold")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[Evaluator] Saved ROC curve figure to: {save_path}")

    def _print_summary_table(self, report: dict):
        """
        Prints clean markdown summary table of results.
        """
        print("\n" + "=" * 65)
        print(f"       OPEN-SET RE-ID EVALUATION REPORT: {report['experiment_name']}")
        print("=" * 65)
        print(f" Closed-Set Rank-1 Accuracy : {report['rank1']:.2%}")
        print(f" Closed-Set Rank-5 Accuracy : {report['rank5']:.2%}")
        print(f" Mean Average Precision(mAP): {report['map']:.2%}")
        print(f" Area Under ROC Curve (AUC) : {report['auc']:.4f}")
        print("-" * 65)
        print(" Detection & Identification Rate (DIR) at Fixed Security FAR:")
        for k, v in report["dir_at_far"].items():
            print(f"   * {k:<18} : {v:.2%}")
        print("-" * 65)
        print(" Impostor False-Accept Rate (FAR) at Fixed Verification Recall:")
        for k, v in report["far_at_tar"].items():
            print(f"   * {k:<18} : {v:.2%}")
        print("=" * 65 + "\n")
