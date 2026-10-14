"""
Ablation Study Runner isolating each algorithmic contribution:
1. M0: Baseline (Standard Softmax + Global Average Pooling)
2. M1: + Sub-Center ArcFace Angular Margin
3. M2: + Batch-Hard Triplet Loss (Hard-Negative Mining)
4. M3: + Fine-Grained Part Attention (Spatial micro-cue localization)
5. M4: + EVT Weibull Confidence Calibration & Ambiguity Margin Rejection (Proposed Full System)
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import Dict, List


class AblationSuiteRunner:
    """
    Automates ablation experiments and generates side-by-side comparative trade-off tables and overlaid ROC curves.
    """
    def __init__(self, ablation_config_path: str = "./configs/ablation_configs.yaml", save_dir: str = "./results/ablations"):
        self.ablation_config_path = ablation_config_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        if os.path.exists(ablation_config_path):
            with open(ablation_config_path, "r") as f:
                self.configs = yaml.safe_load(f).get("ablations", {})
        else:
            self.configs = {}

    def generate_comparative_roc_plot(self, results: List[Dict], output_path: str):
        """
        Plots overlaid ROC curves for all ablation models on a single figure.
        """
        plt.figure(figsize=(9, 6.5), dpi=160)
        colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        linestyles = ["--", "-.", ":", "-", "-"]

        for i, res in enumerate(results):
            c = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]
            roc = res["roc_data"]
            name = res["experiment_name"]
            auc = res["auc"]
            plt.plot(roc["far"], roc["dir"], label=f"{name} (AUC={auc:.3f})", color=c, linestyle=ls, linewidth=2.0)

        plt.xscale("log")
        plt.xlim([1e-4, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Accept Rate (FAR, log scale)", fontsize=12, fontweight="bold")
        plt.ylabel("Detection & Identification Rate (DIR @ Rank-1)", fontsize=12, fontweight="bold")
        plt.title("Ablation Study: Component Impact on Open-Set Trade-Off\n[Low Inter-Class Variance Benchmark]", fontsize=13, fontweight="bold")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[AblationSuite] Comparative ROC plot saved to: {output_path}")

    def format_ablation_table(self, results: List[Dict]) -> str:
        """
        Formats comparative table in GitHub Markdown.
        """
        headers = [
            "Model Variant",
            "Rank-1 (%)",
            "mAP (%)",
            "DIR@FAR=0.1% (%)",
            "DIR@FAR=1% (%)",
            "DIR@FAR=5% (%)",
            "FAR@TAR=90% (%)",
            "AUC"
        ]
        table_rows = []
        for r in results:
            dir_far = r["dir_at_far"]
            far_tar = r["far_at_tar"]
            row = [
                r["experiment_name"],
                f"{r['rank1'] * 100:.1f}",
                f"{r['map'] * 100:.1f}",
                f"{dir_far.get('DIR@FAR=0.1%', 0.0) * 100:.1f}",
                f"{dir_far.get('DIR@FAR=1.0%', 0.0) * 100:.1f}",
                f"{dir_far.get('DIR@FAR=5.0%', 0.0) * 100:.1f}",
                f"{far_tar.get('FAR@TAR=90%', 0.0) * 100:.1f}",
                f"{r['auc']:.4f}"
            ]
            table_rows.append(row)

        md_table = tabulate(table_rows, headers=headers, tablefmt="github")
        return md_table
