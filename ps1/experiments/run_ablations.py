"""
Ablation Experiment Suite Runner.
Executes experiments across all 5 configurations in configs/ablation_configs.yaml,
evaluates on the low-variance open-set benchmark, saves comparative ROC curves,
and prints the final ablation summary table.
"""

import os
import sys
import copy
import yaml
import torch
import numpy as np

# Enable direct execution from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(parent_dir, ".."))
for p in [parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from data.low_variance_generator import LowVarianceDatasetGenerator
    from data.dataset import build_dataloaders
    from models.open_set_reid_model import OpenSetReIDNet
    from losses.combined_loss import CombinedReIDLoss
    from evaluation.benchmark_evaluator import OpenSetBenchmarkEvaluator
    from evaluation.ablation_runner import AblationSuiteRunner
    from calibration.open_set_matcher import OpenSetReIDMatcher
except ImportError:
    from ps1.data.low_variance_generator import LowVarianceDatasetGenerator
    from ps1.data.dataset import build_dataloaders
    from ps1.models.open_set_reid_model import OpenSetReIDNet
    from ps1.losses.combined_loss import CombinedReIDLoss
    from ps1.evaluation.benchmark_evaluator import OpenSetBenchmarkEvaluator
    from ps1.evaluation.ablation_runner import AblationSuiteRunner
    from ps1.calibration.open_set_matcher import OpenSetReIDMatcher


def run_all_ablations():
    default_config_path = "./configs/default_config.yaml"
    ablation_config_path = "./configs/ablation_configs.yaml"

    with open(default_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    with open(ablation_config_path, "r") as f:
        ablation_cfgs = yaml.safe_load(f).get("ablations", {})

    # Ensure dataset is generated
    data_dir = base_cfg.get("dataset", {}).get("data_dir", "./data_storage/low_variance_uniform")
    if not os.path.exists(os.path.join(data_dir, "train")):
        print("[Ablations] Generating benchmark dataset...")
        generator = LowVarianceDatasetGenerator(output_dir=data_dir, num_enrolled_ids=30, num_unknown_ids=20, instances_per_id=6)
        generator.generate()

    dataloaders = build_dataloaders(base_cfg)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    results = []
    ablation_runner = AblationSuiteRunner(ablation_config_path=ablation_config_path)

    print("\n" + "=" * 70)
    print("        RUNNING ABLATION STUDIES (5 MODEL CONFIGURATIONS)")
    print("=" * 70)

    for key, var_cfg in ablation_cfgs.items():
        exp_name = var_cfg.get("name", key)
        print(f"\n--> Running Ablation: {exp_name}")

        # Construct specific config for this ablation
        cfg = copy.deepcopy(base_cfg)
        cfg["loss"]["margin_loss_type"] = var_cfg.get("margin_loss_type", "softmax")
        cfg["loss"]["use_hard_triplet"] = var_cfg.get("use_hard_triplet", False)
        cfg["model"]["use_part_attention"] = var_cfg.get("use_part_attention", False)
        cfg["calibration"]["method"] = var_cfg.get("calibration_method", "none")
        cfg["calibration"]["ambiguity_margin"] = var_cfg.get("ambiguity_margin", 0.0)

        # Build model
        model = OpenSetReIDNet(
            backbone_name=cfg["model"]["backbone"],
            embedding_dim=cfg["model"]["embedding_dim"],
            use_part_attention=cfg["model"]["use_part_attention"],
            num_parts=cfg["model"]["num_parts"]
        ).to(device)

        # Fast training on synthetic mini-benchmark for ablation validation
        num_classes = dataloaders["num_classes"]
        criterion = CombinedReIDLoss(
            num_classes=num_classes,
            feature_dim=cfg["model"]["embedding_dim"],
            margin_loss_type=cfg["loss"]["margin_loss_type"],
            margin=var_cfg.get("margin", 0.35),
            use_hard_triplet=cfg["loss"]["use_hard_triplet"]
        ).to(device)

        optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=4e-4)

        # Train for 5 quick epochs to converge on the manifold
        model.train()
        for epoch in range(1, 6):
            for imgs, pids, _, _ in dataloaders["train_loader"]:
                imgs, pids = imgs.to(device), pids.to(device)
                optimizer.zero_grad()
                metric_feats, class_feats, part_feats = model(imgs)
                loss, _ = criterion(metric_feats, class_feats, part_feats, pids)
                loss.backward()
                optimizer.step()

        # Build matcher with ablation's calibration settings
        matcher = OpenSetReIDMatcher(
            calibration_method=var_cfg.get("calibration_method", "none"),
            threshold=0.70,
            ambiguity_margin=var_cfg.get("ambiguity_margin", 0.0)
        )

        evaluator = OpenSetBenchmarkEvaluator(cfg, matcher=matcher)
        report = evaluator.evaluate(
            model=model,
            gallery_loader=dataloaders["gallery_loader"],
            probe_known_loader=dataloaders["probe_known_loader"],
            probe_unknown_loader=dataloaders["probe_unknown_loader"],
            device=device_str,
            experiment_name=exp_name
        )
        results.append(report)

    # Generate comparative plot
    plot_path = "./results/ablations/comparative_ablation_roc.png"
    ablation_runner.generate_comparative_roc_plot(results, plot_path)

    # Format and display markdown table
    md_table = ablation_runner.format_ablation_table(results)
    print("\n" + "=" * 70)
    print("                     FINAL ABLATION RESULTS TABLE")
    print("=" * 70)
    print(md_table)
    print("=" * 70 + "\n")

    # Save table to markdown
    table_file = "./results/ablations/ablation_summary.md"
    with open(table_file, "w") as f:
        f.write("# Ablation Study Analysis: PS-1 Open-Set Person Re-ID\n\n")
        f.write("Evaluation on Low Inter-Class Variance Benchmark (Same Uniform Attire)\n\n")
        f.write(md_table + "\n\n")
        f.write("![Comparative ROC Curve](./comparative_ablation_roc.png)\n")
    print(f"[Ablations] Saved summary report to: {table_file}")

    return results


if __name__ == "__main__":
    run_all_ablations()
