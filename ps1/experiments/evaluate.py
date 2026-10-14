"""
Standalone Evaluation Script for Open-Set Re-Identification.
Loads a trained checkpoint, executes open-set verification across gallery and known/unknown probes,
and outputs ROC curves, DIR@FAR metrics, and CMC accuracy.
"""

import os
import sys
import argparse
import yaml
import torch

# Enable direct execution from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(parent_dir, ".."))
for p in [parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from data.dataset import build_dataloaders
    from models.open_set_reid_model import build_model
    from evaluation.benchmark_evaluator import OpenSetBenchmarkEvaluator
except ImportError:
    from ps1.data.dataset import build_dataloaders
    from ps1.models.open_set_reid_model import build_model
    from ps1.evaluation.benchmark_evaluator import OpenSetBenchmarkEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Open-Set Person Re-ID Model")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml", help="Path to config")
    parser.add_argument("--checkpoint", type=str, default="./weights/best_openset_reid_model.pth", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load Model
    model = build_model(config).to(device)
    if os.path.exists(args.checkpoint):
        print(f"[Evaluate] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    else:
        print(f"[Evaluate] Checkpoint not found at {args.checkpoint}. Running evaluation with freshly initialized weights...")

    # Load Data
    dataloaders = build_dataloaders(config)
    gallery_loader = dataloaders["gallery_loader"]
    probe_known_loader = dataloaders["probe_known_loader"]
    probe_unknown_loader = dataloaders["probe_unknown_loader"]

    evaluator = OpenSetBenchmarkEvaluator(config)
    report = evaluator.evaluate(
        model=model,
        gallery_loader=gallery_loader,
        probe_known_loader=probe_known_loader,
        probe_unknown_loader=probe_unknown_loader,
        device=str(device),
        experiment_name="Proposed_OpenSet_Framework"
    )

    return report


if __name__ == "__main__":
    main()
