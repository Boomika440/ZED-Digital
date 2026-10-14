"""
End-to-End Training Pipeline for Open-Set Person Re-Identification under Low Inter-Class Variance.
Features:
- AdamW optimizer with Warmup Cosine Annealing learning rate schedule.
- Multi-loss supervision: Sub-Center ArcFace + Batch-Hard Triplet + Part-Level contrastive loss.
- Periodic evaluation and checkpoint saving.
"""

import os
import sys
import argparse
import yaml
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    from models.open_set_reid_model import build_model
    from losses.combined_loss import CombinedReIDLoss
    from evaluation.benchmark_evaluator import OpenSetBenchmarkEvaluator
except ImportError:
    from ps1.data.low_variance_generator import LowVarianceDatasetGenerator
    from ps1.data.dataset import build_dataloaders
    from ps1.models.open_set_reid_model import build_model
    from ps1.losses.combined_loss import CombinedReIDLoss
    from ps1.evaluation.benchmark_evaluator import OpenSetBenchmarkEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Open-Set Person Re-ID Model")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--save_dir", type=str, default="./weights", help="Directory to save checkpoints")
    parser.add_argument("--generate_data_if_missing", action="store_true", default=True, help="Auto generate data")
    return parser.parse_args()


def train(args):
    # 1. Load Configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    dataset_cfg = config.get("dataset", {})

    epochs = args.epochs or train_cfg.get("epochs", 40)
    base_lr = args.lr or train_cfg.get("base_lr", 0.00035)
    weight_decay = train_cfg.get("weight_decay", 0.0005)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Dataset Preparation
    data_dir = dataset_cfg.get("data_dir", "./data_storage/low_variance_uniform")
    if not os.path.exists(os.path.join(data_dir, "train")) and args.generate_data_if_missing:
        print("[Train] Low-variance dataset not found. Generating synthetic uniform benchmark...")
        generator = LowVarianceDatasetGenerator(
            output_dir=data_dir,
            num_enrolled_ids=dataset_cfg.get("num_classes", 40),
            num_unknown_ids=dataset_cfg.get("num_unknown_identities", 25),
            instances_per_id=dataset_cfg.get("samples_per_id", 8)
        )
        generator.generate()

    dataloaders = build_dataloaders(config)
    train_loader = dataloaders["train_loader"]
    gallery_loader = dataloaders["gallery_loader"]
    probe_known_loader = dataloaders["probe_known_loader"]
    probe_unknown_loader = dataloaders["probe_unknown_loader"]
    num_classes = dataloaders["num_classes"]

    print(f"[Train] Initialized dataloaders. Num Enrolled Classes: {num_classes}, Device: {device}")

    # 3. Model & Loss Setup
    model = build_model(config).to(device)
    feature_dim = config.get("model", {}).get("embedding_dim", 512)

    criterion = CombinedReIDLoss(
        num_classes=num_classes,
        feature_dim=feature_dim,
        margin_loss_type=loss_cfg.get("margin_loss_type", "subcenter_arcface"),
        margin=loss_cfg.get("margin", 0.35),
        scale=loss_cfg.get("scale", 30.0),
        sub_centers=loss_cfg.get("sub_centers", 3),
        use_hard_triplet=loss_cfg.get("use_hard_triplet", True),
        triplet_margin=loss_cfg.get("triplet_margin", 0.3),
        lambda_margin=loss_cfg.get("lambda_margin", 1.0),
        lambda_triplet=loss_cfg.get("lambda_triplet", 1.0),
        lambda_part=loss_cfg.get("lambda_part", 0.5)
    ).to(device)

    # 4. Optimizer & Scheduler
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = AdamW(params, lr=base_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=train_cfg.get("min_lr", 1e-5))

    evaluator = OpenSetBenchmarkEvaluator(config)

    # 5. Training Loop
    best_dir = 0.0
    best_checkpoint_path = os.path.join(args.save_dir, "best_openset_reid_model.pth")

    print(f"[Train] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        step_count = 0

        for imgs, pids, _, _ in train_loader:
            imgs = imgs.to(device)
            pids = pids.to(device)

            optimizer.zero_grad()
            metric_feats, class_feats, part_feats = model(imgs)
            loss, loss_dict = criterion(metric_feats, class_feats, part_feats, pids)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            total_loss_epoch += loss.item()
            step_count += 1

        scheduler.step()
        avg_loss = total_loss_epoch / max(1, step_count)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Avg Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

            # Intermediate evaluation
            report = evaluator.evaluate(
                model=model,
                gallery_loader=gallery_loader,
                probe_known_loader=probe_known_loader,
                probe_unknown_loader=probe_unknown_loader,
                device=device_str,
                experiment_name=f"Epoch_{epoch}"
            )

            current_dir01 = report["dir_at_far"].get("DIR@FAR=1.0%", 0.0)
            if current_dir01 >= best_dir:
                best_dir = current_dir01
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "criterion_state_dict": criterion.state_dict(),
                    "config": config,
                    "best_dir01": best_dir
                }, best_checkpoint_path)
                print(f"[Train] --> Best checkpoint updated! DIR@FAR=1%: {best_dir:.2%}")

    print(f"\n[Train] Training complete. Best model checkpoint saved to: {best_checkpoint_path}")
    return best_checkpoint_path


if __name__ == "__main__":
    args = parse_args()
    train(args)
