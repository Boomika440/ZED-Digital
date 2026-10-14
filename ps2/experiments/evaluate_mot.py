"""
Standalone MOTChallenge Format Sequence Evaluator.
Loads pre-recorded or exported MOTChallenge datasets (gt.txt and det.txt),
executes tracking, and logs standard MOT metrics.
"""

import os
import sys
import argparse
import yaml
import numpy as np

# Enable direct script execution from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(parent_dir, ".."))
for p in [parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from tracking.tracker import OcclusionAwareReIDTracker
    from simulation.dataset_loader import MOTDatasetManager
    from evaluation.mot_metrics import MOTMetricsEvaluator
except ImportError:
    from ps2.tracking.tracker import OcclusionAwareReIDTracker
    from ps2.simulation.dataset_loader import MOTDatasetManager
    from ps2.evaluation.mot_metrics import MOTMetricsEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on MOT Sequence")
    parser.add_argument("--seq_dir", type=str, required=True, help="Path to sequence directory containing gt/gt.txt and det/det.txt")
    parser.add_argument("--config", type=str, default="./configs/default_tracker.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    gt_file = os.path.join(args.seq_dir, "gt", "gt.txt")
    det_file = os.path.join(args.seq_dir, "det", "det.txt")

    if not os.path.exists(gt_file):
        print(f"[Error] Ground truth file not found: {gt_file}")
        return

    gt_by_frame = MOTDatasetManager.load_mot_gt(gt_file)

    # Parse detections
    dets_by_frame = {}
    if os.path.exists(det_file):
        with open(det_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    f_idx = int(parts[0])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    score = float(parts[6])
                    box = np.array([x, y, x + w, y + h], dtype=np.float32)

                    if f_idx not in dets_by_frame:
                        dets_by_frame[f_idx] = {"boxes": [], "scores": []}
                    dets_by_frame[f_idx]["boxes"].append(box)
                    dets_by_frame[f_idx]["scores"].append(score)

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    tracker = OcclusionAwareReIDTracker(config=config)
    evaluator = MOTMetricsEvaluator(iou_threshold=0.50)
    pred_by_frame = {}

    all_frames = sorted(list(gt_by_frame.keys()))
    for f in all_frames:
        frame_dets = dets_by_frame.get(f, {"boxes": [], "scores": []})
        boxes = np.array(frame_dets["boxes"], dtype=np.float32) if frame_dets["boxes"] else np.empty((0, 4), dtype=np.float32)
        scores = np.array(frame_dets["scores"], dtype=np.float32) if frame_dets["scores"] else np.empty(0, dtype=np.float32)

        tracks = tracker.update(boxes, scores)
        pred_by_frame[f] = [{"id": t.track_id, "bbox_xyxy": t.to_xyxy} for t in tracks]

    metrics = evaluator.evaluate(gt_by_frame, pred_by_frame)
    print("\n" + "=" * 55)
    print(f"  MOT EVALUATION RESULTS: {os.path.basename(args.seq_dir)}")
    print("=" * 55)
    print(f" MOTA  : {metrics['MOTA']:.2%}")
    print(f" IDF1  : {metrics['IDF1']:.2%}")
    print(f" HOTA  : {metrics['HOTA']:.2%}")
    print(f" IDSW  : {metrics['IDSW']}")
    print(f" MT    : {metrics['MT_pct']:.1%}")
    print(f" ML    : {metrics['ML_pct']:.1%}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
