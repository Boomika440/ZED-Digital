"""
End-to-End Online Multi-Object Tracking Pipeline Execution.
Executes OAR-Tracker over a simulated crowded gate surveillance sequence,
computes official MOTChallenge metrics, and optionally exports annotated video.
"""

import os
import sys
import argparse
import time
import yaml
import cv2
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
    from simulation.crowded_gate_generator import CrowdedGateBenchmarkGenerator
    from evaluation.mot_metrics import MOTMetricsEvaluator
    from evaluation.visualizer import TrackingVisualizer
except ImportError:
    from ps2.tracking.tracker import OcclusionAwareReIDTracker
    from ps2.simulation.crowded_gate_generator import CrowdedGateBenchmarkGenerator
    from ps2.evaluation.mot_metrics import MOTMetricsEvaluator
    from ps2.evaluation.visualizer import TrackingVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run OAR Multi-Object Tracker")
    parser.add_argument("--config", type=str, default="./configs/default_tracker.yaml")
    parser.add_argument("--num_people", type=int, default=20, help="Number of simulated pedestrians in gate")
    parser.add_argument("--seq_length", type=int, default=300, help="Sequence length in frames")
    parser.add_argument("--render_video", action="store_true", default=False, help="Render annotated video to disk")
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    print("=" * 65)
    print("  Zed Digital - PS-2: Identity-Preserving Multi-Object Tracker")
    print("=" * 65)

    # 1. Generate Realistic Crowded Gate Scenario
    print(f"[RunTracker] Generating crowded gate benchmark ({args.num_people} pedestrians, {args.seq_length} frames)...")
    generator = CrowdedGateBenchmarkGenerator(
        num_pedestrians=args.num_people,
        seq_length=args.seq_length
    )
    sim_data = generator.generate()

    # Reorganize Ground Truth by frame
    gt_by_frame = {}
    for rec in sim_data["gt_records"]:
        f = rec["frame"]
        if f not in gt_by_frame:
            gt_by_frame[f] = []
        gt_by_frame[f].append({"id": rec["id"], "bbox_xyxy": rec["bbox_xyxy"]})

    # 2. Initialize Tracker and Evaluator
    tracker = OcclusionAwareReIDTracker(config=config)
    evaluator = MOTMetricsEvaluator(iou_threshold=0.50)
    visualizer = TrackingVisualizer(img_w=1920, img_h=1080) if args.render_video else None

    video_writer = None
    if args.render_video:
        video_path = os.path.join(args.output_dir, "crowded_gate_tracking_demo.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1920, 1080))
        print(f"[RunTracker] Video rendering enabled: {video_path}")

    # 3. Online Causal Tracking Loop
    print("[RunTracker] Running online causal tracking...")
    pred_by_frame = {}
    latencies = []
    total_start = time.time()

    for det_rec in sim_data["det_records"]:
        f_idx = det_rec["frame"]
        boxes = det_rec["bboxes"]
        scores = det_rec["scores"]
        features = det_rec["features"]

        t0 = time.perf_counter()
        active_tracks = tracker.update(boxes, scores, features)
        t1 = time.perf_counter()

        frame_lat_ms = (t1 - t0) * 1000.0
        latencies.append(frame_lat_ms)

        pred_by_frame[f_idx] = [
            {"id": t.track_id, "bbox_xyxy": t.to_xyxy}
            for t in active_tracks
        ]

        if video_writer is not None and visualizer is not None:
            canvas = visualizer.render_frame(active_tracks, f_idx, latency_ms=frame_lat_ms)
            video_writer.write(canvas)

    if video_writer is not None:
        video_writer.release()
        print(f"[RunTracker] Completed video rendering.")

    total_time = time.time() - total_start
    mean_latency = float(np.mean(latencies))
    fps = 1000.0 / max(1e-3, mean_latency)

    # 4. Standard MOTChallenge Evaluation
    print("\n[RunTracker] Computing official MOTChallenge metrics...")
    metrics = evaluator.evaluate(gt_by_frame, pred_by_frame)

    print("\n" + "=" * 60)
    print("        MOTCHALLENGE BENCHMARK EVALUATION REPORT")
    print("=" * 60)
    print(f" MOTA (Tracking Accuracy)      : {metrics['MOTA']:.2%}")
    print(f" IDF1 (Identification F1)      : {metrics['IDF1']:.2%}")
    print(f" HOTA (Higher-Order Accuracy)  : {metrics['HOTA']:.2%}")
    print(f" Identity Switches (IDSW)      : {metrics['IDSW']} occurrences")
    print("-" * 60)
    print(f" Mostly Tracked (MT >= 80%)    : {metrics['MT']} / {metrics['MT'] + metrics['ML'] + metrics['PT']} ({metrics['MT_pct']:.1%})")
    print(f" Mostly Lost (ML <= 20%)       : {metrics['ML']} ({metrics['ML_pct']:.1%})")
    print(f" Detection Recall              : {metrics['Recall']:.2%}")
    print(f" Detection Precision           : {metrics['Precision']:.2%}")
    print("-" * 60)
    print(f" Average Latency per Frame     : {mean_latency:.2f} ms")
    print(f" Real-Time Throughput          : {fps:.1f} FPS (Target > 30 FPS)")
    print(f" Total Processed Time          : {total_time:.2f} s")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    main()
