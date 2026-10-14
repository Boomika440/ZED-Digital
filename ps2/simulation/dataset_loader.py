"""
MOTChallenge Data Format Importer and Exporter.
Handles loading and formatting of standard MOTChallenge annotation files:
- gt/gt.txt: [frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility]
- det/det.txt: [frame, -1, bb_left, bb_top, bb_width, bb_height, score]
"""

import os
from typing import Dict, List, Tuple
import numpy as np


class MOTDatasetManager:
    """
    Manages MOTChallenge format ground truth and detection streams.
    """
    @staticmethod
    def export_to_mot_challenge(sim_data: dict, output_dir: str):
        """
        Exports simulation data to standard MOTChallenge directory hierarchy.
        """
        gt_dir = os.path.join(output_dir, "gt")
        det_dir = os.path.join(output_dir, "det")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(det_dir, exist_ok=True)

        gt_file = os.path.join(gt_dir, "gt.txt")
        det_file = os.path.join(det_dir, "det.txt")

        # Export Ground Truth
        with open(gt_file, "w") as f:
            for rec in sim_data["gt_records"]:
                f_idx = rec["frame"]
                pid = rec["id"]
                x1, y1, x2, y2 = rec["bbox_xyxy"]
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                vis = rec["visibility"]
                # Format: frame, id, x, y, w, h, mark=1, label=1 (pedestrian), visibility
                f.write(f"{f_idx},{pid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,{vis:.2f}\n")

        # Export Detections
        with open(det_file, "w") as f:
            for rec in sim_data["det_records"]:
                f_idx = rec["frame"]
                for box, score in zip(rec["bboxes"], rec["scores"]):
                    x1, y1, x2, y2 = box
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    # Format: frame, -1, x, y, w, h, score, -1, -1, -1
                    f.write(f"{f_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.3f},-1,-1,-1\n")

        print(f"[MOTDatasetManager] Exported MOT sequence to: {output_dir}")

    @staticmethod
    def load_mot_gt(gt_file_path: str) -> Dict[int, List[dict]]:
        """
        Loads MOTChallenge gt.txt file.
        Returns: Dict mapping frame_id -> list of {'id': pid, 'bbox_xyxy': array, 'visibility': float}
        """
        frames_gt = {}
        if not os.path.exists(gt_file_path):
            return frames_gt

        with open(gt_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                frame_idx = int(parts[0])
                pid = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                vis = float(parts[8]) if len(parts) > 8 else 1.0

                box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

                if frame_idx not in frames_gt:
                    frames_gt[frame_idx] = []
                frames_gt[frame_idx].append({
                    "id": pid,
                    "bbox_xyxy": box_xyxy,
                    "visibility": vis
                })

        return frames_gt
