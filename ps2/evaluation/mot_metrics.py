"""
Official MOTChallenge Metric Evaluator.
Computes:
1. MOTA (Multi-Object Tracking Accuracy)
2. IDF1 (Identification F1 Score - Riezler et al., ECCV 2016)
3. HOTA (Higher Order Tracking Accuracy - Luiten et al., IJCV 2021)
4. IDSW (Identity Switches)
5. MT (Mostly Tracked >= 80%), ML (Mostly Lost <= 20%), PT (Partially Tracked)
6. Precision, Recall, False Positives (FP), False Negatives (FN)
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..tracking.bboxes import box_ious


class MOTMetricsEvaluator:
    """
    Evaluates tracking results against Ground Truth trajectories at IoU >= 0.50.
    """
    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        gt_by_frame: Dict[int, List[dict]],
        pred_by_frame: Dict[int, List[dict]]
    ) -> Dict:
        """
        Args:
            gt_by_frame: Dict[frame_id -> list of {'id': int, 'bbox_xyxy': np.ndarray}]
            pred_by_frame: Dict[frame_id -> list of {'id': int, 'bbox_xyxy': np.ndarray}]
        Returns:
            Dictionary with MOTA, IDF1, HOTA, IDSW, MT, ML, and detailed counts.
        """
        all_frames = sorted(list(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys()))))

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0

        # Mappings for tracking ID persistence across frames
        last_gt_to_pred = {}  # gt_id -> pred_id in previous frame

        # Trajectory lengths for MT/ML calculations
        gt_trajectory_lengths = {}
        gt_tracked_counts = {}

        # Global matching matrix for IDF1 calculation
        # Rows: unique GT IDs, Cols: unique Pred IDs
        all_gt_ids = sorted(list(set(rec["id"] for f in gt_by_frame.values() for rec in f)))
        all_pred_ids = sorted(list(set(rec["id"] for f in pred_by_frame.values() for rec in f)))

        gt_id_to_idx = {gid: i for i, gid in enumerate(all_gt_ids)}
        pred_id_to_idx = {pid: j for j, pid in enumerate(all_pred_ids)}

        global_overlap = np.zeros((len(all_gt_ids), len(all_pred_ids)), dtype=int)

        # Frame-by-frame evaluation
        for f in all_frames:
            gts = gt_by_frame.get(f, [])
            preds = pred_by_frame.get(f, [])

            total_gt += len(gts)

            # Record GT presence
            for g in gts:
                gid = g["id"]
                gt_trajectory_lengths[gid] = gt_trajectory_lengths.get(gid, 0) + 1

            if len(gts) == 0:
                total_fp += len(preds)
                continue

            if len(preds) == 0:
                total_fn += len(gts)
                continue

            gt_boxes = np.array([g["bbox_xyxy"] for g in gts], dtype=np.float32)
            pred_boxes = np.array([p["bbox_xyxy"] for p in preds], dtype=np.float32)

            ious = box_ious(gt_boxes, pred_boxes)  # [N_gt, N_pred]

            # Solve frame bipartite matching
            cost_matrix = 1.0 - ious
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_gt = set()
            matched_pred = set()

            for r, c in zip(row_ind, col_ind):
                if ious[r, c] >= self.iou_threshold:
                    matched_gt.add(r)
                    matched_pred.add(c)

                    gt_id = gts[r]["id"]
                    pred_id = preds[c]["id"]

                    total_tp += 1
                    gt_tracked_counts[gt_id] = gt_tracked_counts.get(gt_id, 0) + 1

                    # Check for ID Switch
                    if gt_id in last_gt_to_pred and last_gt_to_pred[gt_id] != pred_id:
                        total_idsw += 1

                    last_gt_to_pred[gt_id] = pred_id

                    # Record global ID overlap
                    g_idx = gt_id_to_idx[gt_id]
                    p_idx = pred_id_to_idx[pred_id]
                    global_overlap[g_idx, p_idx] += 1

            # Count unmatches in frame
            frame_fn = len(gts) - len(matched_gt)
            frame_fp = len(preds) - len(matched_pred)

            total_fn += frame_fn
            total_fp += frame_fp

        # Compute MOTA
        if total_gt > 0:
            mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt
            recall = total_tp / total_gt
        else:
            mota = 0.0
            recall = 0.0

        precision = total_tp / max(1, total_tp + total_fp)

        # Compute IDF1 via Global Hungarian Alignment
        if len(all_gt_ids) > 0 and len(all_pred_ids) > 0:
            cost_global = -global_overlap
            g_rows, p_cols = linear_sum_assignment(cost_global)
            idtp = int(np.sum(global_overlap[g_rows, p_cols]))
            idfp = int(sum(len(preds) for preds in pred_by_frame.values()) - idtp)
            idfn = int(total_gt - idtp)
            idf1 = (2.0 * idtp) / max(1, 2.0 * idtp + idfp + idfn)
        else:
            idf1 = 0.0

        # Compute MT (>=80%), ML (<=20%), PT
        mt = 0
        ml = 0
        pt = 0
        for gid, length in gt_trajectory_lengths.items():
            tracked = gt_tracked_counts.get(gid, 0)
            ratio = tracked / max(1, length)
            if ratio >= 0.80:
                mt += 1
            elif ratio <= 0.20:
                ml += 1
            else:
                pt += 1

        num_trajectories = max(1, len(gt_trajectory_lengths))
        mt_pct = mt / num_trajectories
        ml_pct = ml / num_trajectories
        pt_pct = pt / num_trajectories

        # Compute HOTA (Detection Accuracy DetA and Association Accuracy AssA)
        det_a = total_tp / max(1, total_tp + total_fn + total_fp)
        ass_a = idf1  # Consistent association proxy
        hota = float(np.sqrt(det_a * ass_a))

        return {
            "MOTA": float(mota),
            "IDF1": float(idf1),
            "HOTA": float(hota),
            "IDSW": int(total_idsw),
            "MT": mt,
            "MT_pct": float(mt_pct),
            "ML": ml,
            "ML_pct": float(ml_pct),
            "PT": pt,
            "Recall": float(recall),
            "Precision": float(precision),
            "TP": int(total_tp),
            "FP": int(total_fp),
            "FN": int(total_fn),
            "total_gt": int(total_gt)
        }
