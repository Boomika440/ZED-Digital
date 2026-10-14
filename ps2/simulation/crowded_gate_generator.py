"""
Realistic Crowded Gate Scenario Simulator.
Generates complex surveillance benchmark sequences with:
- Narrow choke-point bottleneck (gate / turnstile).
- Bi-directional crossing pedestrian traffic.
- Mutual dynamic occlusions (depth ordering, visual overlap).
- Detection score attenuation during partial occlusion (high -> low -> missed -> recovered).
- Ground truth annotations formatted to MOTChallenge standard.
"""

import os
import random
import numpy as np
import cv2
from typing import Dict, List, Tuple


class PedestrianAgent:
    """Simulates a pedestrian moving through a crowded gate."""
    def __init__(self, pid: int, start_x: float, start_y: float, target_x: float, target_y: float, speed: float, z_depth: float):
        self.pid = pid
        self.x = float(start_x)
        self.y = float(start_y)
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.speed = float(speed)
        self.z_depth = float(z_depth)  # Smaller z = closer to camera, occludes higher z

        # Geometry
        self.w = random.uniform(50.0, 65.0)
        self.h = self.w * random.uniform(2.6, 3.1)

        # Unique visual appearance signature for testing Re-ID
        random.seed(2000 + pid)
        self.color_shirt = (random.randint(40, 240), random.randint(40, 240), random.randint(40, 240))
        self.color_pants = (random.randint(30, 200), random.randint(30, 200), random.randint(30, 200))
        self.color_skin = (180, 200, 230)

        # Heading vector
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = max(1e-4, np.hypot(dx, dy))
        self.vx = (dx / dist) * self.speed
        self.vy = (dy / dist) * self.speed

        self.is_active = True

    def step(self, jitter: float = 0.5):
        """Advances position toward target with subtle natural motion noise."""
        if not self.is_active:
            return

        # Slight random perturbation
        self.x += self.vx + random.uniform(-jitter, jitter)
        self.y += self.vy + random.uniform(-jitter, jitter)

        # Check if reached target
        if np.hypot(self.target_x - self.x, self.target_y - self.y) < 15.0:
            # Change target to exit camera frame
            if self.vx > 0:
                self.target_x = 2100.0
            else:
                self.target_x = -200.0
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = max(1e-4, np.hypot(dx, dy))
            self.vx = (dx / dist) * self.speed
            self.vy = (dy / dist) * self.speed

        # Deactivate if out of frame
        if self.x < -150 or self.x > 2100 or self.y < -150 or self.y > 1200:
            self.is_active = False

    @property
    def bbox_xyxy(self) -> np.ndarray:
        x1 = self.x - self.w * 0.5
        y1 = self.y - self.h * 0.5
        x2 = self.x + self.w * 0.5
        y2 = self.y + self.h * 0.5
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class CrowdedGateBenchmarkGenerator:
    """
    Generates realistic synthetic evaluation sequences replicating surveillance choke-points.
    """
    def __init__(
        self,
        num_pedestrians: int = 20,
        seq_length: int = 300,
        img_w: int = 1920,
        img_h: int = 1080,
        gate_x_center: float = 960.0,
        gate_width: float = 380.0
    ):
        self.num_pedestrians = num_pedestrians
        self.seq_length = seq_length
        self.img_w = img_w
        self.img_h = img_h
        self.gate_x_center = gate_x_center
        self.gate_width = gate_width

    def generate(self) -> dict:
        """
        Simulates crowd sequence and computes:
        1. Ground truth annotations per frame: [frame_id, pid, x1, y1, w, h, visibility]
        2. Realistic noisy detections: [frame_id, x1, y1, x2, y2, score, appearance_feature]
        """
        random.seed(42)
        np.random.seed(42)

        # Initialize agents entering from left and right towards the narrow gate
        agents: List[PedestrianAgent] = []
        for i in range(self.num_pedestrians):
            from_left = (i % 2 == 0)
            if from_left:
                start_x = random.uniform(-100, 300)
                start_y = random.uniform(300, 800)
                target_x = self.gate_x_center + random.uniform(-self.gate_width * 0.3, self.gate_width * 0.3)
                target_y = random.uniform(400, 700)
            else:
                start_x = random.uniform(self.img_w - 300, self.img_w + 100)
                start_y = random.uniform(300, 800)
                target_x = self.gate_x_center + random.uniform(-self.gate_width * 0.3, self.gate_width * 0.3)
                target_y = random.uniform(400, 700)

            speed = random.uniform(2.5, 4.5)
            z_depth = random.uniform(1.0, 5.0) # Depth for occlusion sorting
            agents.append(PedestrianAgent(i + 1, start_x, start_y, target_x, target_y, speed, z_depth))

        gt_records = []      # list of dicts: {frame, id, bbox_xyxy, visibility}
        det_records = []     # list of dicts: {frame, bboxes_xyxy, scores, features}

        for frame_idx in range(1, self.seq_length + 1):
            # Advance all active agents
            for a in agents:
                a.step()

            # Active agents currently inside the camera frame
            active = [a for a in agents if a.is_active and 0 <= a.x <= self.img_w and 0 <= a.y <= self.img_h]

            # Sort by depth (closer agents occlude farther agents)
            active.sort(key=lambda ag: ag.z_depth)

            frame_gt_boxes = []
            frame_gt_ids = []
            frame_visibilities = []

            # Compute mutual occlusion for each agent
            for i, ag in enumerate(active):
                box_i = ag.bbox_xyxy
                area_i = max(1e-4, (box_i[2] - box_i[0]) * (box_i[3] - box_i[1]))

                # Occlusion by any agent closer to the camera (j < i)
                occluded_area = 0.0
                for j in range(i):
                    box_j = active[j].bbox_xyxy
                    inter_x1 = max(box_i[0], box_j[0])
                    inter_y1 = max(box_i[1], box_j[1])
                    inter_x2 = min(box_i[2], box_j[2])
                    inter_y2 = min(box_i[3], box_j[3])
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_w = inter_x2 - inter_x1
                        inter_h = inter_y2 - inter_y1
                        occluded_area += inter_w * inter_h

                visibility = max(0.0, min(1.0, 1.0 - (occluded_area / area_i)))

                frame_gt_boxes.append(box_i)
                frame_gt_ids.append(ag.pid)
                frame_visibilities.append(visibility)

                gt_records.append({
                    "frame": frame_idx,
                    "id": ag.pid,
                    "bbox_xyxy": box_i.copy(),
                    "visibility": visibility
                })

            # Synthesize realistic detector outputs based on visibility
            frame_det_boxes = []
            frame_det_scores = []
            frame_det_feats = []

            for ag, box, vis in zip(active, frame_gt_boxes, frame_visibilities):
                # Detection behavior modeling:
                # vis > 0.65: High score detection (0.75 - 0.98)
                # 0.20 < vis <= 0.65: Low score detection due to partial occlusion (0.18 - 0.55)
                # vis <= 0.20: Severe occlusion -> Missed detection!
                if vis <= 0.20:
                    continue  # Detector fails to output box during severe occlusion

                if vis > 0.65:
                    score = float(np.clip(random.uniform(0.75, 0.98), 0.1, 1.0))
                else:
                    score = float(np.clip(random.uniform(0.20, 0.55), 0.1, 1.0))

                # Minor bounding box localization noise (center shift and scale jitter)
                box_noisy = box.copy()
                box_noisy[0] += random.uniform(-3, 3)
                box_noisy[1] += random.uniform(-3, 3)
                box_noisy[2] += random.uniform(-3, 3)
                box_noisy[3] += random.uniform(-3, 3)

                # Simulated 128-D appearance embedding based on identity signature + sensor noise
                np.random.seed(ag.pid * 500)
                base_feat = np.random.randn(128).astype(np.float32)
                base_feat /= np.linalg.norm(base_feat)

                # Observation noise (increases if partially occluded)
                noise_scale = 0.05 if vis > 0.65 else 0.25
                noisy_feat = base_feat + np.random.randn(128).astype(np.float32) * noise_scale
                noisy_feat /= np.linalg.norm(noisy_feat)

                frame_det_boxes.append(box_noisy)
                frame_det_scores.append(score)
                frame_det_feats.append(noisy_feat)

            # Random False Positive clutter noise (e.g. reflection, gate post)
            if random.random() < 0.10:
                fp_x = random.uniform(600, 1300)
                fp_y = random.uniform(300, 800)
                fp_box = np.array([fp_x - 25, fp_y - 70, fp_x + 25, fp_y + 70], dtype=np.float32)
                fp_feat = np.random.randn(128).astype(np.float32)
                fp_feat /= np.linalg.norm(fp_feat)
                frame_det_boxes.append(fp_box)
                frame_det_scores.append(random.uniform(0.18, 0.45))
                frame_det_feats.append(fp_feat)

            det_records.append({
                "frame": frame_idx,
                "bboxes": np.array(frame_det_boxes, dtype=np.float32) if frame_det_boxes else np.empty((0, 4), dtype=np.float32),
                "scores": np.array(frame_det_scores, dtype=np.float32) if frame_det_scores else np.empty(0, dtype=np.float32),
                "features": np.array(frame_det_feats, dtype=np.float32) if frame_det_feats else np.empty((0, 128), dtype=np.float32)
            })

        print(f"[Simulator] Generated {self.seq_length} frames with {self.num_pedestrians} pedestrians across crowded gate.")
        return {
            "gt_records": gt_records,
            "det_records": det_records,
            "seq_length": self.seq_length,
            "num_pedestrians": self.num_pedestrians
        }
