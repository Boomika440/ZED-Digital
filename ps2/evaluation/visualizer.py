"""
Real-Time Multi-Object Tracking Visualizer.
Renders:
- Unique color per pedestrian ID.
- Trajectory trails displaying motion history.
- Occlusion state tags ('ACTIVE', 'OCCLUDED').
- Information HUD (Frame, FPS, Active Tracks, Occlusion Count).
"""

from typing import List, Optional
import numpy as np
import cv2

try:
    from ..tracking.track import Track
except (ImportError, ValueError):
    try:
        from tracking.track import Track
    except ImportError:
        from ps2.tracking.track import Track


class TrackingVisualizer:
    """
    Renders tracking overlays on video frames or blank canvas.
    """
    def __init__(self, img_w: int = 1920, img_h: int = 1080):
        self.img_w = img_w
        self.img_h = img_h
        # Deterministic palette of 100 visually distinct colors
        np.random.seed(42)
        self.palette = np.random.randint(50, 255, (1000, 3), dtype=np.uint8)

    def _get_color(self, track_id: int) -> tuple:
        c = self.palette[track_id % len(self.palette)]
        return int(c[0]), int(c[1]), int(c[2])

    def render_frame(
        self,
        tracks: List[Track],
        frame_idx: int,
        background_img: Optional[np.ndarray] = None,
        latency_ms: float = 0.0
    ) -> np.ndarray:
        """
        Renders bounding boxes, IDs, and trajectory trails on the frame.
        """
        if background_img is not None:
            canvas = background_img.copy()
        else:
            # Create dark surveillance theme canvas
            canvas = np.full((self.img_h, self.img_w, 3), 32, dtype=np.uint8)
            # Draw gate choke-point boundaries
            cv2.line(canvas, (750, 0), (750, self.img_h), (60, 60, 60), 2)
            cv2.line(canvas, (1170, 0), (1170, self.img_h), (60, 60, 60), 2)
            cv2.putText(canvas, "[GATE CHOKE-POINT]", (830, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        occluded_count = sum(1 for t in tracks if t.is_occluded)

        # 1. Draw Trajectory Trails
        for trk in tracks:
            color = self._get_color(trk.track_id)
            pts = trk.trajectory
            for i in range(1, len(pts)):
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                thickness = max(1, int(i * 3.0 / len(pts)))
                cv2.line(canvas, pt1, pt2, color, thickness)

        # 2. Draw Bounding Boxes and Status Badges
        for trk in tracks:
            box = trk.to_xyxy
            x1, y1, x2, y2 = [int(v) for v in box]
            color = self._get_color(trk.track_id)

            if trk.is_occluded:
                # Warning border for occluded tracks
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 3)
                label = f"ID:{trk.track_id} [OCCLUDED]"
                bg_color = (0, 140, 255)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{trk.track_id}"
                bg_color = color

            # Text badge
            (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - t_h - 6)), (x1 + t_w + 6, y1), bg_color, -1)
            cv2.putText(canvas, label, (x1 + 3, max(t_h + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. Information HUD
        hud_bg = (20, 20, 20)
        cv2.rectangle(canvas, (20, 20), (450, 140), hud_bg, -1)
        cv2.rectangle(canvas, (20, 20), (450, 140), (80, 80, 80), 1)

        fps = 1000.0 / max(1e-3, latency_ms) if latency_ms > 0 else 0.0
        cv2.putText(canvas, f"Zed Digital OAR-Tracker | Frame: {frame_idx:04d}", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.putText(canvas, f"Active Tracks: {len(tracks)} | Occluded: {occluded_count}", (35, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        cv2.putText(canvas, f"Latency: {latency_ms:.2f} ms ({fps:.1f} FPS)", (35, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

        return canvas
