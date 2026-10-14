"""
Curated Low Inter-Class Variance Benchmark Generator.
Generates or curates a benchmark where identities share near-identical global appearance
(same uniform, similar build, same workplace clothing), while differing only in subtle,
fine-grained discriminative features (badges, shoes, watch, accessories, facial contour).

This enables exact, reproducible evaluation of:
1. False-Accept Rate (FAR) under low inter-class variance
2. True-Accept Rate (TAR / Recall)
3. Open-set rejection of unenrolled impostors in identical attire.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class LowVarianceDatasetGenerator:
    """
    Synthesizes a realistic low-variance open-set Person Re-ID dataset.
    All subjects wear the exact same organizational uniform:
      - Top: Dark Navy Work Shirt (#1E2A38)
      - Bottom: Charcoal Work Trousers (#2B2B2B)
    Discriminative cues are localized to micro-regions:
      - Badge / Pocket ID (upper chest)
      - Footwear / Shoe type and accent (bottom 15% of frame)
      - Wrist accessory / watch (mid-left/right)
      - Hair style / headwear (top 15% of frame)
      - Belt buckle accent
    """
    def __init__(
        self,
        output_dir: str = "./data_storage/low_variance_uniform",
        num_enrolled_ids: int = 60,
        num_unknown_ids: int = 40,
        instances_per_id: int = 10,
        img_h: int = 256,
        img_w: int = 128
    ):
        self.output_dir = output_dir
        self.num_enrolled_ids = num_enrolled_ids
        self.num_unknown_ids = num_unknown_ids
        self.instances_per_id = instances_per_id
        self.img_h = img_h
        self.img_w = img_w

    def _generate_identity_profile(self, id_idx: int) -> dict:
        """
        Generates unique fine-grained attributes for an identity.
        """
        np.random.seed(1000 + id_idx)
        random.seed(1000 + id_idx)

        # Uniform base (subtle intra-class shading)
        shirt_base = np.array([28, 42, 60], dtype=np.int32)
        pants_base = np.array([40, 42, 45], dtype=np.int32)

        # Micro-attributes that distinguish this specific person
        badge_palette = [
            (220, 20, 60),   # Red lanyard
            (30, 144, 255),  # Cyan badge
            (255, 215, 0),   # Gold visitor pass
            (50, 205, 50),   # Green pass
            (255, 140, 0),   # Orange badge
            (148, 0, 211),   # Violet badge
            (200, 200, 200)  # Silver card
        ]
        shoe_palette = [
            (20, 20, 20),    # All-black steel-toe
            (230, 230, 230), # White safety sneakers
            (139, 69, 19),   # Tan work boots
            (70, 70, 70),    # Gray athletic
            (30, 30, 80)     # Navy runners
        ]

        hair_colors = [
            (40, 30, 25),    # Dark brown
            (15, 15, 15),    # Jet black
            (110, 80, 50),   # Light brown
            (160, 140, 110), # Blonde
            (90, 90, 95)     # Gray
        ]

        skin_tones = [
            (235, 195, 160),
            (210, 165, 130),
            (185, 135, 100),
            (140, 95, 65),
            (245, 215, 185)
        ]

        return {
            "id": id_idx,
            "shirt_rgb": shirt_base,
            "pants_rgb": pants_base,
            "badge_color": badge_palette[id_idx % len(badge_palette)],
            "badge_pos_x": 40 + (id_idx % 3) * 12,
            "shoe_color": shoe_palette[(id_idx // 3) % len(shoe_palette)],
            "shoe_accent": ((id_idx * 50) % 255, (id_idx * 80) % 255, 200),
            "hair_color": hair_colors[(id_idx // 2) % len(hair_colors)],
            "hair_length": 15 + (id_idx % 5) * 4,
            "skin_tone": skin_tones[(id_idx // 4) % len(skin_tones)],
            "has_watch": (id_idx % 2 == 0),
            "watch_color": (210, 210, 210) if id_idx % 4 == 0 else (180, 140, 40),
            "height_scale": 0.95 + ((id_idx % 7) / 70.0)
        }

    def _render_person_image(self, profile: dict, instance_idx: int) -> Image.Image:
        """
        Renders an individual sample with natural camera variations (pose, noise, lighting).
        """
        # Seed for repeatable variation per instance
        inst_seed = profile["id"] * 100 + instance_idx
        np.random.seed(inst_seed)
        random.seed(inst_seed)

        img = Image.new("RGB", (self.img_w, self.img_h), color=(225, 228, 232))
        draw = ImageDraw.Draw(img)

        # Lighting perturbation for low-variance challenge
        light_jitter = np.random.randint(-15, 15)

        # 1. Background gradient (simulating indoor hallway)
        for y in range(self.img_h):
            val = int(220 + 20 * (y / self.img_h) + np.random.randint(-3, 3))
            draw.line([(0, y), (self.img_w, y)], fill=(val, val, val + 2))

        # Person body geometry
        scale = profile["height_scale"]
        head_top = int(20 * (2.0 - scale))
        head_bottom = head_top + int(36 * scale)
        torso_bottom = head_bottom + int(95 * scale)
        legs_bottom = self.img_h - 12

        # 2. Head & Hair
        skin = tuple(np.clip(np.array(profile["skin_tone"]) + light_jitter, 0, 255).astype(int))
        hair = tuple(profile["hair_color"])
        draw.ellipse([48, head_top, 80, head_bottom], fill=skin)
        # Hair top
        draw.arc([46, head_top - 2, 82, head_top + profile["hair_length"]], 0, 360, fill=hair, width=5)
        draw.chord([48, head_top - 3, 80, head_top + 14], 180, 360, fill=hair)

        # 3. Torso (Uniform Shirt)
        shirt_col = tuple(np.clip(profile["shirt_rgb"] + light_jitter, 0, 255).astype(int))
        shoulder_w = 42 + np.random.randint(-2, 3)
        draw.polygon([
            (self.img_w // 2 - shoulder_w // 2, head_bottom - 2),
            (self.img_w // 2 + shoulder_w // 2, head_bottom - 2),
            (self.img_w // 2 + shoulder_w // 2 + 5, torso_bottom),
            (self.img_w // 2 - shoulder_w // 2 - 5, torso_bottom)
        ], fill=shirt_col)

        # Subtle collar V-neck
        draw.polygon([(60, head_bottom - 2), (68, head_bottom + 12), (64, head_bottom - 2)], fill=skin)

        # Micro-discriminative cue: ID Badge / Lanyard
        bx = profile["badge_pos_x"]
        by = head_bottom + 15
        draw.line([(64, head_bottom + 5), (bx + 6, by)], fill=profile["badge_color"], width=2)
        draw.rectangle([bx, by, bx + 12, by + 18], fill=profile["badge_color"])
        draw.rectangle([bx + 2, by + 3, bx + 10, by + 10], fill=(255, 255, 255))

        # 4. Arms & Wrist Watch
        draw.rectangle([22, head_bottom + 6, 32, torso_bottom - 10], fill=shirt_col)
        draw.rectangle([96, head_bottom + 6, 106, torso_bottom - 10], fill=shirt_col)
        # Hands
        draw.ellipse([22, torso_bottom - 12, 32, torso_bottom + 6], fill=skin)
        draw.ellipse([96, torso_bottom - 12, 106, torso_bottom + 6], fill=skin)

        # Micro-cue: Watch
        if profile["has_watch"]:
            draw.rectangle([22, torso_bottom - 15, 32, torso_bottom - 11], fill=profile["watch_color"])

        # 5. Legs (Uniform Trousers)
        pants_col = tuple(np.clip(profile["pants_rgb"] + light_jitter, 0, 255).astype(int))
        # Left leg
        draw.polygon([(36, torso_bottom), (61, torso_bottom), (58, legs_bottom - 18), (34, legs_bottom - 18)], fill=pants_col)
        # Right leg
        draw.polygon([(67, torso_bottom), (92, torso_bottom), (94, legs_bottom - 18), (70, legs_bottom - 18)], fill=pants_col)

        # 6. Micro-discriminative cue: Shoes / Footwear
        shoe_col = tuple(profile["shoe_color"])
        draw.rectangle([32, legs_bottom - 18, 59, legs_bottom], fill=shoe_col)
        draw.rectangle([69, legs_bottom - 18, 96, legs_bottom], fill=shoe_col)
        # Shoe sole accent
        draw.line([(32, legs_bottom), (59, legs_bottom)], fill=profile["shoe_accent"], width=3)
        draw.line([(69, legs_bottom), (96, legs_bottom)], fill=profile["shoe_accent"], width=3)

        # Camera blur and subtle noise
        if instance_idx % 3 == 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

        return img

    def generate(self) -> dict:
        """
        Generates full open-set evaluation benchmark partitioned into:
          - Train set (enrolled IDs)
          - Gallery set (enrolled IDs enrolled in database)
          - Probe Known (enrolled returning IDs)
          - Probe Unknown (unenrolled impostor IDs wearing the same uniform)
        """
        paths = {
            "train": os.path.join(self.output_dir, "train"),
            "gallery": os.path.join(self.output_dir, "gallery"),
            "probe_known": os.path.join(self.output_dir, "probe_known"),
            "probe_unknown": os.path.join(self.output_dir, "probe_unknown")
        }

        for p in paths.values():
            os.makedirs(p, exist_ok=True)

        stats = {"train_samples": 0, "gallery_samples": 0, "probe_known_samples": 0, "probe_unknown_samples": 0}

        # 1. Enrolled identities (Training & Known Evaluation)
        for id_idx in range(self.num_enrolled_ids):
            profile = self._generate_identity_profile(id_idx)
            # Generate instances
            for inst in range(self.instances_per_id):
                img = self._render_person_image(profile, inst)
                fname = f"id_{id_idx:04d}_c1_inst_{inst:03d}.jpg"

                if inst < self.instances_per_id // 2:
                    # Half for training
                    img.save(os.path.join(paths["train"], fname))
                    stats["train_samples"] += 1
                elif inst == self.instances_per_id // 2:
                    # One instance for gallery registration
                    img.save(os.path.join(paths["gallery"], fname))
                    stats["gallery_samples"] += 1
                else:
                    # Remaining instances for known query probes
                    img.save(os.path.join(paths["probe_known"], fname))
                    stats["probe_known_samples"] += 1

        # 2. Unknown Impostor Identities (Open-Set Evaluation: Must be rejected!)
        for id_idx in range(self.num_enrolled_ids, self.num_enrolled_ids + self.num_unknown_ids):
            profile = self._generate_identity_profile(id_idx)
            for inst in range(max(2, self.instances_per_id // 3)):
                img = self._render_person_image(profile, inst)
                fname = f"unknown_{id_idx:04d}_c1_inst_{inst:03d}.jpg"
                img.save(os.path.join(paths["probe_unknown"], fname))
                stats["probe_unknown_samples"] += 1

        print(f"[LowVarianceBenchmark] Successfully generated dataset at {self.output_dir}")
        print(f"[LowVarianceBenchmark] Statistics: {stats}")
        return {
            "paths": paths,
            "stats": stats,
            "num_enrolled": self.num_enrolled_ids,
            "num_unknown": self.num_unknown_ids
        }


if __name__ == "__main__":
    generator = LowVarianceDatasetGenerator(
        output_dir="./data_storage/low_variance_uniform",
        num_enrolled_ids=40,
        num_unknown_ids=25,
        instances_per_id=8
    )
    generator.generate()
