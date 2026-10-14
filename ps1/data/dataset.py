"""
PyTorch Dataset and Batch Sampler implementations for Open-Set Person Re-Identification.
Includes RandomIdentitySampler for PxK mini-batch construction required for hard negative mining.
"""

import os
import re
import glob
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image

try:
    from .transforms import build_transforms
except (ImportError, ValueError):
    try:
        from data.transforms import build_transforms
    except ImportError:
        from ps1.data.transforms import build_transforms


class PersonReIDDataset(Dataset):
    """
    General Person Re-ID Dataset.
    Loads image paths and parses identity labels from filename patterns.
    """
    def __init__(self, folder_path: str, transform=None, is_unknown: bool = False):
        self.folder_path = folder_path
        self.transform = transform
        self.is_unknown = is_unknown
        self.samples = []  # List of (filepath, pid, camid)
        self.pid_to_idx = {}
        self._load_samples()

    def _load_samples(self):
        extensions = ("*.jpg", "*.jpeg", "*.png")
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(self.folder_path, ext)))

        pids = set()
        raw_samples = []

        for p in img_paths:
            fname = os.path.basename(p)
            # Match standard patterns: 'id_0001_...' or '0001_c1s1_...' or 'unknown_0001_...'
            match_id = re.search(r'(?:id_|unknown_)?(\d+)', fname)
            if match_id:
                pid = int(match_id.group(1))
            else:
                pid = -1

            match_cam = re.search(r'c(\d+)', fname)
            camid = int(match_cam.group(1)) if match_cam else 1

            if not self.is_unknown and pid != -1:
                pids.add(pid)
            raw_samples.append((p, pid, camid))

        if not self.is_unknown:
            self.pid_to_idx = {pid: idx for idx, pid in enumerate(sorted(list(pids)))}
            self.samples = [
                (path, self.pid_to_idx[pid], camid)
                for path, pid, camid in raw_samples if pid in self.pid_to_idx
            ]
        else:
            # Unknown samples receive label -1 to distinguish from gallery enrolled IDs
            self.samples = [(path, -1, camid) for path, _, camid in raw_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, pid, camid = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, path


class RandomIdentitySampler(Sampler):
    """
    Randomly samples P identities, and K instances for each identity in every batch.
    Batch size = P * K.
    Guarantees that hard triplet mining always has positive pairs and diverse negative candidates.
    """
    def __init__(self, data_source: PersonReIDDataset, batch_size: int, num_instances: int):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source.samples):
            pid = item[1]
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = self.index_dic[pid].copy()
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            np.random.shuffle(idxs)
            batch_idxs = []
            for i in range(0, len(idxs) - len(idxs) % self.num_instances, self.num_instances):
                batch_idxs.append(idxs[i:i + self.num_instances])
            batch_idxs_dict[pid] = batch_idxs

        avai_pids = self.pids.copy()
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_dataloaders(config: dict):
    """
    Builds data loaders for training, gallery, known probes, and unknown impostor probes.
    """
    dataset_cfg = config.get("dataset", {})
    data_dir = dataset_cfg.get("data_dir", "./data_storage/low_variance_uniform")
    img_h = dataset_cfg.get("img_height", 256)
    img_w = dataset_cfg.get("img_width", 128)
    batch_p = dataset_cfg.get("batch_p", 8)
    batch_k = dataset_cfg.get("batch_k", 4)
    batch_size = batch_p * batch_k
    num_workers = dataset_cfg.get("num_workers", 0)

    train_tf = build_transforms(img_h, img_w, is_train=True)
    val_tf = build_transforms(img_h, img_w, is_train=False)

    train_path = os.path.join(data_dir, "train")
    gallery_path = os.path.join(data_dir, "gallery")
    probe_known_path = os.path.join(data_dir, "probe_known")
    probe_unknown_path = os.path.join(data_dir, "probe_unknown")

    train_dataset = PersonReIDDataset(train_path, transform=train_tf, is_unknown=False)
    gallery_dataset = PersonReIDDataset(gallery_path, transform=val_tf, is_unknown=False)
    probe_known_dataset = PersonReIDDataset(probe_known_path, transform=val_tf, is_unknown=False)
    probe_unknown_dataset = PersonReIDDataset(probe_unknown_path, transform=val_tf, is_unknown=True)

    train_sampler = RandomIdentitySampler(train_dataset, batch_size=batch_size, num_instances=batch_k)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    probe_known_loader = DataLoader(probe_known_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    probe_unknown_loader = DataLoader(probe_unknown_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

    return {
        "train_loader": train_loader,
        "gallery_loader": gallery_loader,
        "probe_known_loader": probe_known_loader,
        "probe_unknown_loader": probe_unknown_loader,
        "num_classes": len(train_dataset.pid_to_idx),
        "train_dataset": train_dataset,
        "gallery_dataset": gallery_dataset,
        "probe_known_dataset": probe_known_dataset,
        "probe_unknown_dataset": probe_unknown_dataset
    }
