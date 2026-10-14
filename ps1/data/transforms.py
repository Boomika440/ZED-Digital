"""
Data augmentations and transformations designed for Person Re-Identification.
Specifically engineered for low inter-class variance where global appearance is identical,
forcing the network to attend to fine-grained spatial cues via Random Erasing.
"""

import math
import random
import torch
import torchvision.transforms as T
from PIL import Image


class RandomErasing:
    """
    Random Erasing Data Augmentation (Zhong et al., AAAI 2020).
    Randomly masks rectangular regions in person crops to prevent overfitting to
    uniform base clothing and force the backbone to learn multiple localized discriminative tokens.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) > self.probability:
            return img_tensor

        for _ in range(100):
            area = img_tensor.size(1) * img_tensor.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_tensor.size(2) and h < img_tensor.size(1):
                x1 = random.randint(0, img_tensor.size(1) - h)
                y1 = random.randint(0, img_tensor.size(2) - w)

                if img_tensor.size(0) == 3:
                    img_tensor[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img_tensor[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img_tensor[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img_tensor[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img_tensor

        return img_tensor


def build_transforms(img_height: int = 256, img_width: int = 128, is_train: bool = True):
    """
    Constructs the preprocessing pipeline for training and evaluation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        transform = T.Compose([
            T.Resize((img_height, img_width), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10, padding_mode='edge'),
            T.RandomCrop((img_height, img_width)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            RandomErasing(probability=0.5, mean=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((img_height, img_width), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return transform
