"""ResNet18 glyph classifier."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def eval_transforms(img_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transforms(img_size: int = 128) -> transforms.Compose:
    """Moderate augmentation: small affine jitter and colour jitter, no flips or erasing."""
    return transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.05, 0.05), shear=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_resnet18(num_classes: int, dropout: float | None = None, pretrained: bool = False) -> nn.Module:
    """ResNet18 with a new head: ``Linear``, or ``Dropout -> Linear`` when ``dropout`` is set."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    if dropout is None:
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def load_classifier(path: str, num_classes: int, device: str | torch.device) -> nn.Module:
    """Load a checkpoint written by either head variant and return it in eval mode.

    Older checkpoints have a plain ``fc`` layer (keys ``fc.weight``); those trained
    with dropout store it as ``fc.1.weight``.
    """
    state = torch.load(path, map_location="cpu")
    has_dropout_head = "fc.1.weight" in state
    model = build_resnet18(num_classes, dropout=0.4 if has_dropout_head else None)
    model.load_state_dict(state)
    return model.to(device).eval()


def class_names_from_dir(train_dir: str) -> list[str]:
    """Class names in ``ImageFolder`` order (sorted sub-directory names)."""
    return sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
