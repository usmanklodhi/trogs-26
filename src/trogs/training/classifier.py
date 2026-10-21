"""Train the ResNet18 glyph classifier.

ImageNet-pretrained ResNet18 with a dropout head, class-weighted cross-entropy
(square-root inverse frequency, to soften extreme weights for rare letters),
SGD with momentum and a step learning-rate schedule. The best epoch by
validation accuracy is kept; a confusion matrix and classification report are
written next to the weights.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from trogs.models.classifier import build_resnet18, eval_transforms, train_transforms
from trogs.utils import resolve_device, timestamp

log = logging.getLogger(__name__)


@dataclass
class ClassifierTrainConfig:
    data_dir: str = "data/processed/greek_letters_classification_stratified"
    output_dir: str = "models"
    run_name: str = "v4"
    img_size: int = 128
    batch_size: int = 32
    epochs: int = 30
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    step_size: int = 7
    gamma: float = 0.1
    dropout: float = 0.4
    pretrained: bool = True  # ImageNet weights
    num_workers: int = 4
    device: str = "auto"


class FixedLabelsImageFolder(datasets.ImageFolder):
    """ImageFolder that reuses a given class->index map, so splits missing a class stay aligned."""

    def __init__(self, root, transform=None, class_to_idx=None):
        self.forced_class_to_idx = class_to_idx
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        if self.forced_class_to_idx is None:
            return super().find_classes(directory)
        present = {d.name for d in os.scandir(directory) if d.is_dir()}
        class_to_idx = {c: i for c, i in self.forced_class_to_idx.items() if c in present}
        return list(class_to_idx), class_to_idx


def sqrt_class_weights(targets: list[int], num_classes: int) -> list[float]:
    """``sqrt(N / (K * n_c))`` per class; 1.0 for classes absent from training."""
    counts = Counter(targets)
    total = len(targets)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        weights.append(float(np.sqrt(total / (num_classes * count))) if count > 0 else 1.0)
    return weights


def _run_epochs(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, epochs, device):
    since = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        log.info("Epoch %d/%d", epoch, epochs - 1)
        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            pbar = tqdm(dataloaders[phase], desc=phase, leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix({"loss": loss.item()})

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            log.info("%s Loss: %.4f Acc: %.4f", phase, epoch_loss, epoch_acc)
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

    elapsed = time.time() - since
    log.info("Training complete in %.0fm %.0fs. Best val acc: %.4f", elapsed // 60, elapsed % 60, best_acc)
    model.load_state_dict(best_weights)
    return model, history


def _write_report(model, loader, class_names, device, cm_path, report_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            _, preds = torch.max(model(inputs.to(device)), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Pass every class explicitly so a class missing from the val split cannot misalign the labels.
    labels = list(range(len(class_names)))
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(cm_path)
    log.info("Confusion matrix saved to %s", cm_path)

    report = classification_report(
        all_labels, all_preds, labels=labels, target_names=class_names, zero_division=0
    )
    log.info("\n%s", report)
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)


def train(cfg: ClassifierTrainConfig) -> str:
    device = resolve_device(cfg.device)
    ts = timestamp()
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_path = os.path.join(cfg.output_dir, f"greek_classifier_resnet18_{cfg.run_name}_{ts}.pth")
    cm_path = os.path.join(cfg.output_dir, f"confusion_matrix_{cfg.run_name}_{ts}.png")
    report_path = os.path.join(cfg.output_dir, f"classification_report_{cfg.run_name}_{ts}.txt")

    train_dir = os.path.join(cfg.data_dir, "train")
    val_dir = os.path.join(cfg.data_dir, "val")
    reference = datasets.ImageFolder(train_dir, transform=train_transforms(cfg.img_size))
    class_to_idx, class_names = reference.class_to_idx, reference.classes
    log.info("Training on %s with %d classes", device, len(class_names))

    weights = sqrt_class_weights(reference.targets, len(class_names))
    log.info("Class weights (first 5): %s", weights[:5])

    image_datasets = {
        "train": FixedLabelsImageFolder(train_dir, train_transforms(cfg.img_size), class_to_idx=class_to_idx),
        "val": FixedLabelsImageFolder(val_dir, eval_transforms(cfg.img_size), class_to_idx=class_to_idx),
    }
    dataloaders = {
        split: DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        for split, ds in image_datasets.items()
    }
    dataset_sizes = {split: len(ds) for split, ds in image_datasets.items()}

    model = build_resnet18(len(class_names), dropout=cfg.dropout, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    model, _ = _run_epochs(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, cfg.epochs, device
    )
    torch.save(model.state_dict(), model_path)
    log.info("Model saved to %s", model_path)

    _write_report(model, dataloaders["val"], class_names, device, cm_path, report_path)
    return model_path
