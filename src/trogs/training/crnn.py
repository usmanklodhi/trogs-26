"""Train the CRNN line recogniser with CTC loss.

The line dataset is split randomly (85/15) into train and validation. The
model with the lowest validation loss is kept. After training, metrics are
written as CSV, a loss-curve plot and a short text summary for the thesis.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trogs.models.crnn import GreekCRNN, Vocab, greedy_decode, read_labels_file
from trogs.utils import resolve_device, seed_everything, setup_logging, timestamp

log = logging.getLogger(__name__)


@dataclass
class CrnnTrainConfig:
    data_dir: str = "data/processed/crnn_training_data"
    output_dir: str = "models"
    log_dir: str = "logs"
    split_ratio: float = 0.85
    batch_size: int = 32
    epochs: int = 30
    lr: float = 0.001
    num_workers: int = 4
    device: str = "auto"
    seed: int | None = None


class LineDataset(Dataset):
    """Grayscale line crops normalised to [-1, 1], with encoded text targets."""

    def __init__(self, entries: list[tuple[str, str]], img_dir: str, vocab: Vocab):
        self.entries = entries
        self.img_dir = img_dir
        self.vocab = vocab

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filename, text = self.entries[idx]
        img = cv2.imread(os.path.join(self.img_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((32, 100), dtype=np.uint8)  # missing crop: blank line instead of crashing
        img = (img / 127.5) - 1.0
        return torch.FloatTensor(img).unsqueeze(0), self.vocab.encode(text)


def collate_lines(batch):
    """Right-pad images to the widest in the batch (with -1, i.e. black) and flatten targets for CTC."""
    images, targets = zip(*batch, strict=True)
    max_width = max(img.shape[2] for img in images)
    images = torch.stack([F.pad(img, (0, max_width - img.shape[2], 0, 0), value=-1.0) for img in images])
    target_lengths = torch.IntTensor([len(t) for t in targets])
    targets_flat = torch.IntTensor([c for t in targets for c in t])
    return images, targets_flat, target_lengths, targets


def _ctc_loss(model, criterion, images, targets_flat, target_lengths):
    outputs = model(images)
    log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
    input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(
        images.device
    )
    return outputs, criterion(log_probs, targets_flat, input_lengths, target_lengths)


def evaluate(model, loader, criterion, device, idx2char) -> tuple[float, float]:
    """Return ``(mean validation loss, character error rate)``."""
    import Levenshtein

    model.eval()
    total_distance = total_chars = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, targets_flat, target_lengths, targets_raw in loader:
            outputs, loss = _ctc_loss(
                model, criterion, images.to(device), targets_flat.to(device), target_lengths.to(device)
            )
            total_loss += loss.item()
            for pred, target in zip(greedy_decode(outputs, idx2char), targets_raw, strict=True):
                true = "".join(idx2char[c] for c in target)
                total_distance += Levenshtein.distance(pred, true)
                total_chars += len(true)
    return total_loss / len(loader), total_distance / max(total_chars, 1)


def _write_thesis_assets(history: dict, best_val_loss: float, epochs: int, log_dir: str, ts: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(history)
    csv_path = os.path.join(log_dir, f"metrics_{ts}.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue", marker="o", markersize=4)
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="red", marker="x", markersize=4)
    plt.title("CRNN + CTC Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("CTC Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plot_path = os.path.join(log_dir, f"loss_curves_{ts}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    val_loss = history["val_loss"]
    mean_val_loss = sum(val_loss) / len(val_loss)
    if any(diff > mean_val_loss * 0.2 for diff in np.diff(val_loss)):
        behaviour = "experienced some volatility or spikes before settling"
    elif val_loss[-1] > best_val_loss * 1.15:
        behaviour = "decreased initially but showed signs of overfitting (spiking) towards the end"
    else:
        behaviour = "smoothly decreased and plateaued"

    with open(os.path.join(log_dir, f"thesis_summary_partB_{ts}.txt"), "w") as f:
        f.write("====================================================\n")
        f.write("Part B: Pathway 2 Results (CRNN + CTC)\n")
        f.write("====================================================\n\n")
        f.write("Final CER:\n")
        f.write(
            f"- The final Character Error Rate (CER) on the validation set was "
            f"{history['val_cer'][-1] * 100:.2f}% at epoch {epochs}.\n"
        )
        f.write(
            f"- Note: The absolute best CER achieved during training was {min(history['val_cer']) * 100:.2f}%.\n\n"
        )
        f.write("Loss Curves:\n")
        f.write(f"- Based on the saved metrics, the validation loss {behaviour}.\n")
        f.write(f"- Initial Validation Loss: {val_loss[0]:.4f}\n")
        f.write(f"- Final Validation Loss: {val_loss[-1]:.4f}\n\n")
        f.write("Files generated for your thesis:\n")
        f.write(f"1. Graph: {plot_path}\n")
        f.write(f"2. Data (Excel/CSV): {csv_path}\n")
    log.info("Thesis assets written to %s", log_dir)


def train(cfg: CrnnTrainConfig) -> str:
    ts = timestamp()
    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_logging(os.path.join(cfg.log_dir, f"training_crnn_{ts}.log"))
    seed_everything(cfg.seed)
    log.info("=== Starting CRNN training ===")

    entries = read_labels_file(os.path.join(cfg.data_dir, "labels.txt"))
    vocab = Vocab.from_entries(entries)
    img_dir = os.path.join(cfg.data_dir, "images")

    random.shuffle(entries)
    split_idx = int(len(entries) * cfg.split_ratio)
    train_loader = DataLoader(
        LineDataset(entries[:split_idx], img_dir, vocab),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_lines,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        LineDataset(entries[split_idx:], img_dir, vocab),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_lines,
        num_workers=cfg.num_workers,
    )

    device = resolve_device(cfg.device)
    model = GreekCRNN(num_chars=len(vocab)).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    best_model_path = os.path.join(cfg.output_dir, f"crnn_greek_{ts}_best.pth")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_cer": []}

    log.info("Starting training for %d epochs...", cfg.epochs)
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, targets_flat, target_lengths, _) in enumerate(train_loader):
            optimizer.zero_grad()
            _, loss = _ctc_loss(
                model, criterion, images.to(device), targets_flat.to(device), target_lengths.to(device)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                log.info(
                    "Epoch %02d/%d | Batch %03d/%d | Train Loss: %.4f",
                    epoch + 1,
                    cfg.epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                )

        train_loss = total_loss / len(train_loader)
        val_loss, val_cer = evaluate(model, val_loader, criterion, device, vocab.idx2char)
        log.info(
            "--- Epoch %d --- Train Loss: %.4f | Val Loss: %.4f | Val CER: %.2f%%",
            epoch + 1,
            train_loss,
            val_loss,
            val_cer * 100,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            log.info(">>> New best val loss, model saved to %s", best_model_path)

    log.info("=== Training complete ===")
    _write_thesis_assets(history, best_val_loss, cfg.epochs, cfg.log_dir, ts)
    return best_model_path
