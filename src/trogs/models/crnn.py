"""CRNN line recogniser with CTC output."""

from __future__ import annotations

import torch
import torch.nn as nn

BLANK = 0


class GreekCRNN(nn.Module):
    """VGG-style CNN -> 2-layer BiLSTM -> per-timestep classifier (``num_chars`` + CTC blank).

    Expects grayscale input of height 32 normalised to [-1, 1]; the CNN reduces
    the height to 1 and the width by 4.
    """

    def __init__(self, img_channels: int = 1, num_chars: int = 35, hidden_size: int = 256):
        super().__init__()
        # Layer order defines the state-dict keys; keep it stable for existing checkpoints.
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(features)
        return self.fc(rnn_out)


class Vocab:
    """Character <-> index mapping; index 0 is reserved for the CTC blank."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.char2idx[c] for c in text]

    @classmethod
    def from_entries(cls, entries: list[tuple[str, str]]) -> Vocab:
        return cls(c for _, text in entries for c in text)

    @classmethod
    def from_labels_file(cls, path: str) -> Vocab:
        return cls.from_entries(read_labels_file(path))


def read_labels_file(path: str) -> list[tuple[str, str]]:
    """Read ``<file name>\\t<text>`` rows; malformed rows are skipped."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
    return entries


def greedy_decode(outputs: torch.Tensor, idx2char: dict[int, str]) -> list[str]:
    """Best-path CTC decoding of ``[batch, time, classes]`` scores: collapse repeats, drop blanks."""
    _, best = torch.max(outputs, dim=2)
    best = best.cpu().numpy()
    decoded = []
    for sequence in best:
        chars = []
        prev = -1
        for idx in sequence:
            if idx != BLANK and idx != prev:
                chars.append(idx2char[idx])
            prev = idx
        decoded.append("".join(chars))
    return decoded


def load_crnn(path: str, num_chars: int, device: str | torch.device) -> GreekCRNN:
    model = GreekCRNN(num_chars=num_chars)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()
