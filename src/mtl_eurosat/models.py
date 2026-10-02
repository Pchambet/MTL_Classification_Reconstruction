"""The three architectures of the original course notebooks, unchanged.

Every model returns a :class:`Output`, so one training loop serves all three: models
without a decoder leave ``recon`` empty and the reconstruction term drops out.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class Output(NamedTuple):
    logits: torch.Tensor
    recon: torch.Tensor | None = None
    z_cls: torch.Tensor | None = None  # soft sharing: pooled task features, pulled together
    z_rec: torch.Tensor | None = None


def _block(c_in: int, c_out: int, pool: bool = True) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out)]
    layers.append(nn.ReLU())
    if pool:
        layers.append(nn.MaxPool2d(2))
    return layers


def _decoder(c_in: int) -> nn.Sequential:
    """16x16 feature map -> 64x64 RGB image in [-1, 1]."""
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 3, 3, padding=1),
        nn.Tanh(),
    )


class CNNClassifier(nn.Module):
    """Single-task baseline: three conv layers, global average pooling, dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> Output:
        return Output(self.fc(self.encoder(x)))


class HardShareMTL(nn.Module):
    """Hard parameter sharing: one encoder, a classification head and a decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(*_block(3, 32), *_block(32, 64))
        self.cls_head = nn.Sequential(
            *_block(64, 128, pool=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )
        self.dec_head = _decoder(64)

    def forward(self, x: torch.Tensor) -> Output:
        z = self.shared(x)
        return Output(self.cls_head(z), self.dec_head(z))


class SoftShareMTL(nn.Module):
    """Soft sharing: one shared block, then a branch per task.

    The two branches are tied by an alignment loss between pooled projections of
    their features, and the reconstruction branch is trained as a masked autoencoder.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared1 = nn.Sequential(*_block(3, 32))
        self.enc_cls2 = nn.Sequential(*_block(32, 64))
        self.enc_rec2 = nn.Sequential(*_block(32, 64))
        self.cls_top = nn.Sequential(*_block(64, 128, pool=False))
        self.rec_top = nn.Sequential(*_block(64, 128, pool=False))
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, 2)
        )
        self.dec_head = _decoder(128)
        self.proj_cls = _projection()
        self.proj_rec = _projection()

    def forward(self, x: torch.Tensor) -> Output:
        s = self.shared1(x)
        f_cls, f_rec = self.enc_cls2(s), self.enc_rec2(s)
        return Output(
            logits=self.cls_head(self.cls_top(f_cls)),
            recon=self.dec_head(self.rec_top(f_rec)),
            z_cls=self.proj_cls(f_cls),
            z_rec=self.proj_rec(f_rec),
        )


def _projection() -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "cnn": CNNClassifier,
    "hard": HardShareMTL,
    "soft": SoftShareMTL,
}


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
