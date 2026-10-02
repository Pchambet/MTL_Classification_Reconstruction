"""One training loop for every variant, with checkpoint selection on validation loss.

Two deliberate departures from the course notebooks:

* the best checkpoint is deep-copied. The notebooks kept ``model.state_dict()``,
  which is a view on the live weights, so "restore the best epoch" silently
  restored the last one;
* the out-of-distribution set is scored after every epoch for diagnosis only. It
  never influences checkpoint selection, which uses the in-distribution validation
  loss exactly as the notebooks did.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mtl_eurosat import metrics
from mtl_eurosat.config import BATCH_SIZE, EPOCHS, LR
from mtl_eurosat.models import ARCHITECTURES, Output


@dataclass(frozen=True)
class Variant:
    """A model and its loss: ``alpha * CE + (1 - alpha) * MSE`` unless ``soft``.

    The soft-sharing variant keeps the notebook's own objective:
    ``alpha * CE + beta * masked-L1 + gamma * alignment`` on inputs whose pixels are
    zeroed with probability ``mask_ratio``.
    """

    name: str
    arch: str
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    mask_ratio: float = 0.0
    label: str = ""


@dataclass
class Fit:
    model: nn.Module
    best_epoch: int
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def loss_terms(
    out: Output, x: torch.Tensor, y: torch.Tensor, v: Variant, mask: torch.Tensor | None
) -> torch.Tensor:
    ce = F.cross_entropy(out.logits, y)
    if v.arch == "cnn":
        return ce
    if v.arch == "hard":
        return v.alpha * ce + (1 - v.alpha) * F.mse_loss(out.recon, x)
    assert mask is not None and out.z_cls is not None
    l1 = (out.recon - x).abs() * mask
    masked_l1 = l1.sum() / (mask.sum() * x.size(1) + 1e-6)
    align = F.mse_loss(out.z_cls, out.z_rec)
    return v.alpha * ce + v.beta * masked_l1 + v.gamma * align


def mask_pixels(
    x: torch.Tensor, ratio: float, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero each pixel (all channels) with probability ``ratio``; return input and mask."""
    # Drawn on the CPU so the masks depend on the seed only, not on the device.
    mask = (torch.rand(x.size(0), 1, *x.shape[2:], generator=gen) < ratio).float().to(x.device)
    return x * (1 - mask), mask


def _batch_loss(model: nn.Module, x, y, v: Variant, gen: torch.Generator) -> torch.Tensor:
    if v.mask_ratio > 0:
        xm, mask = mask_pixels(x, v.mask_ratio, gen)
        return loss_terms(model(xm), x, y, v, mask)
    return loss_terms(model(x), x, y, v, None)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """P(residential) and per-image reconstruction MSE (NaN without a decoder)."""
    model.eval()
    probs, mse = [], []
    for i in range(0, len(x), batch):
        xb = x[i : i + batch]
        out = model(xb)
        probs.append(out.logits.softmax(1)[:, 1].cpu())
        if out.recon is None:
            mse.append(torch.full((len(xb),), float("nan")))
        else:
            mse.append(((out.recon - xb) ** 2).mean(dim=(1, 2, 3)).cpu())
    return torch.cat(probs).numpy(), torch.cat(mse).numpy()


@torch.no_grad()
def _val_loss(model: nn.Module, x, y, v: Variant, seed: int) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(seed)  # same masks every epoch
    total = 0.0
    for i in range(0, len(x), 500):
        xb, yb = x[i : i + 500], y[i : i + 500]
        total += _batch_loss(model, xb, yb, v, gen).item() * len(xb)
    return total / len(x)


def fit(
    v: Variant,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    x_ood: torch.Tensor | None = None,
    y_ood: np.ndarray | None = None,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> Fit:
    """Train ``v`` with Adam for ``epochs`` and keep the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ARCHITECTURES[v.arch]().to(x_train.device)  # initialised on CPU: device-independent
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=gen)
        run = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size].to(x_train.device)
            opt.zero_grad()
            loss = _batch_loss(model, x_train[idx], y_train[idx], v, gen)
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
        val_loss = _val_loss(model, x_val, y_val, v, seed)
        p_val, _ = predict(model, x_val)
        row = {
            "epoch": epoch,
            "train_loss": run / len(order),
            "val_loss": val_loss,
            "val_acc": metrics.accuracy(y_val.cpu().numpy(), p_val),
        }
        if x_ood is not None and y_ood is not None:
            row["ood_acc"] = metrics.accuracy(y_ood, predict(model, x_ood)[0])
        history.append(row)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return Fit(model, best_epoch, time.perf_counter() - start, history)
