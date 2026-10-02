"""Image loading, caching and splits.

The images (EuroSAT RGB Forest/Residential and the AID out-of-distribution set) are
committed under ``data/``. Decoding 6,000 JPEGs takes a few seconds, so the decoded
uint8 arrays are cached once in ``data/interim/`` and every run reads that cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mtl_eurosat.config import CLASSES, DATA, IMAGE_SIZE, INTERIM, OOD_GROUPS

CACHE = INTERIM / "images.npz"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Images:
    """Decoded images as uint8 ``(n, H, W, 3)`` arrays with integer labels."""

    x_id: np.ndarray
    y_id: np.ndarray
    x_ood: np.ndarray
    y_ood: np.ndarray
    group_ood: np.ndarray  # AID group name of each OOD image


def read_folder(folder: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Decode every image of ``folder`` (sorted by name) to an RGB ``size`` x ``size`` array."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    out = np.empty((len(files), size, size, 3), dtype=np.uint8)
    for i, path in enumerate(files):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            if rgb.size != (size, size):
                rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
            out[i] = np.asarray(rgb)
    return out


def read_all(root: Path = DATA) -> Images:
    """Read the in-distribution classes and the OOD groups from ``root``."""
    x_id = [read_folder(root / name) for name in CLASSES]
    y_id = [np.full(len(x), label, dtype=np.int64) for label, x in enumerate(x_id)]
    x_ood = [read_folder(root / "OOD" / group) for group in OOD_GROUPS]
    y_ood = [
        np.full(len(x), OOD_GROUPS[g], dtype=np.int64)
        for g, x in zip(OOD_GROUPS, x_ood, strict=True)
    ]
    g_ood = [np.full(len(x), g) for g, x in zip(OOD_GROUPS, x_ood, strict=True)]
    return Images(
        x_id=np.concatenate(x_id),
        y_id=np.concatenate(y_id),
        x_ood=np.concatenate(x_ood),
        y_ood=np.concatenate(y_ood),
        group_ood=np.concatenate(g_ood),
    )


def build_cache(root: Path = DATA, cache: Path = CACHE) -> Images:
    images = read_all(root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **images.__dict__)
    return images


def load(root: Path = DATA, cache: Path = CACHE) -> Images:
    """Cached images; decoded from ``root`` on first use."""
    if not cache.exists():
        return build_cache(root, cache)
    with np.load(cache) as z:
        return Images(**{k: z[k] for k in z.files})


def stratified_split(
    y: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Train/validation indices with the same class balance in both parts.

    The seed drives the split as well as the initialisation, so the spread across
    seeds includes the variance due to which images land in validation.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_val = round(len(idx) * val_fraction)
        val.append(idx[:n_val])
        train.append(idx[n_val:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """uint8 ``(n, H, W, 3)`` -> float ``(n, 3, H, W)`` in [-1, 1] (the decoder ends in tanh)."""
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()
    return t / 127.5 - 1.0


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """Inverse of :func:`to_tensor`, for plotting reconstructions."""
    x = ((t.detach().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return x.permute(0, 2, 3, 1).cpu().numpy()
