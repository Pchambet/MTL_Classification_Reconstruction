"""Paths, experiment grid and the shared visual identity."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
SITE = ROOT / "site"

# In-distribution classes (EuroSAT RGB, Sentinel-2) and their binary label.
CLASSES = ("Forest", "Residential")
# Out-of-distribution groups (AID, Google Earth aerial imagery) and their binary label.
OOD_GROUPS = {"Forest": 0, "DenseResidential": 1, "MediumResidential": 1}

IMAGE_SIZE = 64
VAL_FRACTION = 0.2
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
SEEDS = tuple(range(10))

# Loss weight on classification; (1 - alpha) goes to reconstruction.
# alpha = 1.0 trains the multi-task architecture with the decoder switched off:
# it isolates the effect of the auxiliary loss from the change of architecture.
ALPHAS = (1.0, 0.8, 0.6, 0.4, 0.2)

INK = "#0f172a"
TEAL = "#0d9488"
AMBER = "#d97706"
SLATE = "#64748b"
GRID = "#e2e8f0"
