# MTL — Classification & Reconstruction

Train a single CNN to perform **two tasks simultaneously** — land-use classification (Forest vs Residential) and image reconstruction — to study how Multi-Task Learning improves generalization and reduces overfitting compared to single-task baselines.

## Why Multi-Task Learning?

In standard supervised learning, each task trains an independent model. MTL leverages **shared representations** across related tasks: a classification head forces the encoder to learn discriminative features, while a reconstruction head ensures the encoder preserves fine-grained spatial information. The result is a more robust encoder that generalizes better, especially on small datasets.

## Approach

| Component | Detail |
|---|---|
| **Shared encoder** | Convolutional backbone producing a latent representation |
| **Classification head** | Fully connected layers → binary output (Forest / Residential) |
| **Reconstruction head** | Transposed convolutions → pixel-level reconstruction |
| **Loss** | Weighted sum: `L = α · CrossEntropy + (1-α) · MSE` |
| **Dataset** | [EuroSAT RGB](https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset) — Forest & Residential subsets |

## Project Structure

```
├── data/               # EuroSAT Forest & Residential images
│   ├── Forest/
│   └── Residential/
├── notebooks/          # Per-member experiment notebooks
│   ├── alexi/
│   ├── houssem/
│   ├── mahouna/
│   └── pierre/
├── src/
│   └── resize.py       # Image preprocessing utilities
├── LICENSE
└── README.md
```

## Getting Started

```bash
git clone https://github.com/Pchambet/MTL_Classification_Reconstruction.git
cd MTL_Classification_Reconstruction
```

Download [EuroSAT RGB](https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset), extract the **Forest** and **Residential** folders into `data/`.

## Tech Stack

Python · PyTorch · NumPy · Matplotlib

## Team

Group project — Télécom SudParis, M2 Data Science.

## License

MIT — see [LICENSE](LICENSE).
