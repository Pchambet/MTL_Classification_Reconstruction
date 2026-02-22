# Multi-Task Learning: When One Model Does the Work of Two

**A single neural network learns to classify satellite images and reconstruct them — and in doing so, becomes better at both.**

---

*If you've ever wondered why the human brain can learn to recognize faces, drive a car, and play the piano with the same neural hardware — while a machine typically needs a separate model for each task — this project is for you. We explore **Multi-Task Learning (MTL)**: the idea that learning several related tasks at once can yield a more robust, more generalizable model than training each task in isolation. No magic, no hand-waving — just a carefully designed experiment on real satellite imagery, with results you can run yourself.*

---

## The central question

We ask a simple question:

> **Can forcing a network to solve two problems at once — classification and reconstruction — make it better at the first one?**

Not "better" in the sense of a few extra percentage points on a benchmark. Better in the sense of *understanding more*, *overfitting less*, and *generalizing* to images it has never seen. The answer, as you'll see, is nuanced — and the nuance is exactly where the learning happens.

---

## Part 1 — The intuition: why two tasks might beat one

### The problem with single-task learning

Imagine you're training a convolutional neural network to answer a single question: *Is this patch of Earth a forest or a residential area?*

The network has one job. It will learn whatever features get the training loss down fastest. Sometimes, that means learning the *right* thing: tree canopies look different from rooftops, vegetation has a distinct spectral signature. But sometimes — especially when data is scarce — it means learning shortcuts. A particular shadow pattern. A correlation with image brightness. A memorized patch that happens to appear often in one class.

On the training set, those shortcuts work perfectly. On new images, they fail. This is **overfitting**: the model has optimized for the wrong thing.

### The MTL hypothesis

What if we gave the network a second job — one that *punishes* shortcuts?

**Task 1 — Classification:** *Forest or residential?* This task wants the network to extract discriminative features: *What differentiates these two classes?*

**Task 2 — Reconstruction:** *Reconstruct the input image from the latent representation.* This task wants the network to preserve *everything*: every texture, every edge, every spatial detail. You can't reconstruct a rooftop if you've thrown away the information that makes it a rooftop.

The insight: **classification tends to discard information; reconstruction tends to preserve it.** When both objectives pull on the same shared encoder, the encoder is forced to find a representation that is simultaneously:

- **Discriminative enough** to separate forests from residential areas
- **Rich enough** to reconstruct the original image

There is no room for pure shortcuts. A representation that memorizes training examples without capturing true structure will fail the reconstruction objective. A representation that preserves every pixel but ignores semantics will fail the classification objective. The encoder must find the sweet spot: *the minimal sufficient statistic that satisfies both.* That, in theory, is a better representation.

This is the **auxiliary task** or **multi-task regularization** perspective: the reconstruction task acts as a regularizer, preventing the encoder from collapsing into an overfitting classification machine.

---

## Part 2 — The setup: what we actually built

### The data: EuroSAT RGB

We use the [EuroSAT RGB dataset](https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset): satellite imagery from the Sentinel-2 mission, cropped into 64×64 pixel patches. We restrict to two land-use classes:

| Class | Description | What the network sees |
|-------|--------------|------------------------|
| **Forest** | Wooded areas, canopies, vegetation | Greens, textures, irregular shapes |
| **Residential** | Buildings, roads, urban fabric | Grays, geometric patterns, straight edges |

Satellite imagery is a natural fit for this experiment: the distinction between forest and residential is semantically clear, but both classes contain rich spatial detail (individual trees, roof shapes, shadows). A network that overfits might latch onto artifacts; one that generalizes must learn something about *structure*.

### The architecture: one encoder, two heads

```
                    ┌─────────────────────────┐
                    │   Input image (64×64)   │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Shared CNN Encoder    │  ← The heart: one backbone for both tasks
                    │   (conv layers → latents)│
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 │                 ▼
    ┌─────────────────┐         │         ┌─────────────────┐
    │ Classification  │         │         │  Reconstruction  │
    │     Head        │         │         │      Head        │
    │  (FC → 2 logits)│         │         │ (ConvTranspose → │
    └────────┬────────┘         │         │  64×64 image)    │
             │                  │         └────────┬─────────┘
             ▼                  │                  ▼
    ┌─────────────────┐         │         ┌─────────────────┐
    │  Forest /       │         │         │  Reconstructed   │
    │  Residential    │         │         │     image        │
    └─────────────────┘         │         └─────────────────┘
                                 │
                    Loss = α·CrossEntropy + (1−α)·MSE
```

**Shared encoder:** A convolutional backbone (e.g., a few conv blocks) produces a latent vector or feature map. This representation is the *only* thing both heads see.

**Classification head:** Fully connected layers map the latent representation to a binary output (Forest vs Residential). Trained with cross-entropy loss.

**Reconstruction head:** Transposed convolutions (or a small decoder) map the latent representation back to a 64×64 RGB image. Trained with mean squared error (MSE) between output and input.

**Combined loss:** We use a weighted sum:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{class}} + (1 - \alpha) \cdot \mathcal{L}_{\text{recon}}$$

The hyperparameter \(\alpha\) controls the trade-off. \(\alpha = 1\) means classification only (single-task baseline). \(\alpha = 0.5\) gives equal weight to both. Finding the right \(\alpha\) is part of the experiment.

---

## Part 3 — What we found (and what it means)

*The following reflects the typical behavior observed in MTL experiments; your exact numbers will vary with architecture, data split, and \(\alpha\). The notebooks contain the full experiments.*

### Finding 1: MTL reduces overfitting on small data

When the training set is small (e.g., a few hundred images per class), the single-task classification model often reaches near-perfect training accuracy but drops significantly on the validation set. The MTL model, trained with reconstruction as an auxiliary task, tends to show a smaller gap between train and validation performance.

**Interpretation:** The reconstruction objective prevents the encoder from specializing in spurious training-set patterns. It acts as a structural regularizer.

### Finding 2: The value of \(\alpha\) matters

Not all task weightings are equal. \(\alpha\) too high (e.g., 0.9) and the model almost ignores reconstruction — we're close to single-task. \(\alpha\) too low (e.g., 0.3) and the model focuses on reconstruction at the expense of classification accuracy. There is usually a range (e.g., \(\alpha \in [0.5, 0.7]\)) where both tasks are well-served and generalization is best.

**Interpretation:** MTL is a balancing act. The auxiliary task helps only when it has enough influence to shape the representation.

### Finding 3: Reconstruction quality as a diagnostic

We can inspect the reconstructed images. If they are blurry, the encoder has discarded too much. If they are sharp but the classification is poor, the encoder may have preserved low-level detail at the cost of high-level semantics. A good MTL model often produces reconstructions that are *recognizably* forest or residential — suggesting the latent space has captured the right structure.

**Interpretation:** Reconstruction is not just a regularizer; it is a window into what the encoder has learned.

### Finding 4: When MTL might not help

MTL is not a silver bullet. If the tasks are too unrelated, they can *interfere*: optimizing for one hurts the other. Here, classification and reconstruction are naturally aligned — both depend on understanding spatial structure — so the synergy holds. In other settings, task-specific encoders or more sophisticated MTL methods (e.g., uncertainty weighting, GradNorm) may be needed.

**Interpretation:** MTL works when the auxiliary task provides a useful inductive bias. Choose your auxiliary task with care.

---

## Part 4 — For the professor: the theory in one paragraph

Multi-Task Learning can be framed as optimizing a shared representation \(\phi\) under a multi-objective loss. The key theoretical result (Caruana, 1997; Baxter, 2000) is that when tasks are related, the shared representation benefits from a *larger effective dataset*: the representation must satisfy multiple constraints, which reduces the hypothesis space and improves generalization. The reconstruction task, in particular, imposes a *denseness* constraint on the latent space — it must preserve enough information to recover the input — which prevents the representation from collapsing to a set of task-specific features that overfit. The optimal weighting \(\alpha\) can be interpreted as a Lagrange multiplier balancing the two objectives; in practice, it is usually tuned empirically.

---

## Part 5 — Reproduce the experiments

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy, Matplotlib
- (Optional) A GPU for faster training

### Setup

```bash
git clone https://github.com/Pchambet/MTL_Classification_Reconstruction.git
cd MTL_Classification_Reconstruction
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install torch numpy matplotlib
```

### Data

1. Download [EuroSAT RGB](https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset).
2. Extract the **Forest** and **Residential** folders into `data/`:

```
data/
├── Forest/        ← forest patch images (.jpg)
└── Residential/   ← residential patch images (.jpg)
```

### Run

Each team member has a notebook folder with their experiments. Open the notebooks in `notebooks/<name>/` and run the cells. The pipeline typically includes:

1. Data loading and preprocessing (resize, normalize)
2. Model definition (encoder + classification head + reconstruction head)
3. Training loop with combined loss
4. Evaluation (accuracy, reconstruction MSE, qualitative inspection)

```bash
jupyter notebook notebooks/pierre/
```

---

## Project structure

```
MTL_Classification_Reconstruction/
├── data/                    # EuroSAT Forest & Residential images (you download)
│   ├── Forest/
│   └── Residential/
├── notebooks/               # Per-member experiment notebooks
│   ├── alexi/
│   ├── houssem/
│   ├── mahouna/
│   └── pierre/
├── src/
│   └── resize.py            # Image preprocessing utilities
├── README.md
├── LICENSE
└── arborescence.txt         # Full file tree (if needed)
```

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch |
| Data | NumPy, image I/O |
| Visualization | Matplotlib |
| Preprocessing | Custom utilities in `src/` |

---

## Key references

- **Caruana, R.** (1997). *Multitask learning.* Machine learning, 28(1), 41-75.
- **Ruder, S.** (2017). *An overview of multi-task learning in deep neural networks.* arXiv:1706.05098.
- **Zhang, Y. & Yang, Q.** (2017). *A survey on multi-task learning.* arXiv:1707.08114.

---

## Team

Group project — Télécom SudParis, M2 Data Science.  
Alexi, Houssem, Mahouna, Pierre — 2025/2026.

---

## License

MIT — see [LICENSE](LICENSE).

---

*"The best way to understand a model is to ask it to do two things at once — and see what it chooses to remember."*
