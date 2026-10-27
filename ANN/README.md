# ANN — Binary Classification (`make_moons`)

| Field | Detail |
|-------|--------|
| **Task** | Binary classification |
| **Model** | Fully connected ANN (Keras / TensorFlow) |
| **Dataset** | scikit-learn `make_moons` (synthetic) |
| **Primary metric** | Held-out **test accuracy** |
| **Reported result** | **92.50%** test accuracy (train–val gap −1.72 pp) |
| **Artifact** | [`ann_make_moons.ipynb`](ann_make_moons.ipynb) (executed cells + plots) |

> **Result snapshot:** test **92.50%** · best val **96.88%** · train–val gap **−1.72 pp**

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Dataset & attribution](#2-dataset--attribution)
3. [Data protocol](#3-data-protocol)
4. [Model architecture](#4-model-architecture)
5. [Training protocol](#5-training-protocol)
6. [Evaluation metrics](#6-evaluation-metrics)
7. [Results](#7-results-from-executed-notebook)
8. [Reproducibility](#8-reproducibility)
9. [Highlights](#9-highlights)

---

## 1. Problem statement

Learn a nonlinear decision boundary that separates two interleaving half-moon classes in \(\mathbb{R}^2\). The data are noisy by construction (`noise=0.25`), so the model must generalize rather than memorize point locations.

**Goals**

1. Strong held-out **test** accuracy.
2. Small train–validation accuracy gap (no clear overfitting).
3. Transparent train / validation / test protocol with a fixed seed.

---

## 2. Dataset & attribution

| Item | Value |
|------|-------|
| **Name** | `make_moons` |
| **Provider** | [scikit-learn](https://scikit-learn.org/) |
| **API** | [`sklearn.datasets.make_moons`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) |
| **License** | BSD (scikit-learn) |
| **Type** | Synthetic 2D points (not a third-party image/text corpus) |
| **Parameters** | `n_samples=2000`, `noise=0.25`, `random_state=42` |
| **Features** | 2 continuous coordinates |
| **Labels** | `{0, 1}` |

**Citation (software):** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.

---

## 3. Data protocol

| Split | Ratio | Count | Role |
|-------|-------|-------|------|
| Train | 64% | 1,280 | Parameter updates |
| Validation | 16% | 320 | Early stopping, LR schedule, gap check |
| Test | 20% | 400 | **Final reported accuracy only** |

- Stratified splits; `random_state=42`.
- Test labels are **not** used for model selection or early stopping.

---

## 4. Model architecture

```text
Input(2)
  → Dense(48, ReLU, L2) → Dropout(0.25)
  → Dense(24, ReLU, L2) → Dropout(0.25)
  → Dense(1, sigmoid)
```

| Component | Specification |
|-----------|----------------|
| Hidden layers | **2** |
| Widths | 48 → 24 |
| Activations | ReLU (hidden), sigmoid (output) |
| Regularization | L2 on dense kernels + dropout 0.25 |
| Loss | Binary cross-entropy |
| Optimizer | Adam (`lr=1e-3`) |
| Framework | TensorFlow / Keras `Sequential` |

### Design rationale

| Decision | Why |
|----------|-----|
| **2 hidden layers** | Enough capacity for a smooth 2D nonlinear boundary; deeper nets add little on this toy geometry and raise overfit risk. |
| **48 → 24** | Mild bottleneck encourages compact features without under-fitting the moons. |
| **L2 + dropout** | Noise=`0.25` invites memorization; regularization + early stopping keep train/val aligned. |
| **Sigmoid + BCE** | Standard probabilistic binary classification setup. |

---

## 5. Training protocol

| Setting | Value |
|---------|-------|
| Max epochs | 200 |
| Batch size | 32 |
| Callbacks | `EarlyStopping(monitor=val_loss, restore_best_weights=True)`, `ReduceLROnPlateau` |
| Seed | 42 (NumPy, Python `random`, TensorFlow) |

---

## 6. Evaluation metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **Accuracy** | \((TP+TN)/N\) | Primary score (train / val / test) |
| **Binary cross-entropy** | \(-\mathbb{E}[y\log\hat{p}+(1-y)\log(1-\hat{p})]\) | Optimization objective |
| **Precision / recall / F1** | Per-class from sklearn `classification_report` | Secondary test diagnostics |
| **Train–val gap** | \(\mathrm{acc}_{train}-\mathrm{acc}_{val}\) (percentage points) | Overfitting monitor |

**Headline number:** test accuracy after restoring best validation-loss weights.

---

## 7. Results (from executed notebook)

| Metric | Value |
|--------|-------|
| Best train accuracy (any epoch) | 95.16% |
| Best validation accuracy | 96.88% |
| **Held-out test accuracy** | **92.50%** |
| Train–val gap (at best comparison) | **−1.72 pp** |

Per-class test report (notebook): precision / recall / F1 ≈ 0.92–0.94 for both classes (support 200 / 200).

---

## 8. Reproducibility

```powershell
cd Machine-Learning-Tasks
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab ANN/ann_make_moons.ipynb
```

- Select a kernel with TensorFlow + scikit-learn installed.
- **Run All** — logs, curves, and the classification report are saved in the committed notebook.
- **Colab:** upload the notebook → first `%pip` cell → Run all.

---

## 9. Highlights

- From-scratch feed-forward ANN (not a classical SVM / RF-only baseline)
- Documented data source and generation parameters
- Explicit train / val / test protocol with fixed seed
- L2, dropout, and early stopping for generalization
- Overfitting monitored via train–val accuracy gap
