# ANN — Binary Classification on `make_moons`

Feed-forward neural network (Keras / TensorFlow) for binary classification on a synthetic 2D dataset.

## Contents

| File | Description |
|------|-------------|
| [`ann_make_moons.ipynb`](ann_make_moons.ipynb) | Full notebook with training logs, metrics, and plots (outputs saved) |

## Data attribution

- **Dataset:** [`sklearn.datasets.make_moons`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- **Source / library:** [scikit-learn](https://scikit-learn.org/) (BSD license)
- **Settings used:** `n_samples=2000`, `noise=0.25`, `random_state=42`
- This is a **synthetic** toy dataset (two interleaving half-circles). No third-party image/text corpus is used.

## Architecture

| Piece | Choice |
|-------|--------|
| Input | 2 features `(x, y)` |
| Hidden layer 1 | `Dense(48, ReLU)` + Dropout `0.25` |
| Hidden layer 2 | `Dense(24, ReLU)` + Dropout `0.25` |
| Output | `Dense(1, sigmoid)` |
| **Depth** | **2 hidden layers** (+ input/output) |

### Why this depth / width

- `make_moons` is a low-dimensional nonlinear problem; two modest hidden layers are enough to learn the curved decision boundary without a huge capacity.
- L2 weight decay, dropout, validation-based early stopping, and ReduceLROnPlateau keep the train–val gap small so the model does not overfit the noisy points.

## Metrics

| Metric | Role |
|--------|------|
| **Binary cross-entropy** | Training loss |
| **Accuracy** | Primary score (train / val / held-out test) |
| Precision / recall / F1 | Classification report on the test set |
| Train–val accuracy gap | Overfitting check (target: small gap) |

**Goal:** ≥ 85% held-out **test** accuracy without large train–val gap.

### Result (saved notebook output)

| Split | Accuracy |
|-------|----------|
| Best train (any epoch) | 95.16% |
| Best validation | 96.88% |
| **Held-out test** | **92.50%** |
| Train–val gap | −1.72 pp |

Split: 64% train / 16% validation / 20% test (stratified).

## How to run

```powershell
cd Machine-Learning-Tasks
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab ANN/ann_make_moons.ipynb
```

**Colab:** upload this notebook, run the first install cell, then Run all.
