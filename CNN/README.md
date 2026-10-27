# CNN — Cats vs Dogs (From-Scratch ConvNet)

| Field | Detail |
|-------|--------|
| **Task** | Binary image classification (cat vs dog) |
| **Model** | Custom CNN — **no transfer learning** |
| **Dataset** | Hugging Face `microsoft/cats_vs_dogs` (Asirra / Kaggle-style) |
| **Primary metric** | **Validation accuracy** (best `val_loss` weights) |
| **Acceptance target** | ≥ 85% val accuracy, controlled overfitting |
| **Reported result** | **86.84%** val accuracy (train–val gap −0.74 pp) |
| **Artifact** | [`cnn_cats_dogs.ipynb`](cnn_cats_dogs.ipynb) (executed cells + plots) |

---

## 1. Problem statement

Classify RGB photographs as **cat** or **dog** using a convolutional network trained **from random initialization**. Pretrained ImageNet backbones (MobileNet, ResNet, EfficientNet, etc.) are **out of scope** for this deliverable.

**Success criteria**

1. Validation accuracy **≥ 85%**.
2. Train–validation accuracy gap remains small at the selected checkpoint (no large overfit).
3. End-to-end pipeline: load public data → augment → train → evaluate → plot curves.

---

## 2. Dataset & attribution

| Item | Value |
|------|-------|
| **Name** | Cats vs Dogs |
| **Hub ID** | [`microsoft/cats_vs_dogs`](https://huggingface.co/datasets/microsoft/cats_vs_dogs) |
| **Lineage** | Asirra / [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)–style collection |
| **Samples** | 23,410 RGB images |
| **Labels** | `0` = cat, `1` = dog |
| **Distribution in repo** | **Not vendored** — downloaded at runtime (~700 MB first pull) |
| **License / terms** | Follow the [dataset card](https://huggingface.co/datasets/microsoft/cats_vs_dogs) and original competition rules for redistribution of raw images |

**Attribution:** Dataset hosted by Microsoft on the Hugging Face Hub; underlying images originate from the Asirra / Dogs vs. Cats public challenge collections. This project uses the Hub mirror for reproducible loading via `datasets.load_dataset`.

---

## 3. Data protocol

| Split | Ratio | Count | Role |
|-------|-------|-------|------|
| Train | 80% | 18,728 | SGD updates + augmentation |
| Validation | 20% | 4,682 | Early stopping, LR schedule, reported accuracy |

- Stratified by `labels`, `seed=42`.
- Images resized to **128×128**, scaled to \([0,1]\).
- **Train-only augmentation:** random horizontal flip, brightness, contrast.
- **Performance note:** images are decoded **once** into uint8 arrays, then fed with `tf.data` (avoids re-decoding every epoch).

---

## 4. Model architecture

Three stages × **two** `Conv2D` layers each (**6 convolutions** total):

```
Input(128, 128, 3)
  → [Conv64 → BN → ReLU → Conv64 → BN → ReLU → MaxPool → Dropout]
  → [Conv128 → BN → ReLU → Conv128 → BN → ReLU → MaxPool → Dropout]
  → [Conv256 → BN → ReLU → Conv256 → BN → ReLU → MaxPool → Dropout]
  → GlobalAveragePooling2D
  → Dense(256, ReLU, L2) → Dropout(0.5)
  → Dense(1, sigmoid)   # float32 head under mixed_float16
```

| Component | Specification |
|-----------|----------------|
| Conv stages | **3** (filters 64 → 128 → 256) |
| Conv layers | **6** (`3×3`, `same` padding) |
| Normalization | BatchNorm after each conv |
| Pooling | MaxPool 2×2 per stage |
| Classifier | GAP → Dense(256) → Dense(1) |
| Regularization | Spatial/channel Dropout (increasing 0.2→0.4) + Dense dropout 0.5 + L2 |
| Loss | Binary cross-entropy |
| Optimizer | Adam (`lr=1e-3`) |
| Precision | `mixed_float16` (Colab GPU) |
| Batch size | 64 |

### Design rationale

| Decision | Why |
|----------|-----|
| **From-scratch only** | Demonstrates CNN feature learning, not fine-tuning of ImageNet weights. |
| **2 convs per stage** | A shallower 32→64→128 single-conv stack plateaued ~65% val; double-conv VGG-style blocks + wider filters reach ≥85%. |
| **64→128→256** | Matches capacity to 23k mid-resolution photos without exploding parameters via flatten. |
| **Global average pooling** | Fewer parameters than flatten; regularizes the head. |
| **Dropout + L2 + early stop** | Keeps train/val accuracies aligned (gap ≈ 0 at best epoch). |
| **Decode-once + batch 64 + AMP** | Targets a **~10–15 min** Colab T4 training window after caching. |

---

## 5. Training protocol

| Setting | Value |
|---------|-------|
| Max epochs | 15 |
| EarlyStopping | `monitor=val_loss`, `patience=4`, `restore_best_weights=True` |
| ReduceLROnPlateau | `factor=0.5`, `patience=2`, `min_lr=1e-6` |
| Seed | 42 |

---

## 6. Evaluation metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **Accuracy** | Correct / total | Primary score |
| **Binary cross-entropy** | Standard BCE | Train/val objective |
| **Train–val gap** | \(\mathrm{acc}_{train}-\mathrm{acc}_{val}\) (pp) | Overfitting monitor |
| **Best checkpoint** | `argmin(val_loss)` via EarlyStopping | Weights used for final `evaluate` |

**Reported headline number:** `model.evaluate(val_ds)` after restored best weights.

---

## 7. Results (from executed notebook)

| Metric | Value |
|--------|-------|
| **Validation accuracy (restored)** | **86.84%** |
| Best epoch (by `val_loss`) | 10 |
| Train accuracy @ best epoch | 86.10% |
| Val accuracy @ best epoch | 86.84% |
| Train–val gap | **−0.74 pp** |
| Target (≥ 85% val) | **Met** |
| Overfitting control | **Met** (val ≥ train at selected epoch) |

Training curves in the notebook show mid-run val volatility (LR / batch noise); **selection by `val_loss`** recovers a stable ≥85% checkpoint.

---

## 8. Reproducibility

```powershell
cd Machine-Learning-Tasks
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab CNN/cnn_cats_dogs.ipynb
```

- **GPU strongly recommended** (Colab T4 or better).
- First run downloads ~700 MB and decodes images once (few minutes), then ~30–50 s/epoch.
- Committed notebook retains training logs, final metrics, and loss/accuracy plots.

---

## 9. Deliverable checklist (for review)

- [x] Custom CNN (no pretrained backbone)
- [x] Public dataset with attribution + load path
- [x] Documented architecture (layer count, widths, head)
- [x] Augmentation on train only
- [x] Val accuracy ≥ 85% with saved evidence
- [x] Overfitting assessed via train–val gap at best epoch
