# CNN — Cats vs Dogs

From-scratch convolutional network (Keras / TensorFlow) for binary image classification. **No transfer learning** (no MobileNet / ResNet / EfficientNet backbones).

## Contents

| File | Description |
|------|-------------|
| [`cnn_cats_dogs.ipynb`](cnn_cats_dogs.ipynb) | Full notebook with training logs, evaluation metrics, and plots (outputs saved) |

## Data attribution

- **Dataset:** [microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs) on Hugging Face
- **Original task / collection:** Asirra / [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) style pet images (cats vs dogs)
- **Size:** 23,410 RGB images
- **License / terms:** follow the dataset card on Hugging Face and the original competition rules when redistributing raw images
- Images are **downloaded at runtime** by the notebook (not vendored in this repo). First download is ~700 MB.

## Architecture

Three convolutional **stages**, each with **two** `Conv2D` layers:

| Stage | Filters | Block |
|-------|---------|--------|
| 1 | 64 | Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout |
| 2 | 128 | same pattern |
| 3 | 256 | same pattern |
| Head | — | GlobalAveragePooling → `Dense(256)` → Dropout → `Dense(1, sigmoid)` |

**Conv layers:** 6 × `Conv2D` (2 per stage × 3 stages).  
**Why this design**

- A single tiny 32→64→128 net plateaued ~65% val accuracy; doubling convs per stage and widening to 64→128→256 adds enough capacity for cats vs dogs at 128×128.
- Global average pooling keeps the classifier small vs flattening large feature maps.
- Dropout + light L2 + early stopping control overfitting; mixed float16 + batch 64 keep a Colab T4 run in ~10–15 minutes after a one-time image decode cache.

## Metrics

| Metric | Role |
|--------|------|
| **Binary cross-entropy** | Training / validation loss |
| **Accuracy** | Primary score |
| Train–val accuracy gap | Overfitting check |
| EarlyStopping on `val_loss` | Restores best weights |

**Goal:** ≥ 85% **validation** accuracy with a small train–val gap.

### Result (saved notebook output)

| Metric | Value |
|--------|-------|
| **Validation accuracy (restored best weights)** | **86.84%** |
| Best epoch (by val loss) | 10 |
| Train accuracy at that epoch | 86.10% |
| Val accuracy at that epoch | 86.84% |
| Train–val gap | −0.74 pp |

Split: 80% train / 20% validation (stratified on labels).

## How to run

```powershell
cd Machine-Learning-Tasks
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install datasets
jupyter lab CNN/cnn_cats_dogs.ipynb
```

**Colab (GPU recommended):** upload the notebook → run install cell → Run all. Expect a few minutes to decode images once, then ~30–50 s/epoch on a T4.
