# Machine-Learning-Tasks

End-to-end deep learning deliverables: one **feed-forward ANN**, one **from-scratch CNN**, and one **from-scratch Transformer LM**. Each task is an isolated folder with an executed notebook, a task-level README (problem → method → metrics → results), and—where applicable—trained weights.

| Task | Path | Focus | Headline result |
|------|------|-------|-----------------|
| ANN | [`ANN/`](ANN/) | Binary classification, controlled overfit | **Test acc 92.50%** |
| CNN | [`CNN/`](CNN/) | No transfer learning | **Val acc 86.84%** |
| Transformer | [`Transformer/`](Transformer/) | Hand-written attention | **Val CE 1.354** (~3.87 char PPL) |

> Open each folder’s `README.md` first, then the notebook outputs. Numbers match the committed executed cells / checkpoint metadata.

---

## Contents

- [Repository layout](#repository-layout)
- [Quick comparison](#quick-comparison)
- [README structure](#readme-structure)
- [Environment](#environment)
- [Data attribution](#data-attribution-index)
- [Author](#author)

---

## Repository layout

```text
Machine-Learning-Tasks/
├── README.md                 # This index
├── requirements.txt
├── .gitignore
├── ANN/
│   ├── README.md             # Attribution, architecture, metrics, results
│   └── ann_make_moons.ipynb  # Executed (logs + plots)
├── CNN/
│   ├── README.md
│   └── cnn_cats_dogs.ipynb   # Executed (logs + plots)
└── Transformer/
    ├── README.md
    ├── transformer_next_token.ipynb   # Executed (train log + generation)
    └── next_token_transformer.pt      # Trained WikiText-2 checkpoint (~20 MB)
```

---

## Quick comparison

| Dimension | ANN | CNN | Transformer |
|-----------|-----|-----|-------------|
| Framework | TensorFlow / Keras | TensorFlow / Keras | PyTorch |
| Data | `sklearn.datasets.make_moons` | HF `microsoft/cats_vs_dogs` | HF `Salesforce/wikitext` (`wikitext-2-raw-v1`) |
| Depth | 2 hidden Dense layers | 6× Conv2D (3 stages × 2) | 6 Transformer blocks |
| Width | 48 → 24 | 64 → 128 → 256 | `d_model=256`, 8 heads, `d_ff=1024` |
| Objective | Binary CE | Binary CE | Causal CE (char) |
| Primary metric | Test accuracy | Val accuracy | Val CE / char PPL |
| Regularization | L2, dropout, early stop | BN, dropout, L2, early stop | AdamW weight decay (+ dropout as coded) |
| Result | **92.50%** test | **86.84%** val | **1.354** val CE |
| Overfit check | Gap −1.72 pp | Gap −0.74 pp | Train/val CE tracked in log |

---

## README structure

Every task README follows the same outline:

1. **Summary card** — task, model, metric, result  
2. **Problem statement** — goals  
3. **Dataset & attribution** — source, license / lineage, citation  
4. **Data protocol** — splits, preprocessing  
5. **Architecture** — layer inventory + rationale  
6. **Training protocol** — optimizer, callbacks, seed  
7. **Evaluation metrics** — definitions  
8. **Results** — numbers tied to notebook / checkpoint  
9. **Reproducibility** — run commands  
10. **Highlights** — what the deliverable demonstrates  

---

## Environment

```powershell
cd Machine-Learning-Tasks
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name=ml-venv --display-name="Python (ML .venv)"
```

| Task | Extra notes |
|------|-------------|
| ANN | CPU fine |
| CNN | GPU recommended; first run downloads ~700 MB images |
| Transformer | GPU recommended; or run final cell only with shipped `.pt` |

---

## Data attribution (index)

| Task | Source | Notes |
|------|--------|-------|
| ANN | [scikit-learn `make_moons`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) | Synthetic; BSD via scikit-learn |
| CNN | [`microsoft/cats_vs_dogs`](https://huggingface.co/datasets/microsoft/cats_vs_dogs) | Asirra / Kaggle-style; downloaded at runtime |
| Transformer | [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) `wikitext-2-raw-v1` | Merity et al.; checkpoint included |

Full citation text lives in each task README.

---

## Author

[ramasamy-24-t](https://github.com/ramasamy-24-t)
