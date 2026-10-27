# Machine-Learning-Tasks

Three separate deep-learning exercises. Each task lives in its **own folder** with code (notebook + outputs) and a task `README.md` (data attribution, architecture, metrics).

| Folder | Task | Primary metric (saved run) |
|--------|------|----------------------------|
| [`ANN/`](ANN/) | Feed-forward net on `make_moons` | **Test accuracy 92.50%** |
| [`CNN/`](CNN/) | From-scratch CNN, cats vs dogs | **Val accuracy 86.84%** |
| [`Transformer/`](Transformer/) | From-scratch next-token LM (WikiText-2) | **Val CE 1.354** (+ checkpoint) |

Open each folder’s README for **source attribution**, **layer choices**, and **metric definitions**.

## Repository layout

```
Machine-Learning-Tasks/
├── README.md
├── requirements.txt
├── .gitignore
├── ANN/
│   ├── README.md
│   └── ann_make_moons.ipynb
├── CNN/
│   ├── README.md
│   └── cnn_cats_dogs.ipynb
└── Transformer/
    ├── README.md
    ├── transformer_next_token.ipynb
    └── next_token_transformer.pt
```

## Setup

```powershell
cd Machine-Learning-Tasks
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name=ml-venv --display-name="Python (ML .venv)"
```

Extra packages used by specific tasks:

- CNN: `datasets` (Hugging Face)
- Transformer: `torch`, `datasets`

```powershell
pip install datasets torch
```

Then open the notebook under `ANN/`, `CNN/`, or `Transformer/`.

## Author

[ramasamy-24-t](https://github.com/ramasamy-24-t)
