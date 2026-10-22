# Machine-Learning-Tasks

Collection of machine learning exercises and notebooks.

## Task 1: ANN — Binary classification on `make_moons`

Artificial neural network for binary classification using scikit-learn’s synthetic **make_moons** dataset and a Keras feed-forward model.

| Item | Detail |
|------|--------|
| Notebook | [`notebooks/ann_make_moons.ipynb`](notebooks/ann_make_moons.ipynb) |
| Goal | ≥ 85% held-out **test** accuracy with controlled overfitting |
| Split | 64% train / 16% validation / 20% test (stratified) |
| Regularization | L2, dropout, early stopping on validation loss |

### Setup

```powershell
cd Machine-Learning-Tasks
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name=ml-venv --display-name="Python (ML .venv)"
jupyter lab notebooks/ann_make_moons.ipynb
```

In Jupyter, select kernel **Python (ML .venv)** and run all cells.

### Google Colab

Upload `notebooks/ann_make_moons.ipynb`, run the first cell to install dependencies, then run all cells.

### Dependencies

See [`requirements.txt`](requirements.txt): TensorFlow, scikit-learn, matplotlib, seaborn, Jupyter.

## Repository layout

```
Machine-Learning-Tasks/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── ann_make_moons.ipynb
```

## Author

[ramasamy-24-t](https://github.com/ramasamy-24-t)
