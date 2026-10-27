# Transformer — Character-Level Next-Token LM (From Scratch)

| Field | Detail |
|-------|--------|
| **Task** | Causal next-token language modeling |
| **Model** | Decoder-only Transformer (hand-written attention) |
| **Dataset** | WikiText-2 raw (`Salesforce/wikitext`) |
| **Primary metric** | Validation **cross-entropy** (nats/char) |
| **Secondary** | Character perplexity \(e^{\mathrm{CE}}\); qualitative generation |
| **Reported result** | **Val CE = 1.354** (≈ **3.87** char PPL), 9,879 steps |
| **Artifacts** | [`transformer_next_token.ipynb`](transformer_next_token.ipynb), [`next_token_transformer.pt`](next_token_transformer.pt) |

> **Result snapshot:** val CE **1.354** · char PPL ≈ **3.87** · **9,879** steps · ~**30 min** wall time

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Dataset & attribution](#2-dataset--attribution)
3. [Data protocol](#3-data-protocol)
4. [Model architecture](#4-model-architecture)
5. [Training protocol](#5-training-protocol)
6. [Evaluation metrics](#6-evaluation-metrics)
7. [Results](#7-results-checkpoint--notebook)
8. [Reproducibility](#8-reproducibility)
9. [Highlights](#9-highlights)

---

## 1. Problem statement

Predict the next **character** given a left context, using a causal (autoregressive) Transformer. The network is implemented **from first principles** in PyTorch:

- Explicit scaled dot-product attention
- Multi-head attention, residuals, LayerNorm, FFN
- Sinusoidal positional encoding
- Causal mask for next-token prediction

**Out of scope:** `torch.nn.Transformer`, Hugging Face `transformers` model classes, or other high-level Transformer wrappers for the core architecture.

**Goals**

1. Working from-scratch attention stack with clear, readable modules.
2. Train on **WikiText-2** and report validation cross-entropy.
3. Ship a checkpoint + self-contained generation cell that loads it and samples text.

---

## 2. Dataset & attribution

| Item | Value |
|------|-------|
| **Name** | WikiText-2 (raw) |
| **Hub ID** | [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) |
| **Config** | `wikitext-2-raw-v1` |
| **Original release** | [WikiText long-term dependency LM dataset](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) |
| **Paper** | Merity, Xiong, Bradbury, Socher — *Pointer Sentinel Mixture Models* |
| **Tokenization here** | Character-level vocabulary built from train text (~**1013** symbols) |
| **Text in repo** | Not vendored — downloaded at train time |
| **Weights in repo** | [`next_token_transformer.pt`](next_token_transformer.pt) (~20 MB) trained on this corpus |

**Attribution:** WikiText © Salesforce; accessed via the Hugging Face `datasets` mirror. Please cite Merity et al. when referring to the benchmark.

---

## 3. Data protocol

| Split | Role |
|-------|------|
| Train | Character stream for next-token CE training |
| Validation | Held-out CE every log interval + final CE |

- Sequences of length `seq_len=128` sampled as `(x[t:t+L], x[t+1:t+L+1])`.
- Batch size 128; AMP (`GradScaler`) when CUDA is available.

---

## 4. Model architecture

| Hyperparameter | Value |
|----------------|-------|
| `vocab_size` | ~1013 (data-dependent) |
| `d_model` | **256** |
| `n_heads` | **8** (`d_head = 32`) |
| `n_layers` | **6** Transformer blocks |
| `d_ff` | **1024** |
| `seq_len` | **128** |
| Positional encoding | Fixed sinusoidal |
| Mask | Causal lower-triangular |
| Parameterization | Embedding → \(N\) blocks → linear LM head |

**Block structure** (as implemented in the notebook):

```text
x → MultiHeadSelfAttention(causal) → + residual → LayerNorm
  → FFN(d_model → d_ff → d_model)   → + residual → LayerNorm
```

### Design rationale

| Decision | Why |
|----------|-----|
| **6 × 256** | Fits a single Colab GPU session; enough capacity for WikiText-style character modeling (tiny Shakespeare-scale nets underfit WikiText into gibberish). |
| **Character tokens** | No external BPE dependency; CE is directly interpretable; vocab ~1k keeps the embedding table small. |
| **Hand-written attention** | Every Q / K / V / scale / mask step is visible and inspectable. |
| **Causal mask** | Correct inductive bias for next-token prediction. |
| **Sinusoidal PE** | Standard absolute positions without learned PE overfitting short runs. |

---

## 5. Training protocol

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (`lr=3e-4`, `weight_decay=0.01`) |
| Objective | Token-average cross-entropy |
| Logging | ~every 60 s: train CE (batch) + val CE (`eval_batches=12`) |
| Wall-time run (shipped ckpt) | ~**30 min**, **9,879** steps, **17** epochs traversed |
| Checkpoint | `model_state_dict` + `config` + `stoi` / `itos` + `final_val_loss` |

---

## 6. Evaluation metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **Cross-entropy (CE)** | Mean \(-\log p(x_{t+1}\mid x_{\le t})\) over chars | Primary train/val metric |
| **Perplexity (char)** | \(\mathrm{PPL}=e^{\mathrm{CE}}\) | Summary LM score for this tokenization |
| **Qualitative generation** | Autoregressive sample from prompts | Sanity / readability check |

**Headline number:** `final_val_loss` stored in `next_token_transformer.pt` (**1.354**).

> **Note:** Character-level PPL is **not** comparable to word-level WikiText-2 leaderboard numbers; it is the correct metric for this tokenization.

---

## 7. Results (checkpoint + notebook)

| Metric | Value |
|--------|-------|
| Train steps | 9,879 |
| Train wall time | ~30.0 min |
| Epochs traversed | 17 |
| **Final validation CE** | **1.354** |
| Approx. char perplexity | \(e^{1.354} \approx\) **3.87** |
| Checkpoint size | ~20.08 MB |
| Config | `d_model=256`, `n_layers=6`, `n_heads=8`, `seq_len=128` |

Final notebook cell loads this checkpoint and generates samples (e.g. prompts `"The game follows "` / Wiki-style headings); outputs are saved in the notebook.

---

## 8. Reproducibility

### Train + evaluate

```powershell
cd Machine-Learning-Tasks
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab Transformer/transformer_next_token.ipynb
```

GPU recommended. Training cell writes a checkpoint; the shipped file already matches the reported metrics.

### Inference only

1. Ensure `next_token_transformer.pt` sits beside the notebook (this folder).
2. Run the **final self-contained test cell** (redefines model + loads weights + `generate`).

**Colab:** upload notebook + `.pt`, enable GPU, Run all (or jump to the test cell).

---

## 9. Highlights

- From-scratch self-attention (no `nn.Transformer` / HF model body)
- WikiText-2 with full attribution
- Documented depth / width and design rationale
- Validation CE + implied character perplexity reported
- Trained weights committed (`next_token_transformer.pt`)
- Generation test with visible notebook output
