# Transformer — Next-Token Language Model (from scratch)

Character-level causal Transformer built with **explicit** scaled-dot-product attention (no `nn.Transformer`, no Hugging Face Transformers model classes for the network itself).

## Contents

| File | Description |
|------|-------------|
| [`transformer_next_token.ipynb`](transformer_next_token.ipynb) | Step-by-step model, WikiText-2 training, and self-contained generation test (outputs saved) |
| [`next_token_transformer.pt`](next_token_transformer.pt) | Trained checkpoint (~20 MB) used by the final test cell |

## Data attribution

- **Dataset:** [WikiText-2](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) (raw)
- **Hugging Face mirror used:** [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), config `wikitext-2-raw-v1`
- **Original paper / release:** Merity et al., *Pointer Sentinel Mixture Models* (WikiText)
- Text is downloaded at runtime for training; the **checkpoint** in this folder was trained on that corpus and is included so generation works offline.

## Architecture

| Hyperparameter | Value |
|----------------|-------|
| Tokenization | Character-level (`vocab_size` ≈ 1013 on WikiText-2 raw) |
| `d_model` | 256 |
| Attention heads | 8 |
| **Transformer blocks** | **6 layers** |
| FFN width `d_ff` | 1024 |
| Context `seq_len` | 128 |
| Position encoding | Fixed sinusoidal |
| Mask | Causal (next-token) |

Each block: Multi-Head Self-Attention → residual + LayerNorm → position-wise FFN → residual + LayerNorm.

### Why 6 × 256

- Small enough to train on a single Colab GPU in a bounded session, large enough for readable WikiText-style character prediction (vs a tiny Shakespeare-scale net that collapses to gibberish on WikiText).
- Character LM + CE loss makes progress easy to monitor without a separate tokenizer dependency.

## Metrics

| Metric | Role |
|--------|------|
| **Cross-entropy loss** (train / val) | Primary training objective |
| **Validation CE** | Checkpoint quality / early comparison |
| Implied perplexity | \(e^{\text{val CE}}\) (character-level) |
| Qualitative generation | Sanity check on held prompts |

### Result (saved checkpoint + notebook)

| Metric | Value |
|--------|-------|
| Train steps | 9,879 |
| Train epochs (wall) | 17 |
| Train time | ~30 min |
| **Final validation CE** | **1.354** |
| Approx. char perplexity | \(e^{1.354} \approx 3.87\) |

The last notebook cell loads `next_token_transformer.pt` and generates sample text (outputs included).

## How to run

```powershell
cd Machine-Learning-Tasks
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch datasets
jupyter lab Transformer/transformer_next_token.ipynb
```

**Inference only:** open the notebook and run the final self-contained test cell (needs `next_token_transformer.pt` beside the notebook).

**Colab:** upload the notebook + `.pt` (or train again), enable GPU, run cells.
