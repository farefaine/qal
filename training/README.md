# Training

A minimal training loop for the Tigrinya GPT model: `train_simple.py`.

It is deliberately bare: AdamW, cross-entropy on next-token prediction, gradient
clipping, a progress bar. No validation loss, LR schedule, gradient accumulation,
mixed precision or multi-GPU. It exists to show the core loop end to end.

## Prerequisites

Run everything from the project root.

1. **Download the raw data** (from [farefaine/tigrinya-pretraining](https://huggingface.co/datasets/farefaine/tigrinya-pretraining)):

   ```bash
   hf download farefaine/tigrinya-pretraining --repo-type=dataset --local-dir datasets/raw/
   ```

2. **Build the train/validation/test splits** (writes `datasets/training/`):

   ```bash
   python -m data.build_dataset
   ```

   See `data/README.md` for details.

3. **Train the tokenizer** (writes `artifacts/tokenizers/tigrinya_tokenizer_8k.json`):
   see `tokenization/README.md`.

## Usage

```bash
# Quick check: 1% of the corpus, 2 epochs
python training/train_simple.py --fraction 0.01 --epochs 2

# Defaults: full corpus, 3 epochs, tiny model
python training/train_simple.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--text` | `datasets/training/train.txt` | Raw corpus text |
| `--tokenizer` | `artifacts/tokenizers/tigrinya_tokenizer_8k.json` | Tokenizer file |
| `--model-config` | `config/model/tiny.yaml` | Model hyperparameters |
| `--fraction` | `1.0` | Fraction of the corpus to use |
| `--epochs` | `3` | Passes over the data |
| `--batch-size` | `16` | Sequences per step |
| `--lr` | `3e-4` | AdamW learning rate |
| `--log-every` | `50` | Steps between loss/throughput refreshes |
| `--seed` | `42` | Random seed |
| `--out` | `artifacts/pretrained/simple_model.pt` | Where weights are saved |

## What the loop does

For each batch of `(input_ids, target_ids)`, where targets are the inputs shifted
by one token:

1. `model(x)` returns logits of shape `(batch, seq_len, vocab_size)`
2. Logits and targets are flattened so each token position is one classification example
3. `cross_entropy` gives the loss, `backward()` the gradients
4. Gradients are clipped to norm 1.0, then AdamW updates the weights

Weights are saved to `artifacts/pretrained/` (gitignored; see its README).
