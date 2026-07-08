# Tokenization

BPE tokenizer training for Tigrinya language models.

## Structure

```
tokenization/
├── hf_bpe/
│   ├── train.py          # Production trainer (HuggingFace tokenizers)
│   └── metrics.py        # Tokenizer analysis utilities
└── from_scratch/
    ├── bpe.py            # From-scratch BPE implementation
    └── train.py          # Training script for from-scratch BPE
```

---

## Production Tokenizer

Uses [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) —> fast, battle-tested, and compatible with the broader ecosystem.

### Usage

```bash
# Train with preset config
python tokenization/hf_bpe/train.py --config medium

# Train with custom config
python tokenization/hf_bpe/train.py --config-path config/tokenizer/custom.yaml
```

### Config Options

| Config   | Vocab Size | Description           |
|----------|------------|----------------------|
| `small`  | 8K         | Compact vocabulary   |
| `medium` | 16K        | Moderate vocabulary  |
| `large`  | 32K        | Larger vocabulary    |

### Pipeline

```
Text → Normalize → Punctuation Split → Metaspace → BPE + Special Tokens
```

- **Normalization**: Applies NFC, Tigrinya character and punctuation canonicalization, newline removal for inline autocomplete, whitespace cleanup, and stripping.
- **Punctuation Split**: Keeps punctuation such as `፡`, `።`, and `፧` separate from word pieces.
- **Metaspace**: Preserves word boundaries with the `▁` marker.
- **BPE**: Learns frequent subword merges from the training corpus.
- **Special Tokens**: Keeps the vocabulary small with `[UNK]` and `[PAD]`.

---

## Educational Implementation

BPE from scratch implementation for learning and reference. Not optimized for production, but useful for understanding the algorithm.

### Usage

```bash
cd tokenization/from_scratch
python train.py
```

### The Algorithm

BPE works by iteratively merging the most frequent character pairs:

```
Initial:  ["ሰ", "ላ", "ም"]
Step 1:   ["ሰ", "ላም"]      # merged "ላ" + "ም"
Step 2:   ["ሰላም"]          # merged "ሰ" + "ላም"
```

### Key Classes

**`TigrinyaBPE`** in `bpe.py`:

```python
from bpe import TigrinyaBPE

# Train
tokenizer = TigrinyaBPE(vocab_size=8000)
tokenizer.train(corpus_text)

# Use
tokens = tokenizer.encode("ኣብ ገለ እዋን መዓት ዝጸሓፍ ኣብ ኣእምሮይ ይመጽእ እሞ ፡ ክጽሕፍ ኢለ ብርዕን ወረቐትን ምስ ሓዝኩ ህልም ይብለኒ ።")
text = tokenizer.decode(tokens)

# Save/Load
tokenizer.save("tokenizer.json")
tokenizer.load("tokenizer.json")
```

---

## Output

Production Hugging Face tokenizers are saved under `artifacts/tokenizers/hf_bpe/`:

```
artifacts/tokenizers/hf_bpe/
├── tigrinya_tokenizer_8k.json
├── tigrinya_tokenizer_16k.json
└── tigrinya_tokenizer_32k.json
```

From-scratch tokenizer experiments are saved under `artifacts/tokenizers/from_scratch/`.

## Why BPE for Tigrinya?

BPE gives Qal a compact subword vocabulary for Tigrinya while still handling unseen word forms.

For a longer discussion of the tokenizer design trade-offs, see the Qal technical series:
[farefaine.substack.com](https://farefaine.substack.com).

