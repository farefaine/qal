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
Text → NFC Normalize → Strip → Punctuation Split → Metaspace → BPE
```

- **NFC normalization**: Canonical Unicode form for Ge'ez script
- **Metaspace**: Preserves word boundaries with `▁` prefix
- **BPE**: Byte Pair Encoding for subword tokenization

---

## Educational Implementation

A from-scratch BPE implementation for learning and reference. Not optimized for production, but useful for understanding the algorithm.

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

Trained tokenizers are saved to `artifacts/tokenizers/`:

```
artifacts/tokenizers/
├── tigrinya_tokenizer_8k.json
├── tigrinya_tokenizer_16k.json
└── tigrinya_tokenizer_32k.json
```

## Why BPE for Tigrinya?

Tigrinya uses Ge'ez script with a large syllabary (~300 base characters). BPE handles this well because:

1. **Morphologically rich**: Words have many inflected forms, subword units capture shared roots
2. **Limited corpus size**: BPE generalizes better than word-level tokenization with small datasets
3. **Script structure**: Ge'ez syllables map naturally to subword units

