# Model Architecture

A GPT-2-style decoder-only transformer implementation in PyTorch.

This release provides the architecture. Models start with random weights;
trained weights are not included.

## Overview

```
GPT
├── Token Embedding
├── Positional Embedding  
├── Dropout
├── TransformerBlock (× n_layers)
│   ├── LayerNorm → MultiHeadAttention → Dropout → Residual
│   └── LayerNorm → FeedForward → Dropout → Residual
├── Final LayerNorm
└── Output Head (Linear)
```

## Components

| Module | Description |
|--------|-------------|
| `GPT` | Main model. Combines embeddings, transformer blocks, and output projection. |
| `TransformerBlock` | Pre-norm transformer block with attention + FFN and residual connections. |
| `MultiHeadAttention` | Causal self-attention with configurable heads. |
| `FeedForward` | Two-layer MLP with GELU activation (4× expansion). |
| `LayerNorm` | Layer normalization with learnable scale and shift. |
| `GELU` | Gaussian Error Linear Unit activation function. |

## Design Choices

- Learned token and absolute positional embeddings are added before the transformer blocks.
- Each block uses pre-norm residual connections and a 4× feed-forward network with tanh-approximate GELU.
- LayerNorm uses an epsilon of `1e-5`.
- The input embedding and output head have separate weights (untied), unlike GPT-2.
- Q/K/V projection biases are configurable through `qkv_bias`; the example below disables them, unlike GPT-2.
- The output head is bias-free and returns raw logits for every input position.

## Configuration

The model is configured via a dictionary. These values illustrate how to instantiate
the architecture; they are not a finalized training configuration. Model sizing
will be covered with the training experiments.

```python
cfg = {                       
    "vocab_size": 8192,      # Vocabulary size
    "context_length": 128,   # Maximum sequence length
    "emb_dim": 256,           # Embedding dimension
    "n_heads": 4,            # Number of attention heads
    "n_layers": 2,           # Number of transformer blocks
    "drop_rate": 0.1,         # Dropout probability
    "qkv_bias": False,        # Bias in attention projections
}
```

## Usage

```python
import torch
from model import GPT

batch_size = 2
seq_len = cfg["context_length"]
vocab_size = cfg["vocab_size"]

# Create random input_ids
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

model = GPT(cfg)
logits = model(input_ids)  # (batch, seq_len, vocab_size)
```


