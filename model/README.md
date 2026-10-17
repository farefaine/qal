# Model Architecture

A GPT-style decoder-only transformer implementation in PyTorch.

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

## Configuration

The model is configured via a dictionary (values depend on model size):

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


