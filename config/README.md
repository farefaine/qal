# Configuration

Tokenizer and model YAML configs, plus lightweight config loading helpers.

## Structure

```
config/
├── tokenizer/
│   ├── small.yaml   # 8K vocab
│   ├── medium.yaml  # 16K vocab
│   └── large.yaml   # 32K vocab
└── model/
    ├── tiny.yaml
    ├── small.yaml
    ├── base.yaml
    ├── large.yaml
    └── xlarge.yaml
```

## Model Sizes

All sizes share an 8K vocab, a 128-token context, 64 dimensions per head and a 4x FFN.

| Config | emb_dim | layers | heads | Params |
|---|---|---|---|---|
| `tiny` | 256 | 2 | 4 | 5.8M |
| `small` | 384 | 4 | 6 | 13.4M |
| `base` | 512 | 4 | 8 | 21.1M |
| `large` | 768 | 6 | 12 | 55.2M |
| `xlarge` | 768 | 8 | 12 | 69.4M |

Parameter counts include embeddings (untied input and output tables).

## Usage

### Loading a Tokenizer Config

```python
from config import load_config

cfg = load_config("config/tokenizer/small.yaml")
tokenizer_cfg = cfg["tokenizer"]
```

### Loading a Model Config

```python
import yaml
from model import GPT

cfg = yaml.safe_load(open("config/model/tiny.yaml"))["model"]
model = GPT(cfg)
```

### Saving a Tokenizer Config

```python
from config import save_config

cfg = {"tokenizer": {"vocab_size": 8192}}
save_config(cfg, "config/tokenizer/custom.yaml")
```

## Naming Convention

| Config | Vocab Size |
|---|---|
| `small` | 8K |
| `medium` | 16K |
| `large` | 32K |
