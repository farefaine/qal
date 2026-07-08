# Configuration

Tokenizer YAML configs and lightweight config loading helpers.

## Structure

```
config/
└── tokenizer/
    ├── small.yaml   # 8K vocab
    ├── medium.yaml  # 16K vocab
    └── large.yaml   # 32K vocab
```

## Usage

### Loading a Tokenizer Config

```python
from config import load_config

cfg = load_config("config/tokenizer/medium.yaml")
tokenizer_cfg = cfg["tokenizer"]
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
