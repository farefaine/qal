# artifacts/pretrained/

Pretrained model weights. The weight files themselves are gitignored; they are
published on Hugging Face (tiny, small, base, large) once training is done.

## Contents

| File | Produced by | Notes |
|------|-------------|-------|
| `simple_model.pt` | `training/train_simple.py` | Tiny config, trained on a small fraction of the corpus. A smoke test of the loop, not a usable model. |

## Loading

```python
import torch, yaml
from model import GPT

cfg = yaml.safe_load(open("config/model/tiny.yaml"))["model"]
model = GPT(cfg)
model.load_state_dict(torch.load("artifacts/pretrained/simple_model.pt"))
```
