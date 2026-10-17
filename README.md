# Qal

Tigrinya language model pretraining: data pipeline, tokenization, training, and inference.

## Phases

| Phase | Module | Status |
|---|---|---|
| 1 | `data/` | ✅ Complete |
| 2 | `tokenization/` | ✅ Complete |
| 3 | `model/` | Architecture implemented and published |
| 4 | `training/` | Coming soon |
| 5 | `inference/` | Coming soon |

## Quickstart

**1. Clone and install dependencies**

```bash
git clone https://github.com/farefaine/qal.git
cd qal
pip install -r requirements.txt
```

**2. Download raw data**

```bash
hf download farefaine/tigrinya-pretraining --repo-type=dataset --local-dir datasets/raw/
```

Raw data is hosted on Hugging Face: [farefaine/tigrinya-pretraining](https://huggingface.co/datasets/farefaine/tigrinya-pretraining)

**3. Build the training dataset**

```bash
python -m data.build_dataset
```

Output is written to `datasets/training/`: ~2.4M training sentences, ~25K validation, ~11K test.

## Repository Structure

```
qal/
├── data/           # Data pipeline (Phase 1)
├── datasets/       # Raw and processed data (gitignored, see datasets/README.md)
├── artifacts/      # Generated outputs (gitignored except README docs)
├── tokenization/   # Tokenizer training and analysis
├── model/          # Model architecture
├── training/       # Training loop and configuration (coming soon)
├── inference/      # Inference and serving (coming soon)
├── notebooks/      # Exploration and analysis
└── config/         # Configuration files
```

## Data

See [data/README.md](data/README.md) for full pipeline documentation.

## Tokenization

See [tokenization/README.md](tokenization/README.md) for tokenizer training documentation.

## Model Architecture

Qal uses a GPT-2-style decoder-only transformer implemented in PyTorch.
The architecture is published; training code and trained weights are not included in this release.

See [model/README.md](model/README.md) for the components, an example configuration, and usage.

## Learn More

- Project motivation: [Building Qal: A Tiny Language Model for Tigrinya](https://farefaine.substack.com/p/17-building-qal-a-tiny-language-model)
- Data collection and cleaning: [Building Qal: Collecting and Cleaning Tigrinya Text](https://farefaine.substack.com/p/27-building-qal-collecting-and-cleaning)

## Connect

- Website: [fanus.dev](https://fanus.dev)
- Substack: [farefaine.substack.com](https://farefaine.substack.com)
- LinkedIn: [linkedin.com/in/fanus-arefaine](https://www.linkedin.com/in/fanus-arefaine/)
