# data/

Prepares raw Tigrinya text sources into clean, deduplicated train/validation/test splits for language model pretraining.

## Modules

| File | Responsibility |
|---|---|
| `preprocessing.py` | Text cleaning, Unicode normalization, foreign script filtering, deduplication |
| `loaders.py` | Loads each raw source format (Bible translations, QA parquets, plain text files) |
| `build_dataset.py` | Orchestrates the full pipeline: load → clean → deduplicate → split → save |
| `__init__.py` | Package exports for Phase 1 |

## Raw Data

Raw sources are available on Hugging Face: [farefaine/tigrinya-pretraining](https://huggingface.co/datasets/farefaine/tigrinya-pretraining)

Download and place under `datasets/raw/` before running:

```bash
hf download farefaine/tigrinya-pretraining --repo-type=dataset --local-dir datasets/raw/
```

## Expected Input Structure

```
datasets/raw/
├── en-ti-bible/          # Bible translations (*.target files)
├── tigqa/                # Tigrinya QA dataset
│   ├── train.parquet
│   ├── dev.parquet
│   └── test.parquet
├── tigrinya_sentences.txt  # Pre-segmented sentence corpus
├── train.txt               # Continuous prose training text
└── validation.txt          # Continuous prose validation text
```

## Running the Pipeline

```bash
# Default arguments
python -m data.build_dataset

# Custom arguments
python -m data.build_dataset --raw-dir INPUT_DIR --output-dir OUTPUT_DIR --val-test-split VAL_TEST_SPLIT
```

Run from the project root (`qal/`).

## Output

Processed files are written to `datasets/training/`:

```
datasets/training/
├── train.txt       # ~2.4M sentences
├── validation.txt  # ~25K sentences
└── test.txt        # ~11K sentences
```

The pipeline prints a full summary on completion including sentence counts and duplicates removed per split.

## Coming Soon

- `dataset.py` — PyTorch `TigrinyaDataset` and memory-mapped dataset for training
- `dataloader.py` — DataLoader factories
- `dataset_stats.py` — Statistics, vocabulary analysis, and visualizations
