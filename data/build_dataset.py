"""
Build Tigrinya training dataset from raw sources.

This script consolidates, cleans, deduplicates, and splits raw text data
into train/validation/test sets for Tigrinya language model pretraining.

Usage:
    # Default arguments
    python -m data.build_dataset

    # Or with custom arguments
    python -m data.build_dataset --raw-dir INPUT_DIR --output-dir OUTPUT_DIR --val-test-split VAL_TEST_SPLIT

"""

import argparse
from pathlib import Path
from typing import Dict

from data.preprocessing import deduplicate_sentences_exact
from data.loaders import load_all_raw_data


DEFAULT_RAW_DIR = "datasets/raw"
DEFAULT_OUTPUT_DIR = "datasets/training"


def prepare_training_data(
    raw_dir: str = DEFAULT_RAW_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    val_test_split: float = 0.7,
) -> Dict[str, int]:
    """
    Main data preparation pipeline.

    Args:
        raw_dir: Directory containing raw data files
        output_dir: Directory to save processed train/validation/test files
        val_test_split: Fraction of validation data to use for validation (rest for test)

    Returns:
        Dictionary with counts for each split
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tigrinya data preparation pipeline")
    print("=" * 60)
    print(f"Raw data directory: {raw_dir}")
    print(f"Output directory: {output_dir}")

    training_data, validation_data = load_all_raw_data(raw_dir)

    print("\nDeduplicating training data...")
    dedup_train, _, train_unique, train_dups = deduplicate_sentences_exact(training_data)
    print(f"  Total: {train_unique + train_dups:,} → Unique: {train_unique:,} (removed {train_dups:,} duplicates)")

    print("\nDeduplicating validation data...")
    dedup_val, _, val_unique, val_dups = deduplicate_sentences_exact(validation_data)
    print(f"  Total: {val_unique + val_dups:,} → Unique: {val_unique:,} (removed {val_dups:,} duplicates)")

    split_idx = int(len(dedup_val) * val_test_split)
    final_validation = dedup_val[:split_idx]
    final_test = dedup_val[split_idx:]

    print(f"\nSplit validation → validation ({len(final_validation):,}) + test ({len(final_test):,})")

    print("\nSaving processed files...")

    train_path = output_dir / "train.txt"
    val_path = output_dir / "validation.txt"
    test_path = output_dir / "test.txt"

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dedup_train))
    print(f"  {train_path}: {len(dedup_train):,} sentences")

    with open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_validation))
    print(f"  {val_path}: {len(final_validation):,} sentences")

    with open(test_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_test))
    print(f"  {test_path}: {len(final_test):,} sentences")

    print("\n" + "=" * 60)
    print("Data preparation complete")
    print("=" * 60)

    stats = {
        'train': len(dedup_train),
        'validation': len(final_validation),
        'test': len(final_test),
        'total': len(dedup_train) + len(final_validation) + len(final_test),
    }

    for key, count in stats.items():
        print(f"  {key.capitalize()}: {count:,} sentences")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build Tigrinya training dataset from raw sources"
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing raw data (default: {DEFAULT_RAW_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for processed files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--val-test-split",
        type=float,
        default=0.7,
        help="Fraction of validation data for validation vs test (default: 0.7)"
    )

    args = parser.parse_args()

    prepare_training_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        val_test_split=args.val_test_split,
    )


if __name__ == "__main__":
    main()
