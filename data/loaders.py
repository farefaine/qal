"""
Raw data loaders for Tigrinya pretraining sources.

Each function handles one source format (Bible translations, QA parquets, plain text)
and returns cleaned lines ready for deduplication and splitting.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from data.preprocessing import preprocess_text


def load_bible_data(raw_dir: Path) -> Dict[str, List[str]]:
    """
    Load and clean Tigrinya Bible translated text files.

    Args:
        raw_dir: Root directory containing raw dataset sources.

    Returns:
        Mapping of source filename to cleaned text lines.
    """
    text_dict = defaultdict(list)
    bible_path = raw_dir / "en-ti-bible"

    if not bible_path.exists():
        print(f"Skipping Bible data: {bible_path} not found")
        return text_dict

    for file in os.listdir(bible_path):
        if 'target' in file:
            filepath = bible_path / file
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    cleaned = preprocess_text(line.strip())
                    if cleaned:
                        text_dict[file].append(cleaned)
            print(f"  {file}: {len(text_dict[file]):,} sentences")

    return text_dict


def load_qa_contexts(raw_dir: Path, filename: str, key: str) -> List[str]:
    """
    Load and clean unique QA context paragraphs from a parquet file.

    Args:
        raw_dir: Root directory containing raw dataset sources.
        filename: Relative parquet file path under raw_dir.
        key: Display label used in progress logging.

    Returns:
        List of cleaned context strings.
    """
    filepath = raw_dir / filename

    if not filepath.exists():
        print(f"  Skipping {filename}: not found")
        return []

    df = pd.read_parquet(filepath)
    contexts = df['context'].unique()

    cleaned_contexts = []
    for context in contexts:
        # Replace newlines with spaces for context paragraphs
        context = context.replace('\n', ' ')
        cleaned = preprocess_text(context)
        if cleaned:
            cleaned_contexts.append(cleaned)

    print(f"  {key}: {len(cleaned_contexts):,} cleaned contexts")
    return cleaned_contexts


def load_text_file(filepath: Path, key: str, split_by: str = "lines") -> List[str]:
    """
    Load and clean a text file, splitting either by line or by Tigrinya sentence marker.

    Args:
        filepath: Path to the source text file.
        key: Display label used in progress logging.
        split_by: How to segment the file.
            - "lines"     — split on newlines; for pre-segmented files (e.g. tigrinya_sentences.txt)
            - "sentences" — collapse newlines, split on Tigrinya full stop ።, and re-append it;
                            for continuous prose files (e.g. train.txt, validation.txt)

    Returns:
        List of cleaned text segments.
    """
    if not filepath.exists():
        print(f"  Skipping {key}: {filepath} not found")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()

    if split_by == "sentences":
        segments = raw.replace('\n', ' ').split('።')
    else:
        segments = raw.split('\n')

    cleaned = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        result = preprocess_text(segment)
        if result:
            cleaned.append(result + '።' if split_by == "sentences" else result)

    label = "sentences"
    print(f"  {key}: {len(cleaned):,} {label}")
    return cleaned


def load_all_raw_data(raw_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Load all configured raw sources and assemble train/validation pools.

    Args:
        raw_dir: Root directory containing raw dataset sources.

    Returns:
        Tuple of (training_data, validation_data) lists.
    """
    raw_dir = Path(raw_dir)
    text_dict = defaultdict(list)

    print("\n📖 Loading Bible data...")
    bible_data = load_bible_data(raw_dir)
    for key, lines in bible_data.items():
        text_dict[key] = lines

    print("\n📚 Loading QA contexts...")
    text_dict['train.context'] = load_qa_contexts(raw_dir, "tigqa/train.parquet", "train.context")
    text_dict['dev.context'] = load_qa_contexts(raw_dir, "tigqa/dev.parquet", "dev.context")
    text_dict['test.context'] = load_qa_contexts(raw_dir, "tigqa/test.parquet", "test.context")

    print("\n📝 Loading tigrinya sentences dataset...")
    text_dict['tigrinya_sentences'] = load_text_file(
        raw_dir / "tigrinya_sentences.txt", "tigrinya_sentences"
    )

    print("\n📄 Loading train/validation text files...")
    text_dict['train.txt'] = load_text_file(raw_dir / "train.txt", "train.txt", split_by="sentences")
    text_dict['validation.txt'] = load_text_file(raw_dir / "validation.txt", "validation.txt", split_by="sentences")

    # Collect training data (everything except validation.txt)
    training_data = []
    for key, lines in text_dict.items():
        if key != 'validation.txt':
            training_data.extend(lines)

    validation_data = text_dict['validation.txt']

    return training_data, validation_data
