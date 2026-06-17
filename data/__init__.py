"""
Data processing utilities for Tigrinya language model training.

This package provides:
- Text preprocessing and filtering (preprocessing.py)
- Raw source loaders (loaders.py)
- Training data preparation pipeline (build_dataset.py)
"""

# Preprocessing functions
from data.preprocessing import (
    apply_cleaning_rules,
    has_foreign_scripts,
    normalize_tigrinya_text,
    sanitize_tigrinya_text,
    preprocess_text,
    deduplicate_sentences_exact,
)

# Data preparation pipeline
from data.build_dataset import prepare_training_data

__all__ = [
    # Preprocessing
    'apply_cleaning_rules',
    'has_foreign_scripts',
    'normalize_tigrinya_text',
    'sanitize_tigrinya_text',
    'preprocess_text',
    'deduplicate_sentences_exact',
    # Data preparation
    'prepare_training_data',
]
