"""
Training script for scratch BPE tokenizer.

This script trains the from-scratch implementation of BPE tokenizer
for educational and reference purposes.
"""


from datetime import datetime
from bpe import TigrinyaBPE


def train_scratch_tokenizer(
    corpus_path: str,
    output_path: str,
    vocab_size: int = 8192,
    special_tokens: list = None
):
    """
    Train scratch BPE tokenizer.
    
    Args:
        corpus_path: Path to training text file
        output_path: Path to save trained tokenizer
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens
    
    Returns:
        None
    """
    print("=" * 60)
    print("Training Scratch BPE Tokenizer")
    print("=" * 60)
    
    # Initialize tokenizer
    if special_tokens is None:
        special_tokens = ['[PAD]', '[UNK]']
    
    tokenizer = TigrinyaBPE(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    # Load corpus
    print(f"\nLoading corpus from: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    print(f"Corpus size: {len(corpus)} characters")

    
    # Train
    print(f"\nTraining with target vocab size: {vocab_size}")
    tokenizer.train(corpus, verbose=True)
    
    # Save
    print(f"\nSaving tokenizer to: {output_path}")
    tokenizer.save(output_path)
    
    # Test
    print("\n" + "=" * 60)
    print("Testing tokenizer")
    print("=" * 60)
    test_text = "ኣብ ገለ እዋን መዓት ዝጸሓፍ ኣብ ኣእምሮይ ይመጽእ እሞ ፡ ክጽሕፍ ኢለ ብርዕን ወረቐትን ምስ ሓዝኩ ህልም ይብለኒ ።"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    print("\nTraining complete!")


def main():
    """Main training function."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Usage
    corpus_path = "datasets/training/test.txt"  # Update with your corpus path
    output_path = f"artifacts/tokenizers/experiments/educational_bpe_{timestamp}.json"
    
    train_scratch_tokenizer(
        corpus_path=corpus_path,
        output_path=output_path,
        vocab_size=512
    )


if __name__ == "__main__":
    main()

