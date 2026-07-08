# tokenization/hf_bpe/train.py

import argparse
import sys
from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Replace, Sequence, Strip
from tokenizers.pre_tokenizers import Metaspace, Punctuation
from tokenizers.pre_tokenizers import Sequence as PreSequence
from tokenizers.trainers import BpeTrainer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import load_config

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for the Tigrinya language",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str, 
        choices=["small", "medium", "large"], 
        default="medium",
        help="Tokenizer configuration size"
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to a tokenizer config YAML file (overrides --config)"
    )

    return parser.parse_args()

def config_path(args):
    """Get the config path based on the command line arguments.
    
    Args:
        args: Command line arguments
    
    Returns:
        Path to the config file
    """

    if args.config_path:
        config_path = args.config_path
    else:
        config_path = f"config/tokenizer/{args.config}.yaml"

    if not Path(config_path).exists():
        print(f'Error: Config file not found: {config_path}')
        print(f'Available configs: small, medium, large')
        sys.exit(1)
    
    return config_path

## Tokenizer Config 
def setup_normalizer():
    """Setup the normalizer for the tokenizer."""
    return Sequence([
        NFC(),
        # Canonicalize Tigrinya punctuation variants.
        Replace("፣", "፡"),
        Replace("፤", "፡"),
        Replace("፥", "፡"),
        Replace("፦", "፡"),
        Replace("᎓", "፡"),

        # Canonicalize common Tigrinya character variants.
        Replace("ሠ", "ሰ"),
        Replace("ሡ", "ሱ"),
        Replace("ሢ", "ሲ"),
        Replace("ሣ", "ሳ"),
        Replace("ሤ", "ሴ"),
        Replace("ሥ", "ስ"),
        Replace("ሦ", "ሶ"),
        Replace("ሧ", "ሷ"),
        Replace("ፀ", "ጸ"),
        Replace("ፁ", "ጹ"),
        Replace("ፂ", "ጺ"),
        Replace("ፃ", "ጻ"),
        Replace("ፄ", "ጼ"),
        Replace("ፅ", "ጽ"),
        Replace("ፆ", "ጾ"),

        # Remove newlines because this tokenizer is for inline autocomplete.
        Replace("\n", " "),
        Replace(r"\s+", " "),
        Strip(),
    ])


def setup_pre_tokenizer():
    """Setup the pre-tokenizer for the tokenizer."""
    return PreSequence([
        Punctuation(),
        Metaspace(),
    ])


def setup_trainer(vocab_size: int, special_tokens: List[str], min_frequency: int = 2, show_progress: bool = True):
    """Setup the trainer for the tokenizer."""
    return BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=min_frequency, show_progress=show_progress)



## Data Config 
def batch_iterator(file_paths: List[str], batch_size: int = 50):
    """Create a batch iterator over the files in the given list.
    
    Args:
        file_paths: List of file paths to read
        batch_size: Size of the batch to yield

    Returns:
        Generator that yields batches of lines from the files
    """

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in f:
                line = line.strip()
                line = line.replace('\n', ' ')

                if line:
                    batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch


## Train Tokenizer 
def train_tokenizer(config_path):
    """Train HuggingFace BPE tokenizer.

    Args:
        config_path: Path to the config file

    Returns:
        Trained HuggingFace BPE tokenizer
    """

    # load config 
    config = load_config(config_path)
    tokenizer_config = config['tokenizer']

    print(f"Using config: {config_path}")
    print("Training tokenizer...")
    print(f"Vocab size: {tokenizer_config['vocab_size']}")
    print(f"Special tokens: {tokenizer_config['special_tokens']}")

    # Initialize 
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = setup_normalizer()
    tokenizer.pre_tokenizer = setup_pre_tokenizer()
    tokenizer.decoder = MetaspaceDecoder()

    # Train 
    trainer = setup_trainer(
        vocab_size=tokenizer_config['vocab_size'],
        min_frequency=tokenizer_config['min_frequency'],
        special_tokens=list(tokenizer_config['special_tokens'].values()),
        show_progress=True
    )

    tokenizer.train_from_iterator(
        batch_iterator(tokenizer_config['train_files'], tokenizer_config['batch_size']),
        trainer=trainer
    )

    # Save 
    print(f"Saving tokenizer to {tokenizer_config['tokenizer_path']}")
    output_path = tokenizer_config['tokenizer_path']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)

    return tokenizer


if __name__ == "__main__":

    args = parse_args()
    config_path = config_path(args)
    tokenizer = train_tokenizer(config_path)
    print("Tokenizer trained successfully!")