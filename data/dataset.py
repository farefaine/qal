import torch 
from torch.utils.data import Dataset


def encode_in_chunks(text, tokenizer, chunk_lines=10_000):
    """
    Tokenize a large text in batches of lines instead of one giant string.

    Encoding the whole corpus in a single call is slow and memory-hungry.
    Batches are processed one after another; within a batch, `encode_batch`
    encodes the lines in parallel across CPU cores. Each batch is turned into a
    tensor right away so no huge Python list of ints is built.

    Returns:
        torch.Tensor: 1-D tensor of all token ids.
    """
    lines = text.splitlines(keepends=True)
    chunks = []
    for start in range(0, len(lines), chunk_lines):
        encodings = tokenizer.encode_batch(lines[start: start + chunk_lines])
        ids = [token_id for enc in encodings for token_id in enc.ids]
        chunks.append(torch.tensor(ids, dtype=torch.long))
    return torch.cat(chunks)


class TigrinyaDataset(Dataset):
    """
    PyTorch Dataset for pretraining language models on Tigrinya text using a sliding window.

    Creates input-target token id pairs using a given tokenizer.
    Each sample consists of two sequences:
      - input_ids:  A sequence of tokens of length `max_length`
      - target_ids: The input shifted left by one token, so target[i] is the token
                    that follows input[i] (next-token prediction)

    The dataset slides a window of size `max_length + 1` over the tokenized text with the
    specified `stride` (sequence overlap). Each window produces one input-target example.

    The corpus has one sentence per line and the tokenizer has no end-of-text token.
    Newlines are dropped, but each sentence ends with the Tigrinya full stop (`▁።`),
    which marks the sentence boundary in the token stream. Windows may start mid-sentence
    and run across boundaries, as in standard GPT-style pretraining.

    Args:
        text (str): The raw Tigrinya text corpus as a single string.
        tokenizer (PreTrainedTokenizerFast or Tokenizer): Tokenizer for the text.
        max_length (int): Length of input sequence for each sample.
        stride (int): Step size of each sliding window (controls overlap).

    Example:
        >>> dataset = TigrinyaDataset(text, tokenizer, max_length=128, stride=32)
        >>> input_ids, target_ids = dataset[0]
    """

    def __init__(self, text, tokenizer, max_length, stride):
        # Keep a single token tensor; windows are sliced from it on demand
        # instead of materializing two tensors per sample.
        self.token_ids = encode_in_chunks(text, tokenizer)
        if len(self.token_ids) <= max_length:
            raise ValueError(
                f"Text has {len(self.token_ids)} tokens; need more than "
                f"max_length={max_length} to build at least one sample."
            )
        self.max_length = max_length
        self.stride = stride
        self.num_samples = len(range(0, len(self.token_ids) - max_length, stride))

    def __len__(self):
        """Returns the total number of input-target pairs."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_ids, target_ids) for a given index.
        """
        if not 0 <= idx < self.num_samples:
            raise IndexError(idx)
        i = idx * self.stride
        return (
            self.token_ids[i: i + self.max_length],
            self.token_ids[i + 1: i + self.max_length + 1],
        )
