import torch
from data.dataset import TigrinyaDataset
from torch.utils.data import DataLoader


def create_dataloader(
    text,
    tokenizer,
    batch_size, 
    max_length, 
    stride,
    shuffle, 
    drop_last=True,
    num_workers=0):
    """
    Create a DataLoader for Tigrinya pretraining using TigrinyaDataset.

    Args:
        text (str): Raw corpus text (Tigrinya).
        tokenizer (PreTrainedTokenizerFast or Tokenizer): Hugging Face tokenizer or compatible object.
        batch_size (int): Number of samples per batch.
        max_length (int): Sequence length (input sequence token count).
        stride (int): Number of tokens to slide for the next sequence (overlap = max_length - stride).
        shuffle (bool): If True, randomly shuffle dataset each epoch.
        drop_last (bool, optional): Drop last batch if its size < batch_size. Default: True.
        num_workers (int, optional): Number of data loading worker processes. Default: 0.

    Returns:
        DataLoader: PyTorch DataLoader yielding batches of (input_ids, target_ids).
    """
    dataset = TigrinyaDataset(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    return dataloader
