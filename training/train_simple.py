"""
Minimal training loop for the Tigrinya GPT model.

This simple training loop is deliberately bare. It has no validation loss,
LR schedule, gradient accumulation, mixed precision or multi-GPU support.
The full trainer with all the bells and whistles is coming soon.

Usage:
    python training/train_simple.py
    python training/train_simple.py --fraction 0.01 --epochs 2
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from tqdm import tqdm

from data.dataloader import create_dataloader
from model import GPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="datasets/training/train.txt", help="Path to raw corpus text file")
    parser.add_argument("--tokenizer", default="artifacts/tokenizers/tigrinya_tokenizer_8k.json", help="Path to tokenizer.json")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the corpus to use")
    parser.add_argument("--model-config", default="config/model/tiny.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/pretrained/simple_model.pt")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_dataloader(args, cfg):
    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    # Use only part of the corpus for quick runs
    text = text[: int(len(text) * args.fraction)]

    tokenizer = Tokenizer.from_file(args.tokenizer)
    # Each sample is context_length tokens; stride == context_length means no overlap.
    # The dataset returns (input_ids, target_ids), targets being inputs shifted by one token.
    return create_dataloader(
        text,
        tokenizer,
        batch_size=args.batch_size,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=True,
    )


def train_model(model, dataloader, optimizer, device, epochs, log_every):
    model.train()  
    step = 0
    start = time.time()
    tokens_seen = 0
    for epoch in range(epochs):
        bar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}")
        for x, y in bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # clear gradients from the previous step
            logits = model(x)  # (batch, seq_len, vocab_size)

            # Flatten so every token position is one classification example
            vocab_size = logits.size(-1)
            logits_flat = logits.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
            targets_flat = y.view(-1)  # (batch * seq_len)

            loss = F.cross_entropy(logits_flat, targets_flat)
            loss.backward()  # compute gradients
            # Cap the gradient norm so one bad batch cannot blow up training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # update weights

            tokens_seen += x.numel()
            step += 1
            # loss.item() forces a GPU sync, so only read it every log_every steps
            if step % log_every == 0:
                tok_per_sec = tokens_seen / (time.time() - start)
                bar.set_postfix(loss=f"{loss.item():.4f}", tok_s=f"{tok_per_sec:,.0f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)  # reproducible init and data order
    device = get_device()

    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)["model"]

    dataloader = build_dataloader(args, cfg)
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | params {n_params / 1e6:.1f}M | "
          f"{len(dataloader)} steps/epoch | {args.epochs} epochs")

    train_model(model, dataloader, optimizer, device, args.epochs, args.log_every)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
