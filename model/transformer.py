import torch 
from torch import nn 
from .transformer_block import TransformerBlock
from .normalization import LayerNorm


class GPT(nn.Module):

    """
    GPT model for the Transformer model:
    ----------------------------
    Transformer decoder model that is used to generate text.
    """

    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim']) 
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.out_proj = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(cfg)

    def forward(self, in_idx):
        _, seq_length = in_idx.shape

        token_emb = self.token_emb(in_idx)
        pos_indices = torch.arange(seq_length, device=in_idx.device)
        pos_emb = self.pos_emb(pos_indices)
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_proj(x)

        return logits


