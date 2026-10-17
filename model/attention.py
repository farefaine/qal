import torch
from torch import nn

class MultiHeadAttention(nn.Module):

    """
    MultiHeadAttention layer for the Transformer model
    ----------------------------
    This layer is used to compute the attention among input embeddings for each head.
    """

    def __init__(self, cfg):

        super().__init__()
        
        assert cfg['emb_dim'] % cfg['n_heads'] == 0, "emb_dim must be divisible by n_heads"

        self.dout = cfg['emb_dim']
        self.num_heads = cfg['n_heads']
        self.head_dim = cfg['emb_dim'] // cfg['n_heads']
        self.Wq = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias=cfg['qkv_bias'])
        self.Wk = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias=cfg['qkv_bias'])
        self.Wv = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias=cfg['qkv_bias'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.proj = nn.Linear(cfg['emb_dim'], cfg['emb_dim'])

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg['context_length'], cfg['context_length']),
            diagonal=1
            )
        )


    def forward(self, x):

        b, num_tokens, d_in = x.shape

        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)


        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1,2) ## b, num_heads, num_tokens, head_dim
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1) ## b, num_heads,num_tokens , num_tokens 

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        attn_weights = self.dropout(attn_weights) ## b, num_heads, num_tokens, num_tokens 

        context_vector = attn_weights @ values ## b, num_heads, num_tokens, head_dim

        context_vector = context_vector.transpose(1, 2) ## b, num_tokens, num_heads, head_dim

        context_vector = context_vector.contiguous().view(
            b, num_tokens, self.dout
        )

        context_vector = self.proj(context_vector)

        return context_vector ## b, num_tokens, emb_dim