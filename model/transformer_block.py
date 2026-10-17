from .normalization import LayerNorm 
from .feedforward import FeedForward
from .attention import MultiHeadAttention
from torch import nn 


class TransformerBlock(nn.Module):

    """
    TransformerBlock layer for the Transformer model:
    ----------------------------
    Contains MultiHeadAttention, FeedForward, LayerNorm and Dropout layers.
    It is a building block of the Transformer model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg)
        self.layer_norm2 = LayerNorm(cfg)
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x 
        x = self.layer_norm1(x)
        x = self.multi_head_attention(x)
        x = self.dropout(x)
        x = shortcut + x 

        shortcut = x 
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = shortcut + x 

        return x 
