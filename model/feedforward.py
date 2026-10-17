from torch import nn 
from .activation import GELU

class FeedForward(nn.Module):

    """
    FeedForward layer for the Transformer model:
    ----------------------------
    This layer is used to project the input embeddings into a higher-dimensional space,
    apply a non-linear activation function, and then project back into the original dimension.
    """

    def __init__(self, cfg):

        super().__init__()
        self.layer1 = nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim'])
        self.gelu = GELU()
        self.layer2 = nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        return self.layer2(x)
