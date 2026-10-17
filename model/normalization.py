import torch 
from torch import nn 

class LayerNorm(nn.Module):

    """
    LayerNorm layer for the Transformer model:
    ----------------------------
    This layer is used to normalize the input embeddings by mean and variance.
    It is useful to stabilize the training of the model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(cfg['emb_dim']))
        self.shift = nn.Parameter(torch.zeros(cfg['emb_dim']))


    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - x_mean) / torch.sqrt(x_var + self.eps)

        return norm_x * self.scale + self.shift

