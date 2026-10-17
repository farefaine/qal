import torch
from torch import nn

class GELU(nn.Module):

    """
    GELU activation function:
    ----------------------------
    This is a non-linear activation function that is used to introduce non-linearity into the model.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))