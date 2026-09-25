import torch
from torch import nn


class GatedBroadcastRenderer(nn.Module):
    """Legacy class name; v2 intentionally has no input-copy skip or zero gate."""
    def __init__(self, dim, out_dim):
        super().__init__()
        self.proj_out = nn.Linear(dim, out_dim)

    def forward(self, elements, assignments):
        return self.proj_out(torch.bmm(assignments.to(elements.dtype), elements))
