from torch import nn


class AdaLN(nn.Module):
    """Per-block AdaLN-Zero modulation."""
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, dim * 6)
        self.silu = nn.SiLU()
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond):
        return self.linear(self.silu(cond)).chunk(6, dim=-1)
