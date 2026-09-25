import torch
import torch.nn as nn

class AdaLN(nn.Module):
    """自适应层归一化 (Adaptive Layer Normalization)，用于注入时间和文本条件"""
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, dim * 6)
        
    def forward(self, cond):
        # Chunk 成 6 份，分别对应 MSA 和 MLP 的 shift, scale, gate
        return self.linear(self.silu(cond)).chunk(6, dim=-1)