import torch
import torch.nn as nn
from .adal_n import AdaLN

class ElementDiTBlock(nn.Module):
    """运行在 Element Tokens 层级的核心 Transformer Block"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, ada_params):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_params
        
        # MSA (拓扑与物理关系自注意力)
        x_norm1 = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP (属性非线性演变)
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class ElementDiT(nn.Module):
    """拓扑动力学主干 (Topological Dynamics Engine)"""
    def __init__(self, dim, cond_dim, depth=12, num_heads=8):
        super().__init__()
        self.adaln_generator = AdaLN(cond_dim, dim)
        self.blocks = nn.ModuleList([
            ElementDiTBlock(dim, num_heads) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, element_tokens, condition):
        # 1. 生成调制参数
        ada_params = self.adaln_generator(condition)
        
        # 2. 深度网络前向传播
        x = element_tokens
        for block in self.blocks:
            x = block(x, ada_params)
            
        return self.final_norm(x)