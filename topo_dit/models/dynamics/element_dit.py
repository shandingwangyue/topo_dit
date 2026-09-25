import math
import torch
from torch import nn
from .adal_n import AdaLN


class ElementDiTBlock(nn.Module):
    def __init__(self, dim, cond_dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.modulation = AdaLN(cond_dim, dim)

    def forward(self, x, condition):
        sm, cm, gm, sf, cf, gf = [p.unsqueeze(1) for p in self.modulation(condition)]
        h = self.norm1(x) * (1 + cm) + sm
        x = x + gm * self.attn(h, h, h, need_weights=False)[0]
        return x + gf * self.mlp(self.norm2(x) * (1 + cf) + sf)


class ElementDiT(nn.Module):
    def __init__(self, dim, cond_dim, depth=12, num_heads=8):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(64, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.condition_proj = nn.Linear(cond_dim, cond_dim)
        self.blocks = nn.ModuleList([ElementDiTBlock(dim, cond_dim, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Final modulation is not zero initialized: time/condition influence is
        # measurable at initialization, while residual blocks start as identity.
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(self, elements, t, condition):
        freq = torch.exp(-math.log(10000) * torch.arange(32, device=t.device) / 32)
        phase = t.float().unsqueeze(1) * freq.unsqueeze(0) * 1000
        embed = torch.cat((phase.cos(), phase.sin()), -1).to(elements.dtype)
        c = self.time_mlp(embed) + self.condition_proj(condition)
        x = elements
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.final_modulation(c).chunk(2, -1)
        return self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
