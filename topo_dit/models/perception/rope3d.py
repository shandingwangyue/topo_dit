"""Additive 3D sinusoidal positions (not rotary embeddings)."""
import math
import torch
from torch import nn


class ApplyPosEmb3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim < 1:
            raise ValueError('dim must be positive')
        self.dim = dim

    def forward(self, x, T, H, W):
        if x.shape[1:] != (T * H * W, self.dim):
            raise ValueError('position grid does not match token shape')
        axes = torch.meshgrid(*(torch.arange(s, device=x.device, dtype=torch.float32)
                                for s in (T, H, W)), indexing='ij')
        pairs = math.ceil(self.dim / 6)
        frequencies = torch.exp(-math.log(10000) * torch.arange(pairs, device=x.device) / pairs)
        pe = []
        for axis in axes:
            phase = axis.reshape(-1, 1) * frequencies
            pe.append(torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1))
        return x + torch.cat(pe, -1)[:, :self.dim].to(x.dtype).unsqueeze(0)
