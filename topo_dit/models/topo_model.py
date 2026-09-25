"""Latent-grid flow field with a balanced routing bottleneck. No codec loads here."""
import torch
from torch import nn
from .perception.rope3d import ApplyPosEmb3D
from .router.stratifier import SinkhornStratifier
from .dynamics.element_dit import ElementDiT
from .renderer.broadcast import GatedBroadcastRenderer


class TopoDiT_Model(nn.Module):
    def __init__(self, dim=256, cond_dim=512, num_elements=64, depth=4,
                 num_heads=8, latent_channels=16, sinkhorn_iters=30,
                 epsilon=0.2, routing='sinkhorn'):
        super().__init__()
        if min(dim, cond_dim, num_elements, depth, num_heads, latent_channels) < 1 or dim % num_heads:
            raise ValueError('dimensions must be positive and dim divisible by num_heads')
        if epsilon <= 0 or sinkhorn_iters < 1:
            raise ValueError('invalid routing parameters')
        self.dim, self.cond_dim, self.latent_channels = dim, cond_dim, latent_channels
        self.latent_proj = nn.Linear(latent_channels, dim)
        self.pos_emb = ApplyPosEmb3D(dim)
        self.router = SinkhornStratifier(dim, num_elements, sinkhorn_iters, epsilon, routing)
        self.dynamics = ElementDiT(dim, cond_dim, depth, num_heads)
        self.renderer = GatedBroadcastRenderer(dim, latent_channels)
        self.reconstruction_head = GatedBroadcastRenderer(dim, latent_channels)

    def _encode_grid(self, z):
        if z.ndim != 5 or z.shape[1] != self.latent_channels or min(z.shape) < 1:
            raise ValueError('z must have shape (B, latent_channels, T, H, W)')
        if not torch.isfinite(z).all():
            raise ValueError('z contains non-finite values')
        b, _, t, h, w = z.shape
        tokens = z.permute(0, 2, 3, 4, 1).reshape(b, t*h*w, self.latent_channels)
        return self.router(self.pos_emb(self.latent_proj(tokens), t, h, w))

    def _unflatten(self, tokens, z):
        b, c, t, h, w = z.shape
        return tokens.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

    def reconstruct(self, corrupted_z):
        e, a = self._encode_grid(corrupted_z)
        return self._unflatten(self.reconstruction_head(e, a), corrupted_z), a, e

    def forward(self, z_t, t, condition, edit_index=None, edit_scale=0.0):
        b = z_t.shape[0]
        if t.shape != (b,) or condition.shape != (b, self.cond_dim):
            raise ValueError('expected t=(B,) and condition=(B, cond_dim)')
        if not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
            raise ValueError('flow time must be finite and in [0, 1]')
        if not torch.isfinite(condition).all():
            raise ValueError('condition contains non-finite values')
        e, a = self._encode_grid(z_t)
        features = self.dynamics(e, t, condition)
        if edit_index is not None:
            if not 0 <= edit_index < features.shape[1]:
                raise ValueError('edit_index out of range')
            features = features.clone()
            features[:, edit_index] *= edit_scale
        velocity = self._unflatten(self.renderer(features, a), z_t)
        return velocity, a, features
