"""Optional continuous Cosmos JIT adapter; never instantiated by the latent model."""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F


class CosmosCausal3DVAE(nn.Module):
    def __init__(self, model_dir, device='cpu', dtype=torch.float32, scale_factor=1.0,
                 temporal_compression=8, spatial_compression=8, latent_channels=16):
        super().__init__()
        if not math.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError('scale_factor must be positive and explicitly match the cache')
        if min(temporal_compression, spatial_compression, latent_channels) < 1:
            raise ValueError('invalid codec geometry')
        directory = Path(model_dir)
        for name in ('encoder.jit', 'decoder.jit'):
            if not (directory/name).is_file():
                raise FileNotFoundError(directory/name)
        self.encoder = torch.jit.load(str(directory/'encoder.jit'), map_location=device).to(dtype=dtype)
        self.decoder = torch.jit.load(str(directory/'decoder.jit'), map_location=device).to(dtype=dtype)
        self.register_buffer('_device_dtype', torch.empty(0, device=device, dtype=dtype))
        self.latent_dim, self.scale_factor = latent_channels, scale_factor
        self.temporal_compression, self.spatial_compression = temporal_compression, spatial_compression
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # The pretrained codec stays frozen and in eval mode even under parent.train().
        return super().train(False)

    @torch.no_grad()
    def encode(self, video):
        if video.ndim != 5 or video.shape[1] != 3 or not torch.isfinite(video).all():
            raise ValueError('video must be finite B,3,T,H,W')
        if video.min() < -1.001 or video.max() > 1.001:
            raise ValueError('video range must be [-1,1]')
        video = video.to(self._device_dtype)
        t,h,w = video.shape[-3:]
        tf,sf = self.temporal_compression,self.spatial_compression
        tp = 1 + math.ceil((t-1)/tf)*tf
        hp,wp = math.ceil(h/sf)*sf,math.ceil(w/sf)*sf
        padded = F.pad(video,(0,wp-w,0,hp-h,0,tp-t),mode='replicate')
        out = self.encoder(padded)
        z = out[0] if isinstance(out,(tuple,list)) else out
        expected = (video.shape[0],self.latent_dim,1+(tp-1)//tf,hp//sf,wp//sf)
        if z.shape != expected or not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError(f'continuous codec shape/type mismatch: expected {expected}, got {tuple(z.shape)}')
        return z.float()*self.scale_factor

    @torch.no_grad()
    def decode(self, latent, crop_shape=None):
        if latent.ndim != 5 or latent.shape[1] != self.latent_dim or not torch.isfinite(latent).all():
            raise ValueError('invalid continuous latent')
        out = self.decoder((latent/self.scale_factor).to(self._device_dtype))
        video = out[0] if isinstance(out,(tuple,list)) else out
        if video.ndim != 5 or video.shape[1] != 3 or not torch.isfinite(video).all():
            raise ValueError('invalid decoded video')
        if crop_shape is not None:
            t,h,w = crop_shape
            if min(t,h,w) < 1 or any(a>b for a,b in zip(crop_shape,video.shape[-3:])):
                raise ValueError('crop exceeds decoded shape')
            video = video[:,:,:t,:h,:w]
        return video.float()
