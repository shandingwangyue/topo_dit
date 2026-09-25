"""Cached latent dataset: no random text placeholders or recursive bad-file retries."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


def load_tensor(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f'{path}: expected a finite floating tensor')
    return value.float()


class TopoVideoDataset(Dataset):
    def __init__(self, meta_path, latent_channels, cond_dim, conditioning='unconditional',
                 latent_scale=1.0, codec_id='UNSPECIFIED', condition_id='none'):
        self.meta_path = Path(meta_path).resolve()
        self.base = self.meta_path.parent
        self.records = [json.loads(line) for line in self.meta_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.records:
            raise ValueError('empty latent dataset')
        self.latent_channels, self.cond_dim = latent_channels, cond_dim
        self.conditioning = conditioning
        if conditioning not in ('cached', 'unconditional'):
            raise ValueError('invalid conditioning mode')
        for i, r in enumerate(self.records):
            if r.get('codec_id') != codec_id or r.get('latent_scale') != latent_scale:
                raise ValueError(f'record {i}: codec_id/latent_scale mismatch')
            if conditioning == 'cached' and r.get('condition_id') != condition_id:
                raise ValueError(f'record {i}: condition_id mismatch')
            for key in (['latent_path', 'condition_path'] if conditioning == 'cached' else ['latent_path']):
                if key not in r or not (self.base / r[key]).is_file():
                    raise FileNotFoundError(f'record {i}: missing {key}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        z = load_tensor(self.base / record['latent_path'])
        if z.ndim != 4 or z.shape[0] != self.latent_channels or min(z.shape) < 1:
            raise ValueError('cached latent must be (C, T, H, W) with configured C')
        if self.conditioning == 'cached':
            c = load_tensor(self.base / record['condition_path'])
            if c.shape != (self.cond_dim,):
                raise ValueError('cached condition must be (cond_dim,)')
        else:
            c = torch.zeros(self.cond_dim)
        return {'latent': z, 'condition': c}


def create_dataloader(config, model_config, shuffle=True):
    dataset = TopoVideoDataset(config.meta_path, model_config.latent_channels, model_config.cond_dim,
                               config.conditioning, config.latent_scale, config.codec_id, config.condition_id)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.num_workers, drop_last=False)
