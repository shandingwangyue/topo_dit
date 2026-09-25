from pathlib import Path
import torch
from configs.config import TopoConfig
from .models import TopoDiT_Model


FORMAT_VERSION = 2


def save_checkpoint(path, model, config, epoch, phase, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format_version': FORMAT_VERSION, 'config': config.config_dict,
               'model_state_dict': model.state_dict(), 'epoch': epoch, 'phase': phase,
               'torch_rng_state': torch.get_rng_state()}
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != FORMAT_VERSION:
        raise ValueError('incompatible checkpoint: v2 architecture required; v0 weights cannot be silently migrated')
    config = TopoConfig(config_dict=payload['config'])
    model = TopoDiT_Model(**config.config_dict['model'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.to(device), config, payload
