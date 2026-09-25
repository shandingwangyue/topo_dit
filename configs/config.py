from copy import deepcopy
from pathlib import Path
import yaml


class DotDict(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        value = self[key]
        return DotDict(value) if isinstance(value, dict) else value


DEFAULTS = {
    'model': {'dim': 256, 'cond_dim': 512, 'num_elements': 64, 'depth': 4,
              'num_heads': 8, 'latent_channels': 16, 'sinkhorn_iters': 30,
              'epsilon': 0.2, 'routing': 'sinkhorn'},
    'data': {'meta_path': 'data/latents.jsonl', 'batch_size': 1, 'num_workers': 0,
             'conditioning': 'unconditional', 'latent_scale': 1.0,
             'codec_id': 'UNSPECIFIED', 'condition_id': 'none'},
    'training': {'learning_rate': 0.0001, 'weight_decay': 0.01, 'max_epochs': 10,
                 'lambda_ent': 0.0, 'lambda_ortho': 0.0, 'mask_ratio': 0.5,
                 'grad_clip': 1.0, 'seed': 42},
}


class TopoConfig:
    def __init__(self, yaml_path=None, config_dict=None):
        self.config_dict = deepcopy(DEFAULTS)
        values = config_dict
        if yaml_path is not None:
            with Path(yaml_path).open(encoding='utf-8') as f:
                values = yaml.safe_load(f)
        if values is not None:
            if not isinstance(values, dict):
                raise ValueError('config must be a mapping')
            for section, entries in values.items():
                if section not in DEFAULTS or not isinstance(entries, dict):
                    raise ValueError(f'unknown/invalid config section: {section}')
                unknown = set(entries) - set(DEFAULTS[section])
                if unknown:
                    raise ValueError(f'unknown {section} keys: {sorted(unknown)}')
                self.config_dict[section].update(entries)
        self.cfg = DotDict(self.config_dict)
        if self.data.conditioning not in ('unconditional', 'cached'):
            raise ValueError('conditioning must be unconditional or cached')
        if self.data.conditioning == 'cached' and self.data.condition_id == 'none':
            raise ValueError('cached conditioning requires an explicit condition_id')
        if self.data.latent_scale <= 0 or self.data.batch_size < 1 or self.data.num_workers < 0:
            raise ValueError('invalid data settings')
        if not 0 < self.training.mask_ratio < 1 or self.training.max_epochs < 1:
            raise ValueError('invalid training settings')
        if self.training.learning_rate <= 0 or self.training.grad_clip <= 0:
            raise ValueError('learning_rate and grad_clip must be positive')

    def __getattr__(self, name):
        return getattr(self.cfg, name)
