import argparse
import json
from pathlib import Path
import torch
from configs.config import TopoConfig
from .models import TopoDiT_Model
from .data.video_dataset import create_dataloader
from .training.trainer import TopoDiTTrainer
from .checkpoint import save_checkpoint, load_checkpoint


def main(phase=2):
    parser = argparse.ArgumentParser(description='Train on prepared latent caches; never starts by import.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default=f'checkpoints/phase{phase}')
    parser.add_argument('--ckpt_phase1', help='optional v2 masked-warmup checkpoint')
    parser.add_argument('--resume', help='resume same phase, including optimizer and CPU RNG')
    args = parser.parse_args()
    config = TopoConfig(args.config)
    torch.manual_seed(config.training.seed)
    start, payload = 1, None
    if args.resume and args.ckpt_phase1:
        parser.error('--resume and --ckpt_phase1 are mutually exclusive')
    source = args.resume or args.ckpt_phase1
    if source:
        model, old_config, payload = load_checkpoint(source, args.device)
        if old_config.config_dict['model'] != config.config_dict['model'] or old_config.config_dict['data'] != config.config_dict['data']:
            raise ValueError('checkpoint architecture/data contract differs from config')
        if args.ckpt_phase1 and (phase != 2 or payload['phase'] != 1):
            raise ValueError('warmup transfer requires phase 1 checkpoint and phase 2 entry point')
        if args.resume and payload['phase'] != phase:
            raise ValueError('resume phase mismatch')
    if not source:
        model = TopoDiT_Model(**config.config_dict['model']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)
    if args.resume:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        torch.set_rng_state(payload['torch_rng_state'])
        start = payload['epoch'] + 1
    loader = create_dataloader(config.data, config.model)
    trainer = TopoDiTTrainer(model, optimizer, config.training, args.device)
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config.config_dict, indent=2), encoding='utf-8')
    for epoch in range(start, config.training.max_epochs + 1):
        metrics = trainer.train_epoch(loader, epoch, phase)
        print(json.dumps(metrics))
        with (out/'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')
        save_checkpoint(out/f'epoch_{epoch:04d}.pt', model, config, epoch, phase, optimizer)
