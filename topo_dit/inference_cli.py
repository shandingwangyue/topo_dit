import argparse
from pathlib import Path
import torch
from .checkpoint import load_checkpoint
from .data.video_dataset import load_tensor
from .sampling import euler_flow_matching_sampler


def main():
    parser = argparse.ArgumentParser(description='Sample latent flow; decode video only with an explicit codec.')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--condition', help='cached condition tensor (D,) or (B,D)')
    parser.add_argument('--condition-id', help='must match checkpoint encoder identity')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--latent-shape', type=int, nargs=3, required=True, metavar=('T','H','W'))
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True, help='.pt latent sample output')
    parser.add_argument('--edit-index', type=int, help='experimental slot-feature intervention; no background guarantee')
    parser.add_argument('--edit-scale', type=float, default=0.0)
    parser.add_argument('--codec-dir')
    parser.add_argument('--codec-id')
    parser.add_argument('--video-output', help='.mp4 path, requires --codec-dir and --codec-id')
    parser.add_argument('--temporal-compression', type=int, default=8)
    parser.add_argument('--spatial-compression', type=int, default=8)
    parser.add_argument('--fps', type=int, default=8)
    args = parser.parse_args()
    model, config, payload = load_checkpoint(args.ckpt, args.device)
    if payload['phase'] not in (2,3):
        parser.error('masked-warmup checkpoint is not a trained velocity model')
    if config.data.conditioning == 'cached':
        if not args.condition or args.condition_id != config.data.condition_id:
            parser.error('matching --condition and --condition-id are required')
        c = load_tensor(args.condition)
        c = c.unsqueeze(0) if c.ndim == 1 else c
    else:
        if args.condition or args.condition_id:
            parser.error('this checkpoint was trained unconditionally')
        c = torch.zeros(1,model.cond_dim)
    if args.video_output and (not args.codec_dir or args.codec_id != config.data.codec_id or args.fps < 1):
        parser.error('video output requires matching codec identity, directory and positive fps')
    latent = euler_flow_matching_sampler(model,c.to(args.device),*args.latent_shape,
                                         steps=args.steps,seed=args.seed,
                                         edit_index=args.edit_index,edit_scale=args.edit_scale)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'latent':latent.cpu(),'codec_id':config.data.codec_id,
                'latent_scale':config.data.latent_scale,'seed':args.seed,
                'steps':args.steps,'checkpoint':str(args.ckpt)},output)
    print(f'Saved latent sample: {output}')
    if args.video_output:
        if latent.shape[0] != 1:
            parser.error('MP4 export currently requires batch size 1')
        from .models.perception.causal_3d_vae import CosmosCausal3DVAE
        import imageio.v3 as iio
        codec = CosmosCausal3DVAE(args.codec_dir,device=args.device,scale_factor=config.data.latent_scale,
                                  temporal_compression=args.temporal_compression,
                                  spatial_compression=args.spatial_compression,
                                  latent_channels=model.latent_channels)
        video = codec.decode(latent)[0].clamp(-1,1)
        frames = ((video.permute(1,2,3,0)+1)*127.5).round().byte().cpu().numpy()
        video_path = Path(args.video_output)
        video_path.parent.mkdir(parents=True,exist_ok=True)
        iio.imwrite(video_path,frames,fps=args.fps,macro_block_size=1)
        print(f'Saved decoded video: {video_path}')
