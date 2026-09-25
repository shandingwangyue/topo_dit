"""Explicit optional preprocessing command; not used by CPU tests."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from topo_dit.models.perception.causal_3d_vae import CosmosCausal3DVAE


def main():
    parser=argparse.ArgumentParser(description='Cache continuous video latents with explicit codec metadata')
    parser.add_argument('--input-manifest',required=True,help='JSONL: video_path; optional condition_path and condition_id')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--codec-dir',required=True)
    parser.add_argument('--codec-id',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--scale',type=float,default=1.0)
    parser.add_argument('--frames',type=int,default=17)
    parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--temporal-compression',type=int,default=8)
    parser.add_argument('--spatial-compression',type=int,default=8)
    args=parser.parse_args()
    if min(args.frames,args.stride,args.size)<1:
        parser.error('frames, stride and size must be positive')
    source=Path(args.input_manifest).resolve()
    records=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines() if x.strip()]
    out=Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'latents.jsonl').exists():
        raise FileExistsError('use a new output directory to preserve existing cache metadata')
    codec=CosmosCausal3DVAE(args.codec_dir,device=args.device,scale_factor=args.scale,
                           temporal_compression=args.temporal_compression,
                           spatial_compression=args.spatial_compression)
    import imageio.v3 as iio
    results=[]
    for i,record in enumerate(records):
        frames=[]
        # Stream only the requested prefix; no recursive recovery for bad videos.
        for index,frame in enumerate(iio.imiter(source.parent/record['video_path'])):
            if index % args.stride == 0:
                frames.append(torch.as_tensor(frame[...,:3].copy()).permute(2,0,1))
            if len(frames)==args.frames:
                break
        if len(frames)<args.frames:
            raise ValueError(f'record {i}: video too short; no looping across a false temporal boundary')
        video=torch.stack(frames).float()/127.5-1
        # Aspect-preserving resize then deterministic center crop.
        h,w=video.shape[-2:]
        factor=args.size/min(h,w)
        video=F.interpolate(video,size=(round(h*factor),round(w*factor)),mode='bilinear',align_corners=False,antialias=True)
        h,w=video.shape[-2:]; top,left=(h-args.size)//2,(w-args.size)//2
        video=video[:,:,top:top+args.size,left:left+args.size].permute(1,0,2,3).unsqueeze(0)
        z=codec.encode(video)[0].cpu()
        filename=f'latent_{i:06d}.pt'
        torch.save(z,out/filename)
        item={'latent_path':filename,'codec_id':args.codec_id,'latent_scale':args.scale,
              'source_video':str(source.parent/record['video_path']),
              'source_shape':list(video.shape[-3:]),'latent_shape':list(z.shape),
              'temporal_compression':args.temporal_compression,'spatial_compression':args.spatial_compression}
        if 'condition_path' in record:
            if not record.get('condition_id'):
                raise ValueError('condition_path requires condition_id')
            item.update(condition_path=str((source.parent/record['condition_path']).resolve()),condition_id=record['condition_id'])
        results.append(item)
    (out/'latents.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in results),encoding='utf-8')
    print(f'Cached {len(results)} clips at {out}')


if __name__=='__main__':
    main()
