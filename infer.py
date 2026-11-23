#!/usr/bin/env python3
"""Run Any2Building on a single input or the bundled paper examples."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['point','image','both','image_prior'])
    p.add_argument('--points', type=Path)
    p.add_argument('--image', type=Path)
    p.add_argument('--prior', type=Path)
    p.add_argument('--examples', action='store_true')
    p.add_argument('--case', action='append', help='Select one or more bundled example IDs')
    p.add_argument('--variant', choices=['ema'], default='ema')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--guidance', type=float, default=3.)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--memory-fraction', type=float, default=1.0)
    p.add_argument('--dtype', choices=['auto', 'float32', 'bfloat16'], default='auto',
                   help='Weight storage dtype (auto uses bfloat16 on <12GB GPUs)')
    p.add_argument('--offload', choices=['auto', 'none', 'model', 'block'], default='auto',
                   help='CPU offloading strategy for low-VRAM GPUs')
    p.add_argument('--verbose', action='store_true', help='Show sampling progress bar')
    args = p.parse_args()
    if args.steps < 1 or not 0 < args.memory_fraction <= 1:
        p.error('steps must be positive; memory-fraction must be in (0,1]')
    if 'ATTN_BACKEND' not in os.environ:
        os.environ['ATTN_BACKEND'] = 'flash_attn' if importlib.util.find_spec('flash_attn') else 'sdpa'
    from any2building.pipeline import Any2BuildingPipeline
    from any2building.io import sha256, occupancy_mesh
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the released model')
    if args.memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    if args.examples:
        rows = json.loads((args.root/'examples/manifest.json').read_text())
        if args.case:
            ids = {r['id'] for r in rows}
            if set(args.case)-ids: p.error(f'Unknown example IDs: {set(args.case)-ids}')
            rows = [r for r in rows if r['id'] in args.case]
        rows = [{**r, **{k:args.root/r[k] for k in ('points','image','prior') if k in r}} for r in rows]
    else:
        if args.mode is None: p.error('--mode or --examples is required')
        rows = [dict(id='prediction', mode=args.mode, points=args.points,image=args.image,prior=args.prior)]
    if not rows: p.error('No cases selected')
    for row in rows:
        if (args.outdir/row['id']).exists():
            raise FileExistsError(f'Output already exists: {args.outdir/row["id"]}; choose a fresh output directory')
    pipeline = Any2BuildingPipeline(args.root, args.variant,
                                    image_enabled=any(r['mode']!='point' for r in rows),
                                    dtype=args.dtype, offload=args.offload)
    summary = []
    for row in rows:
        out = args.outdir/row['id']; out.mkdir(parents=True)
        result = pipeline(mode=row['mode'], points=row.get('points'), image=row.get('image'),
                          prior=row.get('prior'), seed=args.seed, steps=args.steps,
                          guidance=args.guidance, verbose=args.verbose)
        np.savez_compressed(out/'occupancy.npz',occupancy=result['occupancy'])
        np.savez_compressed(out/'structure_latent.npz',z=result['latent'])
        np.savez_compressed(out/'query_logits.npz',logits=result['query_logits'])
        occupancy_mesh(result['occupancy'],out/'structure_preview.ply')
        rec = dict(id=row['id'],mode=row['mode'],model='Any2Building',variant=args.variant,
            seed=args.seed,steps=args.steps,guidance=args.guidance,occupied=int(result['occupancy'].sum()),
            input_sha256={k:sha256(row[k]) for k in ('points','image','prior') if row.get(k)},
            output='64^3 surface occupancy; structure_preview.ply is marching cubes, not the LoD2 surface stage')
        (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n')
        summary.append(rec); print(json.dumps(rec),flush=True)
    (args.outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__': main()

