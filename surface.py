#!/usr/bin/env python3
"""Generate a dense surface from Any2Building predicted surface voxels (Sec. 3.4)."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--occupancy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--guidance',type=float,default=3.)
    p.add_argument('--memory-fraction',type=float,default=.8)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Choose a new output file')
    root=Path(__file__).resolve().parent
    os.environ.setdefault('ATTN_BACKEND','flash_attn')
    os.environ.setdefault('SPARSE_ATTN_BACKEND','flash_attn')
    import numpy as np
    import torch
    import torch.utils.checkpoint
    import trimesh
    from any2building.io import load_occupancy,sha256
    from any2building.backbone import models
    from any2building.models.surface_generator import VoxelConditionalSurfaceGenerator
    from any2building.backbone.modules.sparse import SparseTensor
    from any2building.backbone.pipelines.samplers import FlowEulerCfgSampler
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    cfg=json.loads((root/'configs/surface_generator.json').read_text())
    flow=VoxelConditionalSurfaceGenerator(**cfg['models']['denoiser']['args']).cuda().eval().requires_grad_(False)
    flow.load_state_dict(torch.load(root/'weights/surface_generator/surface_flow.pt',map_location='cpu',weights_only=True,mmap=True),strict=True)
    decoder=models.from_pretrained(str(root/'weights/surface_decoder/model')).cuda().eval().requires_grad_(False)
    decoder.set_resolution(1024)
    coords=np.argwhere(load_occupancy(a.occupancy)).astype(np.int32)
    c=torch.from_numpy(coords)
    c4=torch.cat([torch.zeros(len(c),1,dtype=torch.int32),c],1).cuda()
    norm=cfg['dataset']['args']['normalization']
    mean=torch.tensor(norm['mean'],device='cuda')[None]
    std=torch.tensor(norm['std'],device='cuda')[None]
    noise=SparseTensor(coords=c4,feats=torch.randn(len(c4),32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(a.seed)))
    null=torch.zeros(1,1,1024,device='cuda')
    with torch.no_grad():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            latent=FlowEulerCfgSampler(sigma_min=1e-5).sample(flow,noise=noise,cond=null,neg_cond=null,
                steps=a.steps,guidance_strength=a.guidance,verbose=False).samples
        latent=latent.replace(feats=latent.feats.float()*std+mean)
        mesh=decoder(latent)[0]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    trimesh.Trimesh(mesh.vertices.detach().float().cpu().numpy(),mesh.faces.detach().long().cpu().numpy(),process=False).export(a.output)
    rec=dict(model='Any2Building voxel-conditional surface generator',weights='ema',steps=a.steps,seed=a.seed,
        guidance=a.guidance,occupancy_sha256=sha256(a.occupancy),occupied=len(coords),vertices=len(mesh.vertices),faces=len(mesh.faces),
        conditioning='Predicted occupied surface cells only; no image, point tokens, filled interiors, or GT support')
    a.output.with_suffix('.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec),flush=True)


if __name__=='__main__':main()

