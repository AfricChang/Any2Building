#!/usr/bin/env python3
"""Generate a dense surface from Any2Building predicted surface voxels (Sec. 3.4)."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def _install_decoder_stage_offload(decoder, device):
    """Move each decoder block to GPU on demand and back to CPU right after."""
    handles = []
    for stage in decoder.blocks:
        for block in stage:
            def _pre(mod, _inp):
                mod.to(device)
            def _post(mod, _inp, _out):
                mod.to('cpu')
                if device.type == 'cuda':
                    import torch
                    torch.cuda.empty_cache()
            handles.append(block.register_forward_pre_hook(_pre))
            handles.append(block.register_forward_hook(_post))
    return handles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--occupancy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--guidance',type=float,default=3.)
    p.add_argument('--memory-fraction',type=float,default=1.0)
    p.add_argument('--dtype',choices=['auto','float32','bfloat16'],default='auto')
    p.add_argument('--offload',choices=['auto','none','model','block'],default='auto')
    p.add_argument('--verbose',action='store_true')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Choose a new output file')
    root=Path(__file__).resolve().parent
    default_attn='flash_attn' if importlib.util.find_spec('flash_attn') else 'sdpa'
    os.environ.setdefault('ATTN_BACKEND',default_attn)
    os.environ.setdefault('SPARSE_ATTN_BACKEND',default_attn)
    if 'SPARSE_CONV_BACKEND' not in os.environ:
        if importlib.util.find_spec('flex_gemm'):
            os.environ['SPARSE_CONV_BACKEND']='flex_gemm'
        elif importlib.util.find_spec('spconv'):
            os.environ['SPARSE_CONV_BACKEND']='spconv'
    import numpy as np
    import torch
    import torch.utils.checkpoint
    import trimesh
    from any2building.io import load_occupancy,sha256
    from any2building.pipeline import _resolve_memory_strategy,_install_block_offload,_move_denoiser_shell
    from any2building.backbone import models
    from any2building.models.surface_generator import VoxelConditionalSurfaceGenerator
    from any2building.backbone.modules.sparse import SparseTensor
    from any2building.backbone.pipelines.samplers import FlowEulerCfgSampler

    device=torch.device('cuda')
    if a.memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    weight_dtype,offload=_resolve_memory_strategy(device,a.dtype,a.offload)
    print(f'[SURFACE] device={device} weight_dtype={weight_dtype} offload={offload}',flush=True)

    cfg=json.loads((root/'configs/surface_generator.json').read_text())
    flow=VoxelConditionalSurfaceGenerator(**cfg['models']['denoiser']['args']).eval().requires_grad_(False)
    flow.load_state_dict(torch.load(root/'weights/surface_generator/surface_flow.pt',map_location='cpu',weights_only=True,mmap=True),strict=True)
    for block in flow.blocks:
        block.use_checkpoint=False
    if weight_dtype=='bfloat16':
        flow.convert_to(torch.bfloat16)

    coords=np.argwhere(load_occupancy(a.occupancy)).astype(np.int32)
    c=torch.from_numpy(coords)
    c4=torch.cat([torch.zeros(len(c),1,dtype=torch.int32),c],1).cuda()
    norm=cfg['dataset']['args']['normalization']
    mean=torch.tensor(norm['mean'],device='cuda')[None]
    std=torch.tensor(norm['std'],device='cuda')[None]
    noise=SparseTensor(coords=c4,feats=torch.randn(len(c4),32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(a.seed)))
    null=torch.zeros(1,1,1024,device='cuda')

    with torch.no_grad():
        if offload=='block':
            _install_block_offload(flow,device)
            _move_denoiser_shell(flow,device)
        else:
            flow.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            latent=FlowEulerCfgSampler(sigma_min=1e-5).sample(flow,noise=noise,cond=null,neg_cond=null,
                steps=a.steps,guidance_strength=a.guidance,verbose=a.verbose).samples
        del flow,noise,null
        torch.cuda.empty_cache()

        latent=latent.replace(feats=latent.feats.float()*std+mean)
        decoder=models.from_pretrained(str(root/'weights/surface_decoder/model')).eval().requires_grad_(False)
        decoder.set_resolution(1024)
        if offload!='none':
            decoder.from_latent.to(device)
            decoder.output_layer.to(device)
            _install_decoder_stage_offload(decoder,device)
        else:
            decoder.to(device)
        mesh=decoder(latent)[0]
        del decoder,latent
        torch.cuda.empty_cache()

    a.output.parent.mkdir(parents=True,exist_ok=True)
    trimesh.Trimesh(mesh.vertices.detach().float().cpu().numpy(),mesh.faces.detach().long().cpu().numpy(),process=False).export(a.output)
    rec=dict(model='Any2Building voxel-conditional surface generator',weights='ema',steps=a.steps,seed=a.seed,
        guidance=a.guidance,occupancy_sha256=sha256(a.occupancy),occupied=len(coords),vertices=len(mesh.vertices),faces=len(mesh.faces),
        conditioning='Predicted occupied surface cells only; no image, point tokens, filled interiors, or GT support')
    a.output.with_suffix('.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec),flush=True)


if __name__=='__main__':main()

