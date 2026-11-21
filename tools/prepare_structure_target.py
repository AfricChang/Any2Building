#!/usr/bin/env python3
"""Encode a 64^3 complete surface grid with the matching frozen structure VAE."""
import argparse
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--occupancy',type=Path,required=True)
    p.add_argument('--encoder',required=True,help='Prefix of matching ss_enc_conv3d_16l8_fp16 .json/.safetensors files')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import torch,numpy as np
    from any2building.backbone import models
    from any2building.io import load_occupancy
    model=models.from_pretrained(a.encoder).cuda().eval().requires_grad_(False)
    occ=torch.from_numpy(load_occupancy(a.occupancy).astype(np.float32))[None,None].cuda()
    with torch.no_grad():z=model(occ,sample_posterior=False)
    if z.shape!=(1,8,16,16,16):raise ValueError(f'Unexpected encoder output: {z.shape}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,z=z[0].float().cpu().numpy())


if __name__=='__main__':main()
