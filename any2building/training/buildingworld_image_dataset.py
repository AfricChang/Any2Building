import os
from pathlib import Path
import numpy as np
import torch
from .prior_dataset import QueryPriorDataset
from .prior_geometry import prior_grid,features_from_grid

def features(spec,sid,grid=8):
    dense=prior_grid(spec,sid)
    if grid==8:return features_from_grid(dense)
    assert grid==16
    padded=np.pad(dense,1);interior=dense.copy()
    for dx,dy,dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        interior &= padded[1+dx:65+dx,1+dy:65+dy,1+dz:65+dz]
    blocks=dense.reshape(16,4,16,4,16,4)
    shell=(dense & ~interior).reshape(16,4,16,4,16,4).max((1,3,5))
    return np.stack([blocks.max((1,3,5)),shell,blocks.mean((1,3,5))],-1).reshape(4096,3).astype(np.float32)

class BuildingWorldImageDataset(QueryPriorDataset):
    def __init__(self,*args,rough_grid=8,**kwargs):
        self.rough_grid=int(rough_grid)
        super().__init__(*args,**kwargs)

    def visualize_sample(self, *args, **kwargs):
        raise NotImplementedError('Use infer.py and its portable occupancy preview; training snapshots are disabled.')
    def load_query_prior(self,root,sid):
        if self.rough_grid==8:return super().load_query_prior(root,sid)
        spec=root['query_prior_geometry']
        path=Path(spec['cache_dir']+'_xyz16_v1')/(sid+'.npy')
        if path.exists():f=np.load(path,allow_pickle=False)
        else:
            f=features(spec,sid,16)
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
            with temp.open('wb') as stream:np.save(stream,f.astype(np.float16))
            os.replace(temp,path)
        if f.shape!=(4096,3) or not np.isfinite(f).all() or not f[:,0].any() or f.min()<0 or f.max()>1:raise ValueError(path)
        return torch.from_numpy(f.astype(np.float32))
