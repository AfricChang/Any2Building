"""Sparse reference surface latents for the voxel-conditional flow."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..backbone.modules.sparse import SparseTensor
from ..backbone.utils.data_utils import load_balanced_group_indices


class SurfaceLatentDataset(Dataset):
    def __init__(self,manifest,normalization,max_tokens=32768):
        self.root=Path(manifest).resolve().parent
        self.rows=[json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        self.mean=torch.tensor(normalization['mean'])[None]
        self.std=torch.tensor(normalization['std'])[None]
        self.max_tokens=max_tokens
        self.loads=[]
        for row in self.rows:
            with np.load(self.root/row['surface_latent'],allow_pickle=False) as data:
                n=len(data['coords'])
            if not 0<n<=max_tokens:raise ValueError('Surface token count outside configured limits')
            self.loads.append(n)
        if not self.rows:raise ValueError('Empty surface manifest')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        with np.load(self.root/self.rows[index]['surface_latent'],allow_pickle=False) as data:
            c=data['coords'].copy();f=data['feats'].astype(np.float32)
        if c.ndim!=2 or c.shape[1]!=3 or f.shape!=(len(c),32) or not np.isfinite(f).all() or np.any(c!=np.floor(c)) or c.min()<0 or c.max()>=64:
            raise ValueError('Expected integer [N,3] coords in 0..63 and finite [N,32] feats')
        return dict(coords=torch.from_numpy(c.astype(np.int32)),feats=(torch.from_numpy(f)-self.mean)/self.std,
                    cond=torch.zeros(1,1024))

    @staticmethod
    def collate_fn(batch,split_size=None):
        groups=[list(range(len(batch)))] if split_size is None else load_balanced_group_indices([len(b['coords']) for b in batch],split_size)
        packs=[]
        for group in groups:
            subset=[batch[i] for i in group]
            coords=torch.cat([torch.cat([torch.full((len(b['coords']),1),j,dtype=torch.int32),b['coords']],1) for j,b in enumerate(subset)])
            feats=torch.cat([b['feats'] for b in subset])
            latent=SparseTensor(coords=coords,feats=feats)
            latent._shape=torch.Size([len(subset),32])
            packs.append(dict(x_0=latent,cond=torch.stack([b['cond'] for b in subset])))
        return packs[0] if split_size is None else packs

