"""Manifest dataset for homogeneous four-mode training batches."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from ..io import load_points, load_image, load_occupancy, prior_descriptors


class BuildingObservationDataset(Dataset):
    def __init__(self, manifest, pretrained_ss_dec, normalization=None, point_sample_size=16384):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.rows = [json.loads(s) for s in self.manifest.read_text().splitlines() if s.strip()]
        if not self.rows: raise ValueError('Empty training manifest')
        self.pretrained_ss_dec = pretrained_ss_dec
        self.normalization = normalization
        if normalization is not None:
            self.mean = torch.tensor(normalization['mean']).reshape(-1,1,1,1)
            self.std = torch.tensor(normalization['std']).reshape(-1,1,1,1)
        self.point_sample_size = point_sample_size
        self.mode_subtype_indices = defaultdict(lambda:defaultdict(list))
        for j,row in enumerate(self.rows):
            mode = row['mode']
            if mode not in ('point','image','both','rough'): raise ValueError(mode)
            if 'latent' not in row: raise ValueError('Each training row requires a complete-target latent')
            self.mode_subtype_indices[mode][row.get('subtype','default')].append(j)
        self.mode_subtype_indices = {m:dict(v) for m,v in self.mode_subtype_indices.items()}

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]; mode = row['mode']
        with np.load(self.root/row['latent'],allow_pickle=False) as data:
            z = data['z'].astype(np.float32)
        if z.shape != (8,16,16,16) or not np.isfinite(z).all(): raise ValueError('Target z must be [8,16,16,16]')
        target = torch.from_numpy(z)
        if self.normalization is not None: target = (target-self.mean)/self.std
        pack = {'x_0':target,'condition_mode':mode}
        if mode in ('point','both'):
            points = load_points(self.root/row['points'])
            if np.abs(points).max() > .501: raise ValueError('Normalize points with the shared target frame')
            idx = np.random.choice(len(points),self.point_sample_size,replace=len(points)<self.point_sample_size)
            pack['point_cond'] = torch.from_numpy(points[idx])
        if mode in ('image','both','rough'):
            pack['image_cond'] = torch.from_numpy(load_image(self.root/row['image']))
        if mode == 'rough':
            pack['query_prior'] = torch.from_numpy(prior_descriptors(load_occupancy(self.root/row['prior'])))
        return pack

