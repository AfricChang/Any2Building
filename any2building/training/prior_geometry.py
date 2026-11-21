"""Direct geometry -> XYZ 8^3 query activation. No SS encoder/decoder involved."""
from pathlib import Path
import numpy as np

VERSION = 'xyz8-solid-surface-fraction-v1'

def load_coords(path, hk_rotate=False):
    with np.load(path, allow_pickle=False) as pack:
        c = pack['coords'].astype(np.int64)
    if c.ndim != 2 or c.shape[1] != 3 or not len(c) or c.min()<0 or c.max()>=64:
        raise ValueError(f'Invalid 64-grid coordinates: {path}')
    if hk_rotate:
        c = c[:, [0,2,1]].copy()
        c[:,2] = 63-c[:,2]
    return c

def prior_grid(spec, sid):
    c = load_coords(Path(spec['coords_dir'])/(sid+'.npz'), spec.get('hk_rotate',False))
    grid = np.zeros((64,64,64), dtype=bool)
    grid[tuple(c.T)] = True
    kind = spec['kind']
    if kind == 'lod0':
        lo,hi=c.min(0),c.max(0)
        grid[:]=False
        grid[lo[0]:hi[0]+1,lo[1]:hi[1]+1,lo[2]:hi[2]+1]=True
    elif kind == 'lod1':
        footprint=grid.any(2)
        lo,hi=c[:,2].min(),c[:,2].max()
        grid[:]=False
        grid[:,:,lo:hi+1]=footprint[:,:,None]
    elif kind != 'direct':
        raise ValueError(kind)
    return grid

def features_from_grid(grid):
    grid=np.asarray(grid,dtype=bool)
    if grid.shape!=(64,64,64) or not grid.any():raise ValueError('Empty/invalid prior grid')
    # Six-neighbour surface of a filled LOD; direct OSM shells remain shells.
    padded=np.pad(grid,1)
    interior=grid.copy()
    for dx,dy,dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        interior &= padded[1+dx:65+dx,1+dy:65+dy,1+dz:65+dz]
    surface=grid & ~interior
    blocks=grid.reshape(8,8,8,8,8,8)
    active=blocks.max((1,3,5))
    fraction=blocks.mean((1,3,5))
    shell=surface.reshape(8,8,8,8,8,8).max((1,3,5))
    # C-order flatten: j=(x*8+y)*8+z. Exactly spatial_coords meshgrid(ij).
    return np.stack([active,shell,fraction],-1).reshape(512,3).astype(np.float32)

def features(spec,sid):
    return features_from_grid(prior_grid(spec,sid))
