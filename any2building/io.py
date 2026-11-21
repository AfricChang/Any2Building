"""Portable input formats in the common normalized XYZ building frame."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_points(path):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            points = data['points'].copy()
    elif path.suffix.lower() == '.npy':
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in ('.xyz', '.txt'):
        points = np.loadtxt(path)[:, :3]
    elif path.suffix.lower() == '.ply':
        import trimesh
        points = np.asarray(trimesh.load(path, process=False).vertices)
    else:
        raise ValueError('Use NPZ (points key), NPY, XYZ, TXT, or PLY')
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected a nonempty finite [N,3] point cloud')
    return points


def load_image(path, image_size=512):
    image = Image.open(path).convert('RGBA')
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    rgba = np.asarray(image, dtype=np.float32) / 255.
    # Preserve the original alpha-over-black convention; no automatic recrop.
    return (rgba[:, :, :3] * rgba[:, :, 3:4]).transpose(2, 0, 1).copy()


def load_occupancy(path):
    path = Path(path)
    if path.suffix == '.npy':
        occupancy = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as data:
            if 'occupancy' in data:
                occupancy = data['occupancy'].copy()
            elif 'coords' in data:
                c = data['coords']
                if c.ndim != 2 or c.shape[1] != 3 or not len(c) or not np.isfinite(c).all() or np.any(c != np.floor(c)) or c.min() < 0 or c.max() >= 64:
                    raise ValueError('coords must be integer [N,3] indices in [0,63]')
                occupancy = np.zeros((64,64,64), dtype=bool)
                occupancy[tuple(c.astype(np.int64).T)] = True
            else:
                raise ValueError('Expected occupancy or coords in NPZ')
    if occupancy.shape != (64,64,64) or not np.isfinite(occupancy).all() or not np.isin(occupancy,[0,1]).all() or not occupancy.any():
        raise ValueError('Expected a nonempty binary [64,64,64] XYZ occupancy')
    return occupancy.astype(bool)


def prior_descriptors(occupancy):
    """[64,64,64] -> [4096,3]: presence, boundary presence, fraction (Eq. 2)."""
    dense = np.asarray(occupancy, dtype=bool)
    if dense.shape != (64,64,64):
        raise ValueError('Prior grid must be 64 cubed')
    padded = np.pad(dense, 1)
    interior = dense.copy()
    for dx,dy,dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        interior &= padded[1+dx:65+dx,1+dy:65+dy,1+dz:65+dz]
    blocks = dense.reshape(16,4,16,4,16,4)
    boundary = (dense & ~interior).reshape(16,4,16,4,16,4).max((1,3,5))
    return np.stack([blocks.max((1,3,5)),boundary,blocks.mean((1,3,5))],-1).reshape(4096,3).astype(np.float32)


def occupancy_mesh(occupancy, path):
    import trimesh
    from skimage.measure import marching_cubes
    v,f,_,_ = marching_cubes(np.pad(occupancy.astype(np.float32),1), .5)
    # XYZ axes, cell centers (i+.5)/64-.5. Marching cubes is a preview only.
    v = (v-.5)/64-.5
    trimesh.Trimesh(v,f,process=False).export(path)

