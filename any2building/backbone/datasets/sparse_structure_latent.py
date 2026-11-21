import os
import json
from typing import *
import numpy as np
import torch
from PIL import Image
from .components import (
    StandardDatasetBase, ImageConditionedMixin, PointImageConditionedMixin,
    find_point_condition, load_point_condition, sample_point_condition,
    render_point_condition,
)
from .. import models


class SparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        
    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    def visualize_sample(self, x_0):
        raise NotImplementedError('Use infer.py to export structure previews')


class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata['ss_latent_encoded'] == True]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root['ss_latent'], f'{instance}.npz'))
        z = torch.tensor(latent['z']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
        }
        return pack


class ImageConditionedSparseStructureLatent(ImageConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._available_indices = []
        self._availability_misses = 0
        self._refresh_available_indices()

    def _refresh_available_indices(self):
        self._available_indices = [
            i for i, (root, instance) in enumerate(self.instances)
            if os.path.isfile(os.path.join(root['ss_latent'], f'{instance}.npz'))
        ]

    def __getitem__(self, index):
        """Train from files that currently exist while a 10k manifest streams in."""
        root, instance = self.instances[index]
        latent_path = os.path.join(root['ss_latent'], f'{instance}.npz')
        if not os.path.isfile(latent_path):
            self._availability_misses += 1
            if not self._available_indices or self._availability_misses % 128 == 0:
                self._refresh_available_indices()
            if not self._available_indices:
                raise RuntimeError(f'No SS latent files are available under {root["ss_latent"]}')
            index = self._available_indices[np.random.randint(len(self._available_indices))]
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as exc:
            self._refresh_available_indices()
            if not self._available_indices:
                raise
            fallback = self._available_indices[np.random.randint(len(self._available_indices))]
            root, instance = self.instances[fallback]
            print(f'Error loading streamed instance at index {index}: {exc}; fallback={instance}')
            return self.get_instance(root, instance)


class MultimodalConditionedSparseStructureLatent(PointImageConditionedMixin, SparseStructureLatent):
    """Sparse-structure latent paired with an image and incomplete point cloud."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._available_indices = []
        self._availability_misses = 0
        self._refresh_available_indices()

    def _refresh_available_indices(self):
        self._available_indices = [
            i for i, (root, instance) in enumerate(self.instances)
            if os.path.isfile(os.path.join(root['ss_latent'], f'{instance}.npz'))
        ]

    def __getitem__(self, index):
        root, instance = self.instances[index]
        if not os.path.isfile(os.path.join(root['ss_latent'], f'{instance}.npz')):
            self._availability_misses += 1
            if not self._available_indices or self._availability_misses % 128 == 0:
                self._refresh_available_indices()
            if not self._available_indices:
                raise RuntimeError(f'No SS latent files are available under {root["ss_latent"]}')
            index = self._available_indices[np.random.randint(len(self._available_indices))]
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as exc:
            self._refresh_available_indices()
            if not self._available_indices:
                raise
            fallback = self._available_indices[np.random.randint(len(self._available_indices))]
            root, instance = self.instances[fallback]
            print(f'Error loading streamed instance at index {index}: {exc}; fallback={instance}')
            return self.get_instance(root, instance)


class FourWayConditionedSparseStructureLatent(SparseStructureLatent):
    """Heterogeneous, exactly balanced stage-1 training pools.

    Roots declare ``training_mode`` as point, image, rough, or both and a
    ``condition_subtype``. The custom sampler balances modes at 25% and cycles
    subtypes equally within each mode. Only the modalities required by the
    selected homogeneous batch are loaded.
    """

    MODES = ('point', 'image', 'rough', 'both')

    def __init__(
        self,
        roots,
        *,
        image_size=512,
        point_cond_root='partial_xyz',
        point_sample_size=16384,
        normalize_points=False,
        normalize_padding=1e-3,
        sparse_point_manifest=None,
        sparse_point_ratio=0.0,
        require_all_modes=True,
        **kwargs,
    ):
        self.image_size = int(image_size)
        self.point_cond_root = point_cond_root
        self.point_sample_size = int(point_sample_size)
        self.normalize_points = bool(normalize_points)
        self.require_all_modes = bool(require_all_modes)
        self.normalize_padding = float(normalize_padding)
        self.sparse_point_ratio = float(sparse_point_ratio)
        if not 0 <= self.sparse_point_ratio < 1:
            raise ValueError('sparse_point_ratio must be in [0, 1)')
        super().__init__(roots, **kwargs)
        self.mode_subtype_indices = {}
        for index, (root, _) in enumerate(self.instances):
            mode = root.get('training_mode')
            subtype = root.get('condition_subtype', mode)
            if mode not in self.MODES:
                raise ValueError(f'Invalid training_mode={mode!r}')
            self.mode_subtype_indices.setdefault(mode, {}).setdefault(subtype, []).append(index)
        missing = [mode for mode in self.MODES if not self.mode_subtype_indices.get(mode)]
        if missing:
            if self.require_all_modes:
                raise ValueError(f'Missing non-empty four-way training pools: {missing}')
            print(f'Warning: four-way pools active only for modes: {sorted(self.mode_subtype_indices)} (missing {missing})')
        self.sparse_mode_subtype_indices = {}
        self.dense_mode_subtype_indices = {}
        if sparse_point_manifest and self.sparse_point_ratio > 0:
            with open(sparse_point_manifest) as handle:
                sparse_paths = {os.path.abspath(line.strip()) for line in handle if line.strip()}
            point_extensions = ('.xyz', '.npy', '.npz', '.ply', '.pt')
            def is_sparse_path(root, instance):
                directories = []
                if self.point_cond_root in root:
                    directories.append(root[self.point_cond_root])
                if 'base' in root:
                    directories.append(os.path.join(root['base'], self.point_cond_root))
                names = [instance]
                if instance.startswith('bldg_'):
                    names.append(instance.removeprefix('bldg_'))
                # Membership-only lookup avoids millions of network filesystem stat calls.
                return any(
                    os.path.abspath(os.path.join(directory, name + extension)) in sparse_paths
                    for directory in directories for name in names for extension in point_extensions
                )
            for mode in ('point', 'both'):
                for subtype, indices in self.mode_subtype_indices.get(mode, {}).items():
                    sparse, dense = [], []
                    for index in indices:
                        root, instance = self.instances[index]
                        (sparse if is_sparse_path(root, instance) else dense).append(index)
                    self.sparse_mode_subtype_indices.setdefault(mode, {})[subtype] = sparse
                    self.dense_mode_subtype_indices.setdefault(mode, {})[subtype] = dense
                    print(f'Sparse-point stratum {mode}/{subtype}: {len(sparse)} sparse, '
                          f'{len(dense)} dense, target={self.sparse_point_ratio:.1%}')
            self.sparse_mode_indices = {
                mode: [index for values in subtype_map.values() for index in values]
                for mode, subtype_map in self.sparse_mode_subtype_indices.items()
            }
        # Dense SS latents have equal compute load. Kept for sampler API
        # compatibility and diagnostic tooling.
        self.loads = [1] * len(self.instances)

    def __getitem__(self, index):
        # A stale reachable manifest can contain a missing latent. Keep the
        # replacement within the same mode and subtype so a homogeneous
        # four-way batch retains the keys expected by default_collate.
        root, instance = self.instances[index]
        mode = root['training_mode']
        subtype = root.get('condition_subtype', mode)
        pool = self.mode_subtype_indices[mode][subtype]
        for attempt in range(32):
            candidate = index if attempt == 0 else pool[np.random.randint(len(pool))]
            candidate_root, candidate_instance = self.instances[candidate]
            try:
                return self.get_instance(candidate_root, candidate_instance)
            except (OSError, ValueError, KeyError) as exc:
                if attempt == 0:
                    print(f'Error loading {instance} ({mode}/{subtype}): {exc}')
        raise RuntimeError(f'Could not load a valid {mode}/{subtype} instance after 32 attempts: {instance}')

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        root = getattr(self, '_current_root', None)
        if not isinstance(root, dict):
            return metadata, stats
        if root.get('trust_include_instances') and root.get('include_instances'):
            stats['Trusted reachable manifest'] = len(metadata)
            return metadata, stats
        mode = root.get('training_mode')
        if mode in ('point', 'both'):
            metadata = metadata[
                metadata.index.astype(str).map(
                    lambda value: find_point_condition(root, self.point_cond_root, value) is not None
                )
            ]
            stats['With point condition'] = len(metadata)
        if mode in ('image', 'rough', 'both'):
            render_root = root.get('render_cond')
            metadata = metadata[
                metadata.index.astype(str).map(
                    lambda value: render_root is not None
                    and os.path.isfile(os.path.join(render_root, value, 'transforms.json'))
                )
            ]
            stats['With image condition'] = len(metadata)
        if mode == 'rough':
            prior_root = root.get('rough_ss_latent')
            metadata = metadata[
                metadata.index.astype(str).map(
                    lambda value: prior_root is not None
                    and os.path.isfile(os.path.join(prior_root, f'{value}.npz'))
                )
            ]
            stats['With rough SS prior'] = len(metadata)
        return metadata, stats

    @staticmethod
    def _square_crop_box(alpha, margin=0.0):
        """Robust square crop box from an alpha/mask array.

        Thresholds at 50% opacity first (anti-aliased fringes or stray faint
        pixels must not inflate the bbox), falls back to any non-zero alpha,
        then to the full frame.  `margin` enlarges the square side by that
        fraction so the building does not touch the frame edges.
        """
        nz = np.empty((0, 2), dtype=np.int64)
        for thr in (128, 1):
            nz = np.argwhere(alpha >= thr)
            if len(nz):
                break
        if not len(nz):
            h, w = alpha.shape
            return (0, 0, w, h)
        y0, x0 = nz.min(0); y1, x1 = nz.max(0) + 1
        side = max(x1 - x0, y1 - y0) * (1 + margin)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return (int(round(cx - side / 2)), int(round(cy - side / 2)),
                int(round(cx + side / 2)), int(round(cy + side / 2)))

    @staticmethod
    def _auto_expose(rgb, mask=None, dark_thresh=0.28, target_mean=0.45, max_gain=4.0):
        """Lift under-exposed condition images (dark aerial crops / night-like
        renders) instead of dropping them.

        If the mean luminance of the visible pixels is below `dark_thresh`
        (0-1 scale), scale RGB toward `target_mean`; gain is capped at
        `max_gain` to limit noise amplification.  This makes the previous
        lum<45 quality blacklist unnecessary.
        """
        vis = rgb[:, mask] if mask is not None else rgb
        mean = vis.mean().clamp_min(1e-4)
        if mean < dark_thresh:
            gain = min((target_mean / mean).item(), max_gain)
            rgb = (rgb * gain).clamp_(max=1.0)
        return rgb

    @staticmethod
    def _target_instance(root, instance):
        separator = root.get('target_id_separator')
        return instance.split(separator, 1)[0] if separator and separator in instance else instance

    def _load_image(self, root, instance):
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as handle:
            frames = json.load(handle)['frames']
        frame = frames[np.random.randint(len(frames))]
        image = Image.open(os.path.join(image_root, frame['file_path']))
        variant = root.get('image_variant', 'context')
        crop_margin = float(root.get('crop_margin', 0.15))
        if variant in ('osm_cutout', 'alpha_cutout'):
            mask_root = root.get('alpha_mask_root', root.get('osm_mask_root'))
            mask_path = os.path.join(mask_root, f'{instance}.png')
            mask = Image.open(mask_path).convert('L')
            alpha = np.asarray(mask)
            if not (alpha > 0).any():
                raise ValueError(f'Empty cutout alpha mask for {instance}')
            box = self._square_crop_box(alpha, crop_margin)
            image = image.convert('RGB').crop(box).resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )
            mask = mask.crop(box).resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )
            alpha_tensor = torch.from_numpy(np.asarray(mask).copy()).float() / 255
            rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
            out = rgb * alpha_tensor.unsqueeze(0)
            return self._auto_expose(out, alpha_tensor > 0.01)
        if variant == 'cutout' and image.mode == 'RGBA':
            alpha = np.asarray(image.getchannel('A'))
            image = image.crop(self._square_crop_box(alpha, crop_margin))
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = torch.from_numpy(np.asarray(image.getchannel('A')).copy()).float() / 255
            rgb = torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1).float() / 255
            out = rgb * alpha.unsqueeze(0)
            return self._auto_expose(out, alpha > 0.01)
        image = image.convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return self._auto_expose(rgb)

    def get_instance(self, root, instance):
        target = self._target_instance(root, instance)
        latent = np.load(os.path.join(root['ss_latent'], f'{target}.npz'))
        z = torch.tensor(latent['z']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std
        mode = root['training_mode']
        pack = {'x_0': z, 'condition_mode': mode}
        if mode in ('image', 'rough', 'both'):
            pack['image_cond'] = self._load_image(root, instance)
        if mode in ('point', 'both'):
            path = find_point_condition(root, self.point_cond_root, instance)
            if path is None:
                raise FileNotFoundError(f'No point condition for {instance}')
            points = load_point_condition(path)
            if self.normalize_points:
                lo, hi = points.amin(0), points.amax(0)
                span = (hi - lo).amax().clamp_min(1e-6)
                points = (points - (lo + hi) / 2) * ((1 - 2*self.normalize_padding) / span)
            pack['point_cond'] = sample_point_condition(points, self.point_sample_size)
        if mode == 'rough' or (mode == 'both' and root.get('rough_ss_latent')):
            prior = np.load(os.path.join(root['rough_ss_latent'], f'{instance}.npz'))['z']
            pack['rough_prior'] = torch.tensor(prior).float()
        return pack

    def visualize_point_condition(self, points):
        return render_point_condition(points)
