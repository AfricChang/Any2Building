from typing import *
import json
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


_POINT_EXTENSIONS = ('.pt', '.xyz', '.npy', '.npz', '.ply')

_METADATA_CACHE: Dict[str, pd.DataFrame] = {}

def _read_metadata_cached(path: str) -> pd.DataFrame:
    """Read each physical metadata table once per training process."""
    frame = _METADATA_CACHE.get(path)
    if frame is None:
        frame = pd.read_csv(path, dtype={"sha256": str}, low_memory=False).set_index("sha256")
        _METADATA_CACHE[path] = frame
    return frame


def find_point_condition(root, point_cond_root: str, instance: str) -> Optional[str]:
    """Resolve a point condition from a root mapping or a base subdirectory."""
    candidates = []
    if isinstance(root, dict):
        if point_cond_root in root:
            candidates.append(root[point_cond_root])
        if 'base' in root:
            candidates.append(os.path.join(root['base'], point_cond_root))
    else:
        candidates.append(os.path.join(root, point_cond_root))
    instance_names = [instance]
    # BuildingWorld metadata/render/latent IDs use ``bldg_<numeric-id>`` while
    # the paired Calgary XYZ archives use ``<numeric-id>.xyz``.
    if instance.startswith('bldg_'):
        instance_names.append(instance.removeprefix('bldg_'))
    for directory in candidates:
        for instance_name in instance_names:
            for extension in _POINT_EXTENSIONS:
                path = os.path.join(directory, f'{instance_name}{extension}')
                if os.path.isfile(path):
                    return path
    return None


def load_point_condition(path: str) -> torch.Tensor:
    if path.endswith('.pt'):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(data, dict):
            data = data.get('points', data.get('xyz'))
        points = torch.as_tensor(data)
    elif path.endswith('.xyz'):
        points = torch.from_numpy(np.loadtxt(path, dtype=np.float32))
    elif path.endswith('.npy'):
        points = torch.from_numpy(np.load(path))
    elif path.endswith('.npz'):
        data = np.load(path)
        key = 'points' if 'points' in data else ('xyz' if 'xyz' in data else data.files[0])
        points = torch.from_numpy(data[key])
    elif path.endswith('.ply'):
        with open(path, 'r') as handle:
            lines = handle.readlines()
        try:
            end = next(i for i, line in enumerate(lines) if line.strip() == 'end_header')
        except StopIteration as exc:
            raise ValueError(f'Invalid ASCII PLY header: {path}') from exc
        if not any(line.strip() == 'format ascii 1.0' for line in lines[:end]):
            raise ValueError(f'Only ASCII PLY point conditions are supported: {path}')
        points = torch.from_numpy(np.loadtxt(lines[end + 1:], dtype=np.float32))
    else:
        raise ValueError(f'Unsupported point condition: {path}')
    points = points.float().reshape(-1, points.shape[-1])[:, :3].contiguous()
    if points.shape[0] == 0 or not torch.isfinite(points).all():
        raise ValueError(f'Point condition is empty or non-finite: {path}')
    return points


def sample_point_condition(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[0] >= count:
        indices = torch.randperm(points.shape[0])[:count]
    else:
        indices = torch.randint(points.shape[0], (count,))
    return points[indices]


def render_point_condition(points: torch.Tensor, image_size: int = 512) -> torch.Tensor:
    """Render a batch of clouds as four orthographic views in a 2x2 image."""
    device = points.device
    output = torch.zeros(points.shape[0], 3, image_size, image_size, device=device)
    half = image_size // 2
    axes = ((0, 2), (1, 2), (0, 2), (1, 2))
    signs = ((1, 1), (1, 1), (-1, 1), (-1, 1))
    for batch_index, cloud in enumerate(points):
        lo, hi = cloud.amin(dim=0), cloud.amax(dim=0)
        center = (lo + hi) / 2
        scale = (hi - lo).amax().clamp_min(1e-6)
        normalized = ((cloud - center) / scale + 0.5).clamp(0, 1)
        color = normalized.clamp(0.15, 1.0)
        for view, ((u_axis, v_axis), (u_sign, v_sign)) in enumerate(zip(axes, signs)):
            u = normalized[:, u_axis] if u_sign > 0 else 1 - normalized[:, u_axis]
            v = normalized[:, v_axis] if v_sign > 0 else 1 - normalized[:, v_axis]
            px = (u * (half - 3) + 1).long()
            py = ((1 - v) * (half - 3) + 1).long()
            row, col = view // 2, view % 2
            linear = (py + row * half) * image_size + px + col * half
            canvas = output[batch_index].view(3, -1)
            canvas.scatter_reduce_(1, linear.unsqueeze(0).expand(3, -1), color.T, reduce='amax')
    return output


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        try:
            self.roots = json.loads(roots)
            root_type = 'obj'
        except:
            self.roots = roots.split(',')
            root_type = 'list'
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        if root_type == 'obj':
            for key, root in self.roots.items():
                self._current_root = root
                self._stats[key] = {}
                metadata = pd.DataFrame(columns=['sha256']).set_index('sha256')
                for root_key, r in root.items():
                    if root_key in {
                        'include_instances', 'exclude_instances', 'training_mode',
                        'condition_subtype', 'image_variant', 'target_id_separator',
                        'trust_include_instances', 'osm_mask_root', 'alpha_mask_root',
                    }:
                        continue
                    metadata_path = os.path.join(r, 'metadata.csv')
                    # Auxiliary roots (for example partial_xyz) need not carry
                    # their own copy of metadata.csv.
                    if os.path.isfile(metadata_path):
                        metadata = metadata.combine_first(_read_metadata_cached(metadata_path))
                if metadata.empty:
                    raise FileNotFoundError(f'No metadata.csv found in dataset root mapping {key!r}')
                self._stats[key]['Total'] = len(metadata)
                include_instances = self._read_instance_filter(root.get('include_instances'))
                exclude_instances = self._read_instance_filter(root.get('exclude_instances'))
                if include_instances is not None:
                    metadata = metadata[metadata.index.astype(str).isin(include_instances)]
                    self._stats[key]['Included by split'] = len(metadata)
                if exclude_instances is not None:
                    metadata = metadata[~metadata.index.astype(str).isin(exclude_instances)]
                    self._stats[key]['After split exclusion'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata.index.values])
                self.metadata = pd.concat([self.metadata, metadata])
            self._current_root = None
        else:
            for root in self.roots:
                self._current_root = root
                key = os.path.basename(root)
                self._stats[key] = {}
                metadata = pd.read_csv(os.path.join(root, 'metadata.csv'), dtype={'sha256': str})
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
                metadata.set_index('sha256', inplace=True)
                self.metadata = pd.concat([self.metadata, metadata])
            self._current_root = None

    @staticmethod
    def _read_instance_filter(value):
        """Return an ID set from a manifest path or an inline list."""
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            return {str(instance) for instance in value}
        if not isinstance(value, str):
            raise TypeError(f'Instance filter must be a path, string, or list; got {type(value)}')
        if os.path.isfile(value):
            with open(value, 'r') as handle:
                return {line.strip() for line in handle if line.strip()}
        return {instance.strip() for instance in value.split(',') if instance.strip()}
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(f'Error loading {instance}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, context_border_crop_px=0,
                 direct_border_crop_px=0, **kwargs):
        self.image_size = image_size
        self.context_border_crop_px = context_border_crop_px
        self.direct_border_crop_px = direct_border_crop_px
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        if self.direct_border_crop_px:
            crop = self.direct_border_crop_px
            if image.width <= 2 * crop or image.height <= 2 * crop:
                raise ValueError(
                    f'Condition image {image_path} is too small for a direct '
                    f'{crop}px border crop: size={image.size}'
                )
            image = image.crop((crop, crop, image.width - crop, image.height - crop))
            image = image.convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            pack['cond'] = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            return pack

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        if self.context_border_crop_px:
            crop = self.context_border_crop_px
            if bbox[2] - bbox[0] <= 2 * crop or bbox[3] - bbox[1] <= 2 * crop:
                raise ValueError(
                    f'Condition image {image_path} is too small for a {crop}px border crop: '
                    f'alpha bbox={bbox}'
                )
            bbox = [bbox[0] + crop, bbox[1] + crop,
                    bbox[2] - crop, bbox[3] - crop]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_hsize = hsize
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack


class PointImageConditionedMixin(ImageConditionedMixin):
    """Load both the original image condition and an aligned incomplete cloud."""

    def __init__(
        self,
        roots,
        *,
        point_cond_root: str = 'partial_xyz',
        point_sample_size: int = 16384,
        normalize_points: bool = False,
        normalize_padding: float = 1e-3,
        **kwargs,
    ):
        self.point_cond_root = point_cond_root
        self.point_sample_size = int(point_sample_size)
        self.normalize_points = normalize_points
        self.normalize_padding = float(normalize_padding)
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        current_root = getattr(self, '_current_root', None)
        # StandardDatasetBase calls this while iterating roots. Infer that root
        # from the available mapping when older datasets do not set _current_root.
        if current_root is None and isinstance(self.roots, dict) and len(self.roots) == 1:
            current_root = next(iter(self.roots.values()))
        if current_root is not None:
            metadata = metadata[
                metadata.index.astype(str).map(
                    lambda instance: find_point_condition(current_root, self.point_cond_root, instance) is not None
                )
            ]
        stats['With point condition'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        path = find_point_condition(root, self.point_cond_root, instance)
        if path is None:
            raise FileNotFoundError(f'No point condition for {instance} ({self.point_cond_root})')
        points = load_point_condition(path)
        if self.normalize_points:
            lo, hi = points.amin(dim=0), points.amax(dim=0)
            span = (hi - lo).amax().clamp_min(1e-6)
            points = (points - (lo + hi) / 2) * ((1 - 2 * self.normalize_padding) / span)
        pack['image_cond'] = pack.pop('cond')
        pack['point_cond'] = sample_point_condition(points, self.point_sample_size)
        return pack

    def visualize_point_condition(self, point_cond: torch.Tensor) -> torch.Tensor:
        return render_point_condition(point_cond)


class MultiImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, max_image_cond_view = 4, **kwargs):
        self.image_size = image_size
        self.max_image_cond_view = max_image_cond_view
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        n_views = len(metadata['frames'])
        n_sample_views = np.random.randint(1, self.max_image_cond_view+1)

        assert n_views >= n_sample_views, f'Not enough views to sample {n_sample_views} unique images.'

        sampled_views = np.random.choice(n_views, size=n_sample_views, replace=False)

        cond_images = []
        for v in sampled_views:
            frame_info = metadata['frames'][v]
            image_path = os.path.join(image_root, frame_info['file_path'])
            image = Image.open(image_path)

            alpha = np.array(image.getchannel(3))
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize
            aug_center = center
            aug_bbox = [
                int(aug_center[0] - aug_hsize),
                int(aug_center[1] - aug_hsize),
                int(aug_center[0] + aug_hsize),
                int(aug_center[1] + aug_hsize),
            ]

            img = image.crop(aug_bbox)
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = img.getchannel(3)
            img = img.convert('RGB')
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            img = img * alpha.unsqueeze(0)

            cond_images.append(img)

        pack['cond'] = [torch.stack(cond_images, dim=0)]  # (V,3,H,W)
        return pack
