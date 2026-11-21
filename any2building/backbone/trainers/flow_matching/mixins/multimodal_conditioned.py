"""DINOv3 + DGCNN conditioning with three-way modality sampling."""

from typing import Optional

import numpy as np
import torch

from .image_conditioned import ImageConditionedMixin


class MultimodalConditionedMixin(ImageConditionedMixin):
    """
    Select point-only, image-only, or concatenated tokens during training.

    The default probabilities are exactly 1/3 each. Classifier-free dropout is
    applied afterwards by ``ClassifierFreeGuidanceMixin``.
    """

    MODES = ('point', 'image', 'rough', 'both')

    def __init__(
        self,
        *args,
        condition_mode_probs=(0.25, 0.25, 0.25, 0.25),
        inference_condition_mode: str = 'both',
        snapshot_condition_modes=('point', 'image', 'both'),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        probs = np.asarray(condition_mode_probs, dtype=np.float64)
        if probs.shape != (4,) or np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
            raise ValueError('condition_mode_probs must be four non-negative values summing to one')
        if inference_condition_mode not in self.MODES:
            raise ValueError(f'Unknown inference_condition_mode: {inference_condition_mode}')
        snapshot_condition_modes = tuple(snapshot_condition_modes)
        if not snapshot_condition_modes or any(mode not in self.MODES for mode in snapshot_condition_modes):
            raise ValueError(f'snapshot_condition_modes must contain values from {self.MODES}')
        self.condition_mode_probs = probs
        self.inference_condition_mode = inference_condition_mode
        self.snapshot_condition_modes = snapshot_condition_modes

    def _point_encoder(self, training: bool):
        models = self.training_models if training else self.models
        if 'point_cond_encoder' not in models:
            raise KeyError('Config must define models.point_cond_encoder')
        return models['point_cond_encoder']

    def encode_points(self, points: torch.Tensor, point_mask: Optional[torch.Tensor], training: bool):
        return self._point_encoder(training)(points, point_mask=point_mask)

    def _condition_fuser(self, training: bool):
        models = self.training_models if training else self.models
        return models.get('condition_fuser')

    def _encode_multimodal(
        self,
        image_cond,
        point_cond,
        point_mask=None,
        *,
        mode: str,
        training: bool,
    ) -> torch.Tensor:
        if isinstance(mode, (list, tuple)):
            unique = set(mode)
            if len(unique) != 1:
                raise ValueError(f'Each training batch must be homogeneous, got modes={unique}')
            mode = next(iter(unique))
        if mode not in self.MODES:
            raise ValueError(f'condition_mode must be one of {self.MODES}, got {mode!r}')
        image_tokens = None
        point_tokens = None
        if mode in ('image', 'rough', 'both'):
            if image_cond is None:
                raise ValueError(f'{mode} conditioning requires image_cond')
            image_tokens = self.encode_image(image_cond)
        if mode in ('point', 'both'):
            if point_cond is None:
                raise ValueError(f'{mode} conditioning requires point_cond')
            point_tokens = self.encode_points(point_cond, point_mask, training)
        if mode == 'both':
            fuser = self._condition_fuser(training)
            if fuser is not None:
                out = fuser(point_tokens, image_tokens)
                # DDP wrapper does not proxy arbitrary attributes; read the inner module.
                inner = getattr(fuser, 'module', fuser)
                self._last_occ_logits = getattr(inner, 'last_occ_logits', None)
                return out
            return torch.cat([image_tokens, point_tokens], dim=1)
        return point_tokens if mode == 'point' else image_tokens

    def get_cond(
        self,
        cond=None,
        image_cond=None,
        point_cond=None,
        point_mask=None,
        condition_mode=None,
        **kwargs,
    ):
        if condition_mode is None:
            # Keep every DDP rank and every accumulation microbatch on the
            # same branch for the whole optimizer step.
            step = int(getattr(self, 'step', 0))
            rng = np.random.default_rng(step ^ 0x4D554C54)
            mode = rng.choice(self.MODES, p=self.condition_mode_probs)
        else:
            mode = condition_mode
        tokens = self._encode_multimodal(
            image_cond, point_cond, point_mask, mode=mode, training=True
        )
        kwargs['neg_cond'] = torch.zeros_like(tokens)
        return super(ImageConditionedMixin, self).get_cond(tokens, **kwargs)

    def get_inference_cond(
        self,
        cond=None,
        image_cond=None,
        point_cond=None,
        point_mask=None,
        condition_mode=None,
        **kwargs,
    ):
        mode = condition_mode or self.inference_condition_mode
        tokens = self._encode_multimodal(
            image_cond, point_cond, point_mask, mode=mode, training=False
        )
        kwargs.pop('rough_prior', None)
        kwargs.pop('occ8', None)
        kwargs['neg_cond'] = torch.zeros_like(tokens)
        return super(ImageConditionedMixin, self).get_inference_cond(tokens, **kwargs)

    def vis_cond(self, image_cond=None, point_cond=None, **kwargs):
        visualizations = {}
        point_vis = None
        if point_cond is not None and hasattr(self.dataset, 'visualize_point_condition'):
            point_vis = self.dataset.visualize_point_condition(point_cond)
        if image_cond is not None and point_vis is not None:
            visualizations['condition_pairs'] = {
                'value': torch.cat([point_vis, image_cond], dim=-1),
                'type': 'image',
            }
        elif image_cond is not None:
            visualizations['image'] = {'value': image_cond, 'type': 'image'}
        elif point_vis is not None:
            visualizations['point_cloud'] = {'value': point_vis, 'type': 'image'}
        return visualizations
