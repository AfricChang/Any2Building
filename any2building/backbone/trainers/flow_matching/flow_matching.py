from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin
from .mixins.multimodal_conditioned import MultimodalConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        rough_noise_mode (str): How the rough SS prior enters the noise in
            'rough' condition mode. 'cosine': per-channel standardized prior
            mixed with pure noise as cos(theta)*prior + sin(theta)*noise with
            theta ~ U[0, pi/2] (variance preserved). 'add_norm': per-channel
            standardized prior added directly to pure N(0,1) noise (variance
            not preserved).
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        rough_inference_lambda: float = 0.35,
        rough_noise_mode: str = 'cosine',
        point_preservation_weight: float = 0.0,
        point_preservation_margin_weight: float = 0.0,
        point_preservation_margin: float = 0.0,
        anchor_occ_weight: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.rough_inference_lambda = float(rough_inference_lambda)
        if rough_noise_mode not in ('cosine', 'add_norm'):
            raise ValueError(f"Unknown rough_noise_mode: {rough_noise_mode!r}")
        self.rough_noise_mode = rough_noise_mode
        self.point_preservation_weight = float(point_preservation_weight)
        self.point_preservation_margin_weight = float(point_preservation_margin_weight)
        self.point_preservation_margin = float(point_preservation_margin)
        self.anchor_occ_weight = float(anchor_occ_weight)

    def _rough_noise(self, rough_prior: torch.Tensor, pure_noise: torch.Tensor, training: bool) -> torch.Tensor:
        """Build the effective noise from a rough SS prior and pure noise."""
        prior = rough_prior.to(pure_noise.device, dtype=pure_noise.dtype)
        # Normalize every latent channel over its spatial support.
        prior_mean = prior.mean(dim=(2, 3, 4), keepdim=True)
        prior_std = prior.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        prior = (prior - prior_mean) / prior_std
        if self.rough_noise_mode == 'add_norm':
            return pure_noise + prior
        # 'cosine': the cos^2+sin^2 identity preserves unit variance.
        if training:
            geometric_lambda = torch.rand(prior.shape[0], device=prior.device)
            angle = geometric_lambda.view(-1, 1, 1, 1, 1) * (torch.pi / 2)
        else:
            angle = self.rough_inference_lambda * (torch.pi / 2)
        return torch.cos(torch.as_tensor(angle, device=prior.device)) * prior + \
               torch.sin(torch.as_tensor(angle, device=prior.device)) * pure_noise

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        rough_prior=None,
        condition_mode=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        mode = condition_mode
        if isinstance(mode, (list, tuple)):
            unique = set(mode)
            if len(unique) != 1:
                raise ValueError(f'Expected a homogeneous condition batch, got {unique}')
            mode = next(iter(unique))
        if mode == 'rough':
            if rough_prior is None:
                raise ValueError('rough mode requires rough_prior')
            noise = self._rough_noise(rough_prior, noise, training=True)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        # Keep raw point inputs for the optional paired point teacher before
        # multimodal encoders consume them.
        raw_point_cond = kwargs.get('point_cond')
        raw_point_mask = kwargs.get('point_mask')
        raw_occ8 = kwargs.get('occ8')
        cond = self.get_cond(cond, condition_mode=condition_mode, **kwargs)
        # Multimodal raw inputs are consumed by the condition encoders and are
        # not valid keyword arguments for the denoiser backbone.
        for key in ('image_cond', 'point_cond', 'point_mask', 'condition_mode', 'rough_prior', 'occ8'):
            kwargs.pop(key, None)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # Variant C: anchor occupancy supervision (both-mode batches only).
        # Target is the GT 8^3 occupancy (dataset pack['occ8']) aligned with the
        # fuser's fixed anchor ordering; logits are stashed by the fuser.
        occ_target = raw_occ8
        occ_logits = getattr(self, '_last_occ_logits', None)
        if mode == 'both' and occ_target is not None and occ_logits is not None:
            occ = F.binary_cross_entropy_with_logits(
                occ_logits.reshape(occ_logits.shape[0], -1),
                occ_target.reshape(occ_target.shape[0], -1).float(),
            )
            terms['anchor_occ'] = occ
            # Keep the occ head in the autograd graph even at zero weight
            # (DDP marks all fuser params as used; zero grads are fine).
            terms['loss'] = terms['loss'] + self.anchor_occ_weight * occ

        # On a both-mode batch, compare against the point-only prediction at
        # exactly the same x_t/t. The teacher is detached, so this regularizes
        # fusion without moving the point branch toward a weaker fused result.
        if (mode == 'both' and raw_point_cond is not None and
                (self.point_preservation_weight > 0 or
                 self.point_preservation_margin_weight > 0)):
            with torch.no_grad():
                point_tokens = self._encode_multimodal(
                    None, raw_point_cond, raw_point_mask,
                    mode='point', training=True,
                )
                point_pred = self.training_models['denoiser'](
                    x_t, t * 1000, point_tokens, **kwargs
                )
                point_mse = F.mse_loss(point_pred, target)
            preserve = F.mse_loss(pred, point_pred)
            margin = F.relu(terms['mse'] - point_mse + self.point_preservation_margin)
            terms['point_teacher_mse'] = point_mse
            terms['point_preserve'] = preserve
            terms['point_margin'] = margin
            terms['loss'] = (terms['loss'] + self.point_preservation_weight * preserve
                             + self.point_preservation_margin_weight * margin)

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        if hasattr(self.dataset, 'mode_subtype_indices'):
            return self._run_fourway_snapshot(num_samples, batch_size, verbose)

        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        snapshot_modes = getattr(self, 'snapshot_condition_modes', None)
        modes = (None,) if snapshot_modes is None else tuple(snapshot_modes)
        sample_by_mode = {
            ('sample' if mode is None else f'sample_{mode}'): []
            for mode in modes
        }
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            for mode in modes:
                key = 'sample' if mode is None else f'sample_{mode}'
                if mode is None:
                    args = self.get_inference_cond(**data)
                else:
                    args = self.get_inference_cond(**data, condition_mode=mode)
                res = sampler.sample(
                    self.models['denoiser'],
                    # Use identical noise for a fair modality comparison.
                    noise=noise,
                    **args,
                    steps=50, guidance_strength=3.0, verbose=verbose,
                )
                sample_by_mode[key].append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
        }
        for key, values in sample_by_mode.items():
            sample_dict[key] = {'value': torch.cat(values, dim=0), 'type': 'sample'}
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict

    @torch.no_grad()
    def _run_fourway_snapshot(self, num_samples, batch_size, verbose=False):
        """Render paired inputs, GT, and all four inference modes.

        We deliberately select BuildingWorld ``both`` examples carrying image,
        point cloud, and OSM prior so every prediction in a row uses identical
        GT and conditions.
        """
        both = []
        for subtype_values in self.dataset.mode_subtype_indices.get('both', {}).values():
            both.extend(
                index for index in subtype_values
                if self.dataset.instances[index][0].get('rough_ss_latent')
            )
        if hasattr(self.dataset,'snapshot_indices'):
            pool=self.dataset.snapshot_indices()
        elif both:
            pool = both
        else:
            # Subset-mode training (e.g. single-paradigm ablations): render
            # examples from whichever snapshot modes actually have data.
            snapshot_modes = getattr(self, 'snapshot_condition_modes', None) or self.MODES
            pool = []
            for mode in ('point', 'image', 'rough', 'both'):
                if mode not in snapshot_modes:
                    continue
                for subtype_values in self.dataset.mode_subtype_indices.get(mode, {}).values():
                    pool.extend(subtype_values)
        if not pool:
            raise RuntimeError('No snapshot examples available for the configured modes')
        render_modes = [
            mode for mode in ('point', 'image', 'rough', 'both')
            if mode in getattr(self, 'snapshot_condition_modes', self.MODES)
            and self.dataset.mode_subtype_indices.get(mode)
        ]
        generator = torch.Generator().manual_seed(int(getattr(self, 'step', 0)) + 0x534E4150)
        order = torch.randperm(len(pool), generator=generator).tolist()
        # Each distributed rank renders a disjoint slice so the gathered
        # 16-example contact sheet contains 16 distinct condition/GT pairs.
        offset = int(getattr(self, 'rank', 0)) * num_samples
        selected = [pool[order[(offset + i) % len(order)]] for i in range(num_samples)]
        sampler = self.get_sampler()
        outputs = {f'sample_{mode}': [] for mode in render_modes}
        sample_gt, cond_vis = [], []
        for start in range(0, len(selected), batch_size):
            if hasattr(self.dataset,'get_snapshot_instance'):
                items = [self.dataset.get_snapshot_instance(index) for index in selected[start:start + batch_size]]
            else:
                items = [self.dataset[index] for index in selected[start:start + batch_size]]
            keys = items[0].keys()
            data = {
                key: (torch.stack([item[key] for item in items])
                      if isinstance(items[0][key], torch.Tensor)
                      else [item[key] for item in items])
                for key in keys
            }
            data = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in data.items()}
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            inference_data = {key: value for key, value in data.items() if key not in ('x_0', 'condition_mode')}
            pure_noise = torch.randn_like(data['x_0'])
            for mode in render_modes:
                noise = pure_noise
                if mode == 'rough':
                    noise = self._rough_noise(inference_data['rough_prior'], pure_noise, training=False)
                args = self.get_inference_cond(**inference_data, condition_mode=mode)
                result = sampler.sample(
                    self.models['denoiser'], noise=noise, **args,
                    steps=50, guidance_strength=3.0, verbose=verbose,
                )
                outputs[f'sample_{mode}'].append(result.samples)
        result = {'sample_gt': {'value': torch.cat(sample_gt), 'type': 'sample'}}
        result.update({key: {'value': torch.cat(value), 'type': 'sample'} for key, value in outputs.items()})
        result.update(dict_reduce(cond_vis, None, {
            'value': lambda value: torch.cat(value, dim=0),
            'type': lambda value: value[0],
        }))
        return result

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass


class MultimodalConditionedFlowMatchingCFGTrainer(MultimodalConditionedMixin, FlowMatchingCFGTrainer):
    """Dense flow trainer with DGCNN/DINOv3/concatenated conditioning."""
    pass
