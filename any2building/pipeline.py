"""Any2Building structure inference from point/image observations."""
import json
from pathlib import Path
import numpy as np
import torch
from . import models
from .backbone import models as primitives
from .backbone.pipelines.samplers import FlowEulerCfgSampler
from .io import load_points, load_image, load_occupancy, prior_descriptors

WEIGHT_NAMES = {
    'denoiser': 'spatial_prompt_dit',
    'point_cond_encoder': 'point_cloud_tokenizer',
    'condition_fuser': 'shared_spatial_query_conditioner',
}


class Any2BuildingPipeline:
    def __init__(self, root, variant='ema', device='cuda', image_enabled=True):
        if variant != 'ema':
            raise ValueError('This release provides only EMA weights')
        self.root = Path(root)
        self.device = torch.device(device)
        self.config = json.loads((self.root/'configs/any2building.json').read_text())
        self.models = {}
        for key,spec in self.config['models'].items():
            net = getattr(models,spec['name'])(**spec['args'])
            path = self.root/f'weights/{WEIGHT_NAMES[key]}.pt'
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True, mmap=True), strict=True)
            self.models[key] = net.to(self.device).eval().requires_grad_(False)
        self.decoder = primitives.from_pretrained(str(self.root/'weights/structure_decoder/model')).to(self.device).eval().requires_grad_(False)
        self.image_tokenizer = None
        if image_enabled:
            from .models.image_tokenizer import ImageTokenizer
            self.image_tokenizer = ImageTokenizer(str(self.root/'weights/image_tokenizer'))
            self.image_tokenizer.to(self.device)
        self.sampler = FlowEulerCfgSampler(sigma_min=1e-5)

    @torch.no_grad()
    def __call__(self, *, mode, points=None, image=None, prior=None, seed=42, steps=50, guidance=3.):
        if mode not in ('point','image','both','image_prior'):
            raise ValueError(mode)
        if mode in ('point','both') and points is None:
            raise ValueError(f'{mode} requires points')
        if mode != 'point' and image is None:
            raise ValueError(f'{mode} requires an image')
        if mode == 'image_prior' and prior is None:
            raise ValueError('image_prior requires a surface occupancy prior')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        p = torch.from_numpy(load_points(points))[None].to(self.device) if mode in ('point','both') else None
        i = torch.from_numpy(load_image(image))[None].to(self.device) if mode != 'point' else None
        r = torch.from_numpy(prior_descriptors(load_occupancy(prior)))[None].to(self.device) if mode == 'image_prior' else None
        if p is not None and p.abs().max() > .501:
            raise ValueError('Points must already share the normalized [-.5,.5] building frame; see docs/DATA.md')
        noise = torch.randn(1,8,16,16,16, device=self.device,
                            generator=torch.Generator(device=self.device).manual_seed(seed))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pt = self.models['point_cond_encoder'](p) if p is not None else None
            im = self.image_tokenizer(i) if i is not None else None
            cond = self.models['condition_fuser'](pt, im, query_prior=r)
            query = self.models['condition_fuser'].occupancy_logits.float().cpu().numpy()[0]
            latent = self.sampler.sample(self.models['denoiser'], noise=noise, cond=cond,
                neg_cond=torch.zeros_like(cond), steps=steps, guidance_strength=guidance, verbose=False).samples
        occupancy = (self.decoder(latent.float())[0,0] > 0).cpu().numpy()
        if not occupancy.any():
            raise RuntimeError('The predicted structure is empty')
        return dict(occupancy=occupancy, query_logits=query, latent=latent[0].float().cpu().numpy())

