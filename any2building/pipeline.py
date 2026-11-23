"""Any2Building structure inference from point/image observations."""
from contextlib import contextmanager
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


def _resolve_memory_strategy(device, dtype, offload):
    if device.type != 'cuda':
        return ('float32' if dtype == 'auto' else dtype), ('none' if offload == 'auto' else offload)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    if offload == 'auto':
        if total_gb >= 12.0 and free_gb >= 8.0:
            offload = 'none'
        elif total_gb > 8.0 and free_gb >= 4.5:
            offload = 'model'
        else:
            offload = 'block'
    if dtype == 'auto':
        dtype = 'bfloat16' if (offload != 'none' or total_gb < 12.0) else 'float32'
    return dtype, offload


def _install_block_offload(denoiser, device):
    handles = []
    for block in denoiser.blocks:
        cpu_params = []
        for p in block.parameters():
            p.data = p.data.pin_memory()
            cpu_params.append((p, p.data))
        cpu_buffers = []
        for mod in block.modules():
            for name, buf in mod._buffers.items():
                if buf is not None:
                    pinned = buf.pin_memory()
                    mod._buffers[name] = pinned
                    cpu_buffers.append((mod, name, pinned))

        def _pre_hook(_module, _inputs, _params=cpu_params, _buffers=cpu_buffers):
            for p, cpu_data in _params:
                p.data = cpu_data.to(device, non_blocking=True)
            for mod, name, cpu_buf in _buffers:
                mod._buffers[name] = cpu_buf.to(device, non_blocking=True)

        def _post_hook(_module, _inputs, _output, _params=cpu_params, _buffers=cpu_buffers):
            for p, cpu_data in _params:
                p.data = cpu_data
            for mod, name, cpu_buf in _buffers:
                mod._buffers[name] = cpu_buf

        handles.append(block.register_forward_pre_hook(_pre_hook))
        handles.append(block.register_forward_hook(_post_hook))
    return handles


def _move_denoiser_shell(denoiser, device):
    for name, child in denoiser.named_children():
        if name != 'blocks':
            child.to(device)
    for name, buf in denoiser._buffers.items():
        if buf is not None:
            denoiser._buffers[name] = buf.to(device)
    for name, param in denoiser._parameters.items():
        if param is not None:
            param.data = param.data.to(device)


class Any2BuildingPipeline:
    def __init__(self, root, variant='ema', device='cuda', image_enabled=True, dtype='auto', offload='auto'):
        if variant != 'ema':
            raise ValueError('This release provides only EMA weights')
        self.root = Path(root)
        self.device = torch.device(device)
        self.weight_dtype, self.offload = _resolve_memory_strategy(self.device, dtype, offload)
        if self.offload not in ('none', 'model', 'block'):
            raise ValueError(f'Unsupported offload mode: {self.offload}')
        if self.weight_dtype not in ('float32', 'bfloat16'):
            raise ValueError(f'Unsupported weight dtype: {self.weight_dtype}')
        print(f'[PIPELINE] device={self.device} weight_dtype={self.weight_dtype} offload={self.offload}', flush=True)

        self.config = json.loads((self.root/'configs/any2building.json').read_text())
        self.models = {}
        self._block_offload_handles = []
        target_device = self.device if self.offload == 'none' else torch.device('cpu')

        for key, spec in self.config['models'].items():
            net = getattr(models, spec['name'])(**spec['args'])
            path = self.root/f'weights/{WEIGHT_NAMES[key]}.pt'
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True, mmap=True), strict=True)
            net = net.eval().requires_grad_(False)
            if key == 'denoiser':
                for block in net.blocks:
                    block.use_checkpoint = False
                if self.weight_dtype == 'bfloat16':
                    net.convert_to(torch.bfloat16)
                    net.control_projections.to(torch.bfloat16)
            self.models[key] = net.to(target_device)

        if self.offload == 'block':
            self._block_offload_handles = _install_block_offload(self.models['denoiser'], self.device)

        self.decoder = primitives.from_pretrained(str(self.root/'weights/structure_decoder/model')).eval().requires_grad_(False)
        self.decoder = self.decoder.to(target_device)
        self.image_tokenizer = None
        if image_enabled:
            from .models.image_tokenizer import ImageTokenizer
            self.image_tokenizer = ImageTokenizer(str(self.root/'weights/image_tokenizer'))
            if self.weight_dtype == 'bfloat16':
                self.image_tokenizer.model.to(torch.bfloat16)
            self.image_tokenizer.to(target_device)
        self.sampler = FlowEulerCfgSampler(sigma_min=1e-5)

    @contextmanager
    def _use_module(self, module, *, block_stream=False):
        if self.offload == 'none':
            yield module
            return
        if block_stream:
            _move_denoiser_shell(module, self.device)
        else:
            module.to(self.device)
        try:
            yield module
        finally:
            if block_stream:
                _move_denoiser_shell(module, torch.device('cpu'))
            else:
                module.to('cpu')
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, *, mode, points=None, image=None, prior=None, seed=42, steps=50, guidance=3., verbose=False):
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
            if p is not None:
                with self._use_module(self.models['point_cond_encoder']) as enc:
                    pt = enc(p)
                del p
            else:
                pt = None
            if i is not None:
                with self._use_module(self.image_tokenizer) as img_enc:
                    im = img_enc(i)
                del i
            else:
                im = None
            with self._use_module(self.models['condition_fuser']) as fuser:
                cond = fuser(pt, im, query_prior=r)
                query = fuser.occupancy_logits.float().cpu().numpy()[0]
            del pt, im, r
            with self._use_module(self.models['denoiser'], block_stream=(self.offload == 'block')) as denoiser:
                latent = self.sampler.sample(denoiser, noise=noise, cond=cond,
                    neg_cond=torch.zeros_like(cond), steps=steps, guidance_strength=guidance, verbose=verbose).samples
            del cond, noise
        with self._use_module(self.decoder) as decoder:
            occupancy = (decoder(latent.float())[0,0] > 0).cpu().numpy()
        if not occupancy.any():
            raise RuntimeError('The predicted structure is empty')
        return dict(occupancy=occupancy, query_logits=query, latent=latent[0].float().cpu().numpy())

