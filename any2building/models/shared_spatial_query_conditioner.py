"""Shared Spatial Query Conditioner, Sec. 3.2, Eqs. (3)-(7).

State-dictionary keys are preserved for released-weight compatibility. The 512
learned completion queries in the base point-image fusion are distinct from
the 16^3 coordinate-initialized shared spatial queries.
"""
import copy
import torch
from torch import nn
from .completion_token_fusion import CompletionTokenFusion

class SharedSpatialQueryConditioner(CompletionTokenFusion):
    def __init__(self, channels=1024, num_heads=16, dropout=0., anchor_grid=8,
                 rough_grid=16, prior_scale=.2, prior_dropout=.1, algorithm='aux'):
        if rough_grid != 16 or algorithm != 'aux':
            raise ValueError('This release implements the paper model: 16^3 queries, auxiliary occupancy.')
        super().__init__(channels, num_heads, dropout, anchor_grid, use_occ_head=False)
        if algorithm not in ('aux', 'gated'):
            raise ValueError(algorithm)
        self.algorithm = algorithm
        channels = self.anchor_pos.shape[-1]
        g = self.anchor_grid
        lin = (torch.arange(g).float() + .5) / g - .5
        # Decoder tensor axes and flatten order are x,y,z (last axis fastest).
        coords = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), -1)
        self.register_buffer('spatial_coords', coords.reshape(1, -1, 3))
        self.coord_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, channels))
        self.spatial_norm = nn.LayerNorm(channels)
        self.spatial_occ = nn.Linear(channels, 1)
        self.spatial_read = copy.deepcopy(self.completion_to_point)
        nn.init.zeros_(self.spatial_read.out_proj.weight)
        nn.init.zeros_(self.spatial_read.out_proj.bias)
        self.occupancy_logits = None
        c = channels
        self.global_point_norm = nn.LayerNorm(c)
        self.global_image_norm = nn.LayerNorm(c)
        self.global_fusion = nn.Sequential(nn.Linear(2*c+2, c), nn.GELU(), nn.Linear(c, c))
        self.query_init = nn.Sequential(nn.Linear(2*c, c), nn.GELU(), nn.Linear(c, c))
        self.read_fusion = nn.Sequential(nn.Linear(2*c+2, c), nn.GELU(), nn.Linear(c, c))
        self.query_self_norm = nn.LayerNorm(c)
        self.query_self = nn.MultiheadAttention(c, num_heads, dropout=dropout, batch_first=True)
        self.query_ffn_norm = nn.LayerNorm(c)
        self.query_ffn = nn.Sequential(nn.Linear(c, 2*c), nn.GELU(), nn.Linear(2*c, c))
        # Start the new inter-query residuals as identity.
        nn.init.zeros_(self.query_self.out_proj.weight)
        nn.init.zeros_(self.query_self.out_proj.bias)
        nn.init.zeros_(self.query_ffn[-1].weight)
        nn.init.zeros_(self.query_ffn[-1].bias)
        self.prior_scale=float(prior_scale)
        self.prior_dropout=float(prior_dropout)
        self.prior_pos_norm=nn.LayerNorm(channels,elementwise_affine=False)
        # Bias-free: an absent prior is EXACTLY a zero residual.
        self.prior_features=nn.Sequential(nn.Linear(3,128,bias=False),nn.GELU(),nn.Linear(128,channels,bias=False))
        self._active_prior=None
        self.rough_grid = int(rough_grid)
        a = (torch.arange(self.rough_grid).float() + .5) / self.rough_grid - .5
        self.register_buffer('rough_coords', torch.stack(torch.meshgrid(a,a,a,indexing='ij'),-1).reshape(1,-1,3), persistent=False)

    def initial_queries(self, point_tokens=None, image_tokens=None):
        ref = point_tokens if point_tokens is not None else image_tokens
        if ref is None:
            raise ValueError('At least one observation modality is required')
        b, _, c = ref.shape
        gp = self.global_point_norm(point_tokens).mean(1) if point_tokens is not None else ref.new_zeros(b,c)
        gi = self.global_image_norm(image_tokens).mean(1) if image_tokens is not None else ref.new_zeros(b,c)
        flags = ref.new_tensor([point_tokens is not None, image_tokens is not None]).expand(b,2)
        g = self.global_fusion(torch.cat([gp,gi,flags],-1))
        raw_e = self.coord_embed(self.rough_coords)
        e = raw_e.to(g.dtype).expand(b,-1,-1)
        q = self.query_init(torch.cat([e,g[:,None].expand_as(e)],-1))
        if self._active_prior is not None:
            if point_tokens is not None or image_tokens is None:
                raise ValueError('The released prior mode uses image + prior')
            p = self._active_prior.to(q.dtype)
            if p.shape != (b,4096,3):
                raise ValueError(f'Expected [B,4096,3] prior descriptors, got {p.shape}')
            if self.training and self.prior_dropout:
                p = p * (torch.rand((b,1,1),device=p.device) >= self.prior_dropout)
            q = q + self.prior_scale * (p[:,:,:1] * self.prior_pos_norm(raw_e).to(q.dtype) + self.prior_features(p))
        return q, flags

    def _forward(self, point_tokens=None, image_tokens=None):
        q0, flags = self.initial_queries(point_tokens, image_tokens)
        if point_tokens is not None and image_tokens is not None:
            # Retain inherited original B, including its legacy A; the NEW
            # spatial branch never uses A to initialize its queries.
            base = CompletionTokenFusion.forward(self, point_tokens, image_tokens)
        else:
            base = point_tokens if point_tokens is not None else image_tokens
        query = self.anchor_query_norm(q0)
        rp = torch.zeros_like(q0)
        ri = torch.zeros_like(q0)
        if point_tokens is not None:
            rp, _ = self.anchor_point_read(query, point_tokens, point_tokens, need_weights=False)
        if image_tokens is not None:
            image = self.image_norm(image_tokens)
            ri, _ = self.anchor_image_read(query, image, image, need_weights=False)
        mask = flags[:,None].expand(-1,q0.shape[1],-1)
        q = q0 + self.read_fusion(torch.cat([rp,ri,mask],-1))
        h = self.query_self_norm(q)
        q = q + self.query_self(h,h,h,need_weights=False)[0]
        q = self.spatial_norm(q + self.query_ffn(self.query_ffn_norm(q)))
        self.occupancy_logits = self.spatial_occ(q).squeeze(-1)
        delta, _ = self.spatial_read(self.completion_read_norm(base),q,q,need_weights=False)
        # DDP must also see modules absent from a single-modality forward.
        zero = sum(p.reshape(-1)[0]*0 for p in self.parameters())
        return torch.cat([base + delta + zero, q], dim=1)

    def forward(self, point_tokens=None, image_tokens=None, query_prior=None):
        self._active_prior = query_prior
        try:
            return self._forward(point_tokens, image_tokens)
        finally:
            self._active_prior = None
