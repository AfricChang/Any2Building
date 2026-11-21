"""Flow DiT with aligned spatial prompts (Sec. 3.3, Eq. 8)."""
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ..backbone.models.sparse_structure_flow import SparseStructureFlowModel
from ..backbone.modules.utils import manual_cast
class SpatialPromptDiT(SparseStructureFlowModel):
    def __init__(self,rough_grid=16,control_blocks=(0,10,20,29),control_scale=1.,main_tokens=1029,**kwargs):
        super().__init__(**kwargs)
        self.rough_grid=int(rough_grid);self.main_tokens=int(main_tokens)
        self.control_scale=float(control_scale)
        self.control_projections=nn.ModuleDict({str(i):nn.Linear(self.cond_channels,self.model_channels,bias=False) for i in control_blocks})
        for layer in self.control_projections.values():nn.init.zeros_(layer.weight)

    def forward(self,x,t,cond):
        self.main_tokens = cond.shape[1] - 4096
        if self.main_tokens <= 0:
            raise ValueError("QE tokens followed by 4096 spatial prompts are required")
        if cond.shape[1]<=self.main_tokens:
            # Zero graph links keep DDP reductions consistent for non-Rough batches.
            zero=sum(p.reshape(-1)[0]*0 for p in self.control_projections.parameters())
            return super().forward(x,t,cond)+zero
        if cond.shape[1]!=self.main_tokens+self.rough_grid**3:raise ValueError(f'Unknown packed condition {cond.shape}')
        cond,q=cond[:,:self.main_tokens],cond[:,self.main_tokens:]
        q=F.layer_norm(q,(q.shape[-1],)) # no learned bias: null CFG stays exactly zero
        if self.rough_grid!=self.resolution:
            q=q.transpose(1,2).reshape(q.shape[0],-1,*([self.rough_grid]*3))
            q=F.interpolate(q,size=(self.resolution,)*3,mode='trilinear',align_corners=False).flatten(2).transpose(1,2)
        h=self.input_layer(x.flatten(2).transpose(1,2).contiguous())
        if self.pe_mode=='ape':h=h+self.pos_emb[None]
        te=self.t_embedder(t)
        if self.share_mod:te=self.adaLN_modulation(te)
        te=manual_cast(te,self.dtype);h=manual_cast(h,self.dtype);cond=manual_cast(cond,self.dtype)
        for i,block in enumerate(self.blocks):
            if str(i) in self.control_projections:
                h=h+self.control_scale*self.control_projections[str(i)](q).to(h.dtype)
            h=block(h,te,cond,self.rope_phases)
        h=self.out_layer(F.layer_norm(manual_cast(h,x.dtype),(h.shape[-1],)))
        return h.transpose(1,2).reshape(x.shape[0],self.out_channels,*([self.resolution]*3)).contiguous()
