"""Any2Building structure training, Sec. 3.3, Eq. (10)."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..backbone.trainers.flow_matching.flow_matching import MultimodalConditionedFlowMatchingCFGTrainer

class Any2BuildingTrainer(MultimodalConditionedFlowMatchingCFGTrainer):
    def __init__(self, *args, query_occ_weight=.1, output_geometry_weight=.02,
                 output_geometry_warmup=2000, geometry_start_step=0, **kwargs):
        self.query_occ_weight = float(query_occ_weight)
        self.output_geometry_weight = float(output_geometry_weight)
        self.output_geometry_warmup = int(output_geometry_warmup)
        self.geometry_start_step = int(geometry_start_step)
        self._occ_decoder = None
        self._batch_query_prior = None
        super().__init__(*args, **kwargs)

    def _rough_noise(self, rough_prior, pure_noise, training):
        if rough_prior.shape[1:] != (4096,3):
            raise ValueError('Prior descriptors must have shape [B,4096,3]')
        return pure_noise

    def _encode_multimodal(self,image_cond,point_cond,point_mask=None,*,mode,training):
        if isinstance(mode,(list,tuple)):
            if len(set(mode))!=1:raise ValueError('Mixed modality batch')
            mode=mode[0]
        if mode not in ('point','image','both','rough'):raise ValueError(mode)
        image=self.encode_image(image_cond) if mode in ('image','both','rough') else None
        point=self.encode_points(point_cond,point_mask,training) if mode in ('point','both') else None
        prior=getattr(self,'_batch_query_prior',None) if mode=='rough' else None
        if mode=='rough' and prior is None:raise ValueError('Rough requires direct geometry query_prior')
        fuser=self._condition_fuser(training)
        out=fuser(point,image,query_prior=prior)
        inner=getattr(fuser,'module',fuser)
        self._query_logits=inner.occupancy_logits
        self._last_occ_logits=None
        self._query_mode=mode
        return out

    @torch.no_grad()
    def occupancy_target(self,x_0):
        if self._occ_decoder is None:
            from ..backbone import models
            self._occ_decoder=models.from_pretrained(self.dataset.pretrained_ss_dec).to(x_0.device).eval().requires_grad_(False)
        z=x_0
        if self.dataset.normalization is not None:z=z*self.dataset.std.to(z.device)+self.dataset.mean.to(z.device)
        logits=self._occ_decoder(z)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite GT occupancy')
        self._surface_target=(logits>0).float()
        if not self._surface_target.flatten(1).any(1).all():raise ValueError('Empty GT surface')
        return F.adaptive_max_pool3d(self._surface_target,16).flatten(1)

    def geometry_loss(self,xt,t,pred,target):
        t=t.float()/1000
        # Algebraically exact for this repository's sigma_min flow convention.
        b=self.sigma_min+(1-self.sigma_min)*t
        clean=(1-self.sigma_min)*xt-b[:,None,None,None,None]*pred.float()
        if self.dataset.normalization is not None:
            clean=clean*self.dataset.std.to(clean.device)+self.dataset.mean.to(clean.device)
        logits=checkpoint(self._occ_decoder,clean,use_reentrant=False).float()
        frac=target.flatten(1).mean(1)
        pw=((1-frac)/frac.clamp_min(1e-4)).clamp(1,10)[:,None,None,None,None]
        bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=pw,reduction='none').flatten(1).mean(1)
        prob=logits.sigmoid()
        dice=1-(2*(prob*target).flatten(1).sum(1)+1)/(prob.flatten(1).sum(1)+target.flatten(1).sum(1)+1)
        # High-noise x0 estimates are unreliable. Do not divide by tiny weights.
        weights=(1-t).square()*(t<=.8)
        return ((bce+dice)*weights).mean(),bce.mean(),dice.mean(),weights.mean()

    def training_losses(self, x_0, query_prior=None, rough_prior=None, **kwargs):
        if rough_prior is not None:
            raise ValueError('Use geometric prior descriptors, never prior latents as noise')
        target = self.occupancy_target(x_0)
        self._batch_query_prior = query_prior
        captured = []
        hook = self.training_models['denoiser'].register_forward_hook(
            lambda module,args,out: captured.append((args[0],args[1],out)))
        try:
            terms,status = super().training_losses(x_0, rough_prior=query_prior, **kwargs)
        finally:
            hook.remove()
            self._batch_query_prior = None
        bce = F.binary_cross_entropy_with_logits(self._query_logits.float(),target)
        terms['loss'] = terms['loss'] + self.query_occ_weight*bce
        terms['query_occ'] = bce
        terms['query_occ_'+self._query_mode] = bce
        if len(captured) != 1:
            raise RuntimeError('Expected one denoiser pass')
        aux,output_bce,dice,coverage = self.geometry_loss(*captured[0],self._surface_target)
        warmup = min(1.,max(0.,(self.step-self.geometry_start_step+1)/max(1,self.output_geometry_warmup)))
        terms['loss'] = terms['loss'] + self.output_geometry_weight*warmup*aux
        terms['output_geometry'] = aux
        terms['output_bce'] = output_bce
        terms['output_dice'] = dice
        terms['output_geometry_scale'] = self.output_geometry_weight*warmup
        return terms,status

    def snapshot(self, **kwargs):
        # Evaluation is an explicit portable infer.py command. No private dataset
        # or hard-coded city selection is accessed from the training loop.
        pass
