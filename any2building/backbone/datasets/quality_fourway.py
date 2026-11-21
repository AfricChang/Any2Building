"""Opt-in B quality-v2 dataset. All repairs are versioned, originals untouched."""
import json, os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image
from .sparse_structure_latent import FourWayConditionedSparseStructureLatent
from .components import find_point_condition, load_point_condition, sample_point_condition
from ..utils.quality_sampling import QualityResumableSampler

def orientation_transform(image,city,instance):
    if city=='Melbourne':return image.transpose(Image.Transpose.ROTATE_180)
    if city=='Yarra' and instance.startswith('richmond__'):return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image

def prepare_quality_image(image,mask,variant,size=512,margin=.15,auto_expose=True):
    """Shared training/evaluation preprocessing; image and mask already aligned."""
    if mask.size!=image.size:raise ValueError('image/mask size mismatch')
    alpha=np.asarray(mask.convert('L'))
    if not (alpha>=128).any():raise ValueError('empty foreground')
    if variant=='context':
        # Keep original deployment context; training examples are explicitly paired.
        rgb=np.asarray(image.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)).copy()
        out=torch.from_numpy(rgb).permute(2,0,1).float()/255
        return FourWayConditionedSparseStructureLatent._auto_expose(out) if auto_expose else out
    box=FourWayConditionedSparseStructureLatent._square_crop_box(alpha,margin)
    rgb=np.asarray(image.convert('RGB').crop(box).resize((size,size),Image.Resampling.LANCZOS)).copy()
    aa=np.asarray(mask.crop(box).resize((size,size),Image.Resampling.LANCZOS)).copy()/255
    out=torch.from_numpy(rgb).permute(2,0,1).float()/255*torch.from_numpy(aa).float()[None]
    return FourWayConditionedSparseStructureLatent._auto_expose(out,torch.from_numpy(aa>.01)) if auto_expose else out

class QualityFourWaySparseStructureLatent(FourWayConditionedSparseStructureLatent):
    sampler_class=QualityResumableSampler

    def __init__(self,*args,quality_plan,**kwargs):
        if args and isinstance(args[0],str) and args[0].startswith('@'):
            args=(Path(args[0][1:]).read_text(),*args[1:])
        self.quality_plan=json.loads(Path(quality_plan).read_text())
        self._quality_counts=Counter();self._quality_errors=[];self._quality_calls=0
        super().__init__(*args,**kwargs)
        expected=set(self.quality_plan['roots'])
        actual={s for m in self.mode_subtype_indices.values() for s in m}
        if actual!=expected:raise ValueError(f'Empty/unplanned QC roots: missing={expected-actual}, extra={actual-expected}')
        if self.quality_plan.get('validate_snapshot_on_startup',False):
            from torch.utils.data import default_collate
            pool=self.snapshot_indices()
            indices=[pool[j] for j in np.linspace(0,len(pool)-1,8,dtype=int)]
            packs=[self.get_snapshot_instance(i) for i in indices]
            batch=default_collate(packs)
            assert {'x_0','image_cond','point_cond','rough_prior'}<=set(batch)
            assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v,torch.Tensor))
            assert self._quality_counts.get('image_variant/both/context',0)==0
            assert self._quality_counts['image_variant/both/alpha_cutout']==8
            print('QUALITY_STARTUP_PAIRED_SNAPSHOT_OK '+json.dumps({'pool':len(pool),'checked':8,'both_context':0}),flush=True)

    def filter_metadata(self,metadata):
        root=getattr(self,'_current_root',None)
        if isinstance(root,dict) and root.get('training_mode') in ('image','both','rough') and root.get('condition_subtype') in self.quality_plan['roots'] and not self.quality_plan['roots'][root['condition_subtype']].get('legacy_full_pool',False):
            # These exact manifest IDs passed a real target-latent load/shape/
            # finite check. Historical metadata misses newly encoded Calgary IDs.
            # Do not apply this override to the unscreened point-only pool.
            metadata=metadata.copy()
            metadata['ss_latent_encoded']=True
        return super().filter_metadata(metadata)

    def _load_image(self,root,instance):
        spec=self.quality_plan['roots'][root['condition_subtype']]
        if spec.get('legacy_full_pool',False):return super()._load_image(root,instance)
        folder=Path(root['render_cond'])/instance
        frames=json.loads((folder/'transforms.json').read_text())['frames']
        with Image.open(folder/frames[np.random.randint(len(frames))]['file_path']) as im:
            image=im.copy()
        if root.get('alpha_mask_root'):
            with Image.open(Path(root['alpha_mask_root'])/(instance+'.png')) as im:mask=im.convert('L')
        elif image.mode=='RGBA':mask=image.getchannel('A')
        else:raise ValueError(f'No building mask: {instance}')
        image=orientation_transform(image,spec['city'],instance)
        mask=orientation_transform(mask,spec['city'],instance)
        mode=root['training_mode']
        if mode=='both':variant='context' if np.random.random()<self.quality_plan.get('both_context_probability',0.) else 'alpha_cutout'
        elif mode=='image':variant='context' if np.random.random()<.5 else 'alpha_cutout'
        else:variant='alpha_cutout'
        self._quality_counts[f'image_variant/{mode}/{variant}']+=1
        return prepare_quality_image(image,mask,variant,self.image_size)

    def get_instance(self,root,instance):
        spec=self.quality_plan['roots'][root['condition_subtype']]
        mode=root['training_mode'];target=self._target_instance(root,instance)
        z=np.load(Path(root['ss_latent'])/(target+'.npz'))['z']
        if z.shape!=(8,16,16,16) or not np.isfinite(z).all():raise ValueError('invalid target latent')
        pack={'x_0':torch.from_numpy(z.astype(np.float32)), 'condition_mode':mode}
        if self.normalization is not None:pack['x_0']=(pack['x_0']-self.mean)/self.std
        if mode in ('image','rough','both'):pack['image_cond']=self._load_image(root,instance)
        if mode in ('point','both'):
            path=find_point_condition(root,self.point_cond_root,instance)
            if path is None:raise FileNotFoundError(f'Missing point {instance}')
            pts=load_point_condition(path)
            if pts.ndim!=2 or pts.shape[1]!=3 or len(pts)==0 or not torch.isfinite(pts).all():
                raise ValueError('invalid point condition')
            if self.normalize_points:
                lo,hi=pts.amin(0),pts.amax(0);span=(hi-lo).amax()
                if span<1e-6:raise ValueError('degenerate point bbox')
                pts=(pts-(lo+hi)/2)*((1-2*self.normalize_padding)/span)
            if mode=='both':
                rates=self.quality_plan['both_keep_rates'];probs=self.quality_plan['both_keep_probs']
                keep=float(np.random.choice(rates,p=probs))
                # Subsample real input rows before repeating; replicas aren't new observations.
                if len(pts)>self.point_sample_size:pts=pts[torch.randperm(len(pts))[:self.point_sample_size]]
                n=max(1,round(len(pts)*keep))
                if n<len(pts):
                    if np.random.random()<.25:
                        direction=torch.randn(3);idx=torch.argsort(pts@direction)[:n]
                        self._quality_counts['drop/spatial']+=1
                    else:idx=torch.randperm(len(pts))[:n];self._quality_counts['drop/random']+=1
                    pts=pts[idx]
                self._quality_counts[f'both_keep/{keep}']+=1
            pack['point_cond']=sample_point_condition(pts,self.point_sample_size)
        if mode=='rough':
            z=np.load(Path(root['rough_ss_latent'])/(instance+'.npz'))['z']
            if z.shape!=(8,16,16,16) or not np.isfinite(z).all():raise ValueError('invalid rough latent')
            pack['rough_prior']=torch.from_numpy(z.astype(np.float32))
        self._quality_counts['success/'+root['condition_subtype']]+=1
        return pack

    def __getitem__(self,index):
        root,sid=self.instances[index];mode=root['training_mode'];sub=root['condition_subtype']
        pool=self.mode_subtype_indices[mode][sub]
        for attempt in range(32):
            rr,ss=self.instances[index if attempt==0 else pool[np.random.randint(len(pool))]]
            try:
                result=self.get_instance(rr,ss)
                self._quality_calls+=1
                if self._quality_calls%32==0:self._flush_quality()
                return result
            except (OSError,ValueError,KeyError) as exc:
                self._quality_counts['error/'+sub]+=1
                self._quality_errors.append(dict(root=sub,instance=ss,error=str(exc)))
                self._flush_quality()
                if attempt==0:print(f'QC_LOAD_ERROR {sub}/{ss}: {exc}',flush=True)
        raise RuntimeError(f'QC source repeatedly failed: {sub}/{sid}')

    def _flush_quality(self):
        directory=Path(self.quality_plan['runtime_dir']);directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'worker_{os.getpid()}.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(counts=dict(self._quality_counts),recent_errors=self._quality_errors[-20:]),indent=2));temp.replace(path)

    def snapshot_indices(self):
        if hasattr(self,'_snapshot_priors'):return list(self._snapshot_priors)
        priors={}
        for sub,indices in self.mode_subtype_indices.get('rough',{}).items():
            spec=self.quality_plan['roots'][sub]
            if spec['city']=='Skylines':continue
            for i in indices:
                root,sid=self.instances[i]
                key=(spec['city'],sid,root['ss_latent'])
                if key not in priors or spec.get('prior_family')=='osm':priors[key]=root
        self._snapshot_priors={}
        for sub,indices in self.mode_subtype_indices.get('both',{}).items():
            city=self.quality_plan['roots'][sub]['city']
            for i in indices:
                root,sid=self.instances[i]
                key=(city,sid,root['ss_latent'])
                if key in priors:self._snapshot_priors[i]=priors[key]
        if not self._snapshot_priors:raise RuntimeError('No aligned image/point/prior snapshot pairs')
        return list(self._snapshot_priors)

    def get_snapshot_instance(self,index):
        self.snapshot_indices()
        root,sid=self.instances[index]
        pack=self.get_instance(root,sid)
        prior=np.load(Path(self._snapshot_priors[index]['rough_ss_latent'])/(sid+'.npz'))['z']
        if prior.shape!=(8,16,16,16) or not np.isfinite(prior).all():raise ValueError('Invalid snapshot prior')
        pack['rough_prior']=torch.from_numpy(prior.astype(np.float32))
        return pack
