import os,json
from pathlib import Path
import numpy as np
import torch
from ..backbone.datasets.quality_fourway import QualityFourWaySparseStructureLatent
from .prior_geometry import features

class QueryPriorDataset(QualityFourWaySparseStructureLatent):
    def __init__(self,roots,**kwargs):
        # The legacy metadata reader interprets each root value as a path.
        # Keep our structured geometry spec outside that reader, then restore
        # it on the same root objects referenced by self.instances.
        data=json.loads(Path(roots[1:]).read_text() if roots.startswith('@') else roots)
        geometry={k:r.pop('query_prior_geometry') for k,r in data.items() if 'query_prior_geometry' in r}
        super().__init__(json.dumps(data),**kwargs)
        for k,spec in geometry.items():self.roots[k]['query_prior_geometry']=spec

    def load_query_prior(self,root,sid):
        spec=root['query_prior_geometry']
        path=Path(spec['cache_dir'])/(sid+'.npy')
        if path.exists():
            f=np.load(path,allow_pickle=False)
        else:
            f=features(spec,sid)
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
            with temp.open('wb') as stream:np.save(stream,f.astype(np.float16))
            os.replace(temp,path)
        if f.shape!=(512,3) or not np.isfinite(f).all() or not f[:,0].any() or f.min()<0 or f.max()>1:
            raise ValueError(f'Invalid query prior: {path}')
        return torch.from_numpy(f.astype(np.float32))

    def get_instance(self,root,instance):
        if root['training_mode']!='rough':return super().get_instance(root,instance)
        target=self._target_instance(root,instance)
        with np.load(Path(root['ss_latent'])/(target+'.npz')) as pack:z=pack['z']
        if z.shape!=(8,16,16,16) or not np.isfinite(z).all():raise ValueError('invalid target latent')
        x=torch.from_numpy(z.astype(np.float32))
        if self.normalization is not None:x=(x-self.mean)/self.std
        pack={'x_0':x,'condition_mode':'rough','image_cond':self._load_image(root,instance),
              'query_prior':self.load_query_prior(root,instance)}
        self._quality_counts['success/'+root['condition_subtype']]+=1
        return pack

    def get_snapshot_instance(self,index):
        self.snapshot_indices()
        root,sid=self.instances[index]
        pack=self.get_instance(root,sid)
        pack['query_prior']=self.load_query_prior(self._snapshot_priors[index],sid)
        return pack
