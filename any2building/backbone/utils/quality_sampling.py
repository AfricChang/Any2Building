"""Explicit source/city sampling, independent of the number of subtype names."""
import hashlib
import json
import math
from collections import defaultdict, Counter
import numpy as np
from .data_utils import ResumableSampler

def bounded_tiers(tierkeys, submap, mode):
    """Prefer useful imagery without oversampling a handful of A-tier objects."""
    if set(tierkeys)=={'legacy'}:return {'legacy':1.}
    preferred={'A':.8,'B':.2} if mode=='both' else {'A':.65,'B':.30,'C':.05}
    # Variant roots may repeat an entity; deduplication is done by the caller's
    # tier manifests, whose variants have the same eligibility by construction.
    ns={t:max(len(submap[k]) for k in ks) for t,ks in tierkeys.items()}
    total=sum(ns.values())
    caps={t:min(1.,3*n/total, .2 if mode=='both' and t=='B' else .1 if mode=='image' and t=='C' else 1.) for t,n in ns.items()}
    out={t:0. for t in ns};active=set(ns);remaining=1.
    while active and remaining>1e-12:
        denom=sum(preferred[t] for t in sorted(active))
        limited=[t for t in sorted(active) if remaining*preferred[t]/denom>=caps[t]-out[t]-1e-12]
        if not limited:
            for t in sorted(active):out[t]+=remaining*preferred[t]/denom
            remaining=0.;break
        for t in limited:
            delta=caps[t]-out[t];out[t]=caps[t];remaining-=delta;active.remove(t)
    if remaining>1e-8:raise ValueError(f'Insufficient {mode} tier capacity: {ns}, {caps}')
    return out

def capped_allocation(counts, total=1., city_cap=.15, repeat_cap=3., reference_n=None):
    """Water-fill sqrt(N), subject to capacity; return unused mass explicitly."""
    counts={k:int(counts[k]) for k in sorted(counts) if counts[k]>0}
    ref=reference_n or sum(counts.values())
    caps={k:min(city_cap, repeat_cap*n/max(ref,1), .005 if n<100 else .01 if n<300 else 1.) for k,n in counts.items()}
    weights={k:math.sqrt(n) for k,n in counts.items()}
    out={k:0. for k in counts};remaining=float(total);active=set(counts)
    while active and remaining>1e-12:
        denom=sum(weights[k] for k in sorted(active))
        limited=[k for k in sorted(active) if remaining*weights[k]/denom>=caps[k]-out[k]-1e-12]
        if not limited:
            for k in sorted(active):out[k]+=remaining*weights[k]/denom
            remaining=0.;break
        for k in limited:
            delta=caps[k]-out[k];out[k]=caps[k];remaining-=delta;active.remove(k)
    return out,max(0.,remaining),caps

def make_weights(dataset):
    specs=dataset.quality_plan['roots'];plan=dataset.quality_plan
    bymode=dataset.mode_subtype_indices
    result={};report={}
    for mode,submap in bymode.items():
        # Count physical building IDs once, not once per representation/LOD.
        entities=defaultdict(set)
        for key,indices in submap.items():
            spec=specs[key]
            for i in indices:
                root,sid=dataset.instances[i]
                target=dataset._target_instance(root,sid)
                entities[spec['city']].add(target)
        counts={c:len(ids) for c,ids in entities.items()}
        keys=sorted(submap)
        if mode=='point':
            family_targets=plan['point_source_shares']
            cityweights={};unused=0.
            for family,target in family_targets.items():
                cities={specs[k]['city'] for k in keys if specs[k]['family']==family}
                ns={c:counts[c] for c in sorted(cities)}
                if family in ('buildingpcc','lidar'):
                    if ns:cityweights.update({c:target*ns[c]/sum(ns.values()) for c in ns})
                    else:unused+=target
                else:
                    # Source allocation is explicit; bound city mass within source.
                    alloc,rest,_=capped_allocation(ns,target,city_cap=.15,reference_n=sum(ns.values())/max(target,1e-12))
                    cityweights.update(alloc);unused+=rest
            if unused>1e-9:
                raise ValueError(f'Point city caps cannot satisfy source quotas: unused={unused}')
        else:
            real={c:n for c,n in counts.items() if c!='Skylines'}
            synthetic=plan['skylines_shares'].get(mode,0.) if 'Skylines' in counts else 0.
            cityweights,rest,caps=capped_allocation(real,1-synthetic,reference_n=sum(real.values()))
            if rest>1e-8:raise ValueError(f'{mode}: infeasible city caps ({rest}); expand data instead of silently relaxing caps')
            if synthetic:cityweights['Skylines']=synthetic
        weights={k:0. for k in keys}
        if mode=='rough':
            # Prior families are allocated first; each family balances available cities.
            for prior,quota in plan['rough_prior_shares'].items():
                kk=[k for k in keys if specs[k]['prior_family']==prior]
                if not kk:raise ValueError(f'Missing required prior family {prior}')
                cityprior=defaultdict(list)
                for k in kk:cityprior[specs[k]['city']].append(k)
                # Fixed family quotas; sqrt entity count and bounded repetition locally.
                ns={c:len(set(dataset._target_instance(dataset.instances[i][0],dataset.instances[i][1]) for k in ks for i in submap[k])) for c,ks in cityprior.items()}
                if prior=='skylines':
                    allocated={c:quota*n/sum(ns.values()) for c,n in ns.items()}
                else:
                    allocated,remaining,_=capped_allocation(ns,quota,reference_n=sum(ns.values())/quota)
                    if remaining>1e-8:raise ValueError(f'{prior} capacity insufficient: {remaining}')
                for city,ks in cityprior.items():
                    mass=allocated[city]
                    tierkeys=defaultdict(list)
                    for k in ks:tierkeys[specs[k]['tier']].append(k)
                    tw=bounded_tiers(tierkeys,submap,mode)
                    for tier,tkeys in tierkeys.items():
                        for k in tkeys:weights[k]+=mass*tw[tier]/len(tkeys)
        else:
            for city,mass in cityweights.items():
                ks=[k for k in keys if specs[k]['city']==city]
                if mode=='point':
                    total_city_records=sum(len(submap[k]) for k in ks)
                    for k in ks:weights[k]=mass*len(submap[k])/total_city_records
                else:
                    tiers=defaultdict(list)
                    for k in ks:tiers[specs[k]['tier']].append(k)
                    tw=bounded_tiers(tiers,submap,mode)
                    for tier,tkeys in tiers.items():
                        for k in tkeys:weights[k]=mass*tw[tier]/len(tkeys)
        total=sum(weights.values())
        if not np.isclose(total,1):raise ValueError((mode,total))
        result[mode]=weights
        report[mode]=dict(root_weights=weights,city_unique_counts=counts,city_weights=dict(Counter({c:sum(weights[k] for k in keys if specs[k]['city']==c) for c in counts})))
    return result,report

class QualityResumableSampler(ResumableSampler):
    def __init__(self,dataset,shuffle=True,seed=0,drop_last=False,batch_size=1):
        super().__init__(dataset,shuffle,seed,drop_last)
        self.batch_size=int(batch_size)
        self.weights,self.report=make_weights(dataset)
        self.sparse_probs={}
        for key,values in dataset.mode_subtype_indices.get('point',{}).items():
            ns=len(dataset.sparse_mode_subtype_indices.get('point',{}).get(key,[]))
            # A 20% target must not repeat a tiny sparse stratum dozens of times.
            self.sparse_probs[key]=min(float(dataset.sparse_point_ratio),3.*ns/len(values))
        self.report['point']['effective_sparse_probs']=self.sparse_probs
        self.signature=hashlib.sha256(json.dumps({'weights':self.weights,'counts':{m:{k:len(v) for k,v in s.items()} for m,s in dataset.mode_subtype_indices.items()},'plan':dataset.quality_plan,'sparse_ratio':dataset.sparse_point_ratio,'sparse_counts':{m:{k:len(v) for k,v in s.items()} for m,s in dataset.sparse_mode_subtype_indices.items()}},sort_keys=True).encode()).hexdigest()
        self.signature=hashlib.sha256((self.signature+json.dumps(self.sparse_probs,sort_keys=True)).encode()).hexdigest()
        if self.rank==0:
            from pathlib import Path
            path=Path(dataset.quality_plan['runtime_dir']);path.mkdir(parents=True,exist_ok=True)
            (path/'effective_sampling.json').write_text(json.dumps(self.report,indent=2))
            print('QUALITY_SAMPLING '+json.dumps({m:r['city_weights'] for m,r in self.report.items()}),flush=True)

    def state_dict(self):
        return dict(epoch=self.epoch,idx=self.idx,quality_signature=self.signature,batch_size=self.batch_size,world_size=self.world_size)

    def load_state_dict(self,state):
        compatible=state.get('quality_signature')==self.signature and state.get('batch_size')==self.batch_size and state.get('world_size')==self.world_size
        self.epoch=int(state.get('epoch',0));self.idx=int(state.get('idx',0)) if compatible else 0
        if self.idx%self.batch_size:self.idx=0
        print(f'QUALITY_SAMPLER_RESUME compatible={compatible} epoch={self.epoch} idx={self.idx}',flush=True)

    def __len__(self):
        gb=self.batch_size*self.world_size
        return max(4,math.ceil(len(self.dataset)/gb/4)*4)*self.batch_size

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        pools={};positions={}
        for mode,subs in self.dataset.mode_subtype_indices.items():
            for key,values in subs.items():
                pools[(mode,key,'all')]=rng.permutation(values) if self.shuffle else np.asarray(values)
                positions[(mode,key,'all')]=0
                if mode=='point':
                    for density,mapping in [('sparse',self.dataset.sparse_mode_subtype_indices),('dense',self.dataset.dense_mode_subtype_indices)]:
                        vals=mapping.get(mode,{}).get(key,[])
                        if vals:pools[(mode,key,density)]=rng.permutation(vals);positions[(mode,key,density)]=0
        def take(key):
            arr=pools[key];pos=positions[key]
            value=int(arr[pos]);pos+=1
            if pos==len(arr):
                pos=0
                if self.shuffle:rng.shuffle(arr)
            positions[key]=pos
            return value
        gb=self.batch_size*self.world_size
        nb=len(self)//self.batch_size
        keys={m:list(w) for m,w in self.weights.items()};probs={m:list(w.values()) for m,w in self.weights.items()}
        exposed=Counter()
        for batch in range(nb):
            mode=('point','image','rough','both')[batch%4]
            selection=[]
            for rootidx in rng.choice(len(keys[mode]),gb,p=probs[mode]):
                key=keys[mode][rootidx];density='all'
                if mode=='point' and (mode,key,'sparse') in pools and (mode,key,'dense') in pools:
                    density='sparse' if rng.random()<self.sparse_probs[key] else 'dense'
                selection.append(take((mode,key,density)))
                if batch*self.batch_size>=self.idx:exposed[(mode,key,density)]+=1
            if batch*self.batch_size<self.idx:continue
            start=self.rank*self.batch_size
            yield from selection[start:start+self.batch_size]
            if self.rank==0 and (batch+1)%40==0:
                from pathlib import Path
                dest=Path(self.dataset.quality_plan['runtime_dir'])/'sampler_exposure.json'
                temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(dict(epoch=self.epoch,global_batches=batch+1,counts={'/'.join(k):v for k,v in exposed.items()}),indent=2));temp.replace(dest)
