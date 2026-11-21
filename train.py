#!/usr/bin/env python3
"""Train/fine-tune Any2Building with the released architecture and objective."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    data=p.add_mutually_exclusive_group(required=True)
    data.add_argument('--manifest',type=Path,help='Portable preprocessed JSONL observations')
    data.add_argument('--reference-roots',type=Path,help='Historical quality-controlled roots, after rebasing the template')
    p.add_argument('--reference-dataset-config',type=Path,default=Path(__file__).resolve().parent/'configs/reference_dataset.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'configs/any2building.json')
    p.add_argument('--init',choices=['ema','scratch'],default='ema')
    p.add_argument('--resume',type=int,help='Resume a checkpoint step saved by this training entry')
    p.add_argument('--additional-steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=4,help='Per GPU, before accumulation splitting')
    p.add_argument('--batch-split',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if min(a.additional_steps,a.batch_size,a.batch_split,a.workers)<1 or a.batch_size%a.batch_split:
        p.error('Positive steps, batches and workers required; batch-size must be divisible by batch-split')
    if a.resume is not None and a.resume < 0:
        p.error('resume must be nonnegative')
    root=Path(__file__).resolve().parent
    os.environ.setdefault('ATTN_BACKEND','flash_attn')
    for key in ('SKIP_INIT_SNAPSHOT','SKIP_DATASET_SNAPSHOT','SKIP_RESUME_SNAPSHOT'):
        os.environ['ANY2BUILDING_'+key]='1'
    os.environ['ANY2BUILDING_NUM_WORKERS']=str(a.workers)
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from any2building import models
    from any2building.training.dataset import BuildingObservationDataset
    from any2building.training.trainer import Any2BuildingTrainer
    from any2building.pipeline import WEIGHT_NAMES
    from any2building.backbone.utils.dist_utils import setup_dist
    rank=int(os.environ.get('RANK',0)); local=int(os.environ.get('LOCAL_RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(local)
    if world>1: setup_dist(rank,local,world,os.environ.get('MASTER_ADDR','localhost'),os.environ.get('MASTER_PORT','29500'))
    random.seed(rank); np.random.seed(rank); torch.manual_seed(rank)
    cfg=json.loads(a.config.read_text())
    dsargs=cfg['dataset']['args']; dsargs['pretrained_ss_dec']=str(root/dsargs['pretrained_ss_dec'])
    if a.manifest:
        dataset=BuildingObservationDataset(a.manifest,**dsargs)
    else:
        from any2building.training.buildingworld_image_dataset import BuildingWorldImageDataset
        dsargs=json.loads(a.reference_dataset_config.read_text())['args']
        dsargs['pretrained_ss_dec']=str(root/'weights/structure_decoder/model')
        dataset=BuildingWorldImageDataset('@'+str(a.reference_roots),**dsargs)
    nets={k:getattr(models,v['name'])(**v['args']).cuda() for k,v in cfg['models'].items()}
    options=cfg['trainer']['args']
    options['image_cond_model']['args']['model_name']=str(root/'weights/image_tokenizer')
    start=a.resume if a.resume is not None else 0
    options.update(max_steps=start+a.additional_steps,batch_size_per_gpu=a.batch_size,batch_split=a.batch_split)
    if a.resume is not None:
        saved=json.loads((a.output/'config.json').read_text())['trainer']['args']
        for key in ('geometry_start_step','output_geometry_warmup'):
            options[key]=saved[key]
    elif a.init=='scratch':
        options['geometry_start_step']=0
    else:
        # Pretrained weights already use the full geometry objective.
        options['geometry_start_step']=-options['output_geometry_warmup']
    if a.resume is None and a.init!='scratch':
        options['finetune_ckpt']={k:str(root/f'weights/{WEIGHT_NAMES[k]}.pt') for k in nets}
        options['finetune_step']=start
    if rank==0 and a.output.exists() and a.resume is None:
        raise FileExistsError('Choose a new training output directory or pass --resume')
    if rank==0:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    if world>1: dist.barrier()
    trainer=Any2BuildingTrainer(nets,dataset,output_dir=str(a.output),load_dir=str(a.output),step=a.resume,**options)
    trainer.run()
    if rank==0: trainer.save(non_blocking=False)
    if world>1: dist.barrier(); dist.destroy_process_group()


if __name__=='__main__':main()
