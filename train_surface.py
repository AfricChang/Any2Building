#!/usr/bin/env python3
"""Train the separate voxel-conditional surface generator (paper Eq. 11)."""
import argparse,json,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--additional-steps',type=int,default=2500)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--batch-split',type=int,default=1)
    p.add_argument('--resume',type=int)
    p.add_argument('--scratch',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if min(a.additional_steps,a.batch_size,a.batch_split,a.workers)<1 or a.batch_size%a.batch_split:
        p.error('Positive arguments required, with batch-size divisible by batch-split')
    if a.resume is not None and a.resume < 0:
        p.error('resume must be nonnegative')
    root=Path(__file__).resolve().parent
    os.environ.setdefault('ATTN_BACKEND','flash_attn')
    os.environ.setdefault('SPARSE_ATTN_BACKEND','flash_attn')
    for key in ('SKIP_INIT_SNAPSHOT','SKIP_DATASET_SNAPSHOT','SKIP_RESUME_SNAPSHOT'):
        os.environ['ANY2BUILDING_'+key]='1'
    os.environ['ANY2BUILDING_NUM_WORKERS']=str(a.workers)
    import torch,numpy as np,random
    import torch.distributed as dist
    from any2building.models.surface_generator import VoxelConditionalSurfaceGenerator
    from any2building.training.surface_dataset import SurfaceLatentDataset
    from any2building.backbone.trainers.flow_matching.sparse_flow_matching import SparseFlowMatchingTrainer
    from any2building.backbone.utils.dist_utils import setup_dist
    class SurfaceTrainer(SparseFlowMatchingTrainer):
        def snapshot(self,**kwargs):pass
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(local)
    if world>1:setup_dist(rank,local,world,os.environ.get('MASTER_ADDR','localhost'),os.environ.get('MASTER_PORT','29500'))
    torch.manual_seed(rank);np.random.seed(rank);random.seed(rank)
    cfg=json.loads((root/'configs/surface_generator.json').read_text())
    ds=SurfaceLatentDataset(a.manifest,cfg['dataset']['args']['normalization'],cfg['dataset']['args']['max_tokens'])
    if len(ds)<a.batch_size*world:p.error('Provide at least batch-size * world-size surface samples')
    flow=VoxelConditionalSurfaceGenerator(**cfg['models']['denoiser']['args']).cuda()
    args=cfg['trainer']['args'];args.pop('elastic',None)
    start=a.resume if a.resume is not None else 0
    args.update(max_steps=start+a.additional_steps,batch_size_per_gpu=a.batch_size,batch_split=a.batch_split)
    if a.resume is None and not a.scratch:
        args.update(finetune_ckpt={'denoiser':str(root/'weights/surface_generator/surface_flow.pt')},finetune_step=start)
    if rank==0:
        if a.output.exists() and a.resume is None:raise FileExistsError('Choose a fresh output directory')
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    if world>1:dist.barrier()
    trainer=SurfaceTrainer({'denoiser':flow},ds,output_dir=str(a.output),load_dir=str(a.output),step=a.resume,**args)
    trainer.run()
    if rank==0:trainer.save(non_blocking=False)
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

