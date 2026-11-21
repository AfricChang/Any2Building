# Training and checkpoint semantics

Download the pretrained models from [Hugging Face](https://huggingface.co/hongye02/Any2Building) and follow the setup instructions in `weights/README.md` before training. Expected model paths are recorded in `weights/manifest.json`.

The released structure weights inherit an earlier trained model and the final output-geometry objective. New training runs count updates from zero and start a new optimizer and sampler. This entry does not reproduce the full historical curriculum from scratch.

The model has a 30-block, width-1536, 12-head DiT; 1,024-wide condition tokens; four EdgeConv stages with k=16; a 16³ shared-query grid; and spatial residuals at blocks 0/10/20/29. The frozen DINOv3 encoder and structure decoder are separate pretrained dependencies.

The retained training engine implements AdamW, distributed data parallel training, gradient accumulation, BF16 autocast, adaptive gradient clipping, EMA0.9999 and complete checkpoint saves. The original final-stage configuration uses LR 1e-5, weight decay 0.01, betas (0.9,0.95), sigma_min 1e-5, a logit-normal timestep distribution with mean/std 1/1, and joint condition dropout 0.1.

Losses match the Any2Building implementation:

- Flow velocity MSE along the paper's sigma_min interpolation.
- Unweighted query BCE against the decoded complete-building 64³ surface grid, max-pooled to 16³, with coefficient 0.1.
- Output geometry BCE plus Dice, coefficient 0.02. The estimated clean latent is `(1-sigma_min)*x_t - (sigma_min+(1-sigma_min)*t)*v_pred`. The target-positive BCE weight is capped to [1,10]; geometry loss is weighted by `(1-t)^2` and masked for t>0.8. Fine-tuning keeps the full geometry weight active from the first update. Scratch training ramps it over the first 2,000 updates; resume preserves the saved warmup settings.

The historical loader retains source/city weighting, quality tiers, point sampling and partial-point degradation. Its data-root and QC templates are path-neutral. The portable JSONL loader is useful for prepared data and the small format example, but does not recreate the historical corpus, split selection, online image augmentation or source mixture by itself.

`train.py` defaults to the released EMA initialization. `--init ema` explicitly selects this same initialization. New checkpoints contain model weights, EMA, optimizer, sampler and other training states. After a run has saved update 2,000, resume it with:

```bash
python train.py --manifest data/train.jsonl --output outputs/finetune --resume 2000 --additional-steps 2000 --batch-size 4 --batch-split 4
```

Historical Any2Building optimizer/sampler states are not distributed, so the release does not claim an identical historical training continuation. Training target latents and the one-case fixtures used for earlier runtime validation are not bundled. Keep demonstration cases out of benchmark training splits.

The inference default is 50 Euler steps, CFG 3, seed 42 and EMA weights. Training snapshots no longer call private benchmark selectors. Evaluate saved models with a separately specified held-out protocol; the six figure cases are qualitative examples, not an unbiased evaluation set.

`train_surface.py` trains the separate voxel-conditional surface flow using complete sparse surface latents at their reference 64³ coordinates, normalized by the recorded 32-channel statistics. It uses a zero condition token and no image or point features. It initializes from the released surface EMA weights and counts new updates from zero; `--resume STEP` restores saved training state.

