# Data contract

All geometry uses XYZ axes, Z up, and a shared building frame in `[-0.5,0.5]`. Grid indexing is `(x,y,z)`, with Z varying fastest in C-order flattening. Cell centers are `(index+0.5)/resolution - 0.5`.

Normalize a building by a known frame: `normalized = (world_xyz - center) / scale`, using one uniform scale for all axes. Use the same center/scale for point observations, complete targets and priors. A partial cloud's own bounding box can differ from the complete building's frame; normalizing each modality independently breaks alignment. Packaged point clouds are already normalized and must not be normalized again. The inference script rejects coordinates outside the expected frame.

Point files can be NPZ (`points`, float32 `[N,3]`), NPY, XYZ/TXT, or PLY. The tokenizer samples 16,384 points, repeats points when necessary, and uses FPS to select 2,048 then 1,024 centers. Sampling is seeded by the CLI seed. No point masks or GT geometry are needed for inference.

Images are target-building crops, RGB or RGBA. They are resized to 512×512 if necessary, without recropping or automatic rotation. Alpha is composited onto black. DINOv3 then applies its original mean/std normalization. Georeferenced world placement and instance cropping are preprocessing tasks.

Priors are NPZ files with binary `occupancy` of shape `[64,64,64]`, or integer `coords` of shape `[N,3]` with values 0–63. The paper representation uses surface occupancy with empty interiors. The portable loader preserves supplied occupancy exactly and never calls a fill operation. Descriptors for each 4³ block are maximum occupancy, maximum six-neighbor boundary occupancy, and mean occupancy. These produce `[4096,3]` descriptors in XYZ order. Legacy synthetic prior construction in the historical loader is retained as recorded, including filled `lod0`/`lod1` intermediates; the descriptor function itself is unchanged.

## Portable training manifests

Each nonempty JSONL line is an observation/target pair. Paths are relative to the manifest file:

```json
{"mode":"point","points":"points/case.npz","latent":"targets/case.npz","subtype":"lidar"}
{"mode":"image","image":"images/case.png","latent":"targets/case.npz","subtype":"overhead"}
{"mode":"both","points":"points/case.npz","image":"images/case.png","latent":"targets/case.npz","subtype":"paired"}
{"mode":"rough","image":"images/case.png","prior":"priors/case.npz","latent":"targets/case.npz","subtype":"lod1"}
```

The historical training name `rough` corresponds to the public inference mode `image_prior`. Each target NPZ contains `z`, a finite `[8,16,16,16]` complete-building structure latent. Use the matching frozen structure encoder when preparing this target; do not substitute occupancy arrays for latent channels. Any2Building uses no extra target normalization. A helper for occupancy-to-latent preparation is described in `tools/prepare_structure_target.py` and requires the matching pretrained structure encoder supplied by the user.

Portable manifests must contain all four modes for a four-mode run. The sampler cycles homogeneous point/image/rough/both batches and balances declared subtypes. A subset of modes is supported for targeted fine-tuning. Images and point degradation are assumed to have been prepared already in this format.

For the historical data preprocessing and source/quality weighting, use `--reference-roots` with `BuildingWorldImageDataset`, `configs/reference_dataset.json`, and a rebased `configs/reference_sampling_plan.json`. The path-neutral layout lists the original 231 root specifications and their required assets. It is a format/reference configuration, not a downloaded training dataset. The full corpus, splits, and QC manifests must be supplied separately.


## Surface-flow training

A surface JSONL row is `{"surface_latent":"targets/case.npz"}`. Its NPZ contains integer `coords` with shape `[N,3]` in 0–63 and finite `feats` with shape `[N,32]` from the matching 1024-resolution shape encoder. Coordinates describe surface support, not filled interiors. Features are normalized inside the loader with `configs/surface_generator.json`.

```bash
python train_surface.py --manifest data/surface_train.jsonl --output outputs/surface_finetune --additional-steps 2500
```

This starts from the released surface EMA with a new optimizer. `--scratch` starts a new surface model; `--resume STEP` restores checkpoints saved by this entry. Full surface training targets and the pretrained shape encoder are external data-preparation inputs.
