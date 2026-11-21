# Paper-to-code mapping

| Paper component | Public implementation | Model weight file |
|---|---|---|
| Point Cloud Tokenizer, Sec. 3.1 / Eq. 1 | `any2building.models.PointCloudTokenizer` | `point_cloud_tokenizer.pt` |
| Image Tokenizer (frozen DINOv3), Sec. 3.1 | `any2building.models.image_tokenizer.ImageTokenizer` | `weights/image_tokenizer/model.safetensors` |
| Prior descriptors, Eq. 2 | `any2building.io.prior_descriptors` | Deterministic descriptors |
| Shared Spatial Query Conditioner, Eqs. 3–6 | `any2building.models.SharedSpatialQueryConditioner` | `shared_spatial_query_conditioner.pt` |
| Condition-Query Read-Back / QE tokens, Eq. 7 | `SharedSpatialQueryConditioner._forward` / `spatial_read` | Same conditioner file |
| Auxiliary completion features, Sec. 3.2 | `CompletionTokenFusion` | Same conditioner file |
| DiT with Spatial Prompts, Eq. 8 | `any2building.models.SpatialPromptDiT` | `spatial_prompt_dit.pt` |
| Flow, query occupancy and output geometry losses, Eqs. 9–10 | `any2building.training.trainer.Any2BuildingTrainer` | Training only |
| 64³ structure decoder | `weights/structure_decoder` + namespaced decoder primitives | `model.safetensors` |
| Voxel-conditional surface generation, Eq. 11 | `surface.py`, `train_surface.py` | `weights/surface_generator/surface_flow.pt` |
| Frozen high-resolution surface decoder | `weights/surface_decoder` + namespaced decoder primitives | `model.safetensors` |
| Low-poly remeshing | `remesh.py` adapter to RoLoPM | External executable |

The shared spatial query grid has **16³ = 4,096** cells. It is distinct from the **512** learned completion queries used to enrich point-image base tokens. Point tokens have length 1,024. The DINOv3 implementation retains the full 1,029-token sequence (1,024 patches plus special tokens). The conditioner transports QE tokens followed by 4,096 spatial tokens; the DiT splits them before cross-attention. Spatial residuals enter blocks 0, 10, 20, and 29, with zero-initialized projections.

The three checkpoint submodels retain their parameter keys so old Any2Building tensors load strictly, without interpolation or weight conversion. Internal training-engine dictionary slots `denoiser`, `point_cond_encoder`, and `condition_fuser` are serialization compatibility keys. Public classes and expected external weight filenames use the paper terminology.

Any2Building's multimodal tokenizer/conditioner, spatial injection and geometry supervision are separated from reused transformer, VAE, sampler and sparse-tensor primitives in `any2building/backbone`. DINOv3, pretrained decoders and RoLoPM retain their upstream names and attribution.

