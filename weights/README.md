# Model weights

Download `Any2Building_weights.zip` from [Hugging Face](https://huggingface.co/hongye02/Any2Building) and extract it into the code-release root.
The archive uses the following paper-module names:

- `weights/spatial_prompt_dit.pt`
- `weights/shared_spatial_query_conditioner.pt`
- `weights/point_cloud_tokenizer.pt`
- `weights/surface_generator/surface_flow.pt`
- `weights/structure_decoder/model.safetensors`
- `weights/surface_decoder/model.safetensors`

The released models use EMA weights. For image-conditioned tasks, obtain the
matching frozen DINOv3 model from its official distribution and place it in
`weights/image_tokenizer` for image-conditioned inference or training.
The manifest and `--weights` validation cover the distributed weights only.
