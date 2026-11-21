# Third-party components

The following upstream components used by Any2Building retain their original attribution and licenses.

- **TRELLIS / TRELLIS.2, Microsoft**: transformer/flow/VAE/sparse primitives and training utilities under `any2building/backbone`; frozen structure and surface decoders under `weights`, available in the Hugging Face model archive. MIT notice: `third_party/licenses/TRELLIS_MIT.txt`. The code is namespaced and the package initializer is kept lightweight. The Any2Building-specific data/conditioning extensions and public entry points are separated from these primitives.
- **DINOv3, Meta**: frozen image-tokenizer weights. Obtain the model and its license from the official distribution. DINOv3 is acknowledged in the paper and used through Transformers 4.57.1.
- **AdaPoinTr-style DGCNN / EdgeConv**: the point tokenizer uses four coordinate-neighbor EdgeConv stages and pure-PyTorch FPS. Parameter names are retained from the trained encoder; no separate AdaPoinTr checkout is required.
- **FlexGEMM**, commit `9f2f050396be3cc48894d15ce308e9672a07c677`, and **CuMesh**, commit `cf1a2f07304b5fe388ed86a16e4a0474599df914`: optional compiled surface-stage dependencies, installed from their upstream repositories through `requirements-surface.txt`.
- **utils3d**, commit `9a4eb15e4021b67b12c460c7057d642626897ec8`: optional surface utility dependency.
- **O-Voxel**: bundled upstream extension source under `third_party/o-voxel`, with its original notices. Compile only for the dense surface stage. Required Eigen headers and licenses are retained; Eigen tests, documentation and benchmarks are omitted.
- **RoLoPM (Robust Low-Poly Meshing)**: separate third-party CLI invoked by `remesh.py`. The executable and Wine runtime are not redistributed. Obtain the CLI from the authors' official Robust Low-Poly Meshing project distribution.

The example inputs are selected from the provided paper figure asset bundle. Dataset case labels are retained to make example matching possible. No dataset license is replaced by a code license. The full BuildingWorld-Image corpus is not part of this artifact.
