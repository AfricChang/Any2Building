"""Voxel-Conditional Surface Generator (Sec. 3.4, Eq. 11)."""
from ..backbone.models.structured_latent_flow import ElasticSLatFlowModel


class VoxelConditionalSurfaceGenerator(ElasticSLatFlowModel):
    """Generate local surface latents at the supplied occupied voxel coordinates.

    The cross-attention architecture is retained for weight compatibility, with
    a single zero condition token. No point or image condition enters this stage.
    """

