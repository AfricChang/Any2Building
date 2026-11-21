"""Point-anchored DINO-to-DGCNN condition fusion."""

import torch
import torch.nn as nn


class PointImageConditionFuser(nn.Module):
    """Add image evidence to point tokens without changing condition length."""

    def __init__(self, channels=1024, num_heads=16, dropout=0.0):
        super().__init__()
        self.point_norm = nn.LayerNorm(channels)
        self.image_norm = nn.LayerNorm(channels)
        self.image_to_point = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        nn.init.zeros_(self.image_to_point.out_proj.weight)
        nn.init.zeros_(self.image_to_point.out_proj.bias)

    def forward(self, point_tokens, image_tokens):
        point_query = self.point_norm(point_tokens)
        image_kv = self.image_norm(image_tokens)
        image_delta, _ = self.image_to_point(
            point_query, image_kv, image_kv, need_weights=False
        )
        return point_tokens + image_delta
