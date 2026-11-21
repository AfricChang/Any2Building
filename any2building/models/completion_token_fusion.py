"""Point-image base tokens with 512 auxiliary completion queries (Sec. 3.2)."""
import torch
from torch import nn
class CompletionTokenFusion(nn.Module):
    def __init__(self, channels=1024, num_heads=16, dropout=0.0, anchor_grid=8,
                 use_occ_head=False):
        super().__init__()
        self.anchor_grid = int(anchor_grid)
        n_anchor = self.anchor_grid ** 3

        # --- variant A blocks (identical names/shapes) ---
        self.point_norm = nn.LayerNorm(channels)
        self.image_norm = nn.LayerNorm(channels)
        self.image_to_point = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )

        # --- completion query path (new) ---
        self.anchor_pos = nn.Parameter(torch.randn(1, n_anchor, channels) * 0.02)
        self.anchor_query_norm = nn.LayerNorm(channels)
        self.anchor_point_read = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.anchor_image_read = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.completion_read_norm = nn.LayerNorm(channels)
        self.completion_to_point = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        # Variant C is opt-in so Variant B checkpoints keep their exact module
        # structure and continue to load strictly.
        self.occ_head = nn.Linear(channels, 1) if use_occ_head else None
        self.last_occ_logits = None

        nn.init.zeros_(self.image_to_point.out_proj.weight)
        nn.init.zeros_(self.image_to_point.out_proj.bias)
        # completion path starts as identity (no-op)
        nn.init.zeros_(self.anchor_image_read.out_proj.weight)
        nn.init.zeros_(self.anchor_image_read.out_proj.bias)
        nn.init.zeros_(self.completion_to_point.out_proj.weight)
        nn.init.zeros_(self.completion_to_point.out_proj.bias)

    def anchor_coords(self, device, dtype):
        g = self.anchor_grid
        lin = torch.linspace(-0.5 + 0.5 / g, 0.5 - 0.5 / g, g, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing='ij')
        return torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)

    def forward(self, point_tokens, image_tokens):
        # variant A fusion
        point_query = self.point_norm(point_tokens)
        image_kv = self.image_norm(image_tokens)
        image_delta, _ = self.image_to_point(
            point_query, image_kv, image_kv, need_weights=False
        )
        fused = point_tokens + image_delta

        # fixed 3D completion queries
        b = point_tokens.shape[0]
        q = self.anchor_pos.expand(b, -1, -1)
        q_read, _ = self.anchor_point_read(
            self.anchor_query_norm(q), point_tokens, point_tokens, need_weights=False
        )
        q = q + q_read
        img_delta, _ = self.anchor_image_read(q, image_kv, image_kv, need_weights=False)
        completion = q + img_delta
        self.last_occ_logits = (
            self.occ_head(completion).squeeze(-1) if self.occ_head is not None else None
        )

        # completion features -> point tokens (length preserved)
        comp_delta, _ = self.completion_to_point(
            self.completion_read_norm(fused), completion, completion, need_weights=False
        )
        return fused + comp_delta
