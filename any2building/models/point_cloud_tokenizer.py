"""AdaPoinTr-style DGCNN encoder for incomplete point-cloud conditions."""

from typing import Optional, Sequence

import torch
import torch.nn as nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def knn_point(k: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    if k > xyz.shape[1]:
        raise ValueError(f"k={k} exceeds the number of points ({xyz.shape[1]})")
    return torch.topk(
        square_distance(query_xyz, xyz), k, dim=-1, largest=False, sorted=False
    ).indices


def gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Pure-PyTorch farthest-point sampling."""
    batch, num_points, _ = xyz.shape
    if npoint > num_points:
        raise ValueError(f"Cannot FPS {npoint} centers from {num_points} points")
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, num_points), 1e10, device=xyz.device)
    farthest = torch.randint(0, num_points, (batch,), device=xyz.device)
    batch_indices = torch.arange(batch, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        distance = torch.minimum(distance, torch.sum((xyz - centroid) ** 2, dim=-1))
        farthest = distance.max(dim=1).indices
    return centroids


class DGCNNGrouper(nn.Module):
    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
        self.num_features = 128
        self.input_trans = nn.Conv1d(3, 8, 1)
        self.layer1 = self._edge_layer(16, 32, 4)
        self.layer2 = self._edge_layer(64, 64, 4)
        self.layer3 = self._edge_layer(128, 64, 4)
        self.layer4 = self._edge_layer(128, 128, 4)

    @staticmethod
    def _edge_layer(in_channels: int, out_channels: int, groups: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    @staticmethod
    def fps_downsample(coor: torch.Tensor, feats: torch.Tensor, count: int):
        idx = furthest_point_sample(coor.transpose(1, 2).contiguous(), count)
        gathered = gather_points(torch.cat([coor, feats], dim=1), idx)
        return gathered[:, :3], gathered[:, 3:]

    def get_graph_feature(self, coor_q, feats_q, coor_k, feats_k):
        batch, dims, num_keys = feats_k.shape
        num_queries = feats_q.shape[2]
        idx = knn_point(
            self.k,
            coor_k.transpose(1, 2).contiguous(),
            coor_q.transpose(1, 2).contiguous(),
        ).transpose(1, 2).contiguous()
        idx += torch.arange(batch, device=feats_q.device).view(-1, 1, 1) * num_keys
        neighbors = feats_k.transpose(2, 1).contiguous().reshape(batch * num_keys, -1)[idx.reshape(-1)]
        neighbors = neighbors.reshape(batch, self.k, num_queries, dims).permute(0, 3, 2, 1)
        centers = feats_q.view(batch, dims, num_queries, 1).expand(-1, -1, -1, self.k)
        return torch.cat((neighbors - centers, centers), dim=1)

    def forward(self, xyz: torch.Tensor, center_num: Sequence[int]):
        coor = xyz.transpose(1, 2).contiguous()
        feats = self.input_trans(coor)
        feats = self.layer1(self.get_graph_feature(coor, feats, coor, feats)).amax(dim=-1)
        coor_q, feats_q = self.fps_downsample(coor, feats, center_num[0])
        feats = self.layer2(self.get_graph_feature(coor_q, feats_q, coor, feats)).amax(dim=-1)
        coor = coor_q
        feats = self.layer3(self.get_graph_feature(coor, feats, coor, feats)).amax(dim=-1)
        coor_q, feats_q = self.fps_downsample(coor, feats, center_num[1])
        feats = self.layer4(self.get_graph_feature(coor_q, feats_q, coor, feats)).amax(dim=-1)
        return coor_q.transpose(1, 2).contiguous(), feats.transpose(1, 2).contiguous()


class PointCloudTokenizer(nn.Module):
    """Encode ``[B, N, 3]`` incomplete clouds as cross-attention tokens."""

    def __init__(
        self,
        in_channels: int = 3,
        num_points: int = 16384,
        center_num: Sequence[int] = (2048, 1024),
        k: int = 16,
        hidden_channels: int = 512,
        out_channels: int = 1024,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("PointCloudTokenizer accepts xyz (3 channels) only")
        if len(center_num) != 2:
            raise ValueError("center_num must contain exactly two downsampling stages")
        self.num_points = int(num_points)
        self.center_num = tuple(int(v) for v in center_num)
        if not self.num_points >= self.center_num[0] >= self.center_num[1] >= k:
            raise ValueError("Expected num_points >= center_num[0] >= center_num[1] >= k")
        self.grouper = DGCNNGrouper(k=k)
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, out_channels)
        ) if use_pos_embed else None

    def _sample_points(self, points: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected point cloud [B,N,3], got {tuple(points.shape)}")
        if mask is None:
            mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        sampled = []
        for cloud, valid_mask in zip(points, mask):
            valid = cloud[valid_mask]
            if valid.shape[0] == 0:
                raise ValueError("An incomplete point-cloud condition contains no valid points")
            if valid.shape[0] >= self.num_points:
                idx = torch.randperm(valid.shape[0], device=valid.device)[:self.num_points]
            else:
                idx = torch.randint(valid.shape[0], (self.num_points,), device=valid.device)
            sampled.append(valid[idx])
        return torch.stack(sampled)

    def forward(self, points: torch.Tensor, point_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        xyz = self._sample_points(points, point_mask)
        coor, feats = self.grouper(xyz, self.center_num)
        tokens = self.input_proj(feats)
        return tokens if self.pos_embed is None else tokens + self.pos_embed(coor)
