import torch
import numpy as np
from dgl.geometry import farthest_point_sampler

def fps_indices(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    points: (N, d) or (B, N, d)
    n_samples: int
    Returns: (n_samples,) or (B, n_samples)
    """
    if points.dim() == 2:
        # (N, d) -> (1, N, d)
        points = points.unsqueeze(0)
        batch_mode = False
    else:
        batch_mode = True
    idx = farthest_point_sampler(points, n_samples)  # (B, n_samples)
    if not batch_mode:
        idx = idx[0]
    return idx

def edge_midpoints(points: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    points: (N, d)
    edge_index: (2, E)
    Returns: (E, d) edge midpoints
    """
    src, dst = edge_index
    return (points[src] + points[dst]) / 2.0

def farthest_point_sample(pos, n_samples, mask=None):
    """
    PyTorch native FPS implementation (Fixed start point version).
    Always sample starting from the 0th node to ensure deterministic results.
    Args:
        pos: (B, N, C) Node coordinates (Dense Batch)
        n_samples: int Number of samples K
        mask: (B, N) bool, True indicates real node, False indicates Padding
    Returns:
        centroids: (B, K) Indices of sampled points
    """
    device = pos.device
    B, N, C = pos.shape
    if mask is not None:
        valid_counts = mask.sum(1)  # (B,)
        max_samples = valid_counts.min().item()
        K = min(n_samples, max_samples)
    else:
        K = min(n_samples, N)
    centroids = torch.zeros(B, K, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = pos[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((pos - centroid) ** 2, -1)
        if mask is not None:
            dist = dist.masked_fill(~mask, -1.0)
        mask_dist = dist < distance
        distance[mask_dist] = dist[mask_dist]
        farthest = torch.max(distance, -1)[1]
    return centroids