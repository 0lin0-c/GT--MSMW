import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
from torch_geometric.utils import to_dense_batch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device=None):
    # Retained for compatibility when positions are not provided (sequence RoPE fallback)
    device = device if device is not None else torch.device('cpu')
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.head_omega = nn.Parameter(torch.linspace(0.5, 2.0, n_heads))
        self.gate_alpha = nn.Parameter(torch.tensor(0.1))

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, distances: torch.Tensor = None, attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None):
        """Scaled dot-product attention with optional distance gating.

        attn_mask: (N, N) or (B, N, N) pairwise mask (True keep, False block).
        key_padding_mask: (B, N) mask for valid tokens (True valid, False pad); avoids building (B, N, N) externally.
        distances: (N, N) or (B, N, N); branched for RoPE/gating.
        """
        bsz, seq_len, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_heads, self.head_dim)

        is_batched_dist = (distances is not None) and (distances.dim() == 3)
        has_kpm = key_padding_mask is not None
        if has_kpm:
            key_padding_mask = key_padding_mask.to(x.device)
            if key_padding_mask.dim() == 1:
                key_padding_mask = key_padding_mask.unsqueeze(0)
            if distances is not None and not is_batched_dist:
                raise ValueError("key_padding_mask expects batched distances (B,N,N)")

        if distances is not None:
            if attn_mask is not None:
                attn_mask = attn_mask.to(distances.device)

            if is_batched_dist:
                if has_kpm:
                    k = key_padding_mask.to(distances.dtype)  # (B,N)
                    pair_weight = k[:, :, None] * k[:, None, :]  # (B,N,N) via broadcast
                    if attn_mask is not None and attn_mask.dim() == 3:
                        pair_weight = pair_weight * attn_mask.to(distances.dtype)
                    numer = (distances * pair_weight).sum(dim=(-1, -2))
                    denom = pair_weight.sum(dim=(-1, -2)).clamp_min(1.0)
                    mean = (numer / denom).clamp_min(1e-6)
                    distances = distances / mean.view(-1, 1, 1)
                elif attn_mask is not None:
                    weight = attn_mask.to(distances.dtype)
                    numer = (distances * weight).sum(dim=(-1, -2))
                    denom = weight.sum(dim=(-1, -2)) + 1e-6
                    mean = (numer / denom).clamp_min(1e-6)
                    distances = distances / mean.view(-1, 1, 1)
                else:
                    distances = distances / (distances.mean(dim=(-1, -2), keepdim=True) + 1e-6)
            else:
                if attn_mask is not None:
                    denom = distances[attn_mask].mean() + 1e-6
                else:
                    denom = distances.mean() + 1e-6
                distances = distances / denom

            if is_batched_dist:
                theta = distances.unsqueeze(1) * self.head_omega.view(1, self.n_heads, 1, 1)  # (B,H,N,N)
            else:
                theta = distances.unsqueeze(0) * self.head_omega.view(self.n_heads, 1, 1)      # (H,N,N)

            q_even, q_odd = xq[..., 0::2].permute(0, 2, 1, 3), xq[..., 1::2].permute(0, 2, 1, 3)
            k_even, k_odd = xk[..., 0::2].permute(0, 2, 1, 3), xk[..., 1::2].permute(0, 2, 1, 3)

            scores_cos = torch.matmul(q_even, k_even.transpose(-1, -2)) + torch.matmul(q_odd, k_odd.transpose(-1, -2))
            scores_sin = torch.matmul(q_even, k_odd.transpose(-1, -2)) - torch.matmul(q_odd, k_even.transpose(-1, -2))
            scores = (scores_cos * torch.cos(theta)) + (scores_sin * torch.sin(theta))
            scores = scores / math.sqrt(self.head_dim)
        else:
            q = xq.permute(0, 2, 1, 3)
            k = xk.permute(0, 2, 1, 3)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                m = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
            elif attn_mask.dim() == 3:
                m = attn_mask.unsqueeze(1)               # (B,1,N,N)
            else:
                raise ValueError("attn_mask must be (N,N) or (B,N,N)")
            scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min)

        if has_kpm:
            if not is_batched_dist:
                raise ValueError("key_padding_mask expects batched distances (B,N,N)")
            # only mask keys to avoid NaN rows; queries for padding will be zeroed later
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)

        v = xv.permute(0, 2, 1, 3)
        if distances is not None:
            alpha = F.softplus(self.gate_alpha)
            gate = torch.exp(-alpha * distances)  # (N,N) or (B,N,N)
            if attn_mask is not None:
                gate = gate * attn_mask.to(gate.dtype)
            if has_kpm:
                k = key_padding_mask.to(gate.dtype)
                if gate.dim() == 3:
                    gate = gate * k[:, None, :]  # mask keys only for batched distances
                elif gate.dim() == 2:
                    gate = gate * k[0][None, :]  # non-batched distances; use first batch entry
                else:
                    raise ValueError("Unexpected gate dimension for key_padding_mask masking")

            if is_batched_dist:
                N = distances.size(-1)
                eye = torch.eye(N, device=distances.device, dtype=gate.dtype).unsqueeze(0)  # (1,N,N)
                gate = gate + eye * (1.0 - gate)                                                            # (B,N,N)
                w = attn_weights * gate.unsqueeze(1)                                                        # (B,H,N,N)
            else:
                N = distances.size(0)
                eye = torch.eye(N, device=distances.device, dtype=gate.dtype)                               # (N,N)
                gate = gate + eye * (1.0 - gate)
                w = attn_weights * gate.unsqueeze(0).unsqueeze(0)                                           # (B,H,N,N)

            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
            output = torch.einsum('bhij,bhjd->bihd', w, v)
        else:
            output = torch.einsum('bhij,bhjd->bihd', attn_weights, v)
        output = output.reshape(bsz, seq_len, -1)
        if has_kpm:
            output = output * key_padding_mask[:, :, None].to(output.dtype)
        return self.wo(output)


class CrossAttention(nn.Module):
    """Distance-aware Cross-Attention: Q comes from one set of tokens, K/V from another.

    Design Goals:
    - Reuses the same head_omega / gate_alpha structure as Attention;
    - Supports padding masks and (B, N_q, N_k) distance matrices for global node-edge coupling.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.head_omega = nn.Parameter(torch.linspace(0.5, 2.0, n_heads))
        self.gate_alpha = nn.Parameter(torch.tensor(0.1))

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        distances: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask_q: torch.Tensor | None = None,
        key_padding_mask_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross attention with distance gating.

        q: (B, N_q, D), k/v: (B, N_k, D)
        distances: (B, N_q, N_k) or None
        attn_mask: (B, N_q, N_k) or (N_q, N_k), True=keep, False=mask
        key_padding_mask_q: (B, N_q); key_padding_mask_k: (B, N_k)
        """

        bsz, q_len, _ = q.shape
        _, k_len, _ = k.shape

        q_proj = self.wq(q).view(bsz, q_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k_proj = self.wk(k).view(bsz, k_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v_proj = self.wv(v).view(bsz, k_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Basic QK^T / sqrt(d)
        scores = torch.matmul(q_proj, k_proj.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # pair-wise mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                m = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,N_q,N_k)
            elif attn_mask.dim() == 3:
                m = attn_mask.unsqueeze(1)               # (B,1,N_q,N_k)
            else:
                raise ValueError("attn_mask must be (N_q,N_k) or (B,N_q,N_k)")
            scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min)

        if key_padding_mask_k is not None:
            k_mask = key_padding_mask_k.to(scores.device)
            if k_mask.dim() == 1:
                k_mask = k_mask.unsqueeze(0)
            scores = scores.masked_fill(~k_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        if key_padding_mask_q is not None:
            q_mask = key_padding_mask_q.to(scores.device)
            if q_mask.dim() == 1:
                q_mask = q_mask.unsqueeze(0)
        else:
            q_mask = None

        attn_weights = F.softmax(scores, dim=-1)

        # Distance gating (same exp(-alpha * d) logic as self-attention)
        if distances is not None:
            # distances: (B, N_q, N_k)
            if attn_mask is not None:
                distances = distances * attn_mask.to(distances.dtype)
            if key_padding_mask_k is not None:
                distances = distances * k_mask[:, None, :].to(distances.dtype)

            # Normalize to near the average scale within the batch to avoid numerical instability
            mean = distances.sum(dim=(-1, -2)) / (distances.numel() / bsz + 1e-6)
            mean = mean.clamp_min(1e-6).view(bsz, 1, 1)
            distances = distances / mean

            alpha = F.softplus(self.gate_alpha)
            gate = torch.exp(-alpha * distances)  # (B,N_q,N_k)

            # head-wise broadcast: (B,1,N_q,N_k)
            gate = gate.unsqueeze(1)
            w = attn_weights * gate
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            w = attn_weights

        # (B,H,N_q,N_k) x (B,H,N_k,D) -> (B,H,N_q,D)
        out = torch.einsum('bhij,bhjd->bhid', w, v_proj)
        out = out.permute(0, 2, 1, 3).reshape(bsz, q_len, -1)

        if q_mask is not None:
            out = out * q_mask[:, :, None].to(out.dtype)

        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads, head_dim)
        self.ffn = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cis=None, distances: torch.Tensor = None, attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None):
        x = x + self.attention(self.attention_norm(x), freqs_cis, distances, attn_mask, key_padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, head_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None, attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None):
        if positions is not None:
            positions = positions.to(x.device)
            distances = torch.cdist(positions, positions)
            for layer in self.layers:
                x = layer(x, distances=distances, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        else:
            freqs_cis = precompute_freqs_cis(self.head_dim, x.shape[1], device=x.device)
            for layer in self.layers:
                x = layer(x, freqs_cis=freqs_cis, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.norm(x)


class SandwichCrossTransformerBlock(nn.Module):
    """Two-stream Sandwich Cross-Transformer Block.

    Structure:
    1) Apply self-attention (spatial smoothing) + FFN to node stream h_V and edge stream h_E respectively;
    2) Apply cross-attention:
       - V update: Q=h_V, K/V=h_E
       - E update: Q=h_E, K/V=h_V

    Expects input in padded format:
    - h_V: (B, N_v, D), h_E: (B, N_e, D)
    - pos_V: (B, N_v, 2), pos_E: (B, N_e, 2)
    - mask_V: (B, N_v) bool, mask_E: (B, N_e) bool
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int, ff_dim: int):
        super().__init__()
        # Self-attention (spatial smoothing)
        self.self_V = Attention(dim, n_heads, head_dim)
        self.self_E = Attention(dim, n_heads, head_dim)
        # Cross-attention (field coupling)
        self.cross_V = CrossAttention(dim, n_heads, head_dim)  # V <- E
        self.cross_E = CrossAttention(dim, n_heads, head_dim)  # E <- V

        # Independent normalization and FFN
        self.norm_V_1 = RMSNorm(dim)
        self.norm_E_1 = RMSNorm(dim)
        self.ffn_V = FeedForward(dim, ff_dim)
        self.ffn_E = FeedForward(dim, ff_dim)
        self.norm_V_2 = RMSNorm(dim)
        self.norm_E_2 = RMSNorm(dim)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        pos_V: torch.Tensor,
        pos_E: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_E: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_v, D = h_V.shape
        _, N_e, _ = h_E.shape

        if mask_V is None:
            mask_V = torch.ones(B, N_v, dtype=torch.bool, device=h_V.device)
        if mask_E is None:
            mask_E = torch.ones(B, N_e, dtype=torch.bool, device=h_E.device)

        # --- 1. Self-attention (Spatial Smoothing) ---
        # Intra-node distance matrix
        dist_VV = torch.cdist(pos_V, pos_V)  # (B,N_v,N_v)
        pair_mask_V = mask_V[:, :, None] & mask_V[:, None, :]

        h_V = h_V + self.self_V(
            self.norm_V_1(h_V),
            distances=dist_VV,
            attn_mask=pair_mask_V,
            key_padding_mask=mask_V,
        )
        h_V = h_V + self.ffn_V(self.norm_V_1(h_V))

        # Intra-edge distance matrix
        dist_EE = torch.cdist(pos_E, pos_E)  # (B,N_e,N_e)
        pair_mask_E = mask_E[:, :, None] & mask_E[:, None, :]

        h_E = h_E + self.self_E(
            self.norm_E_1(h_E),
            distances=dist_EE,
            attn_mask=pair_mask_E,
            key_padding_mask=mask_E,
        )
        h_E = h_E + self.ffn_E(self.norm_E_1(h_E))

        # --- 2. Cross-attention (Field Coupling) ---
        # V <- E
        dist_VE = torch.cdist(pos_V, pos_E)  # (B,N_v,N_e)
        cross_mask_VE = mask_V[:, :, None] & mask_E[:, None, :]
        h_V = h_V + self.cross_V(
            self.norm_V_2(h_V),
            self.norm_E_2(h_E),
            self.norm_E_2(h_E),
            distances=dist_VE,
            attn_mask=cross_mask_VE,
            key_padding_mask_q=mask_V,
            key_padding_mask_k=mask_E,
        )

        # E <- V
        dist_EV = torch.cdist(pos_E, pos_V)  # (B,N_e,N_v)
        cross_mask_EV = mask_E[:, :, None] & mask_V[:, None, :]
        h_E = h_E + self.cross_E(
            self.norm_E_2(h_E),
            self.norm_V_2(h_V),
            self.norm_V_2(h_V),
            distances=dist_EV,
            attn_mask=cross_mask_EV,
            key_padding_mask_q=mask_E,
            key_padding_mask_k=mask_V,
        )

        return h_V, h_E


class SandwichCrossTransformer(nn.Module):
    """Multi-layer Sandwich Cross-Transformer.

    Expects input in padded format:
    - h_V: (B, N_v, D), h_E: (B, N_e, D)
    - pos_V: (B, N_v, 2), pos_E: (B, N_e, 2)
    - mask_V, mask_E: (B, N_v)/(B, N_e) bool

    Usage: After the first stage EH MetaLayer of EH-Evolver, pack node/edge hidden states
    and coordinates into a padded batch, then feed into this module for global context modeling.
    """

    def __init__(self, dim: int, n_layers: int, n_heads: int, head_dim: int, ff_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            SandwichCrossTransformerBlock(dim, n_heads, head_dim, ff_dim)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        pos_V: torch.Tensor,
        pos_E: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_E: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h_V, h_E = layer(h_V, h_E, pos_V, pos_E, mask_V, mask_E)
        return h_V, h_E


class LightAttentionIPA(nn.Module):
    """Inducing-Point Attn (IPA) with distance gating.

    Approximate fully-connected self-attention: All tokens interact through M inducing points, complexity O(N*M).

    Structure (Single-stream):
    1) X -> Z: Aggregate info from all tokens to inducing points Z (pooling, with distance gating)
    2) Z -> X: Broadcast from inducing points back to all tokens (broadcast, with distance gating)

    - h_X: (B, N, D)
    - pos_X: (B, N, 2)
    - mask_X: (B, N) bool, True=valid, False=pad
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int, num_inducing: int = 100):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.num_inducing = num_inducing

        # Use existing distance-aware CrossAttention as core operators for X->Z and Z->X
        self.pool_attn = CrossAttention(dim, n_heads, head_dim)      # Z <- X
        self.broadcast_attn = CrossAttention(dim, n_heads, head_dim) # X <- Z
        # Learnable scorer: scores each token for top-M inducing point selection
        self.scorer = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        h_X: torch.Tensor,
        pos_X: torch.Tensor,
        mask_X: torch.Tensor | None = None,
        inducing_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Single-stream Inducing Point Attention.
        h_X:   (B, N, D)
        pos_X: (B, N, 2)
        mask_X: (B, N) bool, True=valid, False=pad
        inducing_idx: (B, M) or (M,), externally specified inducing point indices (e.g., FPS sampled points)
        Returns: Updated h_X_new: (B, N, D)
        """
        B, N, D = h_X.shape
        device = h_X.device

        if mask_X is None:
            mask_X = torch.ones(B, N, dtype=torch.bool, device=device)

        # Inducing point index logic: Prioritize externally provided inducing_idx, otherwise use learnable scorer to select points
        if inducing_idx is not None:
            # Supports (B, M) or (M,) input
            if inducing_idx.dim() == 1:
                inducing_idx = inducing_idx.unsqueeze(0).expand(B, -1)
            M = inducing_idx.size(1)
            batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, M)
            h_Z = h_X[batch_idx, inducing_idx, :]
            pos_Z = pos_X[batch_idx, inducing_idx, :]
            mask_Z = mask_X[batch_idx, inducing_idx]
        else:
            # Number of inducing points M, not exceeding sequence length
            M = min(self.num_inducing, N)
            # Learnable scoring + batched top-M on valid tokens of each graph
            scores = self.scorer(h_X).squeeze(-1)  # (B, N)
            very_neg = torch.finfo(scores.dtype).min
            scores_masked = scores.masked_fill(~mask_X, very_neg)
            _, idx = torch.topk(scores_masked, k=M, dim=-1)  # (B, M)
            batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(B, M)
            h_Z = h_X[batch_idx, idx, :]
            pos_Z = pos_X[batch_idx, idx, :]
            mask_Z = mask_X[batch_idx, idx]

        # ---------------- 1) X -> Z: Aggregate all tokens to inducing points ----------------
        dist_ZX = torch.cdist(pos_Z, pos_X)  # (B, M, N)
        attn_mask_ZX = mask_Z[:, :, None] & mask_X[:, None, :]  # (B, M, N)
        h_Z_updated = self.pool_attn(
            q=h_Z,
            k=h_X,
            v=h_X,
            distances=dist_ZX,
            attn_mask=attn_mask_ZX,
            key_padding_mask_q=mask_Z,
            key_padding_mask_k=mask_X,
        )  # (B, M, D)

        # ---------------- 2) Z -> X: Broadcast back from inducing points to all tokens ----------------
        dist_XZ = torch.cdist(pos_X, pos_Z)  # (B, N, M)
        attn_mask_XZ = mask_X[:, :, None] & mask_Z[:, None, :]  # (B, N, M)
        h_X_updated = self.broadcast_attn(
            q=h_X,
            k=h_Z_updated,
            v=h_Z_updated,
            distances=dist_XZ,
            attn_mask=attn_mask_XZ,
            key_padding_mask_q=mask_X,
            key_padding_mask_k=mask_Z,
        )  # (B, N, D)

        return h_X_updated


class LightSandwichCrossTransformerBlock(nn.Module):
    """Inducing-point version of Sandwich Cross-Transformer Block.

    Consistent interface with SandwichCrossTransformerBlock, but self-attention uses
    LightAttentionIPA, reducing complexity from O(N^2) to O(N * M), M << N.

    - Self-attention: Node/Edge streams interact via inducing points (LightAttentionIPA)
    - Cross-attention: Still uses fully-connected CrossAttention + explicit distance gating
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int, ff_dim: int, num_inducing: int = 100):
        super().__init__()
        # Number of inducing points for nodes/edges (global fixed hyperparameters)
        self.M_node = 128
        self.M_edge = 512

        # Inducing point self-attention (spatial smoothing)
        self.self_V = LightAttentionIPA(dim, n_heads, head_dim, num_inducing=self.M_node)
        self.self_E = LightAttentionIPA(dim, n_heads, head_dim, num_inducing=self.M_edge)
        # Cross-attention (field coupling) still uses fully-connected distance gating
        self.cross_V = CrossAttention(dim, n_heads, head_dim)  # V <- E
        self.cross_E = CrossAttention(dim, n_heads, head_dim)  # E <- V

        # Scorer for K/V side in cross-attention:
        #   - When V <- E, score edge tokens to select top-M_edge as K/V;
        #   - When E <- V, score node tokens to select top-M_node as K/V.
        self.cross_score_E_for_V = nn.Linear(dim, 1, bias=False)
        self.cross_score_V_for_E = nn.Linear(dim, 1, bias=False)

        # Independent normalization and FFN
        self.norm_V_1 = RMSNorm(dim)
        self.norm_E_1 = RMSNorm(dim)
        self.ffn_V = FeedForward(dim, ff_dim)
        self.ffn_E = FeedForward(dim, ff_dim)
        self.norm_V_2 = RMSNorm(dim)
        self.norm_E_2 = RMSNorm(dim)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        pos_V: torch.Tensor,
        pos_E: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_E: torch.Tensor | None = None,
        node_fps_idx: torch.Tensor | None = None,
        edge_fps_idx: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_v, D = h_V.shape
        _, N_e, _ = h_E.shape

        if mask_V is None:
            mask_V = torch.ones(B, N_v, dtype=torch.bool, device=h_V.device)
        if mask_E is None:
            mask_E = torch.ones(B, N_e, dtype=torch.bool, device=h_E.device)

        # --- 1. Self-attention (Inducing point approximated spatial smoothing) ---
        h_V = h_V + self.self_V(self.norm_V_1(h_V), pos_V, mask_V)
        h_V = h_V + self.ffn_V(self.norm_V_1(h_V))

        h_E = h_E + self.self_E(self.norm_E_1(h_E), pos_E, mask_E)
        h_E = h_E + self.ffn_E(self.norm_E_1(h_E))

        # --- 2. Cross-attention (Field coupling, directly using FPS sampling indices) ---
        assert node_fps_idx is not None and edge_fps_idx is not None, 'FPS indices must be provided.'
        # Supports single batch or multi-batch
        if node_fps_idx.dim() == 1:
            node_fps_idx = node_fps_idx.unsqueeze(0)
        if edge_fps_idx.dim() == 1:
            edge_fps_idx = edge_fps_idx.unsqueeze(0)
        B, N_v, D = h_V.shape
        _, N_e, _ = h_E.shape
        Me = edge_fps_idx.size(1)
        Mv = node_fps_idx.size(1)
        batch_idx = torch.arange(B, device=h_E.device).unsqueeze(-1)
        h_E_sel = h_E[batch_idx, edge_fps_idx, :]
        pos_E_sel = pos_E[batch_idx, edge_fps_idx, :]
        mask_E_sel = mask_E[batch_idx, edge_fps_idx]

        dist_VE = torch.cdist(pos_V, pos_E_sel)
        cross_mask_VE = mask_V[:, :, None] & mask_E_sel[:, None, :]
        h_V = h_V + self.cross_V(
            self.norm_V_2(h_V),
            self.norm_E_2(h_E_sel),
            self.norm_E_2(h_E_sel),
            distances=dist_VE,
            attn_mask=cross_mask_VE,
            key_padding_mask_q=mask_V,
            key_padding_mask_k=mask_E_sel,
        )

        h_V_sel = h_V[batch_idx, node_fps_idx, :]
        pos_V_sel = pos_V[batch_idx, node_fps_idx, :]
        mask_V_sel = mask_V[batch_idx, node_fps_idx]

        dist_EV = torch.cdist(pos_E, pos_V_sel)
        cross_mask_EV = mask_E[:, :, None] & mask_V_sel[:, None, :]
        h_E = h_E + self.cross_E(
            self.norm_E_2(h_E),
            self.norm_V_2(h_V_sel),
            self.norm_V_2(h_V_sel),
            distances=dist_EV,
            attn_mask=cross_mask_EV,
            key_padding_mask_q=mask_E,
            key_padding_mask_k=mask_V_sel,
        )

        return h_V, h_E


class LightSandwichCrossTransformer(nn.Module):
    """Multi-layer Inducing Point Sandwich Cross-Transformer.

    Same interface as SandwichCrossTransformer, but internal Block uses
    LightSandwichCrossTransformerBlock (Self-attention is Inducing Point IPA).
    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        num_inducing: int = 100,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LightSandwichCrossTransformerBlock(dim, n_heads, head_dim, ff_dim, num_inducing=num_inducing)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        pos_V: torch.Tensor,
        pos_E: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_E: torch.Tensor | None = None,
        node_fps_idx: torch.Tensor | None = None,
        edge_fps_idx: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h_V, h_E = layer(h_V, h_E, pos_V, pos_E, mask_V, mask_E, node_fps_idx, edge_fps_idx)
        return h_V, h_E


class TransformerNodeRegressor(nn.Module):
    """Node-only model with two batching strategies:

    mode="mask": concat batch, block-diagonal attn mask (cross-graph blocked).
    mode="pad": per-graph padding to (B, Nmax, C) with key-padding mask; no cross-graph distances.
    """

    def __init__(self, in_dim=9, model_dim=256, n_layers=2, n_heads=8, head_dim=32, ff_dim=256,
                 out_dim=3, mode: str = "mask"):
        super().__init__()
        self.embed = nn.Linear(in_dim, model_dim)
        self.transformer = SimpleTransformer(dim=model_dim, n_layers=n_layers, n_heads=n_heads, head_dim=head_dim, hidden_dim=ff_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, out_dim)
        )
        self.mode = mode.lower()

    def _forward_mask_mode(self, data):
        x = data.x
        positions = getattr(data, 'points', None)
        attn_mask = None
        if hasattr(data, "batch"):
            attn_mask = data.batch.unsqueeze(1).eq(data.batch.unsqueeze(0))

        x = self.embed(x)
        x = self.transformer(x.unsqueeze(0), positions=positions, attn_mask=attn_mask).squeeze(0)
        return self.head(x)

    def _forward_pad_mode(self, data):
        batch = getattr(data, 'batch', torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device))

        x_pad, mask = to_dense_batch(data.x, batch)  # mask: (B, Nmax)
        pos_pad = None
        if hasattr(data, 'points') and data.points is not None:
            pos_pad, _ = to_dense_batch(data.points, batch)

        x_pad = self.embed(x_pad)
        x_pad = self.transformer(x_pad, positions=pos_pad, key_padding_mask=mask)
        x_out = self.head(x_pad)

        return x_out[mask]

    def forward(self, data):
        if self.mode == "pad":
            return self._forward_pad_mode(data)
        return self._forward_mask_mode(data)