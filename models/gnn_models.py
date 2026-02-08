import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GATv2Conv

# Optional DGL import for EGAT
_DGL_AVAILABLE = True
try:
    import dgl  # type: ignore
    try:
        # DGL >= 0.9/2.x
        from dgl.nn.pytorch import EGATConv as DGL_EGATConv  # type: ignore
    except Exception:  # pragma: no cover
        from dgl.nn import EGATConv as DGL_EGATConv  # type: ignore
except Exception:  # pragma: no cover
    _DGL_AVAILABLE = False

# --- 1. Learnable Fourier Time Encoding ---
class LearnableFourierTimeEncoding(nn.Module):
    """Learnable Fourier features for scalar time t.

    Instead of fixed 10000^{-2k/D}, we learn frequencies w_k and encode
    t via [sin(w_k t), cos(w_k t)], so the model can lock onto the
    physical oscillation frequencies of the EM fields.
    """

    def __init__(self, time_dim: int = 32):
        super().__init__()
        assert time_dim % 2 == 0, "time_dim should be even (for sin/cos pairs)"
        half_dim = time_dim // 2
        freq_init = torch.linspace(0.0, 1.0, steps=half_dim)
        freq_init = torch.exp(freq_init * math.log(1000.0))  # ~[1, 1e3]
        self.freq = nn.Parameter(freq_init)  # (half_dim,)

    def forward(self, t: torch.Tensor, device=None) -> torch.Tensor:
        """t: (N_time,) or scalar; returns (N_time, time_dim)."""
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=device or self.freq.device)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        w = self.freq.to(t.device)              # (half_dim,)
        phases = t.unsqueeze(-1) * w.unsqueeze(0)  # (N_time, half_dim)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return emb


def get_sinusoidal_time_encoding(t, dim: int = 32, device=None):
    """Backward-compatible sinusoidal time encoding.

    Kept for models.models.CombinedGNNTransformerModel which still calls
    get_sinusoidal_time_encoding. DeepLeapfrogModel now uses
    LearnableFourierTimeEncoding instead.
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32, device=device)
    if t.ndim == 0:
        t = t.unsqueeze(0)

    half_dim = dim // 2
    inv_freq = math.log(10000.0) / max(half_dim - 1, 1)
    inv_freq = torch.exp(torch.arange(half_dim, device=t.device) * -inv_freq)
    angles = t.unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb

# --- 2. AdaGN: Adaptive Group Normalization ---
class AdaGN(nn.Module):
    def __init__(self, num_features, cond_dim, num_groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_features)
        self.fc = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        # x: (N, C), cond: (N, cond_dim) or (1, cond_dim)
        if cond.dim() == 2 and cond.shape[0] != x.shape[0]:
             cond = cond.expand(x.shape[0], -1) # Broadcast if needed
             
        normed = self.gn(x)
        style = self.fc(cond)
        gamma, beta = style.chunk(2, dim=-1)
        return normed * (1 + gamma) + beta

# --- 3. Maxwell Physics-Informed GNN Block  ---
class MaxwellGNNBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim=32, heads=4, dropout=0.1, mode='H_update'):
        """
        mode: 
          - 'H_update': Uses (h_j - h_i) and orthogonal displacement, simulating gradient drive (E -> H)
          - 'E_update': Uses (h_j + h_i) and parallel displacement, simulating curl drive (H -> E)
        """
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        # 1. Modulation layer: Used to modulate the input "Drive Field" (Source Field)
        # Here cond_dim is fixed to hidden_dim, requiring the input t_emb dimension to match hidden_dim
        self.source_adagn = AdaGN(hidden_dim, hidden_dim)

        # 2. Edge feature dimension definition
        # edge_attr = [ physical quantity difference or sum (D) || geometric vector (2) || (optional) epsilon (1) ]
        if mode == 'E_update':
            # E_update additionally requires epsilon
            edge_input_dim = hidden_dim + 2 + 1 
        else: # H_update
            edge_input_dim = hidden_dim + 2 

        # 3. GATv2Conv
        # Note: The edge_dim parameter of GATv2 is used to receive the physical edge features we need to inject
        self.gat = GATv2Conv(hidden_dim, hidden_dim // heads, 
                             heads=heads, 
                             dropout=dropout, 
                             edge_dim=edge_input_dim, 
                             concat=True)
        
        # Projection after merging multi-heads
        self.post_gat = nn.Linear(hidden_dim, hidden_dim)

        # 4. Update MLP (Residual Update)
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, h_target, h_source, edge_index, pos, t_emb, epsilon=None):
        """
        h_target: The field to be updated (e.g., H_old), dimension (N, D)
        h_source: The driving field (e.g., E_old), dimension (N, D)
        pos: Node coordinates (N, 2)
        t_emb: Time encoding (1, D)
        epsilon: Permittivity (N, 1), only required for E_update
        """
        
        # --- Step 1: Time-modulate the driving field ---
        # Simulates the variation of the wave source with time, or the gain of the operator with time step
        h_source_mod = self.source_adagn(h_source, t_emb)

        # --- Step 2: Construct Physics-Geometric Edge Features ---
        src, dst = edge_index
        
        # Calculate relative position and distance
        rel_pos = pos[src] - pos[dst] # j -> i vector (dx, dy)
        dist_sq = rel_pos.pow(2).sum(dim=-1, keepdim=True) + 1e-8 # d^2 + eta
        
        # Normalized geometric vector (simulates 1/d^2 scaling)
        # Recommended to use 1/d * unit_vec, numerically more stable, strictly following the formula here
        geom_factor = rel_pos / dist_sq # (E, 2)
        
        # Calculate "Driving Term" based on physical mode
        if self.mode == 'H_update':
            # Gradient Mode: Focus on difference between two points (Potential Difference)
            # Formula corresponds to: (E_j - E_i)
            phys_term = h_source_mod[src] - h_source_mod[dst] # (E, D)
            
            # H_update also requires orthogonal rotation [-dy, dx] to simulate curl(E)
            # Here we pass in the original geom and let GAT learn this rotation relationship internally
            # Or explicit rotation:
            ortho_geom = torch.stack([-geom_factor[:, 1], geom_factor[:, 0]], dim=-1)
            
            edge_attr = torch.cat([phys_term, ortho_geom], dim=-1) # (E, D+2)
            
        elif self.mode == 'E_update':
            # Circulation Mode: Focus on average flux on edge (Flux Accumulation)
            # Formula corresponds to: (H_j + H_i)
            phys_term = h_source_mod[src] + h_source_mod[dst] # (E, D)
            
            # E_update must include medium information epsilon
            # epsilon is generally defined on nodes, take the average of the edge
            if epsilon is None:
                raise ValueError("E_update requires epsilon input")
            eps_edge = (epsilon[src] + epsilon[dst]) / 2.0
            
            edge_attr = torch.cat([phys_term, geom_factor, eps_edge], dim=-1) # (E, D+2+1)

        # --- Step 3: Physics GAT Aggregation ---
        # GAT calculates attention weights based on edge_attr (physical driving force)
        # The node feature input to GAT is h_source (driving source)
        aggr_out = self.gat(h_source_mod, edge_index, edge_attr=edge_attr)
        aggr_out = self.post_gat(aggr_out) # Project back to D dim

        # --- Step 4: Residual Update ---
        # h_new = h_old + Delta
        delta = self.update_mlp(aggr_out)
        h_new = h_target + delta
        
        return h_new

"""Deep Leapfrog-style model using MaxwellGNNBlock.

This is a lighter, GNN-only version of the Maxwell-style update, mainly as a
reference implementation of how to chain four MaxwellGNNBlock blocks.
It consumes the same node features as load_data (Ez, Hx, Hy, ..., eps, t, step)
and returns updated embeddings for E and H at time t.
"""

class MaxwellGeometricGNNBlock(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.1, mode='H_update'):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        
        # 1. Determine Edge Feature Input Dimension
        # We do manual multiplication in the code, so edge_attr dimension depends on calculation result
        if mode == 'H_update':
            # Formula: (dE/d^2) * [-dy, dx]
            # Result is a vector, so dimension is hidden_dim * 2 (x component + y component)
            self.edge_input_dim = hidden_dim * 2
        else: # E_update
            # Formula: (Hy*dx - Hx*dy) / d^2
            # H is split into two halves (D/2), result is a scalar (D/2)
            # Also need to add epsilon (1)
            # So dimension is hidden_dim // 2 + 1
            self.edge_input_dim = (hidden_dim // 2) + 1

        # 2. Physics GATv2
        self.gat = GATv2Conv(hidden_dim, hidden_dim // heads, 
                             heads=heads, 
                             edge_dim=self.edge_input_dim, # Receives explicitly calculated physical edge features
                             concat=True,
                             dropout=dropout)
        
        self.post_gat = nn.Linear(hidden_dim, hidden_dim)
        
        # Time modulation layer: Consistent with MaxwellGNNBlock, cond_dim = hidden_dim
        self.source_adagn = AdaGN(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h_target, h_source, edge_index, pos, t_emb, epsilon=None):
        # 1. Modulate Driving Field
        h_src_mod = self.source_adagn(h_source, t_emb)
        
        src, dst = edge_index
        
        # 2. Pre-calculate Geometric Parameters
        # rel_pos: [dx, dy]
        rel_pos = pos[dst] - pos[src]  # Note direction: j(dst) - i(src) or src - dst, depends on edge_index definition
        # Assuming edge_index is [j, i], i.e., source -> target
        # Then pos[src] is j, pos[dst] is i
        d_vec = pos[src] - pos[dst] # j - i
        dx = d_vec[:, 0:1]
        dy = d_vec[:, 1:2]
        dist_sq = d_vec.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8) # d^2
        
        # ==========================================================
        # Core Modification: Strictly execute physical formulas
        # ==========================================================
        
        if self.mode == 'H_update':
            # --- Target Formula: \frac{E_j - E_i}{d^2} * [-dy, dx] ---
            
            # 1. Calculate Scalar Gradient Magnitude
            # h_src_mod here represents E (N, D)
            grad_mag = (h_src_mod[src] - h_src_mod[dst]) / dist_sq # (E, D)
            
            # 2. Explicitly multiply geometric vector [-dy, dx]
            # Result dimension: (E, D) * (E, 1) -> (E, D)
            # We will generate two components, concatenated into 2D vector field features
            vec_x = grad_mag * (-dy) 
            vec_y = grad_mag * (dx)
            
            # 3. Construct Edge Attr: [x_component, y_component] -> (E, 2D)
            edge_attr_phys = torch.cat([vec_x, vec_y], dim=-1) 
            
        elif self.mode == 'E_update':
            # --- Target Formula: \frac{(Hy_j + Hy_i)dx - (Hx_j + Hx_i)dy}{d^2} ---
            
            # 1. Explicitly split vector field H into x, y components
            # Assuming first half of latent dim is Hx, second half is Hy
            half_d = self.hidden_dim // 2
            h_Hx = h_src_mod[:, :half_d]
            h_Hy = h_src_mod[:, half_d:]
            
            # 2. Calculate average field (Trapezoidal integration approximation)
            Hx_avg = h_Hx[src] + h_Hx[dst] # (E, D/2)
            Hy_avg = h_Hy[src] + h_Hy[dst] # (E, D/2)
            
            # 3. Execute Cross Product / Circulation
            # (Hy * dx - Hx * dy) / d^2
            circulation = (Hy_avg * dx - Hx_avg * dy) / dist_sq # (E, D/2)
            
            # 4. Inject permittivity (Mandatory)
            if epsilon is None:
                raise ValueError("E_update requires epsilon")
            # epsilon is defined on nodes, take average on edge
            eps_edge = (epsilon[src] + epsilon[dst]) / 2.0 # (E, 1)
            
            # 5. Construct Edge Attr
            edge_attr_phys = torch.cat([circulation, eps_edge], dim=-1) # (E, D/2 + 1)

        # ==========================================================

        # 3. Input to GATv2
        # GAT attention mechanism now works based on our explicitly calculated "physical driving terms"
        aggr_out = self.gat(h_src_mod, edge_index, edge_attr=edge_attr_phys)
        aggr_out = self.post_gat(aggr_out)
        
        # 4. Residual Update
        h_new = h_target + self.update_mlp(aggr_out)
        
        return h_new

class DeepLeapfrogModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, time_dim: int = 32, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # learnable Fourier time encoder
        self.time_encoder = LearnableFourierTimeEncoding(time_dim)

        # Simple Field Encoders:
        # h_E^0 = encE(Ez, eps), h_H^0 = encH(Hx, Hy, eps)
        self.encE = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.encH = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

        # Time Encoding MLP: tau = MLP_time(PE(t)), where PE(t) is learnable Fourier features
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 4 MaxwellGNNBlock, updated alternately by H/E
        # Original implementation: First two layers also used MaxwellGNNBlock; now commented out for geometric experiment version to avoid DDP unused params error
        # self.block1_H = MaxwellGNNBlock(hidden_dim, time_dim=time_dim, heads=heads, dropout=dropout, mode='H_update')
        # self.block2_E = MaxwellGNNBlock(hidden_dim, time_dim=time_dim, heads=heads, dropout=dropout, mode='E_update')
        self.block3_H = MaxwellGNNBlock(hidden_dim, time_dim=time_dim, heads=heads, dropout=dropout, mode='H_update')
        self.block4_E = MaxwellGNNBlock(hidden_dim, time_dim=time_dim, heads=heads, dropout=dropout, mode='E_update')

        # Geometric Maxwell Blocks: Explicitly inject gradient/curl info in initial phase
        # Experimental usage: First two layers use MaxwellGeometricGNNBlock, last two layers still use MaxwellGNNBlock
        self.geo_block1_H = MaxwellGeometricGNNBlock(hidden_dim, heads=heads, dropout=dropout, mode='H_update')
        self.geo_block2_E = MaxwellGeometricGNNBlock(hidden_dim, heads=heads, dropout=dropout, mode='E_update')

        # Decoding Head: Recover physical quantities Ez, Hx, Hy from latent states
        self.decE = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.decH = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, data):
        """data: PyG Data, must contain x, edge_index, points.

        x: [Ez, Hx, Hy, eps] (Provided by load_data_v2)
        Returns: Predicted fields [Ez_pred, Hx_pred, Hy_pred], shape [N, 3].
        """

        x, edge_index = data.x, data.edge_index
        points = getattr(data, 'points', None)
        if points is None:
            raise ValueError("DeepLeapfrogModel requires data.points (node coordinates)")
        pos = points[:, :2]

        # Extract Physical Quantities
        # Now x only contains Ez, Hx, Hy, eps four channels
        Ez = x[:, 0]
        Hx = x[:, 1]
        Hy = x[:, 2]
        eps = x[:, 3:4]  # (N, 1)

        # Global time t: One scalar per graph (Already made relative time difference from initial frame in load_data_v2)
        if not hasattr(data, "t"):
            raise ValueError("DeepLeapfrogModel expects time attribute 't' on data")

        t_attr = data.t

        # Determine batch info
        if hasattr(data, "batch") and data.batch is not None:
            batch_idx = data.batch.to(x.device)
            num_graphs = int(batch_idx.max()) + 1
        else:
            # Single graph case: Construct an all-zero batch_idx
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            num_graphs = 1

        # Parse t_attr into (num_graphs,) tensor (one scalar time per graph)
        if torch.is_tensor(t_attr):
            t_tensor = t_attr.to(x.device).view(-1)
            if t_tensor.numel() == 1:
                graph_t = t_tensor.expand(num_graphs)
            else:
                assert t_tensor.numel() == num_graphs, \
                    f"Expected data.t to have {num_graphs} elements for {num_graphs} graphs, got {t_tensor.numel()}"
                graph_t = t_tensor
        else:
            t_scalar = float(t_attr)
            graph_t = torch.full((num_graphs,), t_scalar, dtype=torch.float32, device=x.device)

        # Key: Leapfrog needs two sets of time encodings: t/2 and t
        graph_t_full = graph_t              # t
        graph_t_half = graph_t * 0.5        # t/2

        # Encode t/2 (For Block 1 & 2), using learnable Fourier features
        pe_half = self.time_encoder(graph_t_half, device=x.device)   # (num_graphs, time_dim)
        t_embed_half = self.time_mlp(pe_half)                        # (num_graphs, hidden_dim)
        t_vec_half = t_embed_half[batch_idx]                         # (N, hidden_dim)

        # Encode t (For Block 3 & 4), same set of learnable frequencies
        pe_full = self.time_encoder(graph_t_full, device=x.device)   # (num_graphs, time_dim)
        t_embed_full = self.time_mlp(pe_full)                        # (num_graphs, hidden_dim)
        t_vec_full = t_embed_full[batch_idx]                         # (N, hidden_dim)

        # Initial Encoding h_E^0, h_H^0
        hE0_in = torch.stack([Ez, eps.squeeze(-1)], dim=-1)  # (N, 2)
        hH0_in = torch.stack([Hx, Hy, eps.squeeze(-1)], dim=-1)  # (N, 3)
        h_E = self.encE(hE0_in)
        h_H = self.encH(hH0_in)

        # Block 1 & 2: 0 -> t/2, Geometric update H/E (using geometric formulas), time modulation consistent with MaxwellGNNBlock
        # Original: Used MaxwellGNNBlock, injected geometry only via edge_attr
        # h_H_half = self.block1_H(h_target=h_H, h_source=h_E, edge_index=edge_index, pos=pos, t_emb=t_vec_half)
        # h_E_half = self.block2_E(h_target=h_E, h_source=h_H_half, edge_index=edge_index, pos=pos, t_emb=t_vec_half, epsilon=eps)

        # Geometric version now also uses t_vec_half (N, hidden_dim) as cond vector for AdaGN
        h_H_half = self.geo_block1_H(h_target=h_H, h_source=h_E, edge_index=edge_index, pos=pos, t_emb=t_vec_half)
        h_E_half = self.geo_block2_E(h_target=h_E, h_source=h_H_half, edge_index=edge_index, pos=pos, t_emb=t_vec_half, epsilon=eps)

        # (Optional) Insert global Transformer based on h_E_half / h_H_half here, using t_vec_full as condition

        # Block 3: t/2 -> t, Update H (Using t_vec_full)
        h_H_t = self.block3_H(h_target=h_H_half, h_source=h_E_half, edge_index=edge_index, pos=pos, t_emb=t_vec_full)

        # Block 4: t/2 -> t, Update E (Using t_vec_full)
        h_E_t = self.block4_E(h_target=h_E_half, h_source=h_H_t, edge_index=edge_index, pos=pos, t_emb=t_vec_full, epsilon=eps)

        # Decode to physical quantities
        Ez_pred = self.decE(h_E_t)          # (N, 1)
        H_pred = self.decH(h_H_t)          # (N, 2) -> Hx_pred, Hy_pred
        out = torch.cat([Ez_pred, H_pred], dim=-1)  # (N, 3)

        return out


class GAT_layer(nn.Module):
    """Edge-aware GAT using PyG; returns node updates and attention-derived edge scores."""

    def __init__(self, in_vertex_dim, out_vertex_dim, num_heads, edge_dim=None):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.gatconv = GATConv(
            in_channels=in_vertex_dim,
            out_channels=out_vertex_dim,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False,
            bias=True,
        )
        self.node_norm_layers = nn.LayerNorm(num_heads * out_vertex_dim)
        self.edge_norm_layers = nn.LayerNorm(num_heads)
        self.edge_linear = nn.Sequential(nn.Linear(num_heads, 1), nn.ReLU())

    def forward(self, edge_index, point_attr, edge_attr):
        # PyG returns (node_out, (edge_index, attn_weights)) when return_attention_weights=True
        point_attr, attn = self.gatconv(
            point_attr,
            edge_index,
            edge_attr,
            return_attention_weights=True,
        )
        point_attr = self.node_norm_layers(point_attr)
        point_attr = self.act(point_attr)

        attn_weights = attn[1] if isinstance(attn, tuple) else attn  # shape (E, num_heads)
        if attn_weights.dim() == 3:
            attn_weights = attn_weights.mean(dim=1)
        edge_attr_out = self.edge_norm_layers(attn_weights)
        edge_attr_out = self.edge_linear(edge_attr_out).squeeze(1)
        return point_attr, edge_attr_out


class GATConvNetwork(nn.Module):
    """GAT (PyG version) that outputs both node and edge features."""

    def __init__(self, in_vertex_dim=9, in_edge_dim=4, hidden_dim=256, num_heads=2, out_vertex_dim=1, out_edge_dim=2):
        super().__init__()
        self.in_vertex_layers = nn.ModuleList([
            nn.Linear(in_vertex_dim, 32),
            nn.Linear(32, 64),
            nn.Linear(64, hidden_dim)
        ])
        self.in_edge_layers = nn.ModuleList([
            nn.Linear(in_edge_dim, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 1)
        ])
        self.out_node_layers = nn.ModuleList([
            nn.Linear(num_heads * hidden_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, out_vertex_dim)
        ])
        self.out_edge_layers = nn.ModuleList([
            nn.Linear(1, 64),
            nn.Linear(64, 32),
            nn.Linear(32, out_edge_dim)
        ])
        self.gatconv1 = GAT_layer(hidden_dim, hidden_dim, num_heads, edge_dim=1)
        self.gatconv2 = GAT_layer(num_heads * hidden_dim, hidden_dim, num_heads, edge_dim=1)
        self.gatconv3 = GAT_layer(num_heads * hidden_dim, hidden_dim, num_heads, edge_dim=1)
        self.gatconv4 = GAT_layer(num_heads * hidden_dim, hidden_dim, num_heads, edge_dim=1)

    def forward(self, data):
        point_attr, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for layer in self.in_vertex_layers:
            point_attr = F.relu(layer(point_attr))

        for layer in self.in_edge_layers:
            edge_attr = F.relu(layer(edge_attr))
        edge_attr = edge_attr.view(edge_attr.size(0), -1)

        point_attr, edge_attr = self.gatconv1(edge_index, point_attr, edge_attr)
        edge_attr = edge_attr.view(-1, 1)
        point_attr, edge_attr = self.gatconv2(edge_index, point_attr, edge_attr)
        edge_attr = edge_attr.view(-1, 1)
        point_attr, edge_attr = self.gatconv3(edge_index, point_attr, edge_attr)
        edge_attr = edge_attr.view(-1, 1)
        point_attr, edge_attr = self.gatconv4(edge_index, point_attr, edge_attr)

        for layer in self.out_node_layers:
            point_attr = layer(point_attr)

        edge_attr = edge_attr.unsqueeze(1)
        for layer in self.out_edge_layers:
            edge_attr = layer(edge_attr)

        return point_attr, edge_attr


class GAT_layer_node_only(nn.Module):
    """Simplified version: Outputs only node features, using PyG GATConv."""

    def __init__(self, in_vertex_dim, out_vertex_dim, num_heads, edge_dim=None):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.gatconv = GATConv(
            in_channels=in_vertex_dim,
            out_channels=out_vertex_dim,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False,
            bias=True,
        )
        self.node_norm = nn.LayerNorm(num_heads * out_vertex_dim)

    def forward(self, point_attr, edge_index, edge_attr):
        point_attr = self.gatconv(point_attr, edge_index, edge_attr)
        point_attr = self.node_norm(point_attr)
        point_attr = self.activation(point_attr)
        return point_attr


class GATBlock(nn.Module):
    def __init__(self, dim, heads=4, edge_dim=None, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = GATv2Conv(
            in_channels=dim,
            out_channels=dim,      # Key: Dimension remains dim
            heads=heads,
            concat=False,          # Key: No multiplication by heads
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
            bias=True,
        )
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(self.norm(x), edge_index, edge_attr)
        x = x + self.drop(h)       # residual
        return self.act(x)

class GATConvNetworkNodeOnly(nn.Module):
    def __init__(self, out_vertex_dim=3, in_vertex_dim=6, edge_dim=4, hidden_dim=256, 
                 edge_hidden=32, heads=4, dropout=0.1, num_layers=4):
        super().__init__()

        # node encoder
        self.node_in = nn.Sequential(
            nn.Linear(in_vertex_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )

        # edge encoder (Strongly recommended)
        self.edge_in = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, edge_hidden),
        )

        self.blocks = nn.ModuleList([
            GATBlock(hidden_dim, heads=heads, edge_dim=edge_hidden, dropout=dropout)
            for _ in range(num_layers)
        ])

        # output head (Note non-linearity)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_vertex_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_in(x)
        edge_attr = self.edge_in(edge_attr)

        for blk in self.blocks:
            x = blk(x, edge_index, edge_attr)

        return self.out(x)
  

class EGAT_layer(nn.Module):
    """DGL EGATConv version: Synchronously updates node and edge features using dgl.nn.EGATConv.

    Supports two forward interfaces:
    - forward(graph, node_feat, edge_feat)
    - forward(edge_index, node_feat, edge_feat)  // Automatically constructs DGLGraph

    And projects multi-head output (H * out_dim) back to (out_dim) via a linear layer, facilitating subsequent modules to run with fixed dimensions.
    """

    def __init__(self, in_vertex_dim, in_edge_dim, out_vertex_dim, out_edge_dim, num_heads, hidden_dim):
        super().__init__()
        if not _DGL_AVAILABLE:
            raise ImportError(
                "DGL is required for EGAT_layer. Please install a matching wheel, e.g.:\n"
                "pip install tools/dgl-2.4.0+cu118-cp310-cp310-manylinux1_x86_64.whl\n"
                "or tools/dgl-2.2.0+cu118-cp310-cp310-manylinux1_x86_64.whl"
            )

        self.in_vertex_dim = in_vertex_dim
        self.in_edge_dim = in_edge_dim
        self.out_vertex_dim = out_vertex_dim
        self.out_edge_dim = out_edge_dim
        self.num_heads = num_heads

        # DGL EGATConv (Using named arguments, compatible with newer signatures)
        self.gatconv = DGL_EGATConv(
            in_node_feats=in_vertex_dim,
            in_edge_feats=in_edge_dim,
            out_node_feats=out_vertex_dim,
            out_edge_feats=out_edge_dim,
            num_heads=num_heads,
            bias=True,
        )

        # Map back to fixed dimension after multi-head merging
        self.vertex_linear = nn.Sequential(
            nn.Linear(num_heads * out_vertex_dim, out_vertex_dim),
            nn.ReLU(),
        )
        self.edge_linear = nn.Sequential(
            nn.Linear(num_heads * out_edge_dim, out_edge_dim),
            nn.ReLU(),
        )

        # Normalization and Activation (Normalize by final projection dimension)
        self.node_norm_layers = nn.LayerNorm(out_vertex_dim)
        self.edge_norm_layers = nn.LayerNorm(out_edge_dim)
        self.act = nn.LeakyReLU()

    def forward(self, g, point_attr, edge_attr):
        """Use pre-built DGLGraph to avoid performance loss from repeated graph construction."""
        node_out, edge_out = self.gatconv(g, point_attr, edge_attr)

        # Merge multi-head dimensions and linearly project back to out_dim
        if node_out.dim() == 3:
            node_out = node_out.reshape(node_out.size(0), -1)
        if edge_out.dim() == 3:
            edge_out = edge_out.reshape(edge_out.size(0), -1)

        node_out = self.vertex_linear(node_out)
        edge_out = self.edge_linear(edge_out)

        node_out = self.act(self.node_norm_layers(node_out))
        edge_out = self.act(self.edge_norm_layers(edge_out))
        return node_out, edge_out