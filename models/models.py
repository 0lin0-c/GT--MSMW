from models.gnn_models import AdaGN, get_sinusoidal_time_encoding, MaxwellGNNBlock, LearnableFourierTimeEncoding, MaxwellGeometricGNNBlock
from models.gnn_MetaLayers import ExplicitPhysicsMetaLayer, compute_physics_from_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATv2Conv
from models.transformer import SimpleTransformer, SandwichCrossTransformer, LightSandwichCrossTransformer
from models.gnn_models import EGAT_layer, GAT_layer_node_only, GAT_layer, GATConvNetworkNodeOnly, GATBlock

class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, time_emb):
        scale, shift = self.emb(time_emb).chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, n_layers, dropout):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        output = self.dropout(src)
        output = self.encoder(output)
        output = self.dropout(output)
        return output


class EGATTransformerNetwork(nn.Module):
    def __init__(self, in_vertex_dim=9, in_edge_dim=4, num_heads=2, hidden_dim=256, out_vertex_dim=1, out_edge_dim=2):
        super().__init__()
        self.in_vertex_layers = nn.ModuleList([
            nn.Linear(in_vertex_dim, 32),
            nn.Linear(32, 64),
            nn.Linear(64, hidden_dim)
        ])
        self.in_edge_layers = nn.ModuleList([
            nn.Linear(in_edge_dim, 32),
            nn.Linear(32, 64),
            nn.Linear(64, hidden_dim)
        ])

        self.node_transformer = SimpleTransformer(
            dim=hidden_dim * num_heads,
            n_layers=2,
            n_heads=8,
            head_dim=32,
            hidden_dim=256
        )
        self.edge_transformer = SimpleTransformer(
            dim=hidden_dim * num_heads,
            n_layers=2,
            n_heads=8,
            head_dim=32,
            hidden_dim=256
        )

        edge_hidden_dim = hidden_dim
        self.gatconv1 = EGAT_layer(hidden_dim, edge_hidden_dim, hidden_dim, edge_hidden_dim, num_heads, hidden_dim)
        self.gatconv2 = EGAT_layer(hidden_dim * num_heads, edge_hidden_dim, hidden_dim, edge_hidden_dim, num_heads, hidden_dim)
        self.gatconv3 = EGAT_layer(hidden_dim * num_heads, edge_hidden_dim, hidden_dim, edge_hidden_dim, num_heads, hidden_dim)
        self.gatconv4 = EGAT_layer(hidden_dim * num_heads, edge_hidden_dim, hidden_dim, edge_hidden_dim, num_heads, hidden_dim)

        self.node_out_layers = nn.ModuleList([
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, out_vertex_dim)
        ])
        self.edge_out_layers = nn.ModuleList([
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, out_edge_dim)
        ])

    def forward(self, data):
        point_attr, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        points = getattr(data, 'points', None)

        for layer in self.in_vertex_layers:
            point_attr = layer(point_attr)
            point_attr = F.relu(point_attr)

        for layer in self.in_edge_layers:
            edge_attr = layer(edge_attr)
            edge_attr = F.relu(edge_attr)

        point_attr, edge_attr = self.gatconv1(edge_index, point_attr, edge_attr)
        point_attr, edge_attr = self.gatconv2(edge_index, point_attr, edge_attr)
        point_attr, edge_attr = self.gatconv3(edge_index, point_attr, edge_attr)
        point_attr, edge_attr = self.gatconv4(edge_index, point_attr, edge_attr)

        node_positions = points if points is not None else None
        if points is not None:
            src, dst = edge_index
            edge_positions = (points[src] + points[dst]) / 2.0
        else:
            edge_positions = None

        point_attr = self.node_transformer(point_attr.unsqueeze(0), positions=node_positions).squeeze(0)
        edge_attr = self.edge_transformer(edge_attr.unsqueeze(0), positions=edge_positions).squeeze(0)

        for layer in self.node_out_layers:
            point_attr = layer(point_attr)

        for layer in self.edge_out_layers:
            edge_attr = layer(edge_attr)

        return point_attr, edge_attr


class GATTransformerNodeOnly(nn.Module):
    """Node-level GAT encoding + Transformer fusion for node regression.

    The first half reuses the node/edge encoding and GATv2 blocks from `GATConvNetworkNodeOnly`;
    the second half uses `SimpleTransformer` for global interaction, supporting both 'mask' and 'pad' batching strategies.
    """

    def __init__(self, in_vertex_dim=6, edge_dim=4, hidden_dim=256, edge_hidden=32, heads=4, dropout=0.1,
                 num_layers=4, out_vertex_dim=3, transformer_mode: str = "pad", transformer_layers: int = 2,
                 transformer_heads: int = 8, transformer_head_dim: int = 32, transformer_ff: int = 256):
        super().__init__()

        # GATConvNetworkNodeOnly Encoder
        self.node_in = nn.Sequential(
            nn.Linear(in_vertex_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )
        self.edge_in = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, edge_hidden),
        )
        self.blocks = nn.ModuleList([
            GATBlock(hidden_dim, heads=heads, edge_dim=edge_hidden, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Transformer Interaction
        self.transformer_mode = transformer_mode.lower()
        self.transformer = SimpleTransformer(
            dim=hidden_dim,
            n_layers=transformer_layers,
            n_heads=transformer_heads,
            head_dim=transformer_head_dim,
            hidden_dim=transformer_ff,
        )

        # Output head
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_vertex_dim),
        )

    def _forward_mask(self, x, edge_index, edge_attr, positions, batch):
        attn_mask = None
        if batch is not None:
            attn_mask = batch.unsqueeze(1).eq(batch.unsqueeze(0))
        x = self.transformer(x.unsqueeze(0), positions=positions, attn_mask=attn_mask).squeeze(0)
        return x

    def _forward_pad(self, x, positions, batch):
        x_pad, mask = to_dense_batch(x, batch)
        pos_pad = None
        if positions is not None:
            pos_pad, _ = to_dense_batch(positions, batch)
        x_pad = self.transformer(x_pad, positions=pos_pad, key_padding_mask=mask)
        return x_pad[mask]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        positions = getattr(data, 'points', None)

        x = self.node_in(x)
        edge_attr = self.edge_in(edge_attr)
        for blk in self.blocks:
            x = blk(x, edge_index, edge_attr)

        if self.transformer_mode == "pad":
            x = self._forward_pad(x, positions, batch)
        else:
            x = self._forward_mask(x, edge_index, edge_attr, positions, batch)

        return self.out(x)


"""Maxwell-style DeepLeap + Transformer model using MaxwellGNNBlock.

This version reuses the physics-informed MaxwellGNNBlock from gnn_models.py
for the four Leapfrog-style updates, and injects time via learnable Fourier
time encoding (LearnableFourierTimeEncoding) followed by a time MLP,
similar to DeepLeapfrogModel.

Input expectations (for consistency with DeepLeapfrogModel):
- data.x: [Ez, Hx, Hy, eps]
- data.points: node coordinates (N, 2 or 3)
- data.t: per-graph scalar time (already relative to initial frame)
- optional data.batch: graph indices for batched graphs

There are four MaxwellGNNBlock blocks (H/E alternating). A global Transformer
operates (1) between Block 2 and Block 3 on [H_{t/2}, E_{t/2}], and
(2) after Block 4 on [H_t, E_t]. Both Transformer stages use pad mode
similar to TransformerNodeRegressor._forward_pad_mode to support multi-graph parallelization.
"""


class DeepLeapTransModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        time_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # Learnable Fourier time encoder
        self.time_encoder = LearnableFourierTimeEncoding(time_dim)

        # Encoders for initial fields
        # h_E^0 = encE(Ez, eps), h_H^0 = encH(Hx, Hy, eps)
        self.encE = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.encH = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Time MLP: tau = MLP_time(PE(t)) in R^D, with learnable Fourier PE.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Four MaxwellGNNBlock blocks, alternating H and E updates
        # (Original implementation retained for comparison/fallback)
        # Originally, the first two layers also used MaxwellGNNBlock; commented out registration here to avoid DDP unused parameters error, kept as reference.
        # self.block1_H = MaxwellGNNBlock(hidden_dim, time_dim=time_dim, heads=heads, dropout=dropout, mode="H_update")
        # self.block2_E = MaxwellGNNBlock(hidden_dim, time_dim=time_dim, heads=heads, dropout=dropout, mode="E_update")
        self.block3_H = MaxwellGNNBlock(hidden_dim, time_dim=time_dim, heads=heads, dropout=dropout, mode="H_update")
        self.block4_E = MaxwellGNNBlock(hidden_dim, time_dim=time_dim, heads=heads, dropout=dropout, mode="E_update")

        # Geometric Maxwell Blocks: Explicitly inject gradient/curl info in initial phase
        # Experimental: Use MaxwellGeometricGNNBlock for the first two layers, and standard MaxwellGNNBlock for the last two.
        self.geo_block1_H = MaxwellGeometricGNNBlock(hidden_dim, heads=heads, dropout=dropout, mode="H_update")
        self.geo_block2_E = MaxwellGeometricGNNBlock(hidden_dim, heads=heads, dropout=dropout, mode="E_update")

        # Two global Transformers on concatenated [H, E]
        self.time_to_token = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.transformer_mid = SimpleTransformer(
            dim=2 * hidden_dim,
            n_layers=1,
            n_heads=8,
            head_dim=32,
            hidden_dim=4 * hidden_dim,
        )
        self.transformer_final = SimpleTransformer(
            dim=2 * hidden_dim,
            n_layers=1,
            n_heads=8,
            head_dim=32,
            hidden_dim=4 * hidden_dim,
        )

        # Decoders for final Ez(t), Hx(t), Hy(t)
        self.decE = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.decH = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def _time_embeddings(self, graph_t: torch.Tensor, device: torch.device):
        """Compute tau_{t/2} and tau_t from per-graph scalar time using
        learnable Fourier PE + time MLP.

        graph_t: (num_graphs,) tensor
        returns: tau_half, tau_t each of shape (num_graphs, hidden_dim)
        """
        graph_t = graph_t.to(device).view(-1)
        # Learnable Fourier features for t/2 and t
        pe_half = self.time_encoder(graph_t / 2.0, device=device)  # (G, time_dim)
        pe_t = self.time_encoder(graph_t, device=device)           # (G, time_dim)
        tau_half = self.time_mlp(pe_half)  # (G, hidden_dim)
        tau_t = self.time_mlp(pe_t)        # (G, hidden_dim)
        return tau_half, tau_t

    def _transformer_pad(self, h_H, h_E, tau_graph, batch, pos, transformer):
        """Apply a SimpleTransformer to concatenated [H, E] with pad batching.

        - h_H, h_E: (N, hidden_dim)
        - tau_graph: (B, hidden_dim), per-graph time embedding (can be tau_half_graph or tau_t_graph)
        - batch: (N,) graph indices
        - pos: (N, 2) node positions
        - transformer: SimpleTransformer instance on 2*hidden_dim
        """
        HE = torch.cat([h_H, h_E], dim=-1)  # (N, 2D)
        HE_pad, mask = to_dense_batch(HE, batch)  # (B, Nmax, 2D)
        pos_pad, _ = to_dense_batch(pos, batch)

        # Per-graph time bias
        time_bias = self.time_to_token(tau_graph).unsqueeze(1)  # (B, 1, 2D)
        HE_in = HE_pad + time_bias
        HE_out = transformer(HE_in, positions=pos_pad, key_padding_mask=mask)
        HE_flat = HE_out[mask]  # (N, 2D)
        h_H_new = HE_flat[:, : self.hidden_dim]
        h_E_new = HE_flat[:, self.hidden_dim :]
        return h_H_new, h_E_new

    def forward(self, data: torch.Tensor):
        x, edge_index = data.x, data.edge_index
        points = getattr(data, "points", None)

        if points is None:
            raise ValueError("DeepLeapTransModel requires data.points (node coordinates)")

        # Node-wise physical fields from x: [Ez, Hx, Hy, eps]
        Ez = x[:, 0]
        Hx = x[:, 1]
        Hy = x[:, 2]
        eps = x[:, 3:4]  # (N, 1)

        # Batch information (for multi-graph minibatch)
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch.to(x.device)
            num_graphs = int(batch.max()) + 1
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            num_graphs = 1

        # Global time per graph: use data.t (already relative to initial frame, like DeepLeapfrogModel)
        if not hasattr(data, "t"):
            raise ValueError("DeepLeapTransModel expects time attribute 't' on data")

        t_attr = data.t
        if torch.is_tensor(t_attr):
            t_tensor = t_attr.to(x.device).view(-1)
            if t_tensor.numel() == 1:
                graph_t = t_tensor.expand(num_graphs)
            else:
                assert t_tensor.numel() == num_graphs, (
                    f"Expected data.t to have {num_graphs} elements for {num_graphs} graphs, got {t_tensor.numel()}"
                )
                graph_t = t_tensor
        else:
            t_scalar = float(t_attr)
            graph_t = torch.full((num_graphs,), t_scalar, dtype=torch.float32, device=x.device)

        # Time embeddings per graph, then broadcast to nodes via batch index
        device = x.device
        tau_half_graph, tau_t_graph = self._time_embeddings(graph_t, device=device)  # (G, D)
        tau_half = tau_half_graph[batch]  # (N, D)
        tau_t = tau_t_graph[batch]        # (N, D)

        # Initial encodings h_E^0, h_H^0
        hE0_input = torch.stack([Ez, eps.squeeze(-1)], dim=-1)  # (N, 2)
        hH0_input = torch.stack([Hx, Hy, eps.squeeze(-1)], dim=-1)  # (N, 3)
        h_E = self.encE(hE0_input)
        h_H = self.encH(hH0_input)

        # Positions for edge geometry
        pos = points[:, :2]

        # ---------------- Block 1: 0 -> t/2, update H ----------------
        # Original: Use MaxwellGNNBlock
        # h_H_half = self.block1_H(h_target=h_H, h_source=h_E, edge_index=edge_index, pos=pos, t_emb=tau_half)

        # Geometric: Use MaxwellGeometricGNNBlock, time modulation still uses tau_half (N, hidden_dim)
        h_H_half = self.geo_block1_H(h_target=h_H, h_source=h_E, edge_index=edge_index, pos=pos, t_emb=tau_half)

        # ---------------- Block 2: 0 -> t/2, update E ----------------
        # Original: Use MaxwellGNNBlock
        # h_E_half = self.block2_E(h_target=h_E, h_source=h_H_half, edge_index=edge_index, pos=pos, t_emb=tau_half, epsilon=eps)

        # Geometric: Use MaxwellGeometricGNNBlock
        h_E_half = self.geo_block2_E(h_target=h_E, h_source=h_H_half, edge_index=edge_index, pos=pos, t_emb=tau_half, epsilon=eps)

        # ---------------- Global Transformer (mid): [H_{t/2}, E_{t/2}] with pad batching ----------------
        h_H_half, h_E_half = self._transformer_pad(
            h_H_half, h_E_half, tau_half_graph, batch, pos, self.transformer_mid
        )

        # ---------------- Block 3: t/2 -> t, update H ----------------
        h_H_t = self.block3_H(h_target=h_H_half, h_source=h_E_half, edge_index=edge_index, pos=pos, t_emb=tau_t)

        # ---------------- Block 4: t/2 -> t, update E ----------------
        h_E_t = self.block4_E(h_target=h_E_half, h_source=h_H_t, edge_index=edge_index, pos=pos, t_emb=tau_t, epsilon=eps)

        # ---------------- Global Transformer (final): [H_t, E_t] with pad batching ----------------
        h_H_t, h_E_t = self._transformer_pad(
            h_H_t, h_E_t, tau_t_graph, batch, pos, self.transformer_final
        )

        # ---------------- Decoder: final Ez, Hx, Hy at time t ----------------
        Ez_pred = self.decE(h_E_t)  # (N, 1)
        H_pred = self.decH(h_H_t)   # (N, 2) -> [Hx, Hy]
        y_pred = torch.cat([Ez_pred, H_pred], dim=-1)  # (N, 3)
        return y_pred


class EHEvolverSandwichModel(nn.Module):
    """PhysicsAugEGAT-style 4-layer GNN + SandwichCrossTransformer + MLP decoders.

    Current Architecture Sequence:
    - 6 layers of PhysicsAugEGAT-style EGAT GNN (Alternating E/H updates with gradient/curl physical priors).
    - 1 layer of SandwichCrossTransformer (Dual-stream global coupling for nodes/edges).
    - Final MLP decoders to map node/edge latent states to Ez, Hx, Hy.

    Expected Input (Aligned with EHDeepLeapfrogModel / load_data_v3):
    - data.x:         (N, 2)    [Ez, eps]
    - data.edge_attr: (E, 4)    [Hx, Hy, dx, dy]
    - data.points:    (N, 2)    Node coordinates
    - data.edge_index:(2, E)
    - data.batch:     (N,)      Graph indices
    - data.t:         Scalar or (G,) time step per graph (relative to initial frame)
    """

    def __init__(
        self,
        node_in_dim: int = 4,
        edge_in_dim: int = 4,
        gnn_hidden_dim: int = 256,
        transformer_hidden_dim: int = 128,
        time_emb_dim: int = 64,
        gnn_layers_each_stage: int = 3,
        transformer_heads: int = 4,
        transformer_head_dim: int = 32,
        transformer_ff_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """PhysicsAugEGAT-style 4-layer GNN + Single SandwichCrossTransformer.

        - Total 2 * gnn_layers_each_stage EGAT layers, default is 6.
        - GNN structure consistent with PhysicsAugEGATModel (Alternating E/H updates, with gradient/curl physical priors).
        - Followed by a SandwichCrossTransformer for global E/H field coupling, then MLP decoding for Ez, Hx, Hy.
        """
        super().__init__()

        if transformer_ff_dim is None:
            transformer_ff_dim = 4 * transformer_hidden_dim

        self.gnn_hidden_dim = gnn_hidden_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.time_hidden_dim = time_emb_dim
        # Total GNN layers: Follows the semantics of "2 stages * gnn_layers_each_stage layers per stage", default 6 layers
        self.num_layers = 2 * gnn_layers_each_stage

        # ---------------- PhysicsAugEGAT-style Node/Edge Encoding ----------------
        # Node input expected to be [Ez, eps, x, y], fixed dimension 4; keeping node_in_dim arg for backward compatibility
        self.in_vertex_layers = nn.ModuleList([
            nn.Linear(node_in_dim, 64),
            nn.Linear(64, 128),
            nn.Linear(128, gnn_hidden_dim),
        ])
        # Edge input expected to be [Hx_0, Hy_0, dx, dy], fixed dimension 4
        self.in_edge_layers = nn.ModuleList([
            nn.Linear(edge_in_dim, 64),
            nn.Linear(64, 128),
            nn.Linear(128, gnn_hidden_dim),
        ])

        self.node_encode_ln = nn.LayerNorm(gnn_hidden_dim)
        self.edge_encode_ln = nn.LayerNorm(gnn_hidden_dim)
        # AdaLN for time-modulated node and edge encoding
        self.node_adaln = AdaLN(gnn_hidden_dim)
        self.edge_adaln = AdaLN(gnn_hidden_dim)

        # Two sets of physical edge feature encoders for Update E / Update H
        # The common part always includes raw edge features [Hx_0, Hy_0, dx, dy].
        # The extra parts are:
        # - Update E: Append C_ij scalar -> [Hx_0, Hy_0, dx, dy, C_ij]
        # - Update H: Append grad_vec -> [Hx_0, Hy_0, dx, dy, grad_x, grad_y]
        self.edge_phys_E_mlp = nn.Sequential(
            nn.Linear(4, 64),  # Hx, Hy, dx, dy, C_ij
            nn.ReLU(),
            nn.Linear(64, gnn_hidden_dim),
        )
        self.edge_phys_H_mlp = nn.Sequential(
            nn.Linear(4, 64),  # Hx, Hy, dx, dy, grad_vec(2)
            nn.ReLU(),
            nn.Linear(64, gnn_hidden_dim),
        )

        # Multi-layer EGAT: Alternating E/H updates, structure consistent with PhysicsAugEGATModel
        edge_hidden_dim = gnn_hidden_dim
        self.gnn_layers = nn.ModuleList([
            EGAT_layer(
                gnn_hidden_dim,
                edge_hidden_dim,
                gnn_hidden_dim,
                edge_hidden_dim,
                transformer_heads,  # Reusing transformer_heads as EGAT heads
                gnn_hidden_dim,
            )
            for _ in range(self.num_layers)
        ])
        self.node_layer_ln = nn.ModuleList([
            nn.LayerNorm(gnn_hidden_dim) for _ in range(self.num_layers)
        ])
        self.edge_layer_ln = nn.ModuleList([
            nn.LayerNorm(edge_hidden_dim) for _ in range(self.num_layers)
        ])

        # Graph-level Time Encoding: Scalar t -> R^{gnn_hidden_dim}, then mapped to nodes/edges by batch
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.time_hidden_dim, gnn_hidden_dim),
        )

        # Projection from GNN dimensions to Transformer dimensions (separately for E and H)
        self.gnn_to_transformer_E = nn.Linear(gnn_hidden_dim, transformer_hidden_dim)
        self.gnn_to_transformer_H = nn.Linear(gnn_hidden_dim, transformer_hidden_dim)

        # Single Sandwich-style Dual-Stream Transformer for global E/H coupling after GNN
        self.cross_tf = LightSandwichCrossTransformer(
            dim=transformer_hidden_dim,
            n_layers=1,
            n_heads=transformer_heads,
            head_dim=transformer_head_dim,
            ff_dim=transformer_ff_dim,
            num_inducing=100,
        )

        # Final MLP Decoders (Node / Edge decoding separately)
        # Restore to transformer_hidden_dim input
        self.node_decoder = nn.Sequential(
            nn.LayerNorm(transformer_hidden_dim),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_hidden_dim, 1),  # Ez
        )
        self.edge_decoder = nn.Sequential(
            nn.LayerNorm(transformer_hidden_dim),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_hidden_dim, 2),  # Hx, Hy
        )

    def _graph_time_embedding(self, data, batch, device):
        """Construct per-graph time embedding from data.t: tau_graph: (G, time_emb_dim)."""

        if not hasattr(data, "t"):
            raise ValueError("EHEvolverSandwichModel expects time attribute 't' on data")

        t_attr = data.t
        if torch.is_tensor(t_attr):
            t_tensor = t_attr.to(device).view(-1)
            num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1
            if t_tensor.numel() == 1:
                graph_t = t_tensor.expand(num_graphs)
            else:
                assert t_tensor.numel() == num_graphs, (
                    f"Expected data.t to have {num_graphs} elements, got {t_tensor.numel()}"
                )
                graph_t = t_tensor
        else:
            # Scalar float
            num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1
            graph_t = torch.full(
                (num_graphs,), float(t_attr), dtype=torch.float32, device=device
            )

        graph_t_in = graph_t.view(-1, 1)
        tau_graph = self.time_mlp(graph_t_in)  # (G, time_emb_dim)
        return tau_graph

    def _apply_cross_transformer(self, h_E, h_H, edge_index, pos, batch):
        """Pack sparse graph format (h_E, h_H) + geometric info into padded format,
        call a SandwichCrossTransformer, then restore to sparse format.
        """

        device = h_E.device
        B = int(batch.max()) + 1 if batch.numel() > 0 else 1

        # Nodes: (N,D) -> (B,Nv_max,D)
        hV_pad, mask_V = to_dense_batch(h_E, batch)  # (B, Nv_max, D)
        pos_pad, _ = to_dense_batch(pos, batch)      # (B, Nv_max, 2)

        # Edge batch indices and positions (midpoints)
        row, col = edge_index
        edge_batch = batch[row]
        pos_mid = 0.5 * (pos[row] + pos[col])  # (E, 2)

        # (E,D) -> (B,Ne_max,D)
        hE_pad, mask_E = to_dense_batch(h_H, edge_batch)   # (B, Ne_max, D)
        posE_pad, _ = to_dense_batch(pos_mid, edge_batch)  # (B, Ne_max, 2)

        return (
            hV_pad,
            hE_pad,
            pos_pad,
            posE_pad,
            mask_V.to(device),
            mask_E.to(device),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = x.device

        # Node coordinates: Use data.points
        if hasattr(data, "points") and data.points is not None:
            pos = data.points.to(device)
        else:
            raise ValueError("EHEvolverSandwichModel requires data.points or data.pos as node coordinates")

        # Batch info
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch.to(device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # Physics Priors: Compute gradient and curl once using true Ez, Hx, Hy
        # ------------------------------------------------------------------
        Ez = x[:, 0:1]              # (N,1)
        eps = x[:, 1:2]             # (N,1)

        H_edge_true = edge_attr[:, 0:2]          # (E,2)
        grad_vec = getattr(data, 'grad_vec', None)
        C_ij = getattr(data, 'C_ij', None)
        dpos = getattr(data, 'dpos', None)
        assert grad_vec is not None and C_ij is not None and dpos is not None, 'Missing precomputed physics features in data.'

        # ------------------------------------------------------------------
        # Graph-level Time Encoding: data.t -> tau_graph -> tau_nodes / tau_edges
        # ------------------------------------------------------------------
        tau_nodes = None
        tau_edges = None
        if hasattr(data, "t") and data.t is not None:
            t_attr = data.t
            if torch.is_tensor(t_attr):
                t_tensor = t_attr.to(device).view(-1)
                num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1
                if t_tensor.numel() == 1:
                    graph_t = t_tensor.expand(num_graphs)
                else:
                    graph_t = t_tensor
            else:
                num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1
                graph_t = torch.full((num_graphs,), float(t_attr), dtype=torch.float32, device=device)

            tau_graph = self.time_mlp(graph_t.view(-1, 1))  # (G, hidden_dim)
            tau_nodes = tau_graph[batch]                    # (N, hidden_dim)
            src, _ = edge_index
            edge_batch = batch[src]
            tau_edges = tau_graph[edge_batch]               # (E, hidden_dim)

        # ------------------------------------------------------------------
        # Node/Edge Encoding (PhysicsAugEGAT Style)
        # ------------------------------------------------------------------
        node_in = torch.cat([Ez, eps, pos], dim=-1)  # (N,4)
        point_attr = node_in
        for layer in self.in_vertex_layers:
            point_attr = F.relu(layer(point_attr))
        if tau_nodes is not None:
            h_E = self.node_adaln(point_attr, tau_nodes)
        else:
            h_E = self.node_encode_ln(point_attr)

        edge_in = edge_attr
        edge_feat = edge_in
        for layer in self.in_edge_layers:
            edge_feat = F.relu(layer(edge_feat))
        if tau_edges is not None:
            h_H = self.edge_adaln(edge_feat, tau_edges)
        else:
            h_H = self.edge_encode_ln(edge_feat)         # (E, hidden_dim)

        # Static physical edge features
        H0_edge = H_edge_true                        # (E,2)
        dx_dy = dpos                                 # (E,2)
        edge_phys_E_in = torch.cat([H0_edge, dx_dy], dim=-1)  # (E,5) , C_ij
        edge_phys_E = self.edge_phys_E_mlp(edge_phys_E_in)         # (E, hidden_dim)

        edge_phys_H_in = torch.cat([edge_attr[:, :4]], dim=-1)  # (E,6), grad_vec
        edge_phys_H = self.edge_phys_H_mlp(edge_phys_H_in)         # (E, hidden_dim)

        # ------------------------------------------------------------------
        # 4-Layer Alternating EGAT: Update E/H
        # ------------------------------------------------------------------
        # Dynamic DGLGraph construction
        src_idx, dst_idx = edge_index
        g = dgl.graph((src_idx, dst_idx), num_nodes=x.size(0))

        for layer_idx, egat in enumerate(self.gnn_layers):
            if layer_idx % 2 == 0:
                # E-layer: Update Node E, edge features reflect Curl-type physics of H
                if layer_idx == 0:
                    edge_in_layer = h_H + edge_phys_E
                else:
                    edge_in_layer = h_H
                node_out, edge_out = egat(g, h_E, edge_in_layer)
                h_E = self.node_layer_ln[layer_idx](h_E + node_out)
            else:
                # H-layer: Update Edge H, edge features reflect Gradient-type physics of E
                if layer_idx == 1:
                    edge_in_layer = h_H + edge_phys_H
                else:
                    edge_in_layer = h_H
                node_out, edge_out = egat(g, h_E, edge_in_layer)
                h_H = self.edge_layer_ln[layer_idx](h_H + edge_out)

        # ------------------------------------------------------------------
        # Single Sandwich Transformer: Global E/H Coupling
        # ------------------------------------------------------------------
        # Project to Transformer dimensions
        h_E = self.gnn_to_transformer_E(h_E)
        h_H = self.gnn_to_transformer_H(h_H)
        hV_pad, hE_pad, posV_pad, posE_pad, mask_V, mask_E = self._apply_cross_transformer(
            h_E, h_H, edge_index, pos, batch
        )
        # Real-time computation of FPS sampling point indices, sampling count not exceeding effective nodes per batch
        from utils.fps_utils import farthest_point_sample
        node_fps_idx = farthest_point_sample(
            posV_pad, self.cross_tf.layers[0].M_node, mask=mask_V
        )
        edge_fps_idx = farthest_point_sample(
            posE_pad, self.cross_tf.layers[0].M_edge, mask=mask_E
        )
        hV_pad, hE_pad = self.cross_tf(
            hV_pad, hE_pad, posV_pad, posE_pad, mask_V, mask_E,
            node_fps_idx=node_fps_idx, edge_fps_idx=edge_fps_idx
        )
        h_E = hV_pad[mask_V]
        h_H = hE_pad[mask_E]

        # ------------------------------------------------------------------
        # Final MLP Decoder: Output Node Ez and Edge Hx, Hy
        # ------------------------------------------------------------------
        # Residual prediction: Output is current field plus change
        delta_Ez = self.node_decoder(h_E)
        delta_H = self.edge_decoder(h_H)
        pred_Ez_next = Ez + delta_Ez
        pred_H_next = H_edge_true + delta_H
        return pred_Ez_next, pred_H_next