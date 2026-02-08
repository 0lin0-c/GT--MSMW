import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import dgl
from torch_geometric.nn import MetaLayer
from models.gnn_models import EGAT_layer


# ==============================================================================
# 1. Core Component: Multi-Head Physics Edge/Node Evolution Model (Processor Core)
# ==============================================================================

class EH_EdgeModel_MultiHead(nn.Module):
    """
    [Edge Evolution Model]
    Logic: Extract 'Ez-like' physical quantities from node features -> Calculate physical gradients -> Update edge features.
    """
    def __init__(self, hidden_dim, time_emb_dim, num_phys_heads=4, epsilon=1e-8):
        super(EH_EdgeModel_MultiHead, self).__init__()
        self.epsilon = epsilon
        self.num_heads = num_phys_heads
        
        # [Key]: Physics Projection Layer
        # Regardless of what input h_E is (Ez+eps or deep features), project it into K scalars here.
        # The first layer automatically learns to extract Ez, subsequent layers learn to extract latent potentials.
        self.proj_E = nn.Linear(hidden_dim, num_phys_heads)
        
        # MLP Input Dimension: 
        # H_old(d) + Src(d) + Dst(d) + Phys_Grad(K*2) + Time(td)
        input_dim = hidden_dim * 3 + (num_phys_heads * 2) + time_emb_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, hidden_dim + 2] (Includes pos)
        # edge_attr: [E, hidden_dim]
        
        # 1. Split features and coordinates
        h_E_src, pos_src = src[:, :-2], src[:, -2:]
        h_E_dst, pos_dst = dest[:, :-2], dest[:, -2:]
        h_H = edge_attr
        
        # 2. Geometry Calculation
        delta_pos = pos_dst - pos_src         # (E, 2)
        dist_sq = torch.sum(delta_pos**2, dim=1, keepdim=True) # (E, 1)

        # 3. [Core]: Multi-Head Physical Gradient Calculation
        # Projection: (E, D) -> (E, K)
        E_scalar_src = self.proj_E(h_E_src)
        E_scalar_dst = self.proj_E(h_E_dst)
        
        # Difference: (E, K)
        E_diff = E_scalar_dst - E_scalar_src
        
        # Gradient Coefficients: (E, K)
        grad_coeffs = E_diff / (dist_sq + self.epsilon)
        
        # Construct Gradient Vector (Broadcasting): 
        # (E, K, 1) * (E, 1, 2) -> (E, K, 2) -> Flatten -> (E, K*2)
        phys_grads = (grad_coeffs.unsqueeze(-1) * delta_pos.unsqueeze(1)).flatten(1)

        # 4. Feature Fusion and Update
        if u is not None:
            tau_edges = u[batch] # Here 'batch' is the batch index for edges
        else:
            tau_edges = torch.empty(h_H.size(0), 0, device=h_H.device)
            
        edge_input = torch.cat([h_H, h_E_src, h_E_dst, phys_grads, tau_edges], dim=-1)
        
        return self.edge_mlp(edge_input)


class EH_NodeModel_MultiHead(nn.Module):
    """
    [Node Evolution Model]
    Logic: Extract 'H-like' vectors from edge features -> Calculate physical curls -> Update node features.
    """
    def __init__(self, hidden_dim, time_emb_dim, num_phys_heads=4):
        super(EH_NodeModel_MultiHead, self).__init__()
        self.num_heads = num_phys_heads
        
        # [Key]: Physics Projection Layer
        # Project edge features into K 2D vectors (each head has x, y components, so output is 2*K)
        self.proj_H = nn.Linear(hidden_dim, num_phys_heads * 2)
        
        # MLP Input Dimension: 
        # E_old(d) + Agg_H(d) + Phys_Curl(K) + Time(td)
        input_dim = hidden_dim * 2 + num_phys_heads + time_emb_dim
        
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, hidden_dim + 2]
        # edge_attr: [E, hidden_dim] (Already updated edge features)
        
        row, col = edge_index
        h_E = x[:, :-2]
        pos = x[:, -2:]
        h_H_new = edge_attr
        
        # 1. Geometry Calculation (Used for circulation calculation)
        pos_src, pos_dst = pos[row], pos[col]
        delta_pos = pos_dst - pos_src
        dx = delta_pos[:, 0:1]
        dy = delta_pos[:, 1:2]
        
        # 2. [Core]: Multi-Head Physical Curl Calculation
        # Projection: (E, D) -> (E, K*2)
        H_vectors = self.proj_H(h_H_new)
        
        # Reshape to (E, K, 2) to separate x, y components
        H_vectors = H_vectors.view(-1, self.num_heads, 2)
        Hx = H_vectors[:, :, 0] # (E, K)
        Hy = H_vectors[:, :, 1] # (E, K)
        
        # Discrete Circulation Formula: Hx*(-dy) + Hy*(dx)
        # Broadcasting: (E, K) * (E, 1) -> (E, K)
        C_ij = Hx * (-dy) + Hy * dx 
        
        # 3. Aggregation (Scatter Sum)
        # Physical Curl Aggregation: (E, K) -> (N, K)
        phys_curl = scatter(C_ij, col, dim=0, dim_size=x.size(0), reduce='sum')
        # Latent Feature Aggregation: (E, D) -> (N, D)
        m_i = scatter(h_H_new, col, dim=0, dim_size=x.size(0), reduce='sum')
        
        # 4. Feature Fusion and Update
        if u is not None:
            tau_nodes = u[batch]
        else:
            tau_nodes = torch.empty(h_E.size(0), 0, device=h_E.device)

        node_input = torch.cat([h_E, m_i, phys_curl, tau_nodes], dim=-1)
        
        return self.node_mlp(node_input)


class ExplicitPhysicsMetaLayer(nn.Module):
    """
    [Wrapper Layer]
    Responsible for concatenating and stripping 'Pos' (positions), hiding internal details from the user.
    """
    def __init__(self, hidden_dim, time_emb_dim, num_phys_heads=4):
        super(ExplicitPhysicsMetaLayer, self).__init__()
        
        self.meta_layer = MetaLayer(
            edge_model=EH_EdgeModel_MultiHead(hidden_dim, time_emb_dim, num_phys_heads),
            node_model=EH_NodeModel_MultiHead(hidden_dim, time_emb_dim, num_phys_heads)
        )

    def forward(self, h_E, h_H, edge_index, pos, tau, batch):
        # 1. Concatenate pos to h_E and feed to MetaLayer
        #    This allows edge_model and node_model to access coordinates internally.
        x_in = torch.cat([h_E, pos], dim=-1)
        
        # 2. Execute MetaLayer
        #    PyG automatically handles src/dest index extraction.
        h_E_new, h_H_new, _ = self.meta_layer(
            x=x_in, 
            edge_index=edge_index, 
            edge_attr=h_H, 
            u=tau, 
            batch=batch
        )
        
        # 3. Strip output (NodeModel outputs pure h_E_new, without pos)
        return h_E_new, h_H_new

# ==============================================================================
# 2. Overall Architecture: Encoder-Processor-Decoder (EH-Evolver GNN)
# ==============================================================================

class EHDeepLeapfrogModel(nn.Module):
    def __init__(self, 
                 node_in_dim=2,    # Ez, eps
                 edge_in_dim=4,    # Hx, Hy, dx, dy (or L^2)
                 hidden_dim=64, 
                 time_emb_dim=16,
                 num_layers=3,
                 num_phys_heads=4):
        super(EHDeepLeapfrogModel, self).__init__()
        
        # --- 1. Encoder (Translator) ---
        # Map physical inputs to hidden_dim, resolving inconsistent input feature dimensions.
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        
        # Time Encoding (MLP embedding for t)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # --- 2. Processor (Evolution Loop) ---
        # Stack N identical physics Blocks
        self.layers = nn.ModuleList([
            ExplicitPhysicsMetaLayer(hidden_dim, time_emb_dim, num_phys_heads)
            for _ in range(num_layers)
        ])

        # --- 3. Decoder ---
        # Map latent states back to physical quantities
        self.node_decoder = nn.Linear(hidden_dim, 1) # Output: Ez
        self.edge_decoder = nn.Linear(hidden_dim, 2) # Output: Hx, Hy

    def forward(self, data, t):
        """
        data: PyG Data object 
              - data.x: (N, 2) [Ez, eps]
              - data.edge_attr: (E, 4) [Hx, Hy, dx, dy]
              - data.pos: (N, 2) [x, y]
              - data.edge_index: (2, E)
              - data.batch: (N,)
        t: (Batch, 1) float tensor representing time
        """
        x, edge_index, edge_attr, pos, batch = \
            data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        
        # 1. Encoding
        h_E = self.node_encoder(x)          # (N, hidden_dim)
        h_H = self.edge_encoder(edge_attr)  # (E, hidden_dim)
        tau = self.time_mlp(t)              # (Batch, time_emb_dim)

        # 2. Processor Loop
        for layer in self.layers:
            # h_E and h_H always maintain hidden_dim
            # But inside each layer, different physical gradient features are extracted
            h_E, h_H = layer(h_E, h_H, edge_index, pos, tau, batch)

        # 3. Decoding
        pred_Ez = self.node_decoder(h_E)
        pred_H = self.edge_decoder(h_H)

        return pred_Ez, pred_H


# ============================================================================== 
# 3. Physics-Augmented Features & EGAT Fusion Module
# ==============================================================================

class PhysicsAugEGATBlock(nn.Module):
    """Single Stage: Executes only the EGAT joint update, without re-calculating physical features.

    The input `h_E`/`h_H` already contains the initial physical augmentation (calculated and projected by the model at the start of forward).
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        edge_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_hidden_dim = edge_hidden_dim or hidden_dim

        # EGAT Joint Update (Node/Edge outputs are still in hidden_dim family)
        self.egat = EGAT_layer(
            hidden_dim,                      # node_in
            self.edge_hidden_dim,            # edge_in
            hidden_dim,                      # node_out
            self.edge_hidden_dim,            # edge_out
            heads,
            hidden_dim,                      # attn hidden/concat dim (consistent with existing implementation)
        )

    def forward(
        self,
        g: dgl.DGLGraph,             # Pre-built DGL graph (reused within batch)
        h_E: torch.Tensor,           # (N, D)
        h_H: torch.Tensor,           # (E, D or edge_hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Directly execute EGAT joint update (input features already contain physical augmentation)
        hE_new, hH_new = self.egat(g, h_E, h_H)

        # Residual Connection (Add directly when dimensions match)
        h_E = h_E + hE_new
        h_H = h_H + hH_new
        return h_E, h_H


class PhysicsAugEGATModel(nn.Module):
    """EGAT-only Model: References the GNN architecture of EGATTransformerNetwork, no longer doing explicit physics augmentation/time encoding during layers.

    Expected Input (aligned with load_data_v3):
    - data.x: (N,2)   [Ez_0, eps]
    - data.points: (N,2)  Node coordinates
    - data.edge_attr: (E,4) [Hx, Hy, dx, dy]
    - data.edge_index: (2,E)

    Used during encoding phase:
    - Node: [Ez, eps, x, y]
    - Edge: [Hx_0, Hy_0, dx, dy]

        Output:
        - pred_vec: 1D vector, containing only Hx, Hy on all edges (flattened by edge order),
            Corresponds to the y_vec returned by load_data_v3 for the phys_egat model (containing only edge H).
    """

    def __init__(
        self,
        node_in_dim: int = 4,   # Ez(1) + eps(1) + pos(2)
        edge_in_dim: int = 4,   # Hx_0, Hy_0, dx, dy
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 4,
        time_hidden_dim: int = 16,  # Scalar t is mapped to time_hidden_dim first, then to hidden_dim
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.num_layers = layers

        # Node/Edge Input Encoding (Reference: EGATTransformerNetwork.in_vertex_layers/in_edge_layers)
        self.in_vertex_layers = nn.ModuleList([
            nn.Linear(node_in_dim, 32),
            nn.Linear(32, 64),
            nn.Linear(64, hidden_dim),
        ])
        self.in_edge_layers = nn.ModuleList([
            nn.Linear(edge_in_dim, 32),
            nn.Linear(32, 64),
            nn.Linear(64, hidden_dim),
        ])

        # Normalization of initial node/edge features after encoding, to suppress large gradients
        self.node_encode_ln = nn.LayerNorm(hidden_dim)
        self.edge_encode_ln = nn.LayerNorm(hidden_dim)

        # Multi-layer EGAT (GNN part only, mimicking EGATTransformerNetwork.gatconv1-4)
        edge_hidden_dim = hidden_dim
        self.gnn_layers = nn.ModuleList()
        for _ in range(layers):
            self.gnn_layers.append(
                EGAT_layer(
                    hidden_dim,          # node_in
                    edge_hidden_dim,     # edge_in
                    hidden_dim,          # node_out
                    edge_hidden_dim,     # edge_out
                    heads,
                    hidden_dim,          # internal attention hidden dim
                )
            )

        # Two sets of physical edge feature encoders for Update E / Update H:
        # The common part always includes raw edge features [Hx_0, Hy_0, dx, dy].
        # The extra parts are:
        # - Update E: Append C_ij scalar -> [Hx_0, Hy_0, dx, dy, C_ij]
        # - Update H: Append grad_vec -> [Hx_0, Hy_0, dx, dy, grad_x, grad_y]
        self.edge_phys_E_mlp = nn.Sequential(
            nn.Linear(5, 32),  # Hx, Hy, dx, dy, C_ij
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )
        self.edge_phys_H_mlp = nn.Sequential(
            nn.Linear(6, 32),  # Hx, Hy, dx, dy, grad_vec(2)
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )

        # LayerNorm after each residual update, acting on node/edge latent states respectively
        self.node_layer_ln = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])
        self.edge_layer_ln = nn.ModuleList([
            nn.LayerNorm(edge_hidden_dim) for _ in range(layers)
        ])

        # Node Output Head: Predict electric field Ez on points from node latent features
        self.node_out_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1),   # Ez (Electric field scalar on nodes)
        ])

        # Edge Output Head: Predict Hx, Hy on edge midpoints from edge latent features
        self.edge_out_layers = nn.ModuleList([
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.Linear(edge_hidden_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 2),   # Hx, Hy (Magnetic field components on edges)
        ])

        # Graph-level Time Encoding: Scalar t -> R^{hidden_dim}, then mapped to nodes/edges by batch
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, hidden_dim),
        )

    def forward(self, data):
        device = data.x.device
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Ensure edge_index is on the same device as node coordinates (for physics calculation & DGL graph construction)
        edge_index = edge_index.to(device)

        # Get node coordinates: prioritize points, then pos
        if hasattr(data, 'points') and data.points is not None:
            pos = data.points.to(device)
        else:
            raise ValueError('PhysicsAugEGATModel requires data.points or data.pos as node coordinates')

        # Node features use [Ez, eps] concatenated with coordinates [x, y] -> [Ez, eps, x, y]
        Ez = x[:, 0:1]              # (N,1)
        eps = x[:, 1:2]             # (N,1)
        node_in = torch.cat([Ez, eps, pos], dim=-1)  # (N,4)

        # Edge features use initial edge field + geometry info [Hx_0, Hy_0, dx, dy]
        edge_in = edge_attr

        # ------------------------------------------------------------------
        # Physics Priors: Compute gradient and curl once using true Ez, Hx, Hy (No backward pass involved here usually)
        # ------------------------------------------------------------------
        # H_edge_true takes the first two dimensions of edge_attr [Hx_0, Hy_0]
        H_edge_true = edge_attr[:, 0:2]

        # compute_physics_from_data returns:
        #  - grad_vec: (E,2) Discrete gradient vector on edges
        #  - C_ij:     (E,1) Flux/Curl scalar on edges
        #  - dpos:     (E,2) Edge vector (dx, dy)
        E_scalar = Ez.squeeze(-1)  # (N,)
        grad_vec, C_ij, dpos = compute_physics_from_data(E_scalar, H_edge_true, pos, edge_index)

        # Batch info (Required for multi-graph mini-batches)
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch.to(device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # Graph-level Time Encoding (if data.t is provided)
        tau_nodes = None
        tau_edges = None
        if hasattr(data, 't') and data.t is not None:
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

        # Input Encoding: Node/Edge basic latent states (without physics priors yet)
        point_attr = node_in
        for layer in self.in_vertex_layers:
            point_attr = F.relu(layer(point_attr))
        if tau_nodes is not None:
            point_attr = point_attr + tau_nodes
        h_E = self.node_encode_ln(point_attr)        # (N, hidden_dim)

        edge_feat = edge_in
        for layer in self.in_edge_layers:
            edge_feat = F.relu(layer(edge_feat))
        if tau_edges is not None:
            edge_feat = edge_feat + tau_edges
        h_H = self.edge_encode_ln(edge_feat)         # (E, hidden_dim)

        # Static Physical Edge Features (Depend only on initial fields and geometry)
        # 1) Used for Update E Layer: Raw edge features + Curl-type info -> [Hx,Hy, dx, dy, C_ij]
        H0_edge = H_edge_true                        # (E,2)
        dx_dy = dpos                                 # (E,2)
        edge_phys_E_in = torch.cat([H0_edge, dx_dy, C_ij], dim=-1)  # (E,5)
        edge_phys_E = self.edge_phys_E_mlp(edge_phys_E_in)         # (E, hidden_dim)

        # 2) Used for Update H Layer: Raw edge features + Gradient-type info -> [Hx,Hy, dx, dy, grad_x, grad_y]
        edge_phys_H_in = torch.cat([edge_in[:, :4], grad_vec], dim=-1)  # (E,6)
        edge_phys_H = self.edge_phys_H_mlp(edge_phys_H_in)         # (E, hidden_dim)

        # Call EGAT_layer using DGL graph
        src, dst = edge_index
        g = dgl.graph((src, dst), num_nodes=x.size(0), device=device)

        # Alternating update of Node Features (E) and Edge Features (H):
        # - Update E layer uses edge_feat = h_H + edge_phys_E (contains Curl prior of H)
        # - Update H layer uses edge_feat = h_H + edge_phys_H (contains Gradient prior of E)
        for layer_idx, egat in enumerate(self.gnn_layers):
            if layer_idx % 2 == 0:
                # E-layer: Update Node E, edge features reflect Curl-type physics of H
                edge_in_layer = h_H + edge_phys_E
                node_out, edge_out = egat(g, h_E, edge_in_layer)
                h_E = self.node_layer_ln[layer_idx](h_E + node_out)
                # h_H retains the latest value from the previous layer
            else:
                # H-layer: Update Edge H, edge features reflect Gradient-type physics of E
                edge_in_layer = h_H + edge_phys_H
                node_out, edge_out = egat(g, h_E, edge_in_layer)
                h_H = self.edge_layer_ln[layer_idx](h_H + edge_out)
                # h_E retains the latest value from the previous layer

        # Node Output Head: Predict Ez on nodes
        node_out = h_E
        for layer in self.node_out_layers:
            node_out = layer(node_out)
        pred_E_node = node_out  # (N,1)

        # Edge Output Head: Predict Hx, Hy on edges
        edge_out = h_H
        for layer in self.edge_out_layers:
            edge_out = layer(edge_out)
        pred_H_edge = edge_out  # (E,2)

        # Return node and edge prediction tensors, loss is computed against data.y_E / data.y_H in the upper-level training code
        return pred_E_node, pred_H_edge