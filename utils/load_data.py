# load_data.py


import torch
from torch_geometric.data import Data
import numpy as np
import os, pickle
import dgl
from utils.fps_utils import fps_indices, edge_midpoints

def compute_physics_from_data(
    E_scalar: torch.Tensor,           # (N,) true E (e.g., Ez)
    H_edge: torch.Tensor,             # (E,2) true H on edges (Hx, Hy)
    pos: torch.Tensor,                # (N,2)
    edge_index: torch.Tensor,         # (2,E)
    eps_den: float = 1e-8,
    use_curl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute physical edge features once from real E/H.

    Returns:
    - grad_vec: (E,2) discrete gradient of E along each edge
    - C_ij: (E,1) edge scalar (flux or curl)
    - dpos: (E,2) edge delta position
    """
    row, col = edge_index
    dpos = pos[col] - pos[row]                              # (E,2)
    dist_sq = (dpos * dpos).sum(dim=1, keepdim=True)        # (E,1)

    # Discrete gradient of E along each edge
    E_diff = (E_scalar[col] - E_scalar[row]).unsqueeze(-1)  # (E,1)
    grad_scalar = E_diff / (dist_sq + eps_den)              # (E,1)
    grad_vec = grad_scalar * dpos                           # (E,2)

    # Flux vs Curl using true H
    Hx = H_edge[:, 0:1]
    Hy = H_edge[:, 1:2]
    if use_curl:
        C_ij = Hx * dpos[:, 0:1] + Hy * dpos[:, 1:2]       # Curl: H · dl
    else:
        C_ij = Hx * (-dpos[:, 1:2]) + Hy * dpos[:, 0:1]    # Flux: H · n

    return grad_vec/100, C_ij, dpos

def load_data(i, j, args):
    """Load the training dataset (Generic version, includes edge_attr, compatible with all models)."""
    simulation_num = i
    step_num = j
    attributes_file = f"MeshAttributes_{simulation_num}.pkl"
    data_file = f"MeshData_{simulation_num}.pkl"
    attributes_path = os.path.join(args['data_folder'], attributes_file)
    data_path = os.path.join(args['data_folder'], data_file)

    with open(attributes_path, 'rb') as file:
        mesh_attributes = pickle.load(file)
    with open(data_path, 'rb') as file:
        mesh_data = pickle.load(file)

    edge_index = torch.tensor(mesh_data["GraphStructure"], dtype=torch.int64).t().contiguous()

    # Initial moment point features: Ez, Hx, Hy, curlE_x, curlE_y, curlH_z, eps
    point_attr0 = mesh_attributes[0]["PointAttributes"]
    x_idx = [0, 2, 3, 4, 5, 6, 1]  # Restore curl components
    x = torch.from_numpy(point_attr0[:, x_idx]).to(torch.float32)

    x[:, :3] *= 10.0  # scale Ez, Hx, Hy to enlarge signal

    # Target time step field output (frame corresponding to t = step_num+1)
    target_attr = mesh_attributes[step_num + 1]["PointAttributes"]
    y_index = [0, 2, 3]  # Ez, Hx, Hy
    y = torch.from_numpy(target_attr[:, y_index]).to(torch.float32)
    y *= 10.0  # scale targets consistently

    # Edge features: preserve only geometric info; put global time/step into node features
    points = torch.from_numpy(np.array(mesh_attributes[0]["Points"])).to(torch.float32)
    t_val = float(target_attr[0, 10]) if target_attr.shape[0] > 0 else 0.0
    step_val = float(step_num + 1)

    # Write global time and steps into node features
    t_node = torch.full((x.shape[0], 1), t_val, dtype=torch.float32)
    step_node = torch.full((x.shape[0], 1), step_val, dtype=torch.float32)
    x = torch.cat((x, t_node, step_node), dim=1)  # Now x dimension is 9
 
    # Use point coordinates and edge_index to compute relative displacement and distance for each edge
    src, dst = edge_index
    delta = points[dst] - points[src]
    dx_dy = delta[:, :2]  # Take first two dims (assuming planar mesh)
    dist = torch.norm(delta, dim=1, keepdim=True) + 1e-6
    inv_dist = 1.0 / dist
    edge_attr = torch.cat((dx_dy, dist, inv_dist), dim=1)
    
    tensors = {
        "edge_index": edge_index,
        "x": x,
        "edge_attr": edge_attr,
        "y": y,
        "points": points,
    }
    for name, tensor in tensors.items():
        assert isinstance(tensor, torch.Tensor), \
            f"{name} must be a torch.Tensor, got {type(tensor)}"
        assert tensor.numel() > 0, \
            f"{name} is empty, shape={tensor.shape}"
        for dim_idx, dim in enumerate(tensor.shape):
            assert dim > 0, \
                f"{name} dimension {dim_idx} is zero"
    assert isinstance(points, torch.Tensor), \
        f"points must be torch.Tensor, got {type(points)}"
    assert points.numel() > 0, \
        f"points is empty, shape={points.shape}"

    # attach simulation/step metadata for downstream inspection
    sim_id = torch.tensor([simulation_num], dtype=torch.int64)
    step_id = torch.tensor([step_num], dtype=torch.int64)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, points=points, sim_id=sim_id, step_id=step_id)


def load_data_v2(i, j, args):
    """Data loading function tailored for DeepLeapfrogModel.

    Scenario: Predict field at any target time t from initial moment t0 field.

    Design:
    - Node feature x takes initial frame t0 local physics: x = [Ez_0, Hx_0, Hy_0, eps];
    - Target y takes field at step_num+1 frame: [Ez_t, Hx_t, Hy_t];
    - Scalar t takes true time of target frame t = t_{step_num+1} (PointAttributes[:,10]);
    - Keep edge_index, points, sim_id, step_id; do not compute edge_attr (DeepLeapfrogModel only uses edge_index + points for geometry).
    """

    simulation_num = i
    step_num = j
    attributes_file = f"MeshAttributes_{simulation_num}.pkl"
    data_file = f"MeshData_{simulation_num}.pkl"
    attributes_path = os.path.join(args["data_folder"], attributes_file)
    data_path = os.path.join(args["data_folder"], data_file)

    with open(attributes_path, "rb") as file:
        mesh_attributes = pickle.load(file)
    with open(data_path, "rb") as file:
        mesh_data = pickle.load(file)

    edge_index = torch.tensor(mesh_data["GraphStructure"], dtype=torch.int64).t().contiguous()

    # Initial frame t0 point features:
    #  - Old data: [Ez, eps, Hx, Hy, curlE_x, curlE_y, curlH_z, dEz_dt, dHx_dt, dHy_dt, t]
    #  - New data v1 (generate_data_v2 early): [Ez, eps, Hx, Hy, t]
    #  - New data v2 (current generate_data_v2): [Ez, eps, Hx, Hy], time stored separately in 'Time' scalar of each snapshot
    # DeepLeapfrogModel only needs Ez, Hx, Hy, eps from initial step 0
    if step_num + 1 >= len(mesh_attributes):
        raise IndexError(f"step_num+1={step_num+1} exceeds available frames {len(mesh_attributes)} in MeshAttributes_{simulation_num}.pkl")

    point_attr0 = mesh_attributes[0]["PointAttributes"]
    x_idx = [0, 2, 3, 1]  # Ez, Hx, Hy, eps (Layout consistent with old/new, just different total columns)
    x = torch.from_numpy(point_attr0[:, x_idx]).to(torch.float32)
    x[:, :3] *= 10.0  # scale Ez, Hx, Hy

    # Target time step field output (frame corresponding to t = step_num+1)
    target_attr = mesh_attributes[step_num + 1]["PointAttributes"]
    y_index = [0, 2, 3]  # Ez, Hx, Hy
    y = torch.from_numpy(target_attr[:, y_index]).to(torch.float32)
    y *= 10.0

    # Node coordinates and global time (relative to initial frame)
    points = torch.from_numpy(np.array(mesh_attributes[0]["Points"])).to(torch.float32)

    # Time recorded in Meep is absolute time from simulation start.
    # Since we trimmed the initial evolution part in generate_dataset,
    # mesh_attributes[0] actually corresponds to your "DeepLeapfrog initial time t0".
    # To let the model see relative time (t=0 corresponds to initial frame), we shift here:
    #   t_rel = t_abs(target) - t_abs(initial)

    snap0 = mesh_attributes[0]
    snap_tgt = mesh_attributes[step_num + 1]

    if "Time" in snap0 and "Time" in snap_tgt:
        # New generate_data_v2: Read directly fromscalar Time field of each snapshot
        t0_abs = float(snap0["Time"])
        t_abs = float(snap_tgt["Time"])
    else:
        # Compatible with old format: Time is still stored in last column of PointAttributes
        pa0 = snap0["PointAttributes"]
        # Compatible with old 11-column format: Time fixed at index 10
        t_idx = 10

        if pa0.shape[0] > 0:
            t0_abs = float(pa0[0, t_idx])
        else:
            t0_abs = 0.0

        if target_attr.shape[0] > 0:
            t_abs = float(target_attr[0, t_idx])
        else:
            t_abs = 0.0

    t_val = t_abs - t0_abs

    # x only keeps local physics on points, no longer duplicate t and step

    # Basic checks (consistent with original, but skipping edge_attr check)
    tensors = {"edge_index": edge_index, "x": x, "y": y, "points": points}
    for name, tensor in tensors.items():
        assert isinstance(tensor, torch.Tensor), f"{name} must be a torch.Tensor, got {type(tensor)}"
        assert tensor.numel() > 0, f"{name} is empty, shape={tensor.shape}"
        for dim_idx, dim in enumerate(tensor.shape):
            assert dim > 0, f"{name} dimension {dim_idx} is zero"

    sim_id = torch.tensor([simulation_num], dtype=torch.int64)
    step_id = torch.tensor([step_num], dtype=torch.int64)
    t_tensor = torch.tensor(t_val, dtype=torch.float32)

    # Note: No longer attach edge_attr, DeepLeapfrogModel only uses edge_index and points;
    # Global time scalar t is attached to Data as a separate attribute
    return Data(x=x, edge_index=edge_index, y=y, points=points, sim_id=sim_id, step_id=step_id, t=t_tensor)


def load_data_v3(i, j, args):
    """Data loader prepared for EHDeepLeapfrogModel / EH-Evolver GNN (Node E, Edge H).

    Design:
    - Node feat x: [Ez_0, eps] on initial frame 0;
    - Edge feat edge_attr: [Hx_mid, Hy_mid] on initial frame 0 plus geometric vector [dx, dy], i.e., [Hx, Hy, dx, dy];
    - Node target y_E: Point electric field [Ez_t] on target frame step_num+1;
    - Edge target y_H: Midpoint magnetic field [Hx_mid_t, Hy_mid_t] on target frame step_num+1;
    - Time t: Target frame time relative to initial frame t_rel = t_abs(target) - t_abs(initial).
    """

    simulation_num = i
    step_num = j
    attributes_file = f"MeshAttributes_{simulation_num}.pkl"
    data_file = f"MeshData_{simulation_num}.pkl"
    attributes_path = os.path.join(args["data_folder"], attributes_file)
    data_path = os.path.join(args["data_folder"], data_file)

    with open(attributes_path, "rb") as file:
        mesh_attributes = pickle.load(file)
    with open(data_path, "rb") as file:
        mesh_data = pickle.load(file)

    # Current model name (lowercase), used to distinguish phys_egat from EH-Evolver etc.
    model_name = ''
    if isinstance(args, dict):
        model_name = (args.get('model', '') or '').lower()

    if step_num + 1 >= len(mesh_attributes):
        raise IndexError(
            f"step_num+1={step_num+1} exceeds available frames {len(mesh_attributes)} in MeshAttributes_{simulation_num}.pkl"
        )

    edge_index = torch.tensor(mesh_data["GraphStructure"], dtype=torch.int64).t().contiguous()

    # Initial frame 0 point features:
    #  generate_data_v2 current layout is [Ez, eps, Hx, Hy]
    #  All models (including phys_egat) use only [Ez, eps] as input on node side,
    #  Magnetic fields Hx, Hy only participate via edge features.
    point_attr0 = mesh_attributes[0]["PointAttributes"]
    x_idx = [0, 1]  # Ez, eps
    x = torch.from_numpy(point_attr0[:, x_idx]).to(torch.float32)
    x[:, 0:1] *= 10.0  # Enlarge Ez signal

    # Target frame point electric field (Node supervision): [Ez_t]
    target_attr = mesh_attributes[step_num + 1]["PointAttributes"]
    y_E = torch.from_numpy(target_attr[:, 0:1]).to(torch.float32)
    y_E *= 10.0

    # Node coordinates
    points = torch.from_numpy(np.array(mesh_attributes[0]["Points"])).to(torch.float32)

    # Edge features: Unified fetching of edge field and geometry info from FaceAttributes
    # FaceAttributes: [dx, dy, Hx_mid, Hy_mid, curlH_z]
    # For EH-Evolver etc.: Used as explicit physics input;
    # For phys_egat: [Hx_mid, Hy_mid] therein also used as initial edge field input, later only supervised on edges.
    face_attr0 = mesh_attributes[0].get("FaceAttributes", None)
    if face_attr0 is None:
        raise ValueError(
            f"MeshAttributes_{simulation_num}.pkl frame 0 does not contain 'FaceAttributes'; "
            "load_data_v3 requires edge Hx/Hy from generate_data.py"
        )

    face_attr0 = np.asarray(face_attr0, dtype=np.float32)
    if face_attr0.shape[0] != edge_index.size(1):
        raise ValueError(
            f"FaceAttributes edge count {face_attr0.shape[0]} does not match GraphStructure edges {edge_index.size(1)}"
        )

    # dx, dy: vertex1 - vertex2, consistent with order of (src, dst) in GraphStructure
    dx_dy = face_attr0[:, 0:2]  # (E, 2)
    HxHy_0 = face_attr0[:, 2:4]   # (E, 2) at initial frame

    # Consistent with node side, apply same scaling to initial Hx, Hy
    HxHy_0 *= 10.0

    edge_attr_np = np.concatenate([HxHy_0, dx_dy], axis=1)  # [Hx_0, Hy_0, dx, dy]
    edge_attr = torch.from_numpy(edge_attr_np).to(torch.float32)

    # Target frame edge magnetic field (Edge supervision): FaceAttributes_t[:, 2:4]
    face_attr_t = mesh_attributes[step_num + 1].get("FaceAttributes", None)
    if face_attr_t is None:
        raise ValueError(
            f"MeshAttributes_{simulation_num}.pkl frame {step_num+1} does not contain 'FaceAttributes'; "
            "load_data_v3 requires edge Hx/Hy from generate_data.py"
        )
    face_attr_t = np.asarray(face_attr_t, dtype=np.float32)
    if face_attr_t.shape[0] != edge_index.size(1):
        raise ValueError(
            f"FaceAttributes target edge count {face_attr_t.shape[0]} does not match GraphStructure edges {edge_index.size(1)}"
        )
    y_H_np = face_attr_t[:, 2:4]  # (E,2) Hx_mid_t, Hy_mid_t
    y_H_np *= 10.0
    y_H = torch.from_numpy(y_H_np).to(torch.float32)

    # Relative time t_rel = t_abs(target) - t_abs(initial)
    # Compatible with two data formats:
    #  - generate_data_v2.py: Each snapshot has separate scalar Time field;
    #  - Old generate_data.py: Time is still saved in last column of PointAttributes (index 10).
    snap0 = mesh_attributes[0]
    snap_tgt = mesh_attributes[step_num + 1]

    if "Time" in snap0 and "Time" in snap_tgt:
        # generate_data_v2.py style
        t0_abs = float(snap0["Time"])
        t_abs = float(snap_tgt["Time"])
    else:
        # Old version compatibility path: Read from last column of PointAttributes
        pa0 = snap0["PointAttributes"]
        t_idx = 10
        if pa0.shape[0] > 0:
            t0_abs = float(pa0[0, t_idx])
        else:
            t0_abs = 0.0
        if target_attr.shape[0] > 0:
            t_abs = float(target_attr[0, t_idx])
        else:
            t_abs = 0.0
    t_val = t_abs - t0_abs
    t_tensor = torch.tensor(t_val, dtype=torch.float32)

    # Pack supervision targets into 1D tensor y_vec, convenient for old models using single tensor y.
    # Default packing order convention:
    #   [Ez(node 0..N-1), Hx(edge 0..E-1), Hy(edge 0..E-1)]
    # For "phys_egat" model, subsequent training will use data.y_E and data.y_H separately
    # for losses on nodes/edges, y_vec here is only for interface compatibility, not participating in loss calculation.

    if model_name == 'phys_egat':
        # Compatibility placeholder: Still provide a y vector, but real training/eval uses y_E and y_H
        y_vec = torch.cat([y_E.reshape(-1), y_H.reshape(-1)], dim=0)
    else:
        y_vec = torch.cat([y_E.reshape(-1), y_H.reshape(-1)], dim=0)


    # ------------------ Physics Calculation ------------------
    # Node electric field (Ez) and Edge magnetic field (Hx, Hy)
    E_scalar = x[:, 0]  # (N,)
    H_edge = edge_attr[:, 0:2]  # (E,2)
    grad_vec, C_ij, dpos = compute_physics_from_data(E_scalar, H_edge, points, edge_index)

    # Basic checks
    tensors = {
        "edge_index": edge_index,
        "x": x,
        "edge_attr": edge_attr,
        "y_E": y_E,
        "y_H": y_H,
        "points": points,
        "grad_vec": grad_vec,
        "C_ij": C_ij,
        "dpos": dpos,
    }
    for name, tensor in tensors.items():
        assert isinstance(tensor, torch.Tensor), f"{name} must be a torch.Tensor, got {type(tensor)}"
        assert tensor.numel() > 0, f"{name} is empty, shape={tensor.shape}"
        for dim_idx, dim in enumerate(tensor.shape):
            assert dim > 0, f"{name} dimension {dim_idx} is zero"

    sim_id = torch.tensor([simulation_num], dtype=torch.int64)
    step_id = torch.tensor([step_num], dtype=torch.int64)

    # t attached to Data as graph-level scalar, model internally expands/encodes it;
    # y is packed version of (y_E, y_H), supervision targets on nodes and edges respectively
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_vec,
        y_E=y_E,
        y_H=y_H,
        points=points,
        sim_id=sim_id,
        step_id=step_id,
        t=t_tensor,
        grad_vec=grad_vec,
        C_ij=C_ij,
        dpos=dpos,
    )

def load_data_v4(sim_id, src_step, tgt_step, args):
    """
    Modified loading function: Supports arbitrary start src_step and end tgt_step.
    Input: src_step (as x), tgt_step (as y)
    Output: Data object, where x is state at src moment, y is state at tgt moment, t is time difference.
    """
    simulation_num = sim_id
    
    attributes_file = f"MeshAttributes_{simulation_num}.pkl"
    data_file = f"MeshData_{simulation_num}.pkl"
    attributes_path = os.path.join(args["data_folder"], attributes_file)
    data_path = os.path.join(args["data_folder"], data_file)

    with open(attributes_path, "rb") as file:
        mesh_attributes = pickle.load(file)
    with open(data_path, "rb") as file:
        mesh_data = pickle.load(file)

    # --- 1. Get Source Frame (src_step) ---
    edge_index = torch.tensor(mesh_data["GraphStructure"], dtype=torch.int64).t().contiguous()
    
    src_attr = mesh_attributes[src_step]["PointAttributes"]
    x_idx = [0, 1] # Ez, eps
    x = torch.from_numpy(src_attr[:, x_idx]).to(torch.float32)
    x[:, 0:1] *= 10.0 # Scaling

    # Source Edge Attributes (Hx, Hy at src_step)
    src_face = mesh_attributes[src_step]["FaceAttributes"] # Assuming it exists
    src_face = np.asarray(src_face, dtype=np.float32)
    
    dx_dy = src_face[:, 0:2]
    HxHy_src = src_face[:, 2:4] * 10.0 # Hx, Hy scaling
    
    edge_attr_np = np.concatenate([HxHy_src, dx_dy], axis=1)
    edge_attr = torch.from_numpy(edge_attr_np).to(torch.float32)

    # --- 2. Get Target Frame (tgt_step) ---
    tgt_point_attr = mesh_attributes[tgt_step]["PointAttributes"]
    y_E = torch.from_numpy(tgt_point_attr[:, 0:1]).to(torch.float32) * 10.0
    
    tgt_face_attr = mesh_attributes[tgt_step]["FaceAttributes"]
    tgt_face_attr = np.asarray(tgt_face_attr, dtype=np.float32)
    y_H = torch.from_numpy(tgt_face_attr[:, 2:4]).to(torch.float32) * 10.0
    
    # --- 3. Compute Relative Time Delta T ---
    # Assume generate_data_v2 format has "Time" field
    if "Time" in mesh_attributes[src_step] and "Time" in mesh_attributes[tgt_step]:
        t_src = float(mesh_attributes[src_step]["Time"])
        t_tgt = float(mesh_attributes[tgt_step]["Time"])
    else:
        # Compatible with old version, read from PointAttributes
        t_src = float(src_attr[0, 10]) if src_attr.shape[0] > 0 else 0.0
        t_tgt = float(tgt_point_attr[0, 10]) if tgt_point_attr.shape[0] > 0 else 0.0
        
    dt_val = t_tgt - t_src
    t_tensor = torch.tensor([dt_val], dtype=torch.float32) # Note consistency in dimension

    # --- 4. Auxiliary Info ---
    points = torch.from_numpy(np.array(mesh_attributes[0]["Points"])).to(torch.float32)
    # y_vec compatible interface
    y_vec = torch.cat([y_E.reshape(-1), y_H.reshape(-1)], dim=0)
    
    # Physics Calculation (Omit some details, keep original logic)
    E_scalar = x[:, 0]
    H_edge = edge_attr[:, 0:2]
    grad_vec, C_ij, dpos = compute_physics_from_data(E_scalar, H_edge, points, edge_index, use_curl=True)
    # In load_data_v4 of utils/load_data.py
    sim_id_tensor = torch.tensor([sim_id], dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_vec,
        y_E=y_E,
        y_H=y_H,
        points=points,
        t=t_tensor,
        src_step=torch.tensor([src_step]),
        tgt_step=torch.tensor([tgt_step]),
        grad_vec=grad_vec, C_ij=C_ij, dpos=dpos ,# ... other physical tensors
        sim_id=sim_id_tensor
    )