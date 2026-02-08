import torch
import torch.nn.functional as F
from utils.training_utils import Determine_Inf_Nan
from collections import defaultdict
import numpy as np
import os
# Must import load_data_v4 to read ground truth
from utils.load_data import load_data_v4, compute_physics_from_data


def loss_function(net_y: torch.Tensor, real_y: torch.Tensor):
    """Standard MSE loss, assuming net_y and real_y are both tensors and have the same shape."""
    if not torch.is_tensor(net_y) or not torch.is_tensor(real_y):
        raise TypeError("loss_function expects tensor inputs")

    Determine_Inf_Nan(real_y, "real_y")
    Determine_Inf_Nan(net_y, "net_y")

    return F.mse_loss(net_y, real_y)

def evaluate_model(model, loader):
    """
    Accumulate MSE over a data list or loader.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        loss = 0
        for data in loader:
            data = data.to(device)
            net_y = model(data)
            # Compatible with multi-task output (Node Ez, Edge H)
            if isinstance(net_y, (tuple, list)) and len(net_y) == 2 and hasattr(data, 'y_E') and hasattr(data, 'y_H'):
                pred_E, pred_H = net_y
                loss += loss_function(pred_E, data.y_E) + loss_function(pred_H, data.y_H)
            else:
                loss += loss_function(net_y, data.y)
        return loss / len(loader)



def inference_rollout(model, batch, device, stride, args):
    """
    Merged Rollout Function: Supports both "Ground Truth Injection" and "Real-time Calculation" modes.
    
    Argument args['reality'] controls the source of physical quantities:
    - True:  Hybrid Mode. Forces usage of ground truth grad_vec and C_ij from disk.
             Used to diagnose model performance when physical quantities are perfect.
    - False: Self-Consistent Mode. Calculates grad_vec and C_ij in real-time using currently predicted E, H.
             Used for realistic testing and deployment.
             
    Includes memory optimization: inference_mode, detach, empty_cache.
    """
    
    # Get control switch, defaults to False (Real-time calculation)
    use_reality = args.get('reality', False)
    
    # [Memory Optimization 1] Use inference_mode
    with torch.inference_mode():
        
        # --- 1. Dimension Check and Fix ---
        if batch.t.dim() == 1:
            batch.t = batch.t.view(-1, 1)

        # --- 2. Initialize State ---
        start_step = batch.src_step[0].item()
        target_step = batch.tgt_step[0].item()
        curr_step = start_step 
        
        # Clone initial state
        current_x = batch.x.clone()                   # [Ez, eps]
        current_edge_attr = batch.edge_attr.clone()   # [Hx, Hy, dx, dy]

        # Backup time vector
        original_t_vec = batch.t.clone() 
        if original_t_vec.shape[1] != 1:
             original_total_dt = original_t_vec[:, 1:2].clone() if original_t_vec.shape[1] >= 2 else original_t_vec.clone()
        else:
             original_total_dt = original_t_vec.clone()

        # Get sim_id (Required for Reality mode)
        sim_ids = batch.sim_id.cpu().numpy() if hasattr(batch, 'sim_id') else None
        if use_reality and sim_ids is None:
            raise ValueError("args['reality']=True but 'sim_id' not found in batch, cannot load ground truth!")

        # --- 3. Rollout Loop ---
        while curr_step < target_step:
            
            # [Memory Optimization 2] Periodic cleanup
            if (curr_step - start_step) > 0 and (curr_step - start_step) % 20 == 0:
                torch.cuda.empty_cache()

            # --- A. Time Step Update Logic ---
            remainder = target_step % stride
            if curr_step == 0 and remainder != 0:
                next_step = remainder
            else:
                next_step = curr_step + stride
            if next_step > target_step: next_step = target_step

            step_diff = next_step - curr_step
            total_steps_span = target_step - start_step
            ratio = step_diff / total_steps_span if total_steps_span != 0 else 0
            
            current_dt = original_total_dt * ratio
            batch.t = current_dt.view(-1, 1)
            
            # Update Input (Ensure detached from previous round)
            batch.x = current_x
            batch.edge_attr = current_edge_attr
            
            # =========================================================
            # [Core Logic Branch] Physics Injection vs Real-time Calculation
            # =========================================================
            if use_reality:
                # --- Branch 1: Reality Mode (Reality=True) ---
                # Load Ground Truth from disk
                gt_grad_vec_list = []
                gt_C_ij_list = []
                
                # This is an IO-intensive operation
                for sim_id in sim_ids:
                    # Load ground truth at curr_step
                    gt_data = load_data_v4(sim_id, curr_step, curr_step + 1, args)
                    gt_grad_vec_list.append(gt_data.grad_vec)
                    gt_C_ij_list.append(gt_data.C_ij)
                    del gt_data # Memory optimization: burn after use
                
                # Inject ground truth into Batch
                batch.grad_vec = torch.cat(gt_grad_vec_list, dim=0).to(device)
                batch.C_ij = torch.cat(gt_C_ij_list, dim=0).to(device)
                
            else:
                # --- Branch 2: Simulated Mode (Reality=False) ---
                # Real-time calculation based on current predicted E, H
                E_scalar_curr = current_x[:, 0]
                H_edge_curr = current_edge_attr[:, 0:2]
                
                new_grad_vec, new_C_ij, _ = compute_physics_from_data(
                    E_scalar_curr, 
                    H_edge_curr, 
                    batch.points, 
                    batch.edge_index,
                    use_curl=True
                )
                
                # Inject calculated values into Batch
                batch.grad_vec = new_grad_vec
                batch.C_ij = new_C_ij

            # =========================================================
            # [Debug Print] Compare numerical magnitudes
            # =========================================================
            # To prevent screen flooding, only print the 1st step of each Batch, or remove this restriction
            # if (curr_step - start_step) < 2: 
            #     mode_str = "【Reality (GT)】" if use_reality else "【Simulated (Calc)】"
            #     e_mean = current_x[:, 0].abs().mean().item()
            #     h_mean = current_edge_attr[:, 0:2].abs().mean().item()
            #     c_mean = batch.C_ij.abs().mean().item()
            #     grad_mean = batch.grad_vec.abs().mean().item()
                
            #     print(f"Step {curr_step} {mode_str}: "
            #           f"|E|={e_mean:.2e}, |H|={h_mean:.2e}, "
            #           f"|C_ij|={c_mean:.2e}, |Grad|={grad_mean:.2e}")

            # --- B. Model Prediction ---
            net_y = model(batch)
            pred_E, pred_H = net_y

            # [Memory Optimization 3] Detach to cut off history
            pred_E_detached = pred_E.detach()
            pred_H_detached = pred_H.detach()

            # --- C. Update State ---
            current_x[:, 0] = pred_E_detached.squeeze()
            current_edge_attr[:, 0:2] = pred_H_detached
            
            curr_step = next_step

    return pred_E_detached, pred_H_detached


def evaluate_validation_simple(model, loader, args):
    """
    [New] Lightweight validation function.
    Returns only avg_mse and avg_rel_error, no file generation, no redundant printing.
    """
    model.eval()
    device = next(model.parameters()).device
    horizon = args.get('jump_horizon', 10)
    
    total_mse_sum = 0.0
    total_up = 0.0
    total_down = 0.0
    total_graphs = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Still call inference_rollout to ensure inference logic is consistent with testing
            pred_E, pred_H = inference_rollout(model, batch, device, stride=horizon, args=args)
            
            # 1. Calculate MSE (Loss)
            loss = F.mse_loss(pred_E, batch.y_E) + F.mse_loss(pred_H, batch.y_H)
            
            # 2. Calculate Relative Error Components
            diff_E = pred_E - batch.y_E
            diff_H = pred_H - batch.y_H
            
            # Numerator: Sum of squared errors
            sum_sq_diff = diff_E.pow(2).sum().item() + diff_H.pow(2).sum().item()
            # Denominator: Sum of squared ground truths
            sum_sq_real = batch.y_E.pow(2).sum().item() + batch.y_H.pow(2).sum().item()
            
            bs = batch.num_graphs
            total_mse_sum += loss.item() * bs
            total_up += sum_sq_diff
            total_down += sum_sq_real
            total_graphs += bs

    # Aggregate
    avg_mse = total_mse_sum / total_graphs if total_graphs > 0 else 0.0
    avg_rel = total_up / max(total_down, 1e-12)
    
    return avg_mse, avg_rel

def evaluate_rollout_trend(model, loader, args, save_file="loss_trend.txt", epoch=None):
    model.eval()
    device = next(model.parameters()).device
    horizon = args.get('jump_horizon', 10)
    
    # Statistics container
    step_stats = defaultdict(lambda: {'sum_loss': 0.0, 'sum_up': 0.0, 'sum_down': 0.0, 'total_count': 0})
    
    bs = loader.batch_size if hasattr(loader, 'batch_size') else 'Dynamic'
    print(f"Starting Rollout Eval (BS={bs}, Horizon={horizon})...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            if batch.tgt_step.unique().numel() > 1:
                # Compatibility warning
                print(f"Warning: Mixed targets in batch {batch_idx}. Accuracy might be affected.")
            
            target_step = batch.tgt_step[0].item()
            
            # [Key] Call inference with alignment logic
            pred_E, pred_H = inference_rollout(model, batch, device, stride=horizon, args=args)
            
            # Calculate metrics
            loss_E = F.mse_loss(pred_E, batch.y_E)
            loss_H = F.mse_loss(pred_H, batch.y_H)
            total_loss = loss_E + loss_H
            
            # Relative Error Components
            diff_E = pred_E - batch.y_E
            diff_H = pred_H - batch.y_H
            up_val = diff_E.pow(2).sum().item() + diff_H.pow(2).sum().item()
            down_val = batch.y_E.pow(2).sum().item() + batch.y_H.pow(2).sum().item()
            
            current_batch_size = batch.num_graphs
            
            stats = step_stats[target_step]
            stats['sum_loss'] += total_loss.item() * current_batch_size
            stats['sum_up'] += up_val
            stats['sum_down'] += down_val
            stats['total_count'] += current_batch_size
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    print("Aggregating results...")
    
    # Aggregate: Group by Horizon (Binning)
    bins = defaultdict(lambda: {'losses': [], 'ups': [], 'downs': []})
    
    for step, stats in step_stats.items():
        avg_mse = stats['sum_loss'] / stats['total_count'] if stats['total_count'] > 0 else 0.0
        
        # Put into corresponding Bin (e.g., 1-10, 11-20)
        bin_end = ((step - 1) // horizon + 1) * horizon
        bins[bin_end]['losses'].append(avg_mse)
        bins[bin_end]['ups'].append(stats['sum_up'])
        bins[bin_end]['downs'].append(stats['sum_down'])

    sorted_bin_ends = sorted(bins.keys())
    
    global_loss_sum = 0
    global_count = 0
    global_up = 0
    global_down = 0
    lines_to_write = []
    
    os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else '.', exist_ok=True)
    file_exists = os.path.exists(save_file)
    
    with open(save_file, "a") as f:
        if not file_exists:
            f.write(f"{'Epoch':<8}\t{'Range':<12}\t{'Avg_MSE':<12}\t{'Avg_Rel':<12}\t{'Note':<10}\n")
            f.write("=" * 70 + "\n")
        
        for bin_end in sorted_bin_ends:
            bin_start = bin_end - horizon + 1
            if bin_start < 0: bin_start = 0
            
            bin_data = bins[bin_end]
            bin_avg_mse = np.mean(bin_data['losses'])
            
            total_bin_up = sum(bin_data['ups'])
            total_bin_down = sum(bin_data['downs'])
            bin_rel_err = total_bin_up / max(total_bin_down, 1e-12)
            
            global_loss_sum += sum(bin_data['losses'])
            global_count += len(bin_data['losses'])
            global_up += total_bin_up
            global_down += total_bin_down
            
            ep_str = str(epoch) if epoch is not None else "Test"
            range_str = f"{bin_start}-{bin_end}"
            
            line = f"{ep_str:<8}\t{range_str:<12}\t{bin_avg_mse:.2e}    \t{bin_rel_err:.2e}    \t"
            lines_to_write.append(line)
        
        final_mse_avg = global_loss_sum / global_count if global_count > 0 else 0.0
        final_rel_avg = global_up / max(global_down, 1e-12)
        
        for line in lines_to_write:
            f.write(line + "\n")
            
        ep_str = str(epoch) if epoch is not None else "Test"
        f.write(f"{ep_str:<8}\t{'TOTAL':<12}\t{final_mse_avg:.2e}    \t{final_rel_avg:.2e}    \t{'<--'}\n")
        f.write("-" * 70 + "\n")
        
    return final_mse_avg

def maxwell_pde_residual(pred_Ez, pred_Hx, pred_Hy, coords, eps):
    """Calculates Maxwell equation residuals (Autograd)"""
    # 1. Calculate derivatives
    grads_Ez = torch.autograd.grad(pred_Ez, coords, grad_outputs=torch.ones_like(pred_Ez), create_graph=True)[0]
    dEz_dx, dEz_dy, dEz_dt = grads_Ez[:, 0:1], grads_Ez[:, 1:2], grads_Ez[:, 2:3]

    grads_Hx = torch.autograd.grad(pred_Hx, coords, grad_outputs=torch.ones_like(pred_Hx), create_graph=True)[0]
    dHx_dt = grads_Hx[:, 2:3]
    dHx_dy = grads_Hx[:, 1:2] 

    grads_Hy = torch.autograd.grad(pred_Hy, coords, grad_outputs=torch.ones_like(pred_Hy), create_graph=True)[0]
    dHy_dt = grads_Hy[:, 2:3]
    dHy_dx = grads_Hy[:, 0:1]

    # 2. Calculate residuals
    # dEz/dt = (1/eps) * (dHy/dx - dHx/dy)
    res_Ez = dEz_dt - (1.0 / (eps + 1e-6)) * (dHy_dx - dHx_dy)
    # dHx/dt = - dEz/dy
    res_Hx = dHx_dt + dEz_dy
    # dHy/dt = dEz/dx
    res_Hy = dHy_dt - dEz_dx

    return torch.mean(res_Ez**2) + torch.mean(res_Hx**2) + torch.mean(res_Hy**2)

def calc_pinn_loss(model, data, w_pde=0.1):
    """
    Loss calculation process dedicated to PINN / PI-DeepONet
    """
    # 1. Prepare coordinate input with gradients
    device = data.x.device
    pos = data.points.clone()
    t = data.t
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    # Expand t dimension to match pos
    # Corrected logic: Use data.batch to map time t to each node
    # data.batch is an index array [0, 0, ..., 1, 1, ..., 11, 11], indicating which graph each node belongs to
    if hasattr(data, 'batch') and data.batch is not None:
        t_vec = t[data.batch].view(-1, 1)  # [Total_Nodes, 1]
    else:
        # Compatible with single graph case (batch_size=1)
        t_vec = t.view(-1, 1).expand(pos.size(0), 1)
    
    # Concatenate coordinates [x, y, t] and enable gradients
    coords = torch.cat([pos, t_vec], dim=-1)
    coords.requires_grad_(True)
    
    # 2. Forward pass (pass in coords for internal model use)
    # Note: Model forward needs to adapt to query_coords parameter
    pred_Ez, pred_Hx, pred_Hy, _ = model(data, query_coords=coords)
    
    # 3. Data Loss (Supervised Loss)
    # Assuming data.y_E is the ground truth for Ez
    # Note dimension matching, pred might be a 1D vector
    loss_data = F.mse_loss(pred_Ez, data.y_E) 
    
    # 4. Physics Loss (Equation Residuals)
    eps = data.x[:, 1:2] # Assuming eps is in the 2nd dimension of input
    loss_pde = maxwell_pde_residual(pred_Ez, pred_Hx, pred_Hy, coords, eps)
    
    total_loss = loss_data + w_pde * loss_pde
    return total_loss, loss_data.item(), loss_pde.item()

def calc_gpinn_loss(model, data, w_pde=0.1):
    """
    Discrete Loss dedicated to gPINN
    """
    # gPINN's forward returns (Ez_next, H_next) or similar tuple
    pred_Ez, pred_H = model(data)
    
    # 1. Data Loss
    loss_data = F.mse_loss(pred_Ez, data.y_E) + F.mse_loss(pred_H, data.y_H)
    
    # 2. Discrete Physics Loss (Simplified: Check energy conservation or discrete Curl)
    # Here you need to call the discrete_grad you defined in gPINN class
    # For simplicity, you can also just use data loss, or manually calculate discrete curl here
    # Assuming we only use data loss for comparison, or you can call model.discrete_loss(data) if you implemented it
    
    return loss_data, loss_data.item(), 0.0