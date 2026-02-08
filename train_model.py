from html import parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
from tqdm import tqdm
import time
import argparse
# import swanlab  <-- Removed
import copy # Import copy for deep copying args

# Import your custom modules (keep unchanged)
from utils.loss_function import loss_function, evaluate_validation_simple, evaluate_rollout_trend, calc_pinn_loss, calc_gpinn_loss
from utils.training_utils import EarlyStopping, get_next_experiment_dir
from utils.get_data import data_loader_jump_ahead
from utils.save_results import save_loss_results
from utils.relative_error import relative_error_one_batch
# from utils.config_loader import load_secret_config <-- Removed if not used elsewhere
from models.build_model import build_model_from_loader

# [Modification 1] Added preloaded_loaders parameter to train function
def train(args, preloaded_loaders=None):
    # --- 1. Configuration and Environment Initialization ---
    es_patience = int(args.get('es_patience', 50))
    es_min_delta = float(args.get('es_min_delta', 0.001))
    no_es_flag = bool(args.get('no_early_stopping'))
    early_stopping_enabled = (es_patience > 0) and (not no_es_flag)
    early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta, verbose=True) if early_stopping_enabled else None
    com_abl_path = os.path.join('results', 'model', 'com_abl', args['model'])
    model_save_base = com_abl_path 
    save_base = None
    experiment_id = "Exp_Unknown"

    ddp_enabled = bool(args.get('ddp', False))

    # --- 2. Data Loading (Core Modification) ---
    if preloaded_loaders is not None:
        # If data is passed, use it directly (implement data reuse)
        if args.get('local_rank', 0) == 0:
            print(f"[{args['model']}] Using pre-loaded data...")
        train_loader, val_loader, test_loader = preloaded_loaders
    else:
        # Compatible with old mode: if not passed, load it manually
        train_loader, val_loader, test_loader = data_loader_jump_ahead(args)

    # --- Device Settings ---
    local_rank_val = args.get('local_rank', -1)
    try:
        local_rank = int(os.environ.get('LOCAL_RANK', local_rank_val))
    except:
        local_rank = -1

    if ddp_enabled and torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = build_model_from_loader(args, train_loader)
    is_main = (not ddp_enabled) or (torch.distributed.get_rank() == 0)
    model.to(device)

    if ddp_enabled:
        find_unused = bool(args.get('find_unused_parameters'))
        model = DDP(model, device_ids=[local_rank] if local_rank >= 0 else None, find_unused_parameters=find_unused)

    # --- Experiment Directory and ID Generation ---
    if is_main:
        try:
            exp_dir = get_next_experiment_dir(base=model_save_base)
            save_base = str(exp_dir / 'checkpoint')
            experiment_id = exp_dir.name
            with open(exp_dir / "config.txt", "w") as f:
                for key, value in args.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            print(f"Warning: Error creating experiment dir: {e}. Using fallback.")
            os.makedirs(model_save_base, exist_ok=True)
            save_base = os.path.join(model_save_base, "Exp_Default", 'checkpoint')
            experiment_id = "Exp_Default"

    if ddp_enabled:
        obj_list = [save_base, experiment_id]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        save_base, experiment_id = obj_list
    elif save_base is None:
         # Single card Fallback
        save_base = os.path.join(model_save_base, "Exp_Single", 'checkpoint')
        experiment_id = "Exp_Single"

    lr = args.get('lr', 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=args.get('lr_patience', 10))

    if is_main:
        trend_save_dir = os.path.join(args['loss_save_to'], 'epoch_trends')
        os.makedirs(trend_save_dir, exist_ok=True)
    losses = []
    best_val_loss = float('inf')
    model_name = args.get('model', 'gat_conv').lower()
    pde_weight = float(args.get('pde_weight', 0.1))

    # --- 3. Training Loop ---
    for epoch in tqdm(range(args['n_epochs']), desc=f"Train {args['model']}"):
        epoch_loss = 0.0
        
        # [Important] Set sampler epoch in DDP mode, otherwise data shuffle won't work
        if ddp_enabled and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
            
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # --- Model Forward Propagation Logic (Keep Unchanged) ---
            if model_name in ['pideeponet', 'pinn']:
                loss, l_data, l_pde = calc_pinn_loss(model, data, w_pde=pde_weight)
            elif model_name == 'gpinn':
                loss, l_data, l_pde = calc_gpinn_loss(model, data, w_pde=pde_weight)
            else:
                preds = model(data)
                if isinstance(preds, (tuple, list)) and len(preds) == 2:
                    loss_E = loss_function(preds[0], data.y_E)
                    loss_H = loss_function(preds[1], data.y_H)
                    loss = loss_E + loss_H
                else:
                    loss = loss_function(preds, data.y)
            # ----------------------------------

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # --- Validation Logic ---
        val_loss = 0.0
        val_rel_err = 0.0
        train_loss = epoch_loss / max(1, len(train_loader))
        
        if is_main:
            eval_model = model.module if hasattr(model, 'module') else model
            train_rel_err = relative_error_one_batch(train_loader, eval_model, device)
            val_loss, val_rel_err = evaluate_validation_simple(eval_model, val_loader, args)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Simplify log output to prevent screen flooding
            # [Modification 3] Add Train Rel back to log printing
            print(f"Epoch: {epoch} | Train: {train_loss:.2e} (Rel: {train_rel_err:.2e}) | Val (Extra 101-110): {val_loss:.2e} (Rel: {val_rel_err:.2e})")

        if ddp_enabled:
            v_tensor = torch.tensor([val_loss if is_main else 0.0], device=device)
            torch.distributed.broadcast(v_tensor, src=0)
            val_loss = v_tensor.item()

        if is_main:
            losses.append((epoch, train_loss, val_loss))
            save_loss_results(args, losses, experiment_id)

        # Early Stopping
        stopped = False
        if is_main and early_stopper:
            stopped = early_stopper.step(val_loss, model, optimizer=optimizer, epoch=epoch, save_path=save_base)
        
        if ddp_enabled:
            s_tensor = torch.tensor([int(stopped)], device=device)
            torch.distributed.broadcast(s_tensor, src=0)
            stopped = bool(s_tensor.item())

        if stopped:
            break
        else:
            scheduler.step(val_loss)

    # --- 5. Training Finished: Full Trend Test (Keep Detailed Report) ---
    if is_main:
        print("\n" + "="*50)
        print("Training Finished. Running FINAL FULL TREND EVALUATION (0-110)...")
        print("="*50)
        
        # Get original model (remove DDP wrapper)
        eval_model = model.module if hasattr(model, 'module') else model
        
        best_ckpt = f"{save_base}.best.pt"
        if os.path.exists(best_ckpt):
            print(f"Loading best checkpoint from {best_ckpt}...")
            # 1. Load file
            ckpt = torch.load(best_ckpt, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            
            # 2. [Core Fix] Remove 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove leading 'module.' (7 characters)
                else:
                    new_state_dict[k] = v
            
    #         # 3. Load processed weights
            eval_model.load_state_dict(new_state_dict)
            print("Checkpoint loaded successfully (prefix removed).")
            
        else:
            print("Warning: Best checkpoint not found, using last model.")

        # Test Trend Filename
        final_filename = f"{experiment_id}_final_trend_0_110.txt"
        final_file = os.path.join(args['loss_save_to'], args['model'], final_filename)
        
        # Use Test Loader (Lazy Loaded, Dense Step=1)
        # Call heavy function here to generate detailed report
        final_loss = evaluate_rollout_trend(
            eval_model, 
            test_loader, 
            args, 
            save_file=final_file,
            epoch="FINAL"
        )
        
        print(f"Full Trend Evaluation Done. Avg MSE: {final_loss:.4e}")
        print(f"Check trend details in: {final_file}")
        
        # [Critical] Release GPU memory to prevent OOM for the next model
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
    
    return losses

def parse_arguments():
    # ... Keep your parse_arguments unchanged ...
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--find-unused-parameters', action='store_true')
    parser.add_argument('--data-folder', type=str, default='dataset/data_re60_fr1')
    parser.add_argument('--model-save-to', type=str, default='results/model')
    parser.add_argument('--loss-save-to', type=str, default='results/loss')
    parser.add_argument('--num-of-simulations', type=int, default=100)
    parser.add_argument('--num-of-steps', type=int, default=201)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--n-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr-patience', type=int, default=4)
    parser.add_argument('--es-patience', type=int, default=30)
    parser.add_argument('--es-min-delta', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='gat_conv')
    parser.add_argument('--transformer-mode', type=str, default='pad', choices=['mask', 'pad'])
    parser.add_argument('--indices-path', type=str, default='./indices/indices_10x199_160_19_20.pkl')
    parser.add_argument('--config-path', type=str, default='config/local_config.json')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no-early-stopping', action='store_true')
    parser.add_argument('--jump-horizon', type=int, default=10)
    parser.add_argument('--pde-weight', type=float, default=0.1)
    parser.add_argument('--model-list', type=str, nargs='+', default=None, help='List of models to train sequentially')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("program started")
    torch.cuda.empty_cache()
    cmd_args = parse_arguments()
    args = vars(cmd_args)

    if args.get('gpus'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
    
    # --- DDP Initialization (Global, only once) ---
    ddp_enabled = args.get('ddp', False)
    if ddp_enabled:
        if 'LOCAL_RANK' not in os.environ:
             os.environ['LOCAL_RANK'] = str(args.get('local_rank', -1))
        
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        print(f"[Rank {local_rank}] DDP Initialized.")

    # --- [Key Step 1] Unified Data Loading (Load only once) ---
    # Only the main process prints loading info, but all processes need to execute the loading function (because DDP needs respective data shards)
    if not ddp_enabled or torch.distributed.get_rank() == 0:
        print(">>> Loading Data ONCE for all models...")
    
    # Load data
    loaders = data_loader_jump_ahead(args)
    
    # --- [Key Step 2] Determine which models to run ---
    # If --model-list is specified in command line, use list; otherwise run only the one specified by --model
    if args.get('model_list'):
        models_to_run = args['model_list']
    else:
        models_to_run = [args['model']]

    # --- [Key Step 3] Loop Training ---
    for i, model_name in enumerate(models_to_run):
        if not ddp_enabled or torch.distributed.get_rank() == 0:
            print("\n" + "="*60)
            print(f"  SEQUENCE [{i+1}/{len(models_to_run)}]: Training Model '{model_name}'")
            print("="*60)
        
        # Deep copy args to prevent previous loop from modifying parameters affecting the next one
        current_args = copy.deepcopy(args)
        current_args['model'] = model_name
        
        # Special handling for ablation experiments (if model name contains special markers, args can be modified here)
        if "no_attention" in model_name:
            current_args['no_attention'] = True
            # Restore base model name for build_model recognition, or if your build_model can recognize 'ehevolver_no_att'
            # current_args['model'] = 'ehevolver' 
        
        # Call training function, passing pre-loaded data
        train(current_args, preloaded_loaders=loaders)
        
        # Force synchronization after each training to ensure all GPUs finish before starting next model
        if ddp_enabled:
            torch.distributed.barrier()

    if not ddp_enabled or torch.distributed.get_rank() == 0:
        print("All models in list finished.")