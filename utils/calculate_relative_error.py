# calculate_relative_error.py
import numpy as np
import torch
import os, pickle
import time
import argparse
from fvcore.nn import FlopCountAnalysis

from models.build_model import build_model_from_loader
from utils.get_data import data_loader

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, data_loader, num_batches=10):
    """Measure average inference time per batch."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_batches:
                break
            data = data.to(device)
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Release GPU memory
            torch.cuda.empty_cache()
    return np.mean(times) if times else 0.0

def calculate_flops(model, data_loader):
    """Calculate FLOPs using fvcore."""
    try:
        model.eval()
        # Get a sample input (single Data object, not batch)
        sample_batch = next(iter(data_loader))
        if hasattr(sample_batch, 'to_data_list'):
            sample_data = sample_batch.to_data_list()[0]  # Get single Data object
        else:
            sample_data = sample_batch  # Fallback if not batched
        sample_data = sample_data.to(device)
        
        # For PyG Data, try to use the main tensor attributes
        if hasattr(sample_data, 'x'):
            # Use node features as input for FLOPs estimation
            input_tensor = sample_data.x
            flops = FlopCountAnalysis(model, input_tensor)
            total_flops = flops.total()
            return total_flops
        else:
            raise ValueError("Sample data does not have 'x' attribute.")
    except Exception as e:
        print(f"FLOPs calculation failed: {e}. Returning 0.")
        return 0

def calculate_relative_error(model, data_loader, idx_list, prefix, output_file):
    """Calculate relative error for a dataset."""
    model.eval()
    total_loss = 0.0
    total_time = 0.0
    count = 0
    
    with open(output_file, "a") as f:
        for i, data in enumerate(data_loader):
            j = idx_list[i] if i < len(idx_list) else i
            data = data.to(device)
            
            start_time = time.time()
            net_E, net_H = model(data)
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            loss_up = (torch.sum((net_E - data.y_E) ** 2) + torch.sum((net_H - data.y_H) ** 2)).cpu().detach().numpy()
            loss_down = (torch.sum(data.y_E ** 2) + torch.sum(data.y_H ** 2)).cpu().detach().numpy()
            rel_error = loss_up / loss_down
            print(f"{prefix} Index: {j}, Relative Error: {rel_error}, Inference Time: {inference_time}")
            total_loss += rel_error
            count += 1
            
            f.write(f"{prefix} Index: {j}\n")
            f.write(f"Relative Error: {rel_error}\n")
            f.write(f"Inference Time: {inference_time}\n")
            
            # Release GPU memory
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_time = total_time / count if count > 0 else 0.0
    return avg_loss, avg_time


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate relative error, parameter count, inference time, and FLOPs for a trained model.')
    parser.add_argument('--model', type=str, required=True, help='Model type (e.g., ehevolver, deepleapfrog)')
    parser.add_argument('--experiment-id', type=str, required=True, help='Experiment ID (e.g., experiment_21)')
    parser.add_argument('--indices-path', type=str, required=True, help='Path to indices file (e.g., ./indices/indices_10x199_160_19_20.pkl)')
    parser.add_argument('--data-folder', type=str, default='dataset/data_re60_fr1', help='Data folder')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num-of-simulations', type=int, default=100, help='Number of simulations')
    parser.add_argument('--num-of-steps', type=int, default=201, help='Number of steps')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load indices
    with open(args.indices_path, 'rb') as f:
        indices = pickle.load(f)
    train_idx = indices.get('train_idx', [])
    val_idx = indices.get('val_idx', [])
    if not val_idx:
        val_idx = indices.get('test_idx', [])  # Fallback if no val_idx

    # Prepare data loaders
    data_args = {
        'data_folder': args.data_folder,
        'batch_size': args.batch_size,
        'indices_path': args.indices_path,
        'model': args.model,
        'num_of_simulations': args.num_of_simulations,
        'num_of_steps': args.num_of_steps
    }
    train_loader, val_loader, _ = data_loader(data_args)

    # Build and load model
    model = build_model_from_loader(data_args, train_loader)
    model_path = f'./results/model/experiment_{args.experiment_id}/checkpoint.best.pt'
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f'Loaded model from {model_path}')
    else:
        raise FileNotFoundError(f'Model not found at {model_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Calculate metrics
    param_count = count_parameters(model)
    inference_time = measure_inference_time(model, train_loader, num_batches=10)
    flops = calculate_flops(model, train_loader)

    # Create output directory if it doesn't exist
    os.makedirs('./results/relative_error', exist_ok=True)
    output_file = f"./results/relative_error/{args.experiment_id}_metrics.txt"

    # Calculate relative error for train and val sets
    train_loss, train_time = calculate_relative_error(model, train_loader, train_idx, 'train', output_file)
    val_loss, val_time = calculate_relative_error(model, val_loader, val_idx, 'val', output_file)

    # Write summary
    with open(output_file, "a") as f:
        f.write(f"\n=== Summary for {args.experiment_id} ===\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Parameter Count: {param_count}\n")
        f.write(f"FLOPs: {flops}\n")
        f.write(f"Average Inference Time (train): {inference_time:.6f} seconds\n")
        f.write(f"Train Relative Error: {train_loss}\n")
        f.write(f"Val Relative Error: {val_loss}\n")
        f.write(f"Train Inference Time: {train_time:.6f} seconds\n")
        f.write(f"Val Inference Time: {val_time:.6f} seconds\n")

    print(f"Results saved to {output_file}")
    print(f"Parameter Count: {param_count}")
    print(f"FLOPs: {flops}")
    print(f"Average Inference Time: {inference_time:.6f} seconds")
    print(f"Train Relative Error: {train_loss}")
    print(f"Val Relative Error: {val_loss}")

'''
python -m utils.calculate_relative_error \
    --model ehevolver  \
    --experiment-id 30  \
    --indices-path ./indices/indices_tailtest_100x200_10samples_20steps_val0.1.pkl \
    --data-folder ./dataset/point_re60_fr1-2 \
    --batch-size 12 
'''