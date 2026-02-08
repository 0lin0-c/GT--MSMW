import torch
import os
import argparse
from tqdm import tqdm
import sys

# Import project dependencies
from loss_function import evaluate_rollout_trend
from utils.get_data import data_loader_jump_ahead
from models.build_model import build_model_from_loader

def main(args):
    # --- 1. Environment and Device Settings (Single GPU Mode) ---
    # If CUDA_VISIBLE_DEVICES is specified, cuda:0 refers to that specific card
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    print(f"Starts Evaluation for Experiment ID: {args.experiment_id}")
    print(f"Device: {device}")
    print(f"Data Folder: {args.data_folder}")

    # --- 2. Prepare Data ---
    print("Loading datasets...")
    data_args = vars(args)
    
    # Load Test Loader
    _, _, test_loader = data_loader_jump_ahead(data_args, test_only=True)

    # --- 3. Build Model ---
    print("Building model...")
    # Use test_loader to infer dimensions
    model = build_model_from_loader(data_args, test_loader)
    model.to(device)
    
    # Note: DDP wrapper (DistributedDataParallel) is no longer needed here

    # --- 4. Load Checkpoint ---
    ckpt_path = os.path.join(args.model_save_to, args.model, args.experiment_id, "checkpoint.best.pt")
    
    if not os.path.exists(ckpt_path):
        fallback_path = os.path.join(args.model_save_to, args.model, args.experiment_id, "checkpoint_final.pt")
        print(f"Warning: Best checkpoint not found at {ckpt_path}")
        if os.path.exists(fallback_path):
            print(f"Trying fallback: {fallback_path}")
            ckpt_path = fallback_path
        else:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path} or {fallback_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Load weights to current device
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # --- Key Step: Handle Weight Key Names ---
    # If checkpoint was saved using DDP training, key names will have 'module.' prefix
    # Current model is a single-card model without 'module.', so prefix must be removed to match
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.'
        else:
            new_state_dict[k] = v
            
    # Load processed weights
    model.load_state_dict(new_state_dict)
    model.eval() # Switch to evaluation mode
    print("Model loaded successfully.")

    # --- 5. Execute Evaluation ---
    print("\n" + "="*50)
    print("Running FINAL FULL TREND EVALUATION (Single GPU)...")
    print("="*50)
    
    result_dir = os.path.join(args.loss_save_to, args.model, args.experiment_id)
    os.makedirs(result_dir, exist_ok=True)
    final_filename = f"{args.experiment_id}_final_trend_SingleGPU.txt"
    final_file = os.path.join(result_dir, final_filename)
    
    # Execute evaluation
    final_loss = evaluate_rollout_trend(
        model, 
        test_loader, 
        data_args, 
        save_file=final_file,
        epoch="EVAL-ONLY"
    )
    
    print(f"Full Trend Evaluation Done. Avg MSE: {final_loss:.4e}")
    print(f"Results saved to: {final_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Single GPU Evaluation Script')
    
    # Core arguments
    parser.add_argument('--experiment-id', type=str, required=True, 
                        help='Experiment ID folder name')
    parser.add_argument('--data-folder', type=str, required=True, 
                        help='Path to dataset')
    
    # Path arguments
    parser.add_argument('--model-save-to', type=str, default='results/model')
    parser.add_argument('--loss-save-to', type=str, default='results/loss')
    parser.add_argument('--indices-path', type=str, default='./indices/indices_10x199_160_19_20.pkl')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='gat_conv')
    parser.add_argument('--num-of-simulations', type=int, default=100)
    parser.add_argument('--num-of-steps', type=int, default=201)
    parser.add_argument('--jump-horizon', type=int, default=10)
    parser.add_argument('--transformer-mode', type=str, default='pad')
    parser.add_argument('--reality', type=bool, default=True)

    
    # Running arguments
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default="0", help='GPU ID to use (e.g., "0" or "1")')

    # Keep these arguments for compatibility with old commands, but they are not used in the code
    parser.add_argument('--ddp', action='store_true', help='Ignored in single gpu script')
    parser.add_argument('--local-rank', type=int, default=-1, help='Ignored in single gpu script')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    # Set visible GPUs
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        
    main(args)

'''
python -m utils.eval_only\
    --experiment-id experiment_5 \
    --model-save-to results/model/com_abl \
    --loss-save-to results/loss/com_abl \
    --data-folder dataset/point_re60_fr1-2 \
    --model ehevolver \
    --batch-size 12 \
    --gpus 3
'''