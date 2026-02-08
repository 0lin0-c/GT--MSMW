import numpy as np
import os, pickle, random
from utils.load_data import load_data, load_data_v2, load_data_v3, load_data_v4

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, random_split, Subset
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler
from tqdm import tqdm  # Recommended to add a progress bar, as full loading can be slow

class GNNDataset(Dataset):
    def __init__(self, args):
        self.args = args
        model_name = (self.args.get('model', 'gat_conv') or '').lower()
        # Models using specialized data formats:
        # - DeepLeapfrog / DeepLeapTrans: load_data_v2 (x=[Ez,Hx,Hy,eps], has data.t)
        # - EH-Evolver series: load_data_v3 (Nodes Ez/eps + Edges Hx/Hy + data.t)
        models_with_time_v2 = {
            'deepleapfrog',
            'deepleaptrans',
        }
        models_with_time_v3 = {
            'ehevolver',
            'phys_egat',
        }
        if model_name in models_with_time_v3:
            loader_fn = load_data_v3
        elif model_name in models_with_time_v2:
            loader_fn = load_data_v2
        else:
            loader_fn = load_data
        self.data = [loader_fn(i, j, self.args) for i in range(self.args['num_of_simulations']) for j in range(self.args['num_of_steps']-1)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GNNDataset_online_loader(Dataset):
    def __init__(self, args, data_number=20000):
        self.data_number = data_number
        self.args = args

    def __len__(self):
        return self.data_number

    def __getitem__(self, idx):
        simulation_num = idx // (self.args['num_of_steps'] - 1)
        step_num = idx % (self.args['num_of_steps'] - 1)
        model_name = (self.args.get('model', 'gat_conv') or '').lower()
        models_with_time_v2 = {
            'deepleapfrog',
            'deepleaptrans',
        }
        models_with_time_v3 = {
            'ehevolver',
            'phys_egat',
        }
        if model_name in models_with_time_v3:
            loader_fn = load_data_v3
        elif model_name in models_with_time_v2:
            loader_fn = load_data_v2
        else:
            loader_fn = load_data
        return loader_fn(simulation_num, step_num, self.args)

def data_loader(args):
    dataset = GNNDataset_online_loader(args)
       # indices path can be configured via args['indices_path'], fallback to old default
    indices_path = args.get('indices_path', './indices/indices_100x199_160_19_20.pkl') if isinstance(args, dict) else './indices/indices_100_80_10_10.pkl'
    with open(indices_path, 'rb') as f:  # indices_2_15-2-2.pkl indices_100_80_10_10.pkl
        indices = pickle.load(f)
    train_idx = indices['train_idx']
    test_idx = indices['test_idx']
    # val_idx might not exist (e.g., only split into train/test)
    val_idx = indices.get('val_idx', None)
    print(len(train_idx))

    train_dataset = Subset(dataset, train_idx)
    print(len(train_dataset))
    test_dataset = Subset(dataset, test_idx)
    val_dataset = Subset(dataset, val_idx) if val_idx is not None else None

    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # Use DistributedSampler only when distributed is initialized
    num_workers = int(args.get('num_workers', 0)) if isinstance(args, dict) else 0
    pin_memory = torch.cuda.is_available()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    print(len(train_loader))
    # Construct val_loader only if validation set indices exist, otherwise return None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        val_loader = None
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader

def data_loader_index(args):
    # Use preloaded version of GNNDataset, load all samples into memory at once
    # Global sample count = num_of_simulations * (num_of_steps - 1)
    dataset = GNNDataset(args)
    # indices path can be configured via args['indices_path'], fallback to old default
    indices_path = args.get('indices_path', './indices/indices_100x199_160_19_20.pkl') if isinstance(args, dict) else './indices/indices_100_80_10_10.pkl'
    with open(indices_path, 'rb') as f:  # indices_2_15-2-2.pkl indices_100_80_10_10.pkl
        indices = pickle.load(f)
    train_idx = indices['train_idx']
    test_idx = indices['test_idx']
    # val_idx might not exist (e.g., only split into train/test)
    val_idx = indices.get('val_idx', None)
    print(len(train_idx))

    train_dataset = Subset(dataset, train_idx)
    print(len(train_dataset))
    test_dataset = Subset(dataset, test_idx)
    val_dataset = Subset(dataset, val_idx) if val_idx is not None else None

    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # Use DistributedSampler only when distributed is initialized
    num_workers = int(args.get('num_workers', 0)) if isinstance(args, dict) else 0
    pin_memory = torch.cuda.is_available()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    print(len(train_loader))
    # Construct val_loader only if validation set indices exist, otherwise return None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        val_loader = None
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


class GNNDatasetJumpAhead(Dataset):
    def __init__(self, args, split='train', lazy_load=False):
        """
        Args:
            lazy_load (bool): 
                True  -> Store metadata only (sim, src, tgt), IO on read (For Test).
                False -> Load all data directly into RAM (For Train/Val).
        """
        self.args = args
        self.split = split
        self.lazy_load = lazy_load
        
        # Storage container
        self.storage = [] 
        
        # --- Config parameters ---
        horizon = self.args.get('jump_horizon', 10)
        total_sims = self.args['num_of_simulations'] 
        
        # As requested: First 90 full, last 10 for validation
        num_truncated_sims = 10 
        num_full_sims = total_sims - num_truncated_sims 
        
        full_range_sims = list(range(0, num_full_sims))        # 0 - 89
        truncated_range_sims = list(range(num_full_sims, total_sims)) # 90 - 99
        
        global_max_step = 120 
        truncated_max_step = 100
        
        tasks = []
        
        print(f"Building Dataset split='{split}' | Horizon: {horizon} | Lazy Load: {lazy_load}")

        # ==============================================================================
        # 1. Training Set (Train)
        # ==============================================================================
        if split == 'train':
            # Sim 0-89: All the way to 110 (or truncated as needed, here set to global max)
            self._add_train_tasks(tasks, full_range_sims, global_max_step, horizon)
            # Sim 90-99: Truncated to 100
            self._add_train_tasks(tasks, truncated_range_sims, truncated_max_step, horizon)

        # ==============================================================================
        # 2. Validation Set (Val) - Extrapolation
        # Range: Src 91-100 -> Tgt 101-110
        # ==============================================================================
        elif split == 'val':
            start_src = truncated_max_step - horizon + 1 # 91
            end_src = truncated_max_step # 100
            
            for src in range(start_src, end_src + 1):
                tgt = src + horizon 
                if tgt <= global_max_step:
                    for sim_id in truncated_range_sims:
                        tasks.append((sim_id, src, tgt))

        # ==============================================================================
        # 3. Test Set (Test) - Dense Trend (0 -> 1...110)
        # Must generate dense data with Step=1 to test non-divisible steps like 81
        # ==============================================================================
        elif split == 'test':
            print(f"  [Test] Loading ALL Sims for DENSE trend 0 -> 1...{global_max_step} (Step=1)...")
            all_sims = list(range(total_sims))
            src_step = 0
            
            # Generate 1, 2, ..., 110
            for tgt_step in range(src_step + 1, global_max_step + 1, 1):
                for sim_id in all_sims:
                    tasks.append((sim_id, src_step, tgt_step))

        # ==============================================================================
        # Data Loading
        # ==============================================================================
        if self.lazy_load:
            print(f"  [Lazy] Storing {len(tasks)} task metadata only.")
            self.storage = tasks
        else:
            print(f"  [Pre-load] Loading {len(tasks)} samples into RAM...")
            for sim_id, src, tgt in tqdm(tasks):
                try:
                    data_obj = load_data_v4(sim_id, src, tgt, self.args)
                    self.storage.append(data_obj)
                except Exception as e:
                    print(f"Error loading {sim_id}-{src}->{tgt}: {e}")

    def _add_train_tasks(self, tasks, sim_ids, max_step, horizon):
        # Offset
        for target in range(1, horizon):
            for sim_id in sim_ids:
                tasks.append((sim_id, 0, target))
        # Stride
        max_start = max_step - horizon
        for start in range(0, max_start + 1):
            for sim_id in sim_ids:
                tasks.append((sim_id, start, start + horizon))

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, idx):
        if self.lazy_load:
            # Read on demand
            sim_id, src, tgt = self.storage[idx]
            return load_data_v4(sim_id, src, tgt, self.args)
        else:
            # Read from memory
            return self.storage[idx]


# ... (GNNDatasetJumpAhead class remains unchanged) ...

def data_loader_jump_ahead(args, test_only=False):
    """
    Returns Train, Val, Test Loaders.
    Args:
        test_only (bool): If True, skip Train and Val loading, return only Test Loader.
                          Format becomes (None, None, test_loader).
    """
    train_batch_size = args['batch_size']
    total_sims = args['num_of_simulations']
    
    # Dynamically calculate batch size for validation set
    if total_sims > 20:
        num_truncated_sims = 10
    else:
        num_truncated_sims = max(1, total_sims // 2)

    num_workers = int(args.get('num_workers', 4))
    pin = torch.cuda.is_available()
    
    train_loader = None
    val_loader = None

    # =========================================================
    # 1. Train & Val (Load only when test_only=False)
    # =========================================================
    if not test_only:
        print("\n--- Initializing Train Dataset (RAM) ---")
        train_ds = GNNDatasetJumpAhead(args, split='train', lazy_load=False)
        
        print("\n--- Initializing Val Dataset (RAM) ---")
        val_ds = GNNDatasetJumpAhead(args, split='val', lazy_load=False)

        # Train Loader
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
            train_loader = DataLoader(train_ds, batch_size=train_batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin)
        else:
            train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)

        # Val Loader
        val_loader = DataLoader(val_ds, batch_size=num_truncated_sims, shuffle=False, num_workers=num_workers, pin_memory=pin)
    else:
        print("\n[Info] Skipping Train/Val datasets (test_only=True).")

    # =========================================================
    # 2. Test (Always loaded, supports Lazy Load)
    # =========================================================
    print("\n--- Initializing Test Dataset (Disk / Lazy) ---")
    test_ds = GNNDatasetJumpAhead(args, split='test', lazy_load=True)
    
    test_loader = DataLoader(test_ds, batch_size=total_sims, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader

# Encapsulation: Preload + Dataset + Partition + DataLoader, all in one step
def sample_data_loader(args, ddp_enabled=False, epoch=None):
    """
    Returns only train_loader, where each item is a full time sequence for a sample.
    In DDP, automatically uses DistributedSampler and sets epoch.
    """
    dataset = SampleDataset(args)
    if ddp_enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler, shuffle=(sampler is None)) #, collate_fn=identity_collate)
    # If epoch is specified and distributed, set epoch to ensure consistent shuffle
    if ddp_enabled and epoch is not None and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)
    return train_loader


class SampleDataset(Dataset):
    """
    Each item = Full time sequence of one sample
    """
    def __init__(self, args):
        self.args = args
        self.num_sim = args['num_of_simulations']
        self.num_steps = args['num_of_steps']  # = 200

        self.all_data = self._preload_all()

    def _preload_all(self):
        all_data = []
        model_name = (self.args.get('model', 'gat_conv') or '').lower()
        models_with_time_v2 = {
            'deepleapfrog',
            'deepleaptrans',
        }
        models_with_time_v3 = {
            'ehevolver',
            'phys_egat',
        }
        if model_name in models_with_time_v3:
            loader_fn = load_data_v3
        elif model_name in models_with_time_v2:
            loader_fn = load_data_v2
        else:
            loader_fn = load_data
        for sim_id in range(self.num_sim):
            sim_seq = []
            for t in range(self.num_steps):
                data = loader_fn(sim_id, t, self.args)
                sim_seq.append(data)
            all_data.append(sim_seq)
        return all_data

    def __len__(self):
        return self.num_sim

    def __getitem__(self, idx):
        return self.all_data[idx]   # list[Data], len=200