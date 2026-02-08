"""Training utilities: EarlyStopping helper.

Provides an EarlyStopping class you can instantiate in the training script and call
each epoch with the current validation loss and model state.

Behavior:
- Tracks the best validation loss (lower is better).
- When a new best is found it saves a "best" checkpoint (if a path is provided).
- When validation loss hasn't improved for `patience` consecutive epochs, it
  signals early stopping and will save a final checkpoint (if a path is provided).

Example usage in training loop:

    es = EarlyStopping(patience=20, min_delta=1e-6, verbose=True)
    for epoch in range(epochs):
        ... run training/validation ...
        stop = es.step(val_loss, model, optimizer=opt, scheduler=sched,
                       epoch=epoch, save_path=str(exp_dir / 'checkpoint.pt'))
        if stop:
            print('Early stopping triggered')
            break

The saved checkpoint is a dict containing:
- model_state_dict
- optimizer_state_dict (if provided)
- scheduler_state_dict (if provided)
- epoch
- val_loss
- best_val_loss
- best_epoch

"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Optional

import torch
import pathlib
import re
from models.gnn_MetaLayers import compute_physics_from_data


@dataclass
class EarlyStopping:
    patience: int = 20
    min_delta: float = 0.0
    verbose: bool = False
    save_best_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.best_val: float = math.inf
        self.counter: int = 0
        self.best_epoch: Optional[int] = None

    def _save_checkpoint(self, path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                         scheduler: Optional[object], epoch: Optional[int], val_loss: float, note: str = "") -> None:
        # ensure directory exists
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

        state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val,
            "best_epoch": self.best_epoch,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            # scheduler may be an optimizer scheduler object
            try:
                state["scheduler_state_dict"] = scheduler.state_dict()
            except Exception:
                # if scheduler doesn't implement state_dict(), skip
                pass
        if note:
            state["note"] = note

        torch.save(state, path)

    def step(self, val_loss: float, model: torch.nn.Module, *, optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[object] = None, epoch: Optional[int] = None, save_path: Optional[str] = None) -> bool:
        """Notify early-stopper of the latest validation loss.

        Args:
            val_loss: current validation loss (lower is better)
            model: model to save when improvement occurs or when early stopping triggers
            optimizer: optional optimizer to include in saved checkpoint
            scheduler: optional scheduler to include in saved checkpoint
            epoch: current epoch number (for checkpoints)
            save_path: base path to write checkpoints. If provided, two files may be
                written:
                  - <save_path>.best.pt  (when best improved)
                  - <save_path>.earlystop.pt (when stopping triggers)

        Returns:
            stop (bool): True if training should stop (patience exhausted), False otherwise.
        """
        improved = val_loss < (self.best_val - self.min_delta)
        if improved:
            self.best_val = float(val_loss)
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: val_loss improved to {self.best_val:.6f} (epoch={epoch})")
            # save best checkpoint if path provided
            if save_path or self.save_best_path:
                p = self.save_best_path if self.save_best_path else save_path
                best_path = f"{p}.best.pt"
                try:
                    self._save_checkpoint(best_path, model, optimizer, scheduler, epoch, val_loss, note="best")
                    if self.verbose:
                        print(f"Saved best checkpoint to {best_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: failed to save best checkpoint to {best_path}: {e}")
            return False

        # not improved
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping: no improvement in val_loss (counter={self.counter}/{self.patience})")

        if self.counter >= self.patience:
            # trigger early stopping: save final checkpoint (if requested)
            if save_path or self.save_best_path:
                p = self.save_best_path if self.save_best_path else save_path
                early_path = f"{p}.earlystop.pt"
                try:
                    self._save_checkpoint(early_path, model, optimizer, scheduler, epoch, val_loss, note="earlystop")
                    if self.verbose:
                        print(f"Saved early-stop checkpoint to {early_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: failed to save early-stop checkpoint to {early_path}: {e}")
            return True

        return False


__all__ = ["EarlyStopping"]


def get_next_experiment_dir(base="results"):
    base_p = pathlib.Path(base)
    base_p.mkdir(parents=True, exist_ok=True)

    pat = re.compile(r"^experiment_(\d+)$")
    max_num = 0
    for p in base_p.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num

    next_num = max_num + 1
    while True:
        candidate = base_p / f"experiment_{next_num}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            next_num += 1


def Determine_Inf_Nan(data, prefix):
    if torch.isnan(data).any():
        print(f"{prefix} contains NaN!")
    if torch.isinf(data).any():
        print(f"{prefix} contains Inf!")


def _model_summary_for_log(model: torch.nn.Module) -> str:
    """Return a compact string summary for logging."""
    lines = []
    lines.append(str(model))
    try:
        param_count = sum(p.numel() for p in model.parameters())
        lines.append(f"Total parameters: {param_count}")
    except Exception:
        pass
    return "\n".join(lines)


def apply_noise_and_update_physics(data, noise_std=0.02):
    """
    极速版数据增强：
    1. 给 E, H 注入噪声 (In-place 操作，零内存开销)
    2. 复用 edge_attr 中的 (dx, dy) 快速重算 grad_vec 和 C_ij
    3. 全程在 GPU 上完成，无需 CPU 同步
    """
    if data.x is not None:
        noise = torch.randn_like(data.x[:, 0:1]) * noise_std
        data.x[:, 0:1].add_(noise)

    if data.edge_attr is not None:
        noise = torch.randn_like(data.edge_attr[:, 0:2]) * noise_std
        data.edge_attr[:, 0:2].add_(noise)

    E_scalar = data.x[:, 0]  # Noisy Ez
    Hx = data.edge_attr[:, 0:1]  # Noisy Hx
    Hy = data.edge_attr[:, 1:2]  # Noisy Hy


    dpos = data.edge_attr[:, 2:4]  # Static [dx, dy]
    dx = data.edge_attr[:, 2:3]
    dy = data.edge_attr[:, 3:4]


    dist_sq = dx.pow(2) + dy.pow(2) + 1e-8

    row, col = data.edge_index
    E_diff = (E_scalar[col] - E_scalar[row]).unsqueeze(-1)

    new_grad_vec = (E_diff / dist_sq) * dpos
    new_C_ij = Hx * (dy) + Hy * dx

    data.grad_vec = new_grad_vec/100
    data.C_ij = new_C_ij

    return data
