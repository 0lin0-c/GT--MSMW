import time
import torch


def relative_error(loader, model):
    """Compute global relative error over an entire DataLoader.
    rel = sum(||y_hat - y||^2) / sum(||y||^2)
    avg_time = average forward time per output element.
    """
    model.eval()
    device = next(model.parameters()).device
    total_up = 0.0
    total_down = 0.0
    total_samples = 0
    total_time = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            start = time.time()
            pred = model(data)
            total_time += time.time() - start

            # 支持多任务输出 (节点 Ez, 边 H)
            if isinstance(pred, (tuple, list)) and len(pred) == 2 and hasattr(data, 'y_E') and hasattr(data, 'y_H'):
                pred_E, pred_H = pred
                diff_E = pred_E - data.y_E
                diff_H = pred_H - data.y_H
                total_up += diff_E.pow(2).sum().item() + diff_H.pow(2).sum().item()
                total_down += data.y_E.pow(2).sum().item() + data.y_H.pow(2).sum().item()
                total_samples += data.y_E.numel() + data.y_H.numel()
            else:
                diff = pred - data.y
                total_up += diff.pow(2).sum().item()
                total_down += data.y.pow(2).sum().item()
                total_samples += data.y.numel()

    rel = total_up / max(total_down, 1e-12)
    avg_time = total_time / max(total_samples, 1)
    return rel, avg_time


def relative_error_one_batch(loader, model, device=None):
    """Approximate relative error using only a single batch from loader.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    try:
        batch = next(iter(loader))
    except StopIteration:
        return float("nan")
    batch = batch.to(device)
    with torch.no_grad():
        pred = model(batch)

        # 支持多任务输出 (节点 Ez, 边 H)
        if isinstance(pred, (tuple, list)) and len(pred) == 2 and hasattr(batch, 'y_E') and hasattr(batch, 'y_H'):
            pred_E, pred_H = pred
            diff_E = pred_E - batch.y_E
            diff_H = pred_H - batch.y_H
            up = diff_E.pow(2).sum().item() + diff_H.pow(2).sum().item()
            down = batch.y_E.pow(2).sum().item() + batch.y_H.pow(2).sum().item()
        else:
            diff = pred - batch.y
            up = diff.pow(2).sum().item()
            down = batch.y.pow(2).sum().item()
    return up / max(down, 1e-12)
