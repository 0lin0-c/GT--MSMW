import torch
import os

def save_model_results(args, model, number, epoch):
    model_path = os.path.join(args['model_save_to'], args['model'], f"model_{number}")
    # optimizer_path = os.path.join(model_path, "optimizer_states")
    os.makedirs(model_path, exist_ok=True)
    # os.makedirs(optimizer_path, exist_ok=True)
    # Support both DeepSpeed engine and plain torch.nn.Module
    try:
        # DeepSpeed engine
        if hasattr(model, 'save_checkpoint'):
            model.save_checkpoint(model_path)
            return
    except Exception:
        pass

    # Fallback: save state_dict
    try:
        save_path = os.path.join(model_path, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), save_path)
    except Exception as e:
        print('Failed to save model state_dict:', e)

def save_loss_results(args, train_loss, number):
    loss_path = os.path.join(args['loss_save_to'], args['model'])
    os.makedirs(loss_path, exist_ok=True)
    save_path_loss = os.path.join(loss_path, f"loss_{number}.pkl")
    torch.save(train_loss, save_path_loss)