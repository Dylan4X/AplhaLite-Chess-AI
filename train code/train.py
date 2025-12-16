import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

# FIX: Imports adapting to new structure and naming
from CCRLDataset import CCRLDataset
from AlphaLite.AlphaLite import AlphaLite  # Renamed from AlphaZeroNet

# --- Training Configuration (Optimized for RTX 4090) ---
CONFIG = {
    'num_epochs': 50,
    'num_blocks': 20,  # ResNet-20 architecture
    'num_filters': 256,  # 256 filters
    'batch_size': 8192,  # Optimized for 24GB VRAM
    'lr': 0.002,  # Adjusted for large batch size
    'weight_decay': 1e-4,
    'num_workers': 16,  # Adjust based on CPU cores

    # Paths (Modify these to your actual data paths)
    'train_dir': '/root/autodl-tmp/train_dataset',
    'val_dir': '/root/autodl-tmp/test_dataset',
    'model_name': 'AlphaLite_4090'  # Renamed model prefix
}


def train():
    # 1. Hardware Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"🚀 Training on: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("NVIDIA GPU is required for this training script.")

    # 2. Dataset Loading
    print("Loading datasets...")
    train_ds = CCRLDataset(CONFIG['train_dir'])
    val_ds = CCRLDataset(CONFIG['val_dir'])

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True
    )

    # 3. Model Initialization (Using AlphaLite class)
    model = AlphaLite(CONFIG['num_blocks'], CONFIG['num_filters']).to(device)

    # Optional: Compile model for speed (PyTorch 2.0+)
    print("Compiling model...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Warning: torch.compile skipped: {e}")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Mixed Precision Type
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using Mixed Precision: {amp_dtype}")

    # 4. Training Loop
    history = {
        'epoch': [],
        'train_loss': [], 'train_value_loss': [], 'train_policy_loss': [],
        'val_loss': [], 'val_value_loss': [], 'val_policy_loss': []
    }
    best_val_loss = float('inf')

    for epoch in range(CONFIG['num_epochs']):
        # --- Train Phase ---
        model.train()
        t_loss, t_v_loss, t_p_loss = 0, 0, 0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1} [Train]", dynamic_ncols=True)

        for data in pbar:
            pos = data['position'].to(device, non_blocking=True)
            val_tgt = data['value'].to(device, non_blocking=True)
            pol_tgt = data['policy'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                # AlphaLite returns separate losses in training mode
                v_loss, p_loss = model(pos, valueTarget=val_tgt, policyTarget=pol_tgt)
                loss = v_loss + p_loss

            # Scaler is not strictly needed for bfloat16 but good practice generally
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_v_loss += v_loss.item()
            t_p_loss += p_loss.item()
            steps += 1

            pbar.set_postfix({'L': f"{loss.item():.4f}"})

        scheduler.step()

        # --- Validation Phase ---
        model.eval()
        v_loss_sum, v_v_loss, v_p_loss = 0, 0, 0
        v_steps = 0

        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Ep {epoch + 1} [Val]", dynamic_ncols=True):
                pos = data['position'].to(device, non_blocking=True)
                val_tgt = data['value'].to(device, non_blocking=True)
                pol_tgt = data['policy'].to(device, non_blocking=True).view(-1)
                mask = data['mask'].to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    # In eval mode, AlphaLite returns (value, policy_softmax)
                    pred_val, pred_pol = model(pos, policyMask=mask)

                    batch_v_loss = F.mse_loss(pred_val, val_tgt)
                    batch_p_loss = F.nll_loss(torch.log(pred_pol + 1e-8), pol_tgt)
                    total_loss = batch_v_loss + batch_p_loss

                v_loss_sum += total_loss.item()
                v_v_loss += batch_v_loss.item()
                v_p_loss += batch_p_loss.item()
                v_steps += 1

        # --- Logging ---
        avg_metrics = {
            'epoch': epoch + 1,
            'train_loss': t_loss / steps if steps else 0,
            'train_value_loss': t_v_loss / steps if steps else 0,
            'train_policy_loss': t_p_loss / steps if steps else 0,
            'val_loss': v_loss_sum / v_steps if v_steps else 0,
            'val_value_loss': v_v_loss / v_steps if v_steps else 0,
            'val_policy_loss': v_p_loss / v_steps if v_steps else 0
        }

        print(
            f"Summary Ep {epoch + 1}: Train Loss: {avg_metrics['train_loss']:.4f} | Val Loss: {avg_metrics['val_loss']:.4f}")

        for k, v in avg_metrics.items():
            history[k].append(v)

        # Save logs and checkpoints
        pd.DataFrame(history).to_csv(f"{CONFIG['model_name']}_log.csv", index=False)

        if avg_metrics['val_loss'] < best_val_loss:
            best_val_loss = avg_metrics['val_loss']
            torch.save(model.state_dict(), f"{CONFIG['model_name']}_best.pt")
            print(">>> Best Model Saved!")

        torch.save(model.state_dict(), f"{CONFIG['model_name']}_last.pt")


if __name__ == '__main__':
    train()