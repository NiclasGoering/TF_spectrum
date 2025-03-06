#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import os
from functools import partial
from datetime import datetime
import types
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
# Add near the top where other CUDA settings are
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere+ GPUs
# In both files, add:
from utils2 import GPUTensorDataset

from FFNN import DeepNN

print = partial(print, flush=True)

# IMPORTANT: Override pin_memory behavior to prevent errors
# Create a safe wrapper for TensorDataset to ensure tensors are on CPU
class SafeTensorDataset(TensorDataset):
    def __init__(self, *tensors):
        # Ensure all tensors are on CPU before creating the dataset
        cpu_tensors = []
        for tensor in tensors:
            if tensor.device.type != 'cpu':
                cpu_tensors.append(tensor.cpu())
            else:
                cpu_tensors.append(tensor)
        super().__init__(*cpu_tensors)



def evaluate_error_in_batches(
    model: nn.Module, 
    X: torch.Tensor, 
    y: torch.Tensor, 
    eval_batch_size: int = 16384
) -> float:
    """
    Evaluate mean squared error on the provided data in batches.
    Optimized for H100 GPUs.
    """
    model.eval()
    total_err = 0.0
    total_count = 0
    device = next(model.parameters()).device
    
    # Create a dataset and dataloader for evaluation
    X_eval = X.cpu() if X.device.type == 'cuda' else X
    y_eval = y.cpu() if y.device.type == 'cuda' else y
    
    eval_dataset = SafeTensorDataset(X_eval, y_eval)
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )
    
    with torch.no_grad():
        # Use autocast for mixed precision if on CUDA
        context_manager = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()
        with context_manager:
            for batch_X, batch_y in eval_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_preds = model(batch_X)
                total_err += torch.sum((batch_preds - batch_y) ** 2).item()
                total_count += batch_X.size(0)
    
    return total_err / total_count


def create_layer_specific_optimizer(
    model: DeepNN, 
    base_lr: float, 
    weight_decay: float
):
    """
    Create an Adam optimizer with layer-specific learning rates based on model mode.
    
    - For '_lr' suffix modes (standard_lr, ntk_lr, mup_lr), no scaling is applied
    - For regular modes, applies correct theoretical LR scaling with base width
    """
    layer_lrs = model.get_layer_learning_rates(base_lr)

    # Create standard optimizer with base learning rate
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Map parameter-names to the scaling factor from layer_lrs
    param_scale = {}
    linear_layer_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            # Scale factor is the ratio of layer's LR to base LR
            scale_factor = layer_lrs[linear_layer_idx // 2] / base_lr
            param_scale[name] = scale_factor
            if 'bias' in name:
                linear_layer_idx += 1

    def grad_scale_fn():
        # Apply layer-specific LR scaling by scaling gradients
        for name, param in model.named_parameters():
            if param.grad is not None and name in param_scale:
                param.grad.mul_(param_scale[name])
                
    return optimizer, grad_scale_fn



def train_and_evaluate(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
    epochs: int,
    checkpoint_epochs: List[int],
    lr: float,
    weight_decay: float,
    mode: str,
    alignment: bool = False,
    results_dir: str = "",
    timestamp: str = "",
    rank: int = 0,
    experiment_num: int = 0,
    model_prefix: str = "",
    base_width: int = 256,
    eval_interval: int = 250,
    eval_print_interval: int = 500,
    eval_batch_size: int = 16384,
    early_stop_threshold: float = 1e-4,
    fine_tuning_epochs: int = 500,  # Add fine tuning epochs parameter
):
    import math
    
    # Get device from model
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Force use of tensor cores for H100s
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    
    # Ensure all tensors are on the same device as the model
    assert X_train.device == device, "X_train must be on the same device as model"
    assert y_train.device == device, "y_train must be on the same device as model"
    assert X_test.device == device, "X_test must be on the same device as model"
    assert y_test.device == device, "y_test must be on the same device as model"
    
    # For tiny datasets, don't use DataLoader at all - just train directly
    use_direct_training = len(X_train) < 1000

    # Create optimized DataLoader for H100s if not using direct training
    if not use_direct_training:
        # GPU dataset - already on device
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        
        # Configure DataLoader for GPU data - critical changes
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(X_train)),
            shuffle=True,
            pin_memory=False,  # No pin_memory for GPU data
            num_workers=0,     # No workers needed for GPU data
            persistent_workers=False  # No persistent workers needed
        )
    
    # Initialize AMP grad scaler for mixed precision training with better settings
    scaler = torch.amp.GradScaler(
        'cuda',
        init_scale=2**16,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000
    ) if device.type == 'cuda' else None
    
    optimizer, grad_scale_fn = create_layer_specific_optimizer(model, lr, weight_decay)
    
    # Store initial learning rates for warmup
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
    
    # Calculate total training epochs including fine tuning
    main_epochs = epochs - fine_tuning_epochs
    
    # Add warmup period - scale based on dataset size
    warmup_epochs = min(100, main_epochs // 25)
    
    # Adjust cosine annealing cycle length based on dataset size
    if len(X_train) >= 10000:
        # Slower decay for larger datasets
        cycle_epochs = main_epochs - warmup_epochs
        # Use multiple cycles for larger datasets
        n_cycles = max(1, int(len(X_train) / 5000))
        cycle_length = cycle_epochs // n_cycles
    else:
        # Single cycle for smaller datasets
        cycle_length = main_epochs - warmup_epochs
    
    # Use cosine annealing LR with restarts for main training phase
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=cycle_length,
        T_mult=1,  # Keep same period for each restart
        eta_min=lr/50  # Don't let LR go all the way to zero
    )

    # Use fixed indices for evaluation subset - critical for consistency
    subset_size = min(20000, len(X_train))
    if subset_size < len(X_train):
        # Create fixed subset indices with a fixed seed
        torch.manual_seed(42)  # Fixed seed for reproducibility
        subset_indices = torch.randperm(len(X_train), device=device)[:subset_size]
        torch.set_rng_state(torch.get_rng_state())  # Restore RNG state
    else:
        subset_indices = torch.arange(subset_size, device=device)
    
    # Evaluate initial error
    def evaluate_error(model, X, y):
        model.eval()
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    output = model(X)
                    return torch.mean((output - y) ** 2).item()
            else:
                output = model(X)
                return torch.mean((output - y) ** 2).item()
    
    train_error_init = evaluate_error(model, X_train[subset_indices], y_train[subset_indices])
    test_error_init = evaluate_error(model, X_test, y_test)

    print(f"Initial Errors:")
    print(f"   Train Error (subset): {train_error_init:.6f}")
    print(f"   Test Error:           {test_error_init:.6f}")

    error_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    best_test_error = test_error_init
    best_train_error = train_error_init  # Track best train error for early stopping
    checkpoint_epochs = sorted(checkpoint_epochs)
    next_ckpt_idx = 0
    
    # Main training loop with mixed precision
    for epoch in range(main_epochs):
        model.train()
        
        # Apply warmup scaling to learning rate
        if epoch < warmup_epochs:
            warmup_factor = min(1.0, (epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
        
        # Direct training for tiny datasets - no DataLoader
        if use_direct_training:
            optimizer.zero_grad(set_to_none=True)
            
            if device.type == 'cuda':
                # Use mixed precision for forward pass
                with torch.amp.autocast('cuda'):
                    output = model(X_train)
                    loss = torch.mean((output - y_train) ** 2)
                
                # Scale loss and do backward pass
                scaler.scale(loss).backward()
                
                # Unscale the gradients
                scaler.unscale_(optimizer)
                
                # Clip gradients (after unscaling)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Apply custom gradient scaling
                grad_scale_fn()
                
                # Step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU path - no mixed precision
                output = model(X_train)
                loss = torch.mean((output - y_train) ** 2)
                loss.backward()
                grad_scale_fn()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            # Standard training with DataLoader for larger datasets
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                
                if device.type == 'cuda':
                    # Mixed precision training
                    with torch.amp.autocast('cuda'):
                        output = model(batch_X)
                        loss = torch.mean((output - batch_y) ** 2)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scale_fn()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(batch_X)
                    loss = torch.mean((output - batch_y) ** 2)
                    loss.backward()
                    grad_scale_fn()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        
        # Step the LR scheduler - only step if past warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        # Periodically evaluate with the same consistent subset
        if epoch % eval_interval == 0 or epoch == main_epochs - 1:
            model.eval()
            train_error = evaluate_error(model, X_train[subset_indices], y_train[subset_indices])
            test_error = evaluate_error(model, X_test, y_test)
            best_test_error = min(best_test_error, test_error)
            best_train_error = min(best_train_error, train_error)

            error_history['train_errors'].append(train_error)
            error_history['test_errors'].append(test_error)
            error_history['epochs'].append(epoch)

            if epoch % eval_print_interval == 0 or epoch == main_epochs - 1:
                print(f"Epoch {epoch}:")
                print(f"   Training Error (subset): {train_error:.6f}")
                print(f"   Test Error:              {test_error:.6f} (Best: {best_test_error:.6f})")
            
            # Early stopping check
            if train_error < early_stop_threshold:
                print(f"[Early stopping] Training error {train_error:.6f} below threshold {early_stop_threshold}")
                # Save early stopping information
                error_history['early_stopped'] = True
                error_history['stopped_epoch'] = epoch
                
                # Skip to final evaluation
                break

        # Save checkpoint if specified
        if (next_ckpt_idx < len(checkpoint_epochs) 
            and epoch == checkpoint_epochs[next_ckpt_idx]):
            checkpoint_model = DeepNN(
                model.input_dim,
                model.hidden_size,
                model.depth,
                mode=model.mode,
                alignment=model.alignment,
                base_width=model.base_width,
                embed_lr_scale=model.embed_lr_scale,
                hidden_lr_scale=model.hidden_lr_scale,
                readout_lr_scale=model.readout_lr_scale,
                gamma=model.gamma
            ).to(device)
            checkpoint_model.load_state_dict(model.state_dict())

            ckpt_path = os.path.join(
                results_dir,
                f'experiment{experiment_num}',
                f'checkpoint_model_{model_prefix}_epoch{epoch}_{timestamp}_rank{rank}.pt'
            )
            torch.save(checkpoint_model.state_dict(), ckpt_path)
            next_ckpt_idx += 1
    
    # Add fine-tuning phase with full precision for last epochs
    if not (epoch < main_epochs and train_error < early_stop_threshold):  # Only if we didn't early stop
        print(f"Starting fine-tuning phase with full precision for {fine_tuning_epochs} epochs")
        
        # Calculate last training error for reference
        last_train_error = train_error
        
        # Create a fine-tuning scheduler with gradual decay - more conservative
        # Start from a reasonable value (half the original) rather than drastically reducing
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / 2  # Start at half the base rate
            
        ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=fine_tuning_epochs,
            eta_min=lr/20  # Lower bound, but not too small
        )
        
        # Fine-tuning with full precision
        for ft_epoch in range(fine_tuning_epochs):
            epoch = main_epochs + ft_epoch
            model.train()
            
            # Training loop with full precision (no mixed precision)
            if use_direct_training:
                optimizer.zero_grad(set_to_none=True)
                # Full precision training
                output = model(X_train)
                loss = torch.mean((output - y_train) ** 2)
                loss.backward()
                grad_scale_fn()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    # Full precision training
                    output = model(batch_X)
                    loss = torch.mean((output - batch_y) ** 2)
                    loss.backward()
                    grad_scale_fn()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
            # Step the fine-tuning scheduler
            ft_scheduler.step()
            
            # Evaluate at regular intervals during fine-tuning - more frequent evaluation
            if ft_epoch % (eval_interval // 2) == 0 or ft_epoch == fine_tuning_epochs - 1:
                model.eval()
                train_error = evaluate_error(model, X_train[subset_indices], y_train[subset_indices])
                test_error = evaluate_error(model, X_test, y_test)
                best_test_error = min(best_test_error, test_error)
                best_train_error = min(best_train_error, train_error)

                error_history['train_errors'].append(train_error)
                error_history['test_errors'].append(test_error)
                error_history['epochs'].append(epoch)

                # More detailed output during fine-tuning
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Fine-tuning Epoch {ft_epoch}/{fine_tuning_epochs} (Total {epoch}):")
                print(f"   Training Error (subset): {train_error:.8f}")
                print(f"   Test Error:              {test_error:.8f} (Best: {best_test_error:.8f})")
                print(f"   Learning Rate:           {current_lr:.8f}")
                
                # Calculate and print improvement from fine-tuning
                improvement = last_train_error - train_error
                improvement_pct = (improvement / last_train_error) * 100 if last_train_error > 0 else 0
                print(f"   Improvement:             {improvement:.8f} ({improvement_pct:.2f}%)")
                
                # Check if we've reached desired precision - stricter threshold
                if train_error < early_stop_threshold / 10:  # Use stricter threshold for fine-tuning
                    print(f"[Fine-tuning complete] Training error {train_error:.8f} reached desired precision")
                    break
                
                # If we're not making sufficient progress, adjust learning rate dynamically
                if ft_epoch > 0 and ft_epoch % 100 == 0:
                    if improvement < train_error * 0.01:  # Less than 1% improvement
                        # Temporarily boost learning rate to escape potential local minimum
                        print(f"   Progress slow, temporarily increasing learning rate")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 2
    
    # Add early stopping info to history if we completed all epochs
    if not error_history.get('early_stopped', False):
        error_history['early_stopped'] = False
        error_history['stopped_epoch'] = epoch

    # Final evaluation
    model.eval()
    final_train_error = evaluate_error(model, X_train[subset_indices], y_train[subset_indices])
    final_test_error = evaluate_error(model, X_test, y_test)

    print(f"Final Results:")
    print(f"   Initial Train Error:   {train_error_init:.8f}")
    print(f"   Final Train Error:     {final_train_error:.8f}")
    print(f"   Test Error:            {final_test_error:.8f}")
    print(f"   Best Test Error:       {best_test_error:.8f}")

    return best_test_error, train_error_init, final_train_error, error_history, {}




def shuffle_labels(y_train: torch.Tensor, seed: int = None) -> torch.Tensor:
    """Shuffle the training labels randomly."""
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(y_train.size(0))
    return y_train[perm]

def get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates, gammas, alignments=[False, True]):
    """Generate all possible hyperparameter combinations."""
    combinations = []
    if not isinstance(gammas, (list, tuple)):
        gammas = [gammas]

    for hidden_size in hidden_sizes:
        for depth in depths:
            for n_train in n_train_sizes:
                for lr in learning_rates:
                    for gamma in gammas:
                        for alignment in alignments:  # Added alignment loop
                            combinations.append({
                                'hidden_size': hidden_size,
                                'depth': depth,
                                'n_train': n_train,
                                'lr': lr,
                                'gamma': float(gamma),
                                'alignment': alignment  # Added alignment parameter
                            })
    return combinations