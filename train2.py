#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Any, Optional
import os
from functools import partial
from datetime import datetime
import types
from torch.utils.data import TensorDataset, DataLoader

# Ensure prints flush immediately
print = partial(print, flush=True)

# Enhanced GPU dataset that stays on the device
class GPUTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)

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
    
    # Create a dataset directly from GPU tensors
    eval_dataset = GPUTensorDataset(X, y)
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=False,  # No pin_memory needed for GPU data
        num_workers=0      # No workers needed for GPU data
    )
    
    with torch.no_grad():
        # Use autocast for mixed precision if on CUDA
        amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', 
                                dtype=amp_dtype, enabled=device.type == 'cuda'):
            for batch_X, batch_y in eval_loader:
                batch_preds = model(batch_X)
                total_err += torch.sum((batch_preds - batch_y) ** 2).item()
                total_count += batch_X.size(0)
    
    return total_err / total_count


def create_layer_specific_optimizer(
    model: nn.Module, 
    base_lr: float, 
    weight_decay: float
):
    """
    Create an Adam optimizer with layer-specific learning rates based on model mode.
    
    - For '_lr' suffix modes (standard_lr, ntk_lr, mup_lr), no scaling is applied
    - For regular modes, applies correct theoretical LR scaling with base width
    """
    # Check if model has the get_layer_learning_rates method
    if hasattr(model, 'get_layer_learning_rates'):
        layer_lrs = model.get_layer_learning_rates(base_lr)
    else:
        # Fallback to uniform learning rate
        layer_lrs = None

    # Create standard optimizer with base learning rate
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # Store initial learning rates for warmup
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']

    # If layer-specific learning rates are available, set up gradient scaling
    if layer_lrs is not None:
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
    else:
        # No gradient scaling needed
        def grad_scale_fn():
            pass
                
    return optimizer, grad_scale_fn


def optimized_train_and_evaluate(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    mode: str,
    alignment: bool = False,
    device = None,
    base_width: int = 256,
    gamma: float = 1.0,
    eval_interval: int = 1000,
    early_stop_threshold: float = 1e-4,
    fine_tuning_epochs: int = 500,
    original_batch_size: int = 32768  # Reference batch size for LR scaling
):
    """
    Optimized training function specifically for H100 GPUs.
    Uses BF16 mixed precision, larger batches, and less synchronization.
    
    Args:
        model: The neural network model
        X_train, y_train: Training data
        X_test, y_test: Test data
        batch_size: Batch size (should be large for H100s)
        epochs: Total number of epochs
        lr: Learning rate
        weight_decay: Weight decay coefficient
        mode: Training mode (standard, ntk, etc.)
        alignment: Whether to use alignment
        device: Torch device
        base_width: Base width parameter
        gamma: Gamma parameter
        eval_interval: How often to evaluate (should be infrequent)
        early_stop_threshold: Threshold for early stopping
        fine_tuning_epochs: Number of epochs for fine-tuning phase
        original_batch_size: Reference batch size for LR scaling
    
    Returns:
        Tuple of (best_test_error, initial_train_error, final_train_error, error_history)
    """
    import math
    
    # Get device from model if not provided
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    
    # Force use of tensor cores and optimized settings for H100s
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        # Empty cache and create CUDA stream for computation
        torch.cuda.empty_cache()
        compute_stream = torch.cuda.Stream(device=device)
    else:
        compute_stream = None
    
    # Ensure all tensors are on the same device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Dynamically adjust batch size based on dataset size
    # For H100s, we want very large batches when possible
    if len(X_train) < 1000:
        actual_batch_size = len(X_train)  # Use entire dataset for tiny experiments
    elif len(X_train) < 10000:
        actual_batch_size = min(batch_size // 2, len(X_train))
    else:
        # H100s can handle enormous batches
        actual_batch_size = min(batch_size, len(X_train))
    
    # Scale learning rate based on batch size (square root scaling)
    # lr_new = lr_base * sqrt(batch_size_new / batch_size_base)
    batch_size_ratio = actual_batch_size / original_batch_size
    scaled_lr = lr * (batch_size_ratio ** 0.5)
    print(f"Original LR: {lr:.6f}, Scaled LR: {scaled_lr:.6f} (batch size ratio: {batch_size_ratio:.2f})")
    lr = scaled_lr
    
    # For tiny datasets, don't use DataLoader at all
    use_direct_training = len(X_train) < 1000
    
    # Create optimized DataLoader for H100s if not using direct training
    if not use_direct_training:
        # GPU dataset - use optimized loader settings
        train_dataset = GPUTensorDataset(X_train, y_train)
        dataloader_kwargs = {
            'batch_size': actual_batch_size,
            'shuffle': True,
            'pin_memory': False,  # Data already on GPU
            'num_workers': 0,     # No workers needed for GPU data
            'persistent_workers': False
        }
        
        train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    
    # Use BF16 mixed precision for H100s
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    
    # Initialize AMP grad scaler with optimized settings
    scaler = torch.amp.GradScaler(
        enabled=device.type == 'cuda',
        init_scale=2**16,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000
    )
    
    # Create optimizer with layer-specific learning rates
    optimizer, grad_scale_fn = create_layer_specific_optimizer(model, lr, weight_decay)
    
    # Calculate main training epochs (excluding fine-tuning)
    main_epochs = epochs - fine_tuning_epochs
    
    # Add warmup period - smaller for H100s since they converge faster
    warmup_epochs = min(50, main_epochs // 50)
    
    # Use cosine annealing LR for main training phase
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=main_epochs - warmup_epochs,
        eta_min=lr/20  # Don't let LR go all the way to zero
    )
    
    # Use subset for evaluation to reduce overhead
    # Fixed subset for consistency (critical for monitoring convergence)
    subset_size = min(10000, len(X_train))
    torch.manual_seed(42)  # Fixed seed
    subset_indices = torch.randperm(len(X_train), device=device)[:subset_size]
    
    # Function to evaluate model with minimal synchronization
    def evaluate_quickly(model, X, y):
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', 
                                   dtype=amp_dtype, enabled=device.type == 'cuda'):
                output = model(X)
                # Keep result on GPU 
                return ((output - y) ** 2).mean()
    
    # Evaluate initial error
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.cuda.stream(compute_stream):
                train_error_init = evaluate_quickly(model, X_train[subset_indices], y_train[subset_indices])
                test_error_init = evaluate_quickly(model, X_test, y_test)
            # Synchronize to get values
            torch.cuda.current_stream().wait_stream(compute_stream)
            train_error_init = train_error_init.item()
            test_error_init = test_error_init.item()
        else:
            train_error_init = evaluate_quickly(model, X_train[subset_indices], y_train[subset_indices]).item()
            test_error_init = evaluate_quickly(model, X_test, y_test).item()

    print(f"Initial Errors: Train Error: {train_error_init:.6f}, Test Error: {test_error_init:.6f}")

    # Store error history
    error_history = {
        'train_errors': [train_error_init],
        'test_errors': [test_error_init],
        'epochs': [0]
    }
    
    best_test_error = test_error_init
    best_train_error = train_error_init
    
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
            if device.type == 'cuda':
                with torch.cuda.stream(compute_stream):
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Use mixed precision
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                        output = model(X_train)
                        loss = torch.mean((output - y_train) ** 2)
                    
                    # Scale loss and do backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scale_fn()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # CPU path - no mixed precision
                optimizer.zero_grad(set_to_none=True)
                output = model(X_train)
                loss = torch.mean((output - y_train) ** 2)
                loss.backward()
                grad_scale_fn()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            # Standard training with DataLoader for larger datasets
            total_loss = 0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed precision training 
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', 
                                       dtype=amp_dtype, enabled=device.type == 'cuda'):
                    output = model(batch_X)
                    loss = torch.mean((output - batch_y) ** 2)
                    total_loss += loss.item()
                    num_batches += 1
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scale_fn()
                scaler.step(optimizer)
                scaler.update()
        
        # Step the LR scheduler - only step if past warmup
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Drastically reduce evaluation frequency
        # Only evaluate at specific intervals or final epoch
        if epoch % eval_interval == 0 or epoch == main_epochs - 1:
            with torch.no_grad():
                model.eval()
                
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                        train_error = evaluate_quickly(model, X_train[subset_indices], y_train[subset_indices]).item()
                        test_error = evaluate_quickly(model, X_test, y_test).item()
                else:
                    train_error = evaluate_quickly(model, X_train[subset_indices], y_train[subset_indices]).item()
                    test_error = evaluate_quickly(model, X_test, y_test).item()
                
                best_test_error = min(best_test_error, test_error)
                best_train_error = min(best_train_error, train_error)
                
                error_history['train_errors'].append(train_error)
                error_history['test_errors'].append(test_error)
                error_history['epochs'].append(epoch)
                
                print(f"Epoch {epoch}: Train Error: {train_error:.6f}, Test Error: {test_error:.6f}")
                
                # Early stopping check
                if train_error < early_stop_threshold:
                    print(f"Early stopping at epoch {epoch}, train error: {train_error:.6f}")
                    error_history['early_stopped'] = True
                    error_history['stopped_epoch'] = epoch
                    break
    
    # Only do fine-tuning if we haven't reached a very small error already
    do_fine_tuning = not (epoch < main_epochs and train_error < early_stop_threshold)
    
    # Add fine-tuning phase if needed
    if do_fine_tuning and fine_tuning_epochs > 0:
        print(f"Starting fine-tuning phase with {fine_tuning_epochs} epochs")
        
        # Reset learning rate for fine-tuning - more conservative
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / 5  # Much lower LR for fine-tuning
        
        # Fine-tuning with minimal evaluation
        for ft_epoch in range(fine_tuning_epochs):
            model.train()
            
            # Similar logic but with fewer evaluations
            if use_direct_training:
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                        optimizer.zero_grad(set_to_none=True)
                        output = model(X_train)
                        loss = torch.mean((output - y_train) ** 2)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        grad_scale_fn()
                        optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    output = model(X_train)
                    loss = torch.mean((output - y_train) ** 2)
                    loss.backward()
                    grad_scale_fn()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                total_loss = 0
                num_batches = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    
                    with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', 
                                           dtype=amp_dtype, enabled=device.type == 'cuda'):
                        output = model(batch_X)
                        loss = torch.mean((output - batch_y) ** 2)
                        total_loss += loss.item()
                        num_batches += 1
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scale_fn()
                    optimizer.step()
            
            # Evaluate much less frequently during fine-tuning
            if ft_epoch == fine_tuning_epochs - 1 or ft_epoch % (fine_tuning_epochs // 2) == 0:
                with torch.no_grad():
                    model.eval()
                    
                    if device.type == 'cuda':
                        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                            train_error = evaluate_quickly(model, X_train[subset_indices], y_train[subset_indices]).item()
                            test_error = evaluate_quickly(model, X_test, y_test).item()
                    else:
                        train_error = evaluate_quickly(model, X_train[subset_indices], y_train[subset_indices]).item()
                        test_error = evaluate_quickly(model, X_test, y_test).item()
                    
                    best_test_error = min(best_test_error, test_error)
                    best_train_error = min(best_train_error, train_error)
                    
                    error_history['train_errors'].append(train_error)
                    error_history['test_errors'].append(test_error)
                    error_history['epochs'].append(main_epochs + ft_epoch)
                    
                    print(f"Fine-tuning epoch {ft_epoch}: Train Error: {train_error:.6f}, Test Error: {test_error:.6f}")
    
    # Final evaluation
    with torch.no_grad():
        model.eval()
        
        if device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                final_train_error = evaluate_quickly(model, X_train[subset_indices], y_train[subset_indices]).item()
                final_test_error = evaluate_quickly(model, X_test, y_test).item()
        else:
            final_train_error = evaluate_quickly(model, X_train[subset_indices], y_train[subset_indices]).item()
            final_test_error = evaluate_quickly(model, X_test, y_test).item()
    
    print(f"Final Results: Initial Train Error: {train_error_init:.6f}, " +
          f"Final Train Error: {final_train_error:.6f}, Test Error: {final_test_error:.6f}")
    
    # Add early stopping info to history if missing
    if 'early_stopped' not in error_history:
        error_history['early_stopped'] = False
        error_history['stopped_epoch'] = main_epochs + fine_tuning_epochs - 1

    return best_test_error, train_error_init, final_train_error, error_history


def shuffle_labels(y_train: torch.Tensor, seed: int = None) -> torch.Tensor:
    """Shuffle the training labels randomly."""
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(y_train.size(0), device=y_train.device)
    return y_train[perm]