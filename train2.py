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

from FFNN import DeepNN

print = partial(print, flush=True)

def evaluate_error_in_batches(
    model: nn.Module, 
    X: torch.Tensor, 
    y: torch.Tensor, 
    eval_batch_size: int = 1024
) -> float:
    """
    Evaluate mean squared error on the provided data in batches.
    """
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for i in range(0, X.size(0), eval_batch_size):
            batch_X = X[i:i+eval_batch_size]
            batch_y = y[i:i+eval_batch_size]
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
    alignment: bool = False,  # Added alignment parameter
    results_dir: str = "",
    timestamp: str = "",
    rank: int = 0,
    experiment_num: int = 0,
    model_prefix: str = "",
    base_width: int = 256,
    eval_interval: int = 10,
    eval_print_interval: int = 100,
    eval_batch_size: int = 1024
) -> Tuple[float, float, float, dict, Dict[int, nn.Module]]:
    """
    Train the model with Adam + layer-specific LR scaling (depending on mode and alignment).
    """
    device = next(model.parameters()).device
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    optimizer, grad_scale_fn = create_layer_specific_optimizer(model, lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Evaluate initial errors
    subset_size = min(20000, len(X_train))
    train_error_init = evaluate_error_in_batches(model, X_train[:subset_size], y_train[:subset_size], eval_batch_size)
    test_error_init  = evaluate_error_in_batches(model, X_test, y_test, eval_batch_size)

    print(f"Initial Errors:")
    print(f"   Train Error (subset): {train_error_init:.6f}")
    print(f"   Test Error:           {test_error_init:.6f}")

    error_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    best_test_error = test_error_init
    checkpoint_epochs = sorted(checkpoint_epochs)
    next_ckpt_idx = 0
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            output = model(batch_X)
            loss = torch.mean((output - batch_y) ** 2)
            loss.backward()

            # Scale grads for each layer (if relevant):
            grad_scale_fn()

            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
        
        scheduler.step()

        # Periodically evaluate
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            model.eval()
            train_error = evaluate_error_in_batches(
                model, X_train[:subset_size], y_train[:subset_size], eval_batch_size
            )
            test_error = evaluate_error_in_batches(model, X_test, y_test, eval_batch_size)
            best_test_error = min(best_test_error, test_error)

            error_history['train_errors'].append(train_error)
            error_history['test_errors'].append(test_error)
            error_history['epochs'].append(epoch)

            if epoch % eval_print_interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}:")
                print(f"   Training Error (subset): {train_error:.6f}")
                print(f"   Test Error:              {test_error:.6f} (Best: {best_test_error:.6f})")

        # Save checkpoint if specified
        if (next_ckpt_idx < len(checkpoint_epochs) 
            and epoch == checkpoint_epochs[next_ckpt_idx]):
            checkpoint_model = DeepNN(
                model.input_dim,
                model.hidden_size,
                model.depth,
                mode=model.mode,
                alignment=model.alignment,  # Added alignment parameter
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

    # Final evaluation
    model.eval()
    final_train_error = evaluate_error_in_batches(
        model, X_train[:subset_size], y_train[:subset_size], eval_batch_size
    )
    final_test_error  = evaluate_error_in_batches(model, X_test, y_test, eval_batch_size)

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