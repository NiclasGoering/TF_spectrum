#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import random
from functools import partial
import json
from datetime import datetime
import os
from mpi4py import MPI

from FFNN import DeepNN
from utils2 import save_dataset, save_results, save_model

# Ensure prints flush immediately
print = partial(print, flush=True)

def evaluate_error_in_batches(model: nn.Module, X: torch.Tensor, y: torch.Tensor, eval_batch_size: int = 1024) -> float:
    """
    Evaluate the model's mean squared error on the provided data in batches.
    We accumulate the squared error to avoid storing huge tensors.
    """
    model.eval()
    total_error = 0.0
    total_count = 0
    with torch.no_grad():
        for i in range(0, X.size(0), eval_batch_size):
            batch_X = X[i:i+eval_batch_size]
            batch_y = y[i:i+eval_batch_size]
            batch_preds = model(batch_X)
            batch_error = torch.sum((batch_preds - batch_y) ** 2).item()
            total_error += batch_error
            total_count += batch_X.size(0)
    return total_error / total_count

def create_layer_specific_optimizer(model: DeepNN, base_lr: float, weight_decay: float):
    """Create optimizer with layer-specific learning rates."""
    if model.mode not in ['spectral', 'mup_no_align']:
        return optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    layer_lrs = model.get_layer_learning_rates(base_lr)
    param_groups = []
    linear_layer_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            param_groups.append({
                'params': [param],
                'lr': layer_lrs[linear_layer_idx // 2],
                'weight_decay': weight_decay
            })
            if 'bias' in name:
                linear_layer_idx += 1
    if model.mode == 'mup_no_align':
        return optim.Adam(param_groups, lr=base_lr)
    else:
        return optim.SGD(param_groups, lr=base_lr)

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
    results_dir: str,
    timestamp: str,
    rank: int,
    experiment_num: int,
    model_prefix: str,
    gamma: float = 1.0,
    eval_interval: int = 10,       # Evaluation performed every 10 epochs
    eval_print_interval: int = 100, # Print evaluation every 10 epochs
    eval_batch_size: int = 1024     # Batch size for evaluation to avoid OOM
) -> Tuple[float, float, float, dict, Dict[int, nn.Module]]:
    """
    Train the neural network and save checkpoints at specified epochs.
    Evaluates training error (in eval mode) on a fixed subset of the training data (max 20k examples).
    """
    # Ensure everything is on the proper device.
    device = next(model.parameters()).device
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    optimizer = create_layer_specific_optimizer(model, lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Evaluate initial training error on a subset (max 20k)
    subset_size = min(20000, len(X_train))
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    initial_train_error = evaluate_error_in_batches(model, X_train_subset, y_train_subset, eval_batch_size)
    initial_test_error  = evaluate_error_in_batches(model, X_test, y_test, eval_batch_size)
    
    print("Initial Errors:")
    print(f"   Train Error (eval subset of training data): {initial_train_error:.6f}")
    print(f"   Test Error: {initial_test_error:.6f}")
    
    error_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    
    best_test_error = initial_test_error
    checkpoint_epochs = sorted(checkpoint_epochs)
    next_checkpoint_idx = 0
    
    for epoch in range(epochs):
        model.train()
        # Standard training loop
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = torch.mean((output - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Every eval_interval epochs, compute the training error in eval mode on a fixed subset (max 20k)
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            model.eval()
            subset_size = min(20000, len(X_train))
            X_train_subset = X_train[:subset_size]
            y_train_subset = y_train[:subset_size]
            eval_train_error = evaluate_error_in_batches(model, X_train_subset, y_train_subset, eval_batch_size)
            test_error = evaluate_error_in_batches(model, X_test, y_test, eval_batch_size)
            best_test_error = min(best_test_error, test_error)
            
            error_history['train_errors'].append(eval_train_error)
            error_history['test_errors'].append(test_error)
            error_history['epochs'].append(epoch)
            
            if epoch % eval_print_interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}:")
                print(f"   Training Error (eval subset): {eval_train_error:.6f}")
                print(f"   Test Error: {test_error:.6f} (Best so far: {best_test_error:.6f})")
        
        # Save checkpoint if needed.
        if next_checkpoint_idx < len(checkpoint_epochs) and epoch == checkpoint_epochs[next_checkpoint_idx]:
            checkpoint_model = DeepNN(model.input_dim, model.hidden_size, model.depth, 
                                      mode=model.mode, gamma=model.gamma).to(device)
            checkpoint_model.load_state_dict(model.state_dict())
            checkpoint_path = os.path.join(
                results_dir, 
                f'experiment{experiment_num}',
                f'checkpoint_model_{model_prefix}_epoch{epoch}_{timestamp}_rank{rank}.pt'
            )
            torch.save(checkpoint_model.state_dict(), checkpoint_path)
            next_checkpoint_idx += 1
    
    # Final evaluation on the same training subset
    model.eval()
    subset_size = min(20000, len(X_train))
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    final_train_error = evaluate_error_in_batches(model, X_train_subset, y_train_subset, eval_batch_size)
    final_test_error  = evaluate_error_in_batches(model, X_test, y_test, eval_batch_size)
    
    return best_test_error, initial_train_error, final_train_error, error_history, {}

def shuffle_labels(y_train: torch.Tensor, seed: int = None) -> torch.Tensor:
    """Shuffle the training labels randomly."""
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(y_train.size(0))
    return y_train[perm]

def get_parameter_combinations(hidden_sizes, depths, n_train_sizes, learning_rates, gammas):
    """Generate all possible hyperparameter combinations."""
    combinations = []
    if not isinstance(gammas, (list, tuple)):
        gammas = [gammas]
        
    for hidden_size in hidden_sizes:
        for depth in depths:
            for n_train in n_train_sizes:
                for lr in learning_rates:
                    for gamma in gammas:
                        combinations.append({
                            'hidden_size': hidden_size,
                            'depth': depth,
                            'n_train': n_train,
                            'lr': lr,
                            'gamma': float(gamma)
                        })
    return combinations
