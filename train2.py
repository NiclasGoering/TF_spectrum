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

def create_layer_specific_optimizer(model: DeepNN, base_lr: float, weight_decay: float):
    """Create optimizer with layer-specific learning rates"""
    if model.mode not in ['spectral', 'mup_no_align']:
        return optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # For spectral and muP modes, create parameter groups with different learning rates
    layer_lrs = model.get_layer_learning_rates(base_lr)
    param_groups = []
    
    linear_layer_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            param_groups.append({
                'params': [param],
                'lr': layer_lrs[linear_layer_idx // 2],  # Because weight+bias for each layer
                'weight_decay': weight_decay
            })
            if 'bias' in name:
                linear_layer_idx += 1
    
    # Use Adam for muP mode, SGD for spectral mode
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
    gamma: float = 1.0
) -> Tuple[float, float, float, dict, Dict[int, nn.Module]]:
    """Train the neural network and save checkpoints at specified epochs."""
    
    # Get device from model and ensure everything is on that device.
    device = next(model.parameters()).device
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Create the optimizer after the model is on the proper device.
    optimizer = create_layer_specific_optimizer(model, lr, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_test_error = float('inf')
    checkpoint_models = {}
    
    # Evaluate initial training error.
    model.eval()
    with torch.no_grad():
        initial_train_pred = model(X_train)
        initial_test_pred = model(X_test)
        initial_train_error = torch.mean((initial_train_pred - y_train) ** 2).item()
        initial_test_error = torch.mean((initial_test_pred - y_test) ** 2).item()
        
        print("Initial predictions stats:")
        print(f"Train - mean: {torch.mean(initial_train_pred):.6f}, std: {torch.std(initial_train_pred):.6f}")
        print(f"Test  - mean: {torch.mean(initial_test_pred):.6f}, std: {torch.std(initial_test_pred):.6f}")
    
    error_history = {
        'train_errors': [],
        'test_errors': [],
        'epochs': []
    }
    
    checkpoint_epochs = sorted(checkpoint_epochs)
    next_checkpoint_idx = 0
    
    for epoch in range(epochs):
        model.train()
        # Mini-batch training loop.
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
        
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            train_pred = model(X_train)
            train_error = torch.mean((train_pred - y_train) ** 2).item()
            test_error = torch.mean((test_pred - y_test) ** 2).item()
            best_test_error = min(best_test_error, test_error)
            
            error_history['train_errors'].append(train_error)
            error_history['test_errors'].append(test_error)
            error_history['epochs'].append(epoch)
        
        # Save a checkpoint model if needed.
        if next_checkpoint_idx < len(checkpoint_epochs) and epoch == checkpoint_epochs[next_checkpoint_idx]:
            checkpoint_model = DeepNN(model.input_dim, model.hidden_size, model.depth, 
                                      mode=model.mode, gamma=model.gamma).to(device)
            checkpoint_model.load_state_dict(model.state_dict())
            checkpoint_models[epoch] = checkpoint_model
            
            checkpoint_path = os.path.join(
                results_dir, 
                f'experiment{experiment_num}',
                f'checkpoint_model_{model_prefix}_epoch{epoch}_{timestamp}_rank{rank}.pt'
            )
            torch.save(checkpoint_model.state_dict(), checkpoint_path)
            next_checkpoint_idx += 1
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Current Test Error: {test_error:.6f}, Best Test Error: {best_test_error:.6f}")
            print(f"Training Error: {train_error:.6f}")
    
    # Final evaluation on training data.
    model.eval()
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_train_error = torch.mean((final_train_pred - y_train) ** 2).item()
    
    return best_test_error, initial_train_error, final_train_error, error_history, checkpoint_models

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
