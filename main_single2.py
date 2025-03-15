#!/usr/bin/env python3
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
import sys
import glob
import numpy as np
import json
import yaml
import traceback
from datetime import datetime
from functools import partial
import torch.cuda.amp as amp
from typing import List, Dict, Tuple, Any, Optional
import time

# Import your model and helper functions
from FFNN import DeepNN
from utils2 import save_dataset, save_results, save_model

# Ensure prints flush immediately
print = partial(print, flush=True)

# PERFORMANCE CONFIGURATION
# These settings are optimized for maximum throughput on H100s
MAX_PARALLEL_TINY = 16   # For n_train < 1000
MAX_PARALLEL_SMALL = 8   # For 1000 <= n_train < 10000
MAX_PARALLEL_MEDIUM = 4  # For 10000 <= n_train < 100000
BATCH_SIZE_TINY = 1024   # Batch size for tiny experiments
BATCH_SIZE_SMALL = 4096  # Batch size for small experiments
BATCH_SIZE_MEDIUM = 8192 # Batch size for medium experiments
BATCH_SIZE_LARGE = 65536 # Batch size for large experiments
ORIG_BATCH_SIZE = 32768  # Original batch size reference for LR scaling

def load_yaml_config(config_path):
    """Load and return the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_info_from_path(path):
    """Extract parameters from path."""
    path_lower = path.lower()
    path_upper = path.upper()  # Add this line to also check uppercase
    dist_type = None

    # Check for abbreviated identifiers first (NE, NL, NP)
    if 'NE' in path_upper:
        dist_type = "NE"
    elif 'NL' in path_upper:
        dist_type = "NL"
    elif 'NP' in path_upper:
        dist_type = "NP"
    # Fall back to original checks
    elif 'poly' in path_lower:
        dist_type = "NP"
    elif 'lin' in path_lower:
        dist_type = "NL"
    elif 'exp' in path_lower:
        dist_type = "NE"
    else:
        dist_type = "NP"  # Default to "NP"

    basename = os.path.basename(path)
    parts = basename.split('_')

    info = {
        "distribution_type": dist_type
    }

    # Extract order number
    order_num = None
    for i, part in enumerate(parts):
        if part == "O" and i < len(parts) - 1 and parts[i+1].isdigit():
            order_num = parts[i+1]
            break
        if part.startswith('O_') and part[2:].isdigit():
            order_num = part[2:]
            break
    
    if order_num is not None:
        info['order_num'] = order_num

    # Rest of the function remains the same
    for i, part in enumerate(parts):
        if part.startswith('d') and part[1:].isdigit():
            info['input_dim'] = int(part[1:])
        if part.startswith('H') and part[1:].isdigit():
            info['hidden_size'] = int(part[1:])
        if part.startswith('D') and part[1:].isdigit():
            info['depth'] = int(part[1:])
        if part.startswith('a') and i < len(parts) - 1:
            try:
                info['alpha'] = float(part[1:])
            except ValueError:
                pass
    
    return info

def generate_unique_id(config):
    """Generate a unique identifier for this configuration."""
    ds_name = config['ds_name']
    ds_path = config['ds_path']
    base_path = os.path.basename(ds_path)
    
    # Get order number from the dataset parameters
    order_num = None
    if 'order_num' in config:
        order_num = config['order_num']
    else:
        # Extract from path - look for O_X pattern
        path_parts = base_path.split('_')
        for i, part in enumerate(path_parts):
            if part == "O" and i < len(path_parts) - 1 and path_parts[i+1].isdigit():
                order_num = path_parts[i+1]
                break
            if part.startswith('O_') and part[2:].isdigit():
                order_num = part[2:]
                break
    
    # Default to "1" if no order number found
    if not order_num:
        order_num = "1"
    
    align_suffix = "_align" if config['alignment'] else ""
    
    unique_id = (
        f"{ds_name}_O{order_num}"  # Use O1, O2, etc. format
        f"_h{config['hidden_size']}"
        f"_d{config['depth']}"
        f"_n{config['n_train']}"
        f"_lr{config['lr']}"
        f"_mode{config['mode']}"
        f"_exp{config['experiment_num']}"
        f"{align_suffix}"
    )
    
    return unique_id

def load_dataset_info(directory):
    """Load dataset info."""
    if not os.path.isdir(directory):
        return None
    
    dataset_files = glob.glob(os.path.join(directory, "dataset_*.pt"))
    if not dataset_files:
        return None
    
    dataset_path = dataset_files[0]
    result_files = glob.glob(os.path.join(directory, "results_*.json"))
    
    params = extract_info_from_path(directory)
    if not params:
        params = extract_info_from_path(dataset_path)
    
    # Set parameters
    dist_type = params.get("distribution_type", "")
    input_dim = params.get("input_dim", "")
    hidden_size = params.get("hidden_size", "")
    alpha = params.get("alpha", "")
    depth = params.get("depth", None)

    if dist_type and input_dim and hidden_size != "":
        ds_name = f"{dist_type}_d{input_dim}_H{hidden_size}_a{alpha}"
    else:
        ds_name = os.path.basename(dataset_path).replace("dataset_", "").replace(".pt", "")

    file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
    
    return {
        "path": dataset_path,
        "name": ds_name,
        "params": params,
        "directory": directory,
        "size_mb": file_size_mb
    }



def generate_all_combinations(config):
    """Generate all parameter combinations."""
    base_cfg = config["base_config"]
    sweeps = config["sweeps"]
    
    all_combinations = []
    
    for sweep_name, sweep_info in sweeps.items():
        dataset_paths = sweep_info.get("dataset_paths", [])
        sweep_params = sweep_info.get("parameters", {})
        
        dataset_infos = []
        for ds_path in dataset_paths:
            ds_info = load_dataset_info(ds_path)
            if ds_info:
                dataset_infos.append(ds_info)
        
        for ds_info in dataset_infos:
            ds_params = ds_info['params']
            input_dim = ds_params.get('input_dim')
            if not input_dim:
                continue
            
            # Pass the order number to the combinations
            order_num = ds_params.get('order_num')
            
            for n_train in sweep_params.get("n_train", [1024]):
                for lr in sweep_params.get("learning_rates", [0.001]):
                    for hidden_size in sweep_params.get("hidden_sizes", [256]):
                        for depth in sweep_params.get("depths", [1]):
                            for mode in sweep_params.get("modes", ["standard"]):
                                for alignment in sweep_params.get("alignment", [False]):
                                    for exp_num in range(1, base_cfg.get("num_experiments", 1) + 1):
                                        combo = {
                                            'ds_path': ds_info['path'],
                                            'ds_name': ds_info['name'],
                                            'ds_directory': ds_info['directory'],
                                            'hidden_size': hidden_size,
                                            'depth': depth,
                                            'input_dim': input_dim,
                                            'n_train': n_train,
                                            'lr': lr,
                                            'mode': mode,
                                            'gamma': ds_params.get('gamma', 1.0),
                                            'experiment_num': exp_num,
                                            'base_width': sweep_params.get('base_width', 10),
                                            'alignment': alignment,
                                            'sweep_name': sweep_name,
                                            'alpha': ds_params.get('alpha', 1.0),
                                            'size_mb': ds_info.get('size_mb', 0),
                                        }
                                        # Add order_num if available
                                        if order_num:
                                            combo['order_num'] = order_num
                                        all_combinations.append(combo)
    
    return all_combinations

def generate_unique_id(config):
    """Generate a unique identifier for this configuration."""
    ds_name = config['ds_name']
    ds_path = config['ds_path']
    base_path = os.path.basename(ds_path)
    
    order_num = "0"
    path_parts = base_path.split('_')
    for part in path_parts:
        if part.startswith("O_"):
            order_num = part.replace("O_", "")
            break
    
    align_suffix = "_align" if config['alignment'] else ""
    
    unique_id = (
        f"{ds_name}_O{order_num}"
        f"_h{config['hidden_size']}"
        f"_d{config['depth']}"
        f"_n{config['n_train']}"
        f"_lr{config['lr']}"
        f"_mode{config['mode']}"
        f"_exp{config['experiment_num']}"
        f"{align_suffix}"
    )
    
    return unique_id

def load_dataset_directly(ds_path, device):
    """Load dataset directly to GPU."""
    data = torch.load(ds_path, map_location='cpu')
    
    if isinstance(data, dict) and 'X' in data and 'y' in data:
        data['X'] = data['X'].to(device, non_blocking=True)
        data['y'] = data['y'].to(device, non_blocking=True)
    
    return data

def worker_process(gpu_id, num_gpus, all_combinations, config, full_results_dir, timestamp, checkpoint_log_path, completed_configs):
    """
    FASTEST IMPLEMENTATION: Aggressively parallel worker process.
    Runs maximum number of experiments in parallel based on size.
    """
    try:
        start_time = time.time()
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Enable H100 optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
        print(f"[GPU {gpu_id}] Worker started on device {device}")
        
        # Filter combinations for this GPU using modulo assignment
        worker_combinations = [combo for i, combo in enumerate(all_combinations) if i % num_gpus == gpu_id]
        print(f"[GPU {gpu_id}] Assigned {len(worker_combinations)} configurations")
        
        # Remove already completed configurations
        worker_combinations = [combo for combo in worker_combinations 
                              if generate_unique_id(combo) not in completed_configs]
        
        if not worker_combinations:
            print(f"[GPU {gpu_id}] All configurations already completed")
            return
            
        print(f"[GPU {gpu_id}] Processing {len(worker_combinations)} incomplete configurations")
        
        # Group by dataset for more efficient processing
        by_dataset = {}
        for combo in worker_combinations:
            ds_path = combo['ds_path']
            if ds_path not in by_dataset:
                by_dataset[ds_path] = []
            by_dataset[ds_path].append(combo)
        
        # Process each dataset group
        completed_count = 0
        total_to_process = len(worker_combinations)
        
        for ds_path, dataset_combos in by_dataset.items():
            # Skip empty dataset groups
            if not dataset_combos:
                continue
                
            print(f"[GPU {gpu_id}] Loading dataset: {ds_path}")
            try:
                # Load dataset once for all experiments
                data = load_dataset_directly(ds_path, device)
                
                if not (isinstance(data, dict) and 'X' in data and 'y' in data):
                    print(f"[GPU {gpu_id}] WARNING: Unknown dataset format, skipping {ds_path}")
                    continue
                
                X_full = data['X']
                y_full = data['y']
                
                # Group experiments by size for optimal batching
                tiny_exps = [c for c in dataset_combos if c['n_train'] < 1000]
                small_exps = [c for c in dataset_combos if 1000 <= c['n_train'] < 10000]
                medium_exps = [c for c in dataset_combos if 10000 <= c['n_train'] < 100000]
                large_exps = [c for c in dataset_combos if c['n_train'] >= 100000]
                
                # --- Process tiny experiments in large parallel batches ---
                if tiny_exps:
                    for i in range(0, len(tiny_exps), MAX_PARALLEL_TINY):
                        batch = tiny_exps[i:i+MAX_PARALLEL_TINY]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(tiny_exps)} tiny experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_TINY, 10, config["base_config"]["epochs"],  # Use full epochs
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # --- Process small experiments in parallel batches ---
                if small_exps:
                    for i in range(0, len(small_exps), MAX_PARALLEL_SMALL):
                        batch = small_exps[i:i+MAX_PARALLEL_SMALL]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(small_exps)} small experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_SMALL, 20, config["base_config"]["epochs"],  # Use full epochs
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # --- Process medium experiments in smaller parallel batches ---
                if medium_exps:
                    for i in range(0, len(medium_exps), MAX_PARALLEL_MEDIUM):
                        batch = medium_exps[i:i+MAX_PARALLEL_MEDIUM]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(medium_exps)} medium experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_MEDIUM, 30, config["base_config"]["epochs"],  # Use full epochs
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # --- Process large experiments with higher batch sizes ---
                if large_exps:
                    for exp in large_exps:
                        print(f"[GPU {gpu_id}] Processing large experiment with n_train={exp['n_train']}")
                        
                        unique_id = generate_unique_id(exp)
                        if unique_id in completed_configs:
                            print(f"[GPU {gpu_id}] Skipping completed: {unique_id}")
                            continue
                            
                        try:
                            # For large experiments, use single fast training
                            fast_training(
                                exp, device, X_full, y_full, config["base_config"],
                                BATCH_SIZE_LARGE, config["base_config"]["epochs"],  # Use full epochs
                                full_results_dir, timestamp, gpu_id, checkpoint_log_path
                            )
                            completed_configs.add(unique_id)
                            completed_count += 1
                            print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        except Exception as e:
                            print(f"[GPU {gpu_id}] ERROR processing {unique_id}: {str(e)}")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR processing dataset {ds_path}: {str(e)}")
                traceback.print_exc()
                continue
            
            # Clear memory after processing a dataset
            del X_full, y_full, data
            torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        print(f"[GPU {gpu_id}] Completed all experiments in {elapsed_time:.2f} seconds")
        print(f"[GPU {gpu_id}] Average time per experiment: {elapsed_time/max(1, completed_count):.2f} seconds")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {str(e)}")
        traceback.print_exc()

def fast_parallel_training(config_batch, device, X_full, y_full, base_config, 
                          batch_size, eval_interval, max_epochs,
                          full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs):
    """
    Ultra-fast parallel training of multiple models on a single GPU.
    Returns the number of successfully completed experiments.
    """
    # Get test data
    n_test = base_config['n_test']
    fixed_seed = abs(hash(config_batch[0]['ds_path'])) % (2**32)
    generator = torch.Generator(device=device)
    generator.manual_seed(fixed_seed)
    indices = torch.randperm(len(X_full), device=device, generator=generator)
    test_indices = indices[:n_test]
    train_master_indices = indices[n_test:]
    X_test = X_full[test_indices]
    y_test = y_full[test_indices]
    
    # Setup for parallel training
    models = []
    optimizers = []
    train_data = []
    config_items = []
    unique_ids = []
    early_stop_flags = []
    
    # Initialize all models
    for config_item in config_batch:
        unique_id = generate_unique_id(config_item)
        
        if unique_id in completed_configs:
            continue
            
        # Sample training data
        n_train = config_item['n_train']
        sample_seed = hash(f"sample_{n_train}_{config_item['ds_name']}_{config_item['experiment_num']}")
        torch.manual_seed(sample_seed)
        
        if n_train < len(train_master_indices):
            train_indices = train_master_indices[torch.randperm(len(train_master_indices), device=device)[:n_train]]
            X_train = X_full[train_indices]
            y_train = y_full[train_indices]
        else:
            X_train = X_full[train_master_indices]
            y_train = y_full[train_master_indices]
        
        # Initialize model
        model_seed = hash(f"model_{config_item['ds_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}")
        torch.manual_seed(model_seed)
        
        model = DeepNN(
            config_item['input_dim'], 
            config_item['hidden_size'], 
            config_item['depth'], 
            mode=config_item['mode'], 
            alignment=config_item['alignment'],
            base_width=config_item.get('base_width', 10),
            gamma=config_item.get('gamma', 1.0)
        ).to(device)
        
        # Create optimizer with properly scaled learning rate
        # Use sqrt scaling for learning rate
        batch_size_ratio = batch_size / ORIG_BATCH_SIZE
        scaled_lr = config_item["lr"] * (batch_size_ratio ** 0.5)
        optimizer = optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=base_config["weight_decay"])
        
        # Store everything
        models.append(model)
        optimizers.append(optimizer)
        train_data.append((X_train, y_train))
        config_items.append(config_item)
        unique_ids.append(unique_id)
        early_stop_flags.append(False)
    
    if not models:  # All experiments were already completed
        return 0
    
    # Parallel training with BF16 mixed precision
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    
    # Error history tracking
    train_errors = [[] for _ in range(len(models))]
    test_errors = [[] for _ in range(len(models))]
    epoch_numbers = [[] for _ in range(len(models))]
    
    # Track initial errors
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
            for i, model in enumerate(models):
                if early_stop_flags[i]:
                    continue
                    
                X_train, y_train = train_data[i]
                model.eval()
                
                train_output = model(X_train)
                test_output = model(X_test)
                
                train_error = torch.mean((train_output - y_train) ** 2).item()
                test_error = torch.mean((test_output - y_test) ** 2).item()
                
                train_errors[i].append(train_error)
                test_errors[i].append(test_error)
                epoch_numbers[i].append(0)
    
    # Use a less aggressive early stopping threshold
    early_stop_threshold = 1e-7  # Much smaller to avoid premature stopping
    
    # Fast parallel training loop
    for epoch in range(max_epochs):
        # Check if all models have early stopped
        if all(early_stop_flags):
            break
            
        # Train each model with one batch
        for i, model in enumerate(models):
            if early_stop_flags[i]:
                continue
                
            model.train()
            optimizer = optimizers[i]
            X_train, y_train = train_data[i]
            
            # Sample random batch
            if len(X_train) <= batch_size:
                batch_X, batch_y = X_train, y_train
            else:
                batch_indices = torch.randperm(len(X_train), device=device)[:batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
            
            # One training step
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                output = model(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Evaluate periodically
        if (epoch + 1) % eval_interval == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                    for i, model in enumerate(models):
                        if early_stop_flags[i]:
                            continue
                            
                        X_train, y_train = train_data[i]
                        model.eval()
                        
                        train_output = model(X_train)
                        test_output = model(X_test)
                        
                        train_error = torch.mean((train_output - y_train) ** 2).item()
                        test_error = torch.mean((test_output - y_test) ** 2).item()
                        
                        train_errors[i].append(train_error)
                        test_errors[i].append(test_error)
                        epoch_numbers[i].append(epoch + 1)
                        
                        # Early stopping check - less aggressive now
                        if train_error < early_stop_threshold:
                            early_stop_flags[i] = True
    
    # Add fine tuning phase - always run full fine-tuning
    fine_tuning_epochs = base_config.get("fine_tuning_epochs", 500)
    
    # Do fine-tuning phase for all models
    for i, model in enumerate(models):
        if early_stop_flags[i]:
            continue
            
        X_train, y_train = train_data[i]
        optimizer = optimizers[i]
        
        # Reset learning rate for fine-tuning
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5  # Lower LR for fine-tuning
        
        # Fine-tuning loop
        for ft_epoch in range(fine_tuning_epochs):
            model.train()
            
            # Sample batch
            if len(X_train) <= batch_size:
                batch_X, batch_y = X_train, y_train
            else:
                batch_indices = torch.randperm(len(X_train), device=device)[:batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
            
            # One training step
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                output = model(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
            
            loss.backward()
            optimizer.step()
            
            # Evaluate at the end
            if ft_epoch == fine_tuning_epochs - 1:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                        model.eval()
                        train_output = model(X_train)
                        test_output = model(X_test)
                        
                        train_error = torch.mean((train_output - y_train) ** 2).item()
                        test_error = torch.mean((test_output - y_test) ** 2).item()
                        
                        train_errors[i].append(train_error)
                        test_errors[i].append(test_error)
                        epoch_numbers[i].append(max_epochs + ft_epoch + 1)
    
    # Save results
    completed_count = 0
    for i in range(len(models)):
        model = models[i]
        config_item = config_items[i]
        X_train, y_train = train_data[i]
        unique_id = unique_ids[i]
        
        # Final evaluation
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                model.eval()
                train_output = model(X_train)
                test_output = model(X_test)
                
                final_train_error = torch.mean((train_output - y_train) ** 2).item()
                final_test_error = torch.mean((test_output - y_test) ** 2).item()
        
        # Add final epoch if not already added
        if epoch_numbers[i][-1] != max_epochs + fine_tuning_epochs:
            train_errors[i].append(final_train_error)
            test_errors[i].append(final_test_error)
            epoch_numbers[i].append(max_epochs + fine_tuning_epochs)
        
        # Save result
        result = {
            'dataset_name': config_item['ds_name'],
            'dataset_path': config_item['ds_path'],
            'hidden_size': config_item['hidden_size'],
            'depth': config_item['depth'],
            'input_dim': config_item['input_dim'],
            'base_width': config_item.get('base_width', 10),
            'n_train': config_item['n_train'],
            'learning_rate': config_item['lr'],
            'mode': config_item['mode'],
            'alignment': config_item['alignment'],
            'gamma': config_item.get('gamma', 1.0),
            'alpha': config_item.get('alpha', 1.0),
            'test_error': final_test_error,
            'initial_train_error': train_errors[i][0],
            'final_train_error': final_train_error,
            'error_history': {
                'train_errors': train_errors[i],
                'test_errors': test_errors[i],
                'epochs': epoch_numbers[i],
                'early_stopped': early_stop_flags[i],
                'stopped_epoch': epoch_numbers[i][-1]
            },
            'worker_gpu': gpu_id,
            'model_seed': hash(f"model_{config_item['ds_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}"),
            'experiment_num': config_item['experiment_num'],
            'sweep_name': config_item['sweep_name'],
            'parallel_trained': True,
            'batch_size': batch_size,
            'scaled_lr': scaled_lr,  # Include scaled LR in results
            'batch_size_ratio': batch_size_ratio
        }
        
        # Save results
        results_file_path = os.path.join(full_results_dir, f"results_{timestamp}_gpu{gpu_id}.jsonl")
        with open(results_file_path, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()
        
        # Mark as completed
        with open(checkpoint_log_path, "a") as cp_f:
            cp_f.write(unique_id + "\n")
        completed_configs.add(unique_id)
        completed_count += 1
    
    return completed_count

def fast_training(config_item, device, X_full, y_full, base_config, 
                 batch_size, max_epochs, full_results_dir, timestamp, 
                 gpu_id, checkpoint_log_path):
    """Fast single model training for large experiments."""
    unique_id = generate_unique_id(config_item)
    
    # Get test data
    n_test = base_config['n_test']
    fixed_seed = abs(hash(config_item['ds_path'])) % (2**32)
    generator = torch.Generator(device=device)
    generator.manual_seed(fixed_seed)
    indices = torch.randperm(len(X_full), device=device, generator=generator)
    test_indices = indices[:n_test]
    train_master_indices = indices[n_test:]
    X_test = X_full[test_indices]
    y_test = y_full[test_indices]
    
    # Sample training data
    n_train = config_item['n_train']
    sample_seed = hash(f"sample_{n_train}_{config_item['ds_name']}_{config_item['experiment_num']}")
    torch.manual_seed(sample_seed)
    
    if n_train < len(train_master_indices):
        train_indices = train_master_indices[torch.randperm(len(train_master_indices), device=device)[:n_train]]
        X_train = X_full[train_indices]
        y_train = y_full[train_indices]
    else:
        X_train = X_full[train_master_indices]
        y_train = y_full[train_master_indices]
    
    # Initialize model
    model_seed = hash(f"model_{config_item['ds_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}")
    torch.manual_seed(model_seed)
    
    model = DeepNN(
        config_item['input_dim'], 
        config_item['hidden_size'], 
        config_item['depth'], 
        mode=config_item['mode'], 
        alignment=config_item['alignment'],
        base_width=config_item.get('base_width', 10),
        gamma=config_item.get('gamma', 1.0)
    ).to(device)
    
    # Create optimizer with properly scaled learning rate
    # Use sqrt scaling for learning rate
    batch_size_ratio = batch_size / ORIG_BATCH_SIZE
    scaled_lr = config_item["lr"] * (batch_size_ratio ** 0.5)
    optimizer = optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=base_config["weight_decay"])
    
    # Use BF16 mixed precision
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    
    # Error history tracking
    train_errors = []
    test_errors = []
    epoch_numbers = []
    
    # Get initial error
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
            model.eval()
            train_output = model(X_train)
            test_output = model(X_test)
            
            initial_train_error = torch.mean((train_output - y_train) ** 2).item()
            initial_test_error = torch.mean((test_output - y_test) ** 2).item()
            
            train_errors.append(initial_train_error)
            test_errors.append(initial_test_error)
            epoch_numbers.append(0)
    
    # Less aggressive early stopping
    early_stop_threshold = 1e-7  # Much smaller to avoid premature stopping
    
    # Fast training loop
    for epoch in range(max_epochs):
        model.train()
        
        # Process data in batches
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            batch_X = X_train[i:end_idx]
            batch_y = y_train[i:end_idx]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                output = model(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
                total_loss += loss.item()
                num_batches += 1
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Evaluate periodically
        eval_interval = max(1, max_epochs // 10)  # At least 10 evaluations
        if (epoch + 1) % eval_interval == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                    model.eval()
                    train_output = model(X_train)
                    test_output = model(X_test)
                    
                    train_error = torch.mean((train_output - y_train) ** 2).item()
                    test_error = torch.mean((test_output - y_test) ** 2).item()
                    
                    train_errors.append(train_error)
                    test_errors.append(test_error)
                    epoch_numbers.append(epoch + 1)
                    
                    # Early stopping with less aggressive threshold
                    if train_error < early_stop_threshold:
                        print(f"[GPU {gpu_id}] Early stopping at epoch {epoch+1} with train error {train_error:.8f}")
                        break
    
    # Do fine-tuning
    fine_tuning_epochs = base_config.get("fine_tuning_epochs", 500)
    
    # Reset learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = scaled_lr / 5  # Lower LR for fine-tuning
    
    # Fine-tuning loop
    for ft_epoch in range(fine_tuning_epochs):
        model.train()
        
        # Process data in batches
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            batch_X = X_train[i:end_idx]
            batch_y = y_train[i:end_idx]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                output = model(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
            
            loss.backward()
            optimizer.step()
        
        # Evaluate at the end
        if ft_epoch == fine_tuning_epochs - 1:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                    model.eval()
                    train_output = model(X_train)
                    test_output = model(X_test)
                    
                    train_error = torch.mean((train_output - y_train) ** 2).item()
                    test_error = torch.mean((test_output - y_test) ** 2).item()
                    
                    train_errors.append(train_error)
                    test_errors.append(test_error)
                    epoch_numbers.append(max_epochs + ft_epoch + 1)
    
    # Final evaluation
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
            model.eval()
            train_output = model(X_train)
            test_output = model(X_test)
            
            final_train_error = torch.mean((train_output - y_train) ** 2).item()
            final_test_error = torch.mean((test_output - y_test) ** 2).item()
    
    # Save result
    result = {
        'dataset_name': config_item['ds_name'],
        'dataset_path': config_item['ds_path'],
        'hidden_size': config_item['hidden_size'],
        'depth': config_item['depth'],
        'input_dim': config_item['input_dim'],
        'base_width': config_item.get('base_width', 10),
        'n_train': config_item['n_train'],
        'learning_rate': config_item['lr'],
        'mode': config_item['mode'],
        'alignment': config_item['alignment'],
        'gamma': config_item.get('gamma', 1.0),
        'alpha': config_item.get('alpha', 1.0),
        'test_error': final_test_error,
        'initial_train_error': initial_train_error,
        'final_train_error': final_train_error,
        'error_history': {
            'train_errors': train_errors,
            'test_errors': test_errors,
            'epochs': epoch_numbers,
            'early_stopped': epoch < max_epochs - 1,
            'stopped_epoch': epoch if epoch < max_epochs - 1 else max_epochs + fine_tuning_epochs - 1
        },
        'worker_gpu': gpu_id,
        'model_seed': model_seed,
        'experiment_num': config_item['experiment_num'],
        'sweep_name': config_item['sweep_name'],
        'batch_size': batch_size,
        'scaled_lr': scaled_lr,  # Include scaled LR in results
        'batch_size_ratio': batch_size_ratio
    }
    
    # Save results
    results_file_path = os.path.join(full_results_dir, f"results_{timestamp}_gpu{gpu_id}.jsonl")
    with open(results_file_path, "a") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()
    
    # Mark as completed
    with open(checkpoint_log_path, "a") as cp_f:
        cp_f.write(unique_id + "\n")
    
    return True

def main():
    try:
        start_time = time.time()
        print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        if len(sys.argv) < 2:
            print("Usage: python main_single.py <config_file.yaml>")
            sys.exit(1)
        
        config_path = sys.argv[1]
        print(f"Loading config from: {config_path}")
        
        config = load_yaml_config(config_path)
        
        # Extract Base Config
        base_cfg = config["base_config"]
        base_results_dir = base_cfg["base_results_dir"]
        restart_checkpoint = base_cfg.get("restart_checkpoint")
        
        # Don't automatically change epochs - use what's in the config file
        # Just make sure fine_tuning_epochs is set
        if "fine_tuning_epochs" not in base_cfg:
            base_cfg["fine_tuning_epochs"] = 500
        
        # Create experiment name
        sweep_names = list(config["sweeps"].keys())
        experiment_name = f"{'_'.join(sweep_names)}_exp_{datetime.now().strftime('%Y%m%d')}"
        
        # Set up Results Directory
        full_results_dir = os.path.join(base_results_dir, experiment_name)
        os.makedirs(full_results_dir, exist_ok=True)
        
        # Set up Checkpointing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_log_path = os.path.join(full_results_dir, f"checkpoint_{timestamp}.txt")
        
        # Handle restart logic
        if restart_checkpoint is not None:
            checkpoint_log_path = restart_checkpoint
            with open(checkpoint_log_path, "r") as f:
                completed_configs = set(line.strip() for line in f if line.strip())
            timestamp = os.path.basename(restart_checkpoint).replace("checkpoint_", "").replace(".txt", "")
            print(f"Restarting from checkpoint with {len(completed_configs)} completed configurations")
        else:
            if os.path.exists(checkpoint_log_path):
                with open(checkpoint_log_path, "r") as f:
                    completed_configs = set(line.strip() for line in f if line.strip())
                print(f"Using existing checkpoint with {len(completed_configs)} completed configs")
            else:
                completed_configs = set()
                print(f"Starting new run")
            
            # Save hyperparameters
            hyperparams_path = os.path.join(full_results_dir, f"hyperparameters_{timestamp}.yaml")
            with open(hyperparams_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # Generate all combinations
        print("Generating parameter combinations...")
        all_combinations = generate_all_combinations(config)
        print(f"Generated {len(all_combinations)} combinations")
        
        # Filter out completed configurations
        remaining = [c for c in all_combinations if generate_unique_id(c) not in completed_configs]
        print(f"Remaining configurations to process: {len(remaining)}/{len(all_combinations)}")
        
        if not remaining:
            print("All configurations already completed!")
            return
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available. Running on CPU.")
            num_gpus = 1
        
        print(f"Using {num_gpus} GPU(s)")
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            print("spawn method already set")
        
        # Launch one process per GPU
        mp.spawn(
            worker_process,
            args=(num_gpus, all_combinations, config, full_results_dir, timestamp, checkpoint_log_path, completed_configs),
            nprocs=num_gpus,
            join=True
        )
        
        total_time = time.time() - start_time
        print(f"All processes completed in {total_time:.2f} seconds")
        print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()