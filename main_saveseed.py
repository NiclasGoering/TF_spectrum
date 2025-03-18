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
import fcntl  # Added for file locking
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
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.45"
torch.cuda.set_per_process_memory_fraction(0.45)

# PERFORMANCE CONFIGURATION
# These settings are optimized for maximum throughput on H100s
MAX_PARALLEL_TINY = 16   # For n_train < 1000
MAX_PARALLEL_SMALL = 8   # For 1000 <= n_train < 10000
MAX_PARALLEL_MEDIUM = 4  # For 10000 <= n_train < 100000
MAX_PARALLEL_LARGE = 4   # For 100000 <= n_train < 1000000
MAX_PARALLEL_HUGE = 2    # For n_train >= 1000000
BATCH_SIZE_TINY = 1024   # Batch size for tiny experiments
BATCH_SIZE_SMALL = 4096  # Batch size for small experiments
BATCH_SIZE_MEDIUM = 8192 # Batch size for medium experiments (IMPORTANT: Same as main1.py)
BATCH_SIZE_LARGE = 65536 # Batch size for large experiments
BATCH_SIZE_HUGE = 131072  # Batch size for huge experiments
ORIG_BATCH_SIZE = 32768  # Original batch size reference for LR scaling

# Global stats for tracking
global_stats = {
    "total_experiments": 0,
    "completed_experiments": 0,
    "skipped_experiments": 0,
    "failed_experiments": 0,
    "failed_models": set(),
    "error_types": {}
}

def load_yaml_config(config_path):
    """Load and return the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_info_from_path(path):
    """Extract parameters from path."""
    path_lower = path.lower()
    path_upper = path.upper()
    basename = os.path.basename(path)
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

    parts = basename.split('_')
    
    info = {
        "distribution_type": dist_type
    }

    # Extract order number - handle different formats
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

    # Extract key parameters - handle multiple formats
    for i, part in enumerate(parts):
        # Input dimension - match d8, d128, etc.
        if part.startswith('d') and part[1:].isdigit():
            info['input_dim'] = int(part[1:])
        
        # Hidden size - match H8, H128, etc.
        if part.startswith('H') and part[1:].isdigit():
            info['hidden_size'] = int(part[1:])
        
        # Depth - match D2, D3, etc.
        if part.startswith('D') and part[1:].isdigit():
            info['depth'] = int(part[1:])
        
        # Alpha value - match a0.0, a0.5, a1.0, etc.
        if part.startswith('a') and i < len(parts) - 1:
            try:
                info['alpha'] = float(part[1:])
            except ValueError:
                pass
    
    return info

def generate_unique_id(config):
    """Generate a unique identifier for this configuration."""
    model_name = config.get('model_name', '')
    
    # Get order number
    order_num = config.get('order_num', '1')
    
    # Remove the timestamp component from the model name if present
    # This matches pattern like PT_NP_d128_H128_D2_a0.0_O_1_20250315_232919
    parts = model_name.split('_')
    if len(parts) > 8 and len(parts[8]) == 8 and parts[8].isdigit():
        model_name = '_'.join(parts[:8])
    
    # Remove any "PT_" prefix
    if model_name.startswith("PT_"):
        model_name = model_name[3:]
    
    # Make sure the model name doesn't already include O_{order_num}
    if f"_O_{order_num}" in model_name:
        model_base = model_name.split(f"_O_{order_num}")[0]
    else:
        model_base = model_name.split("_O_")[0] if "_O_" in model_name else model_name
    
    align_suffix = "_align" if config['alignment'] else ""
    
    unique_id = (
        f"{model_base}_O{order_num}"
        f"_h{config['hidden_size']}"
        f"_d{config['depth']}"
        f"_n{config['n_train']}"
        f"_lr{config['lr']}"
        f"_mode{config['mode']}"
        f"_exp{config['experiment_num']}"
        f"{align_suffix}"
    )
    
    return unique_id

def find_model_file(directory):
    """Find model file in the directory based on directory name pattern."""
    # Get directory basename
    dir_basename = os.path.basename(directory)
    
    # Extract parameters from directory name
    dir_params = extract_info_from_path(dir_basename)
    
    # Construct expected model filename pattern
    if 'distribution_type' in dir_params and 'input_dim' in dir_params and 'hidden_size' in dir_params and 'depth' in dir_params and 'alpha' in dir_params and 'order_num' in dir_params:
        dist_type = dir_params['distribution_type']
        input_dim = dir_params['input_dim']
        hidden_size = dir_params['hidden_size']
        depth = dir_params['depth']
        alpha = dir_params['alpha']
        order_num = dir_params['order_num']
        
        # Expected filename pattern
        model_pattern = f"model_{dist_type}_d{input_dim}_H{hidden_size}_D{depth}_a{alpha}_O_{order_num}.pt"
        model_path = os.path.join(directory, model_pattern)
        
        if os.path.exists(model_path):
            return model_path
    
    # If not found with exact pattern, try more general model_*.pt pattern
    model_files = glob.glob(os.path.join(directory, "model_*.pt"))
    if model_files:
        return model_files[0]
    
    # If still not found, look for any .pt file that might be a model
    pt_files = glob.glob(os.path.join(directory, "*.pt"))
    for pt_file in pt_files:
        if os.path.basename(pt_file).startswith("model_"):
            return pt_file
    
    return None

def find_dataset_directories(base_dir):
    """Find all directories with PT_ prefix."""
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory {base_dir} does not exist!")
        return []
        
    # Look for PT_ subdirectories
    pt_dirs = glob.glob(os.path.join(base_dir, "PT_*"))
    pt_dirs = [d for d in pt_dirs if os.path.isdir(d)]
    
    return pt_dirs

def load_model_info(directory):
    """Load model info from directory."""
    if not os.path.isdir(directory):
        return None
    
    dir_basename = os.path.basename(directory)
    
    # Extract parameters directly from directory name
    if dir_basename.startswith("PT_"):
        params = extract_info_from_path(directory)
        
        # Set parameters from directory name
        dist_type = params.get("distribution_type", "")
        input_dim = params.get("input_dim", "")
        hidden_size = params.get("hidden_size", "")
        depth = params.get("depth", "")
        alpha = params.get("alpha", "")
        order_num = params.get("order_num", "1")
        
        # Find the model file
        model_file = find_model_file(directory)
        if not model_file:
            print(f"Warning: No model file found in {directory}")
            return None
        
        # Construct model name
        model_name = f"{dist_type}_d{input_dim}_H{hidden_size}_D{depth}_a{alpha}_O_{order_num}"
        
        file_size_mb = os.path.getsize(model_file) / (1024 * 1024)
        
        return {
            "model_path": model_file,
            "model_directory": directory,
            "name": model_name,
            "params": params,
            "directory": directory,
            "size_mb": file_size_mb
        }
    
    return None


def load_model_from_path(model_path, device):
    """Load a pretrained model from path to device, including data generation parameters."""
    try:
        print(f"Loading model from {model_path}")
        model_data = torch.load(model_path, map_location=device)
        
        # Handle different save formats
        if isinstance(model_data, dict) and 'model_state_dict' in model_data:
            # Extract model state and parameters
            model_state = model_data['model_state_dict']
            
            # Extract saved parameters
            input_dim = model_data.get('input_dim')
            hidden_size = model_data.get('hidden_size')
            depth = model_data.get('depth')
            mode = model_data.get('mode', 'standard')
            alignment = model_data.get('alignment', False)
            
            # Extract data generation information
            data_generation_seed = model_data.get('data_generation_seed')
            dist_type = model_data.get('distribution_type', 'normal')
            dist_params = model_data.get('distribution_params', {})
            r_value = dist_params.get('r_value')
            alpha = model_data.get('alpha', 1.0)  # Extract alpha value for kernel shape
            
            print(f"Found parameters in model dict: input_dim={input_dim}, hidden_size={hidden_size}, depth={depth}")
            if data_generation_seed is not None:
                print(f"Found data generation seed: {data_generation_seed}, dist_type={dist_type}")
        else:
            # Assume it's just the state dict
            model_state = model_data
            
            # Extract parameters from filename
            filename = os.path.basename(model_path)
            params = extract_info_from_path(filename)
            input_dim = params.get('input_dim')
            hidden_size = params.get('hidden_size')
            depth = params.get('depth', 2)  # Default to 2 if not specified
            mode = 'standard'
            alignment = False
            
            # Set default data generation parameters
            data_generation_seed = None
            dist_type = 'normal'
            r_value = None
            alpha = params.get('alpha', 1.0)
            
            print(f"Extracted parameters from filename: input_dim={input_dim}, hidden_size={hidden_size}, depth={depth}")
            print(f"No data generation seed found, will use random generation")
        
        # If parameters are missing, try to extract from directory name
        if input_dim is None or hidden_size is None or depth is None:
            dir_name = os.path.basename(os.path.dirname(model_path))
            params = extract_info_from_path(dir_name)
            
            if input_dim is None:
                input_dim = params.get('input_dim')
            if hidden_size is None:
                hidden_size = params.get('hidden_size')
            if depth is None:
                depth = params.get('depth', 2)
            if alpha is None:
                alpha = params.get('alpha', 1.0)
                
            print(f"Extracted parameters from directory: input_dim={input_dim}, hidden_size={hidden_size}, depth={depth}")
        
        # Check if we have all the required parameters
        if input_dim is None or hidden_size is None or depth is None:
            raise ValueError(f"Could not determine model parameters from {model_path}")
        
        # Create model instance
        model = DeepNN(
            input_dim, 
            hidden_size, 
            depth, 
            mode=mode, 
            alignment=alignment
        ).to(device)
        
        # Load state dict
        model.load_state_dict(model_state)
        model.eval()  # Set to evaluation mode
        
        print(f"Successfully loaded model with input_dim={input_dim}, hidden_size={hidden_size}, depth={depth}")
        
        return model, input_dim, hidden_size, depth, alpha, data_generation_seed, dist_type, r_value
    
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None

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

def generate_dataset_from_model_with_seed(model, input_dim, n_samples, device, seed, dist_type, r_value):
    # Set exact seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate the EXACT same data as during pretraining
    if dist_type == 'normal':
        X = torch.randn(n_samples, input_dim, device=device)
    
    # Get outputs
    with torch.no_grad():
        y = model(X)
        
    return {'X': X, 'y': y}

def safe_update_checkpoint(checkpoint_path, unique_id):
    """
    Safely update the checkpoint file using file locking.
    Returns True if successful, False otherwise.
    """
    try:
        with open(checkpoint_path, "a") as cp_f:
            # Acquire an exclusive lock
            fcntl.flock(cp_f, fcntl.LOCK_EX)
            
            # Write the unique ID
            cp_f.write(unique_id + "\n")
            
            # Ensure data is written to disk
            cp_f.flush()
            os.fsync(cp_f.fileno())
            
            # Release the lock
            fcntl.flock(cp_f, fcntl.LOCK_UN)
        
        return True
    except Exception as e:
        print(f"Error updating checkpoint file: {str(e)}")
        traceback.print_exc()
        return False

def load_checkpoint_set(checkpoint_path):
    """Load checkpoint file into a set, using file locking for safety."""
    completed_configs = set()
    
    if not os.path.exists(checkpoint_path):
        return completed_configs
    
    try:
        with open(checkpoint_path, "r") as f:
            # Acquire a shared lock for reading
            fcntl.flock(f, fcntl.LOCK_SH)
            
            # Read all lines
            completed_configs = set(line.strip() for line in f if line.strip())
            
            # Release the lock
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Error reading checkpoint file: {str(e)}")
        traceback.print_exc()
    
    return completed_configs

def sync_checkpoint_file(checkpoint_path, gpu_id):
    """
    Sync the in-memory completed_configs with the checkpoint file.
    This helps ensure a GPU has the latest information from other GPUs.
    """
    try:
        # Get the latest completed experiments from the checkpoint file
        updated_configs = load_checkpoint_set(checkpoint_path)
        print(f"[GPU {gpu_id}] Synced checkpoint file. Now tracking {len(updated_configs)} completed experiments")
        return updated_configs
    except Exception as e:
        print(f"[GPU {gpu_id}] Error syncing checkpoint file: {str(e)}")
        traceback.print_exc()
        return set()  # Return empty set on error

def generate_all_combinations(config):
    """Generate all parameter combinations."""
    base_cfg = config["base_config"]
    sweeps = config["sweeps"]
    
    all_combinations = []
    
    for sweep_name, sweep_info in sweeps.items():
        # Read directory paths or dataset paths
        model_dirs = []
        
        if "dataset_dir" in sweep_info:
            # Find all directories that might contain model information
            for dir_path in sweep_info["dataset_dir"]:
                print(f"Scanning directory: {dir_path}")
                subdirs = find_dataset_directories(dir_path)
                if not subdirs:
                    print(f"WARNING: No PT_* subdirectories found in {dir_path}")
                else:
                    print(f"Found {len(subdirs)} potential model directories")
                model_dirs.extend(subdirs)
        elif "dataset_paths" in sweep_info:
            model_dirs = sweep_info["dataset_paths"]
        
        if not model_dirs:
            print(f"WARNING: No model directories found for sweep '{sweep_name}'")
            continue
            
        sweep_params = sweep_info.get("parameters", {})
        
        model_infos = []
        for dir_path in model_dirs:
            model_info = load_model_info(dir_path)
            if model_info:
                model_infos.append(model_info)
            else:
                print(f"WARNING: Failed to load model info from {dir_path}")
        
        print(f"Found {len(model_infos)}/{len(model_dirs)} valid model configurations")
        
        for model_info in model_infos:
            model_params = model_info['params']
            input_dim = model_params.get('input_dim')
            if not input_dim:
                print(f"Skipping {model_info['directory']} - missing input dimension")
                continue
            
            # Get order number
            order_num = model_params.get('order_num', '1')
            
            for n_train in sweep_params.get("n_train", [1024]):
                for lr in sweep_params.get("learning_rates", [0.001]):
                    for hidden_size in sweep_params.get("hidden_sizes", [256]):
                        for depth in sweep_params.get("depths", [1]):
                            for mode in sweep_params.get("modes", ["standard"]):
                                for alignment in sweep_params.get("alignment", [False]):
                                    for exp_num in range(1, base_cfg.get("num_experiments", 1) + 1):
                                        combo = {
                                            'model_path': model_info['model_path'],
                                            'model_directory': model_info['directory'],
                                            'model_name': model_info['name'],
                                            'hidden_size': hidden_size,
                                            'depth': depth,
                                            'input_dim': input_dim,
                                            'n_train': n_train,
                                            'lr': lr,
                                            'mode': mode,
                                            'gamma': model_params.get('gamma', 1.0),
                                            'experiment_num': exp_num,
                                            'base_width': sweep_params.get('base_width', 10),
                                            'alignment': alignment,
                                            'sweep_name': sweep_name,
                                            'alpha': model_params.get('alpha', 1.0),
                                            'size_mb': model_info.get('size_mb', 0),
                                            'order_num': order_num,
                                        }
                                        all_combinations.append(combo)
    
    return all_combinations

def worker_process(gpu_id, num_gpus, all_combinations, config, full_results_dir, timestamp, checkpoint_log_path, completed_configs):
    """
    Modified worker process that:
    1. Loads model and data generation parameters
    2. Recreates the exact dataset used during pretraining if possible
    3. Uses consistent data sampling to preserve learning curve shape
    """
    try:
        start_time = time.time()
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
        # Initialize tracking variables
        completed_count = 0
        skipped_count = 0
        failed_count = 0
        failed_models = set()
        model_stats = {}
        
        print(f"[GPU {gpu_id}] Worker started on device {device}")
        
        # Sync with checkpoint file to get latest state from all GPUs
        completed_configs = sync_checkpoint_file(checkpoint_log_path, gpu_id)
        
        # Filter combinations for this GPU using modulo assignment
        worker_combinations = [combo for i, combo in enumerate(all_combinations) if i % num_gpus == gpu_id]
        print(f"[GPU {gpu_id}] Assigned {len(worker_combinations)} configurations")
        
        # Remove already completed configurations
        initial_count = len(worker_combinations)
        worker_combinations = [combo for combo in worker_combinations 
                              if generate_unique_id(combo) not in completed_configs]
        
        already_completed = initial_count - len(worker_combinations)
        if already_completed > 0:
            print(f"[GPU {gpu_id}] Skipping {already_completed} already completed configurations")
        
        if not worker_combinations:
            print(f"[GPU {gpu_id}] All configurations already completed")
            return
            
        print(f"[GPU {gpu_id}] Processing {len(worker_combinations)} incomplete configurations")
        
        # Group by model for more efficient processing
        by_model = {}
        for combo in worker_combinations:
            model_path = combo['model_path']
            if model_path not in by_model:
                by_model[model_path] = []
            by_model[model_path].append(combo)
        
        # Process each model group
        total_to_process = len(worker_combinations)
        
        # Periodically sync checkpoint file
        last_sync_time = time.time()
        sync_interval = 300  # seconds (5 minutes)
        
        for model_idx, (model_path, model_combos) in enumerate(by_model.items()):
            # Skip empty model groups
            if not model_combos:
                continue
            
            # Sync checkpoint file periodically
            current_time = time.time()
            if current_time - last_sync_time > sync_interval:
                completed_configs = sync_checkpoint_file(checkpoint_log_path, gpu_id)
                last_sync_time = current_time
                
                # Filter out any newly completed configurations
                model_combos = [combo for combo in model_combos 
                               if generate_unique_id(combo) not in completed_configs]
                
                if not model_combos:
                    print(f"[GPU {gpu_id}] All configs for model {os.path.basename(model_path)} completed by other GPUs, skipping")
                    continue
                
            # Track stats for this model
            model_basename = os.path.basename(model_path)
            if model_basename not in model_stats:
                model_stats[model_basename] = {
                    "total": len(model_combos),
                    "completed": 0,
                    "skipped": 0,
                    "failed": 0
                }
                
            print(f"[GPU {gpu_id}] Loading model [{model_idx+1}/{len(by_model)}]: {model_path}")
            
            # Load model with data generation parameters
            model, input_dim, hidden_size, depth, alpha, data_generation_seed, dist_type, r_value = load_model_from_path(model_path, device)
            
            if model is None or input_dim is None:
                error_msg = f"[GPU {gpu_id}] ERROR: Failed to load model from {model_path}, skipping {len(model_combos)} experiments"
                print(error_msg)
                skipped_count += len(model_combos)
                model_stats[model_basename]["skipped"] += len(model_combos)
                failed_models.add(model_basename)
                
                # Log detailed error
                error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                with open(error_log_path, "a") as err_f:
                    fcntl.flock(err_f, fcntl.LOCK_EX)
                    err_f.write(f"{datetime.now()}: {error_msg}\n")
                    fcntl.flock(err_f, fcntl.LOCK_UN)
                
                continue
            
            try:
                # Find the maximum n_train value to determine how much data to generate
                max_n_train = max([c['n_train'] for c in model_combos])
                n_test = config["base_config"]['n_test']
                total_samples_needed = max_n_train + n_test
                
                # Generate dataset based on whether we have a seed
                if data_generation_seed is not None:
                    print(f"[GPU {gpu_id}] Generating dataset with {total_samples_needed} samples using original seed {data_generation_seed}")
                    
                    # Set the exact same seed used during pretraining
                    torch.manual_seed(data_generation_seed)
                    np.random.seed(data_generation_seed)
                    
                    # Generate data with the same distribution type and parameters
                    if dist_type == 'normal':
                        X = torch.randn(total_samples_needed, input_dim, device=device)
                    elif dist_type == 'uniform':
                        X = 2 * torch.rand(total_samples_needed, input_dim, device=device) - 1
                    elif dist_type == 'spiked_normal':
                        # Spiked normal with r_value
                        theta = torch.randn(input_dim, device=device)
                        theta = theta / torch.norm(theta)
                        X = torch.randn(total_samples_needed, input_dim, device=device)
                        spike_scale = input_dim**(r_value or 0.5)
                        Z = torch.randn(total_samples_needed, 1, device=device) * torch.sqrt(torch.tensor(spike_scale, device=device))
                        X = X + Z * theta
                    else:
                        print(f"[GPU {gpu_id}] Unknown distribution type: {dist_type}, falling back to normal")
                        X = torch.randn(total_samples_needed, input_dim, device=device)
                    
                    # Forward pass through model
                    with torch.no_grad():
                        y = model(X)
                    
                    data = {'X': X, 'y': y}
                    print(f"[GPU {gpu_id}] Successfully recreated original training dataset")
                else:
                    # Fallback to random generation
                    print(f"[GPU {gpu_id}] No seed information available. Generating random dataset with {total_samples_needed} samples")
                    data = generate_dataset_from_model(model, input_dim, total_samples_needed, device)
                
                X_full = data['X']
                y_full = data['y']
                
                print(f"[GPU {gpu_id}] Generated dataset: X_full shape {X_full.shape}, y_full shape {y_full.shape}")
                
                # Create a fixed random permutation of all data indices
                fixed_seed = abs(hash(model_path)) % (2**32)
                generator = torch.Generator(device=device)
                generator.manual_seed(fixed_seed)
                
                # Create a permutation for ordering samples
                # This is critical: we use the SAME permutation for ALL experiments
                # to ensure consistent subsampling
                all_indices = torch.randperm(len(X_full), device=device, generator=generator)
                
                # Put test data first, followed by training data
                # This reorders X_full and y_full according to our permutation
                X_full = X_full[all_indices]
                y_full = y_full[all_indices]
                
                # Create master indices for training data (these are already relative to the permuted data)
                train_master_indices = torch.arange(len(X_full) - n_test, device=device)
                
                print(f"[GPU {gpu_id}] Created train_master_indices with shape {train_master_indices.shape}")
                
                # Sort model combinations by n_train (ascending) to make the training progression clearer
                model_combos.sort(key=lambda x: x['n_train'])
                
                # Group experiments by size for optimal batching
                tiny_exps = [c for c in model_combos if c['n_train'] < 1000]
                small_exps = [c for c in model_combos if 1000 <= c['n_train'] < 10000]
                medium_exps = [c for c in model_combos if 10000 <= c['n_train'] < 100000]
                large_exps = [c for c in model_combos if 100000 <= c['n_train'] < 1000000]
                huge_exps = [c for c in model_combos if c['n_train'] >= 1000000]  # For very large experiments
                
                # --- Process tiny experiments in large parallel batches ---
                if tiny_exps:
                    for i in range(0, len(tiny_exps), MAX_PARALLEL_TINY):
                        batch = tiny_exps[i:i+MAX_PARALLEL_TINY]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(tiny_exps)} tiny experiments")
                        try:
                            n_completed = fast_parallel_training(
                                batch, device, X_full, y_full, train_master_indices, config["base_config"],
                                BATCH_SIZE_TINY, 10, config["base_config"]["epochs"],
                                full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs
                            )
                            completed_count += n_completed
                            model_stats[model_basename]["completed"] += n_completed
                            
                            # Update with any failed experiments
                            n_failed = len(batch) - n_completed
                            if n_failed > 0:
                                failed_count += n_failed
                                model_stats[model_basename]["failed"] += n_failed
                                
                            print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        except Exception as e:
                            print(f"[GPU {gpu_id}] ERROR processing tiny batch: {str(e)}")
                            traceback.print_exc()
                            
                            # Log the failure
                            failed_count += len(batch)
                            model_stats[model_basename]["failed"] += len(batch)
                            
                            # Log detailed error
                            error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                            with open(error_log_path, "a") as err_f:
                                fcntl.flock(err_f, fcntl.LOCK_EX)
                                err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error processing tiny batch from {model_basename}: {str(e)}\n")
                                err_f.write(traceback.format_exc() + "\n")
                                fcntl.flock(err_f, fcntl.LOCK_UN)
                
                # --- Process small experiments in parallel batches ---
                if small_exps:
                    for i in range(0, len(small_exps), MAX_PARALLEL_SMALL):
                        batch = small_exps[i:i+MAX_PARALLEL_SMALL]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(small_exps)} small experiments")
                        try:
                            n_completed = fast_parallel_training(
                                batch, device, X_full, y_full, train_master_indices, config["base_config"],
                                BATCH_SIZE_SMALL, 20, config["base_config"]["epochs"],
                                full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs
                            )
                            completed_count += n_completed
                            model_stats[model_basename]["completed"] += n_completed
                            
                            # Update with any failed experiments
                            n_failed = len(batch) - n_completed
                            if n_failed > 0:
                                failed_count += n_failed
                                model_stats[model_basename]["failed"] += n_failed
                                
                            print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        except Exception as e:
                            print(f"[GPU {gpu_id}] ERROR processing small batch: {str(e)}")
                            traceback.print_exc()
                            
                            # Log the failure
                            failed_count += len(batch)
                            model_stats[model_basename]["failed"] += len(batch)
                            
                            # Log detailed error
                            error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                            with open(error_log_path, "a") as err_f:
                                fcntl.flock(err_f, fcntl.LOCK_EX)
                                err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error processing small batch from {model_basename}: {str(e)}\n")
                                err_f.write(traceback.format_exc() + "\n")
                                fcntl.flock(err_f, fcntl.LOCK_UN)
                
                # --- Process medium experiments in smaller parallel batches ---
                if medium_exps:
                    for i in range(0, len(medium_exps), MAX_PARALLEL_MEDIUM):
                        batch = medium_exps[i:i+MAX_PARALLEL_MEDIUM]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(medium_exps)} medium experiments")
                        try:
                            n_completed = fast_parallel_training(
                                batch, device, X_full, y_full, train_master_indices, config["base_config"],
                                BATCH_SIZE_MEDIUM, 30, config["base_config"]["epochs"],
                                full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs
                            )
                            completed_count += n_completed
                            model_stats[model_basename]["completed"] += n_completed
                            
                            # Update with any failed experiments
                            n_failed = len(batch) - n_completed
                            if n_failed > 0:
                                failed_count += n_failed
                                model_stats[model_basename]["failed"] += n_failed
                                
                            print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        except Exception as e:
                            print(f"[GPU {gpu_id}] ERROR processing medium batch: {str(e)}")
                            traceback.print_exc()
                            
                            # Log the failure
                            failed_count += len(batch)
                            model_stats[model_basename]["failed"] += len(batch)
                            
                            # Log detailed error
                            error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                            with open(error_log_path, "a") as err_f:
                                fcntl.flock(err_f, fcntl.LOCK_EX)
                                err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error processing medium batch from {model_basename}: {str(e)}\n")
                                err_f.write(traceback.format_exc() + "\n")
                                fcntl.flock(err_f, fcntl.LOCK_UN)
                
                # --- Process large experiments with limited parallelism ---
                if large_exps:
                    for i in range(0, len(large_exps), MAX_PARALLEL_LARGE):
                        batch = large_exps[i:i+MAX_PARALLEL_LARGE]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(large_exps)} large experiments")
                        try:
                            n_completed = fast_parallel_training(
                                batch, device, X_full, y_full, train_master_indices, config["base_config"],
                                BATCH_SIZE_LARGE, 40, config["base_config"]["epochs"],
                                full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs
                            )
                            completed_count += n_completed
                            model_stats[model_basename]["completed"] += n_completed
                            
                            # Update with any failed experiments
                            n_failed = len(batch) - n_completed
                            if n_failed > 0:
                                failed_count += n_failed
                                model_stats[model_basename]["failed"] += n_failed
                                
                            print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        except Exception as e:
                            print(f"[GPU {gpu_id}] ERROR processing large batch: {str(e)}")
                            traceback.print_exc()
                            
                            # Log the failure
                            failed_count += len(batch)
                            model_stats[model_basename]["failed"] += len(batch)
                            
                            # Log detailed error
                            error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                            with open(error_log_path, "a") as err_f:
                                fcntl.flock(err_f, fcntl.LOCK_EX)
                                err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error processing large batch from {model_basename}: {str(e)}\n")
                                err_f.write(traceback.format_exc() + "\n")
                                fcntl.flock(err_f, fcntl.LOCK_UN)
                
                # --- Process huge experiments with limited parallelism and larger batch size ---
                if huge_exps:
                    for i in range(0, len(huge_exps), MAX_PARALLEL_HUGE):
                        batch = huge_exps[i:i+MAX_PARALLEL_HUGE]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(huge_exps)} huge experiments")
                        try:
                            n_completed = fast_parallel_training(
                                batch, device, X_full, y_full, train_master_indices, config["base_config"],
                                BATCH_SIZE_HUGE, 50, config["base_config"]["epochs"],
                                full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs
                            )
                            completed_count += n_completed
                            model_stats[model_basename]["completed"] += n_completed
                            
                            # Update with any failed experiments
                            n_failed = len(batch) - n_completed
                            if n_failed > 0:
                                failed_count += n_failed
                                model_stats[model_basename]["failed"] += n_failed
                                
                            print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        except Exception as e:
                            print(f"[GPU {gpu_id}] ERROR processing huge batch: {str(e)}")
                            traceback.print_exc()
                            
                            # Log the failure
                            failed_count += len(batch)
                            model_stats[model_basename]["failed"] += len(batch)
                            
                            # Log detailed error
                            error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                            with open(error_log_path, "a") as err_f:
                                fcntl.flock(err_f, fcntl.LOCK_EX)
                                err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error processing huge batch from {model_basename}: {str(e)}\n")
                                err_f.write(traceback.format_exc() + "\n")
                                fcntl.flock(err_f, fcntl.LOCK_UN)
            
            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR processing model {model_path}: {str(e)}")
                traceback.print_exc()
                
                # Log details
                error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                with open(error_log_path, "a") as err_f:
                    fcntl.flock(err_f, fcntl.LOCK_EX)
                    err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error processing model {model_path}: {str(e)}\n")
                    err_f.write(traceback.format_exc() + "\n")
                    fcntl.flock(err_f, fcntl.LOCK_UN)
                
                # Count all these as failures
                total_exps = len(tiny_exps) + len(small_exps) + len(medium_exps) + len(large_exps) + len(huge_exps)
                failed_count += total_exps
                model_stats[model_basename]["failed"] += total_exps
                failed_models.add(model_basename)
            
            finally:
                # Clear memory after processing a model's dataset
                if 'model' in locals() and model is not None:
                    del model
                if 'X_full' in locals():
                    del X_full
                if 'y_full' in locals():
                    del y_full
                if 'data' in locals():
                    del data
                if 'train_master_indices' in locals():
                    del train_master_indices
                # Force garbage collection
                torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        
        # Generate summary statistics
        summary = {
            "gpu_id": gpu_id,
            "total_assigned": total_to_process,
            "completed": completed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "already_completed": already_completed,
            "elapsed_time": elapsed_time,
            "average_time_per_experiment": elapsed_time / max(1, completed_count) if completed_count > 0 else 0,
            "failed_models": list(failed_models),
            "model_stats": model_stats
        }
        
        # Save summary
        summary_path = os.path.join(full_results_dir, f"summary_{timestamp}_gpu{gpu_id}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"[GPU {gpu_id}] SUMMARY:")
        print(f"  Completed: {completed_count}/{total_to_process} ({completed_count/max(1, total_to_process):.1%})")
        print(f"  Skipped: {skipped_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Already completed: {already_completed}")
        print(f"  Elapsed time: {elapsed_time:.2f} seconds")
        print(f"  Average time per experiment: {elapsed_time/max(1, completed_count):.2f} seconds")
        print(f"  Failed models: {len(failed_models)}")
        print(f"  Detailed summary saved to {summary_path}")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {str(e)}")
        traceback.print_exc()
        
        try:
            # Log the fatal error
            error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
            with open(error_log_path, "a") as err_f:
                fcntl.flock(err_f, fcntl.LOCK_EX)
                err_f.write(f"{datetime.now()}: [GPU {gpu_id}] FATAL ERROR: {str(e)}\n")
                err_f.write(traceback.format_exc() + "\n")
                fcntl.flock(err_f, fcntl.LOCK_UN)
        except:
            # If even the error logging fails, just continue
            pass


def fast_parallel_training(config_batch, device, X_full, y_full, train_master_indices, base_config, 
                          batch_size, eval_interval, max_epochs,
                          full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs):
    """
    Ultra-fast parallel training of multiple models on a single GPU.
    Returns the number of successfully completed experiments.
    Modified to use consistent data sampling with train_master_indices.
    """
    try:
        # Get test data - no need to create test_indices tensor directly
        n_test = base_config['n_test']
        
        # Print diagnostic information
        print(f"[GPU {gpu_id}] X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
        print(f"[GPU {gpu_id}] train_master_indices shape: {train_master_indices.shape}")
        print(f"[GPU {gpu_id}] n_test: {n_test}")
        
        # Test data is already at the beginning of our arrays
        X_test = X_full[:n_test]
        y_test = y_full[:n_test]
        
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
            
            # Sync with checkpoint file to get the latest state
            if unique_id in completed_configs:
                continue
                
            try:
                # Get training data - VERY IMPORTANT: Use the first n_train indices consistently
                n_train = config_item['n_train']
                
                # Use the first n_train indices from the master training indices
                # This ensures smaller datasets are proper subsets of larger ones
                # if n_train <= len(train_master_indices):
                #     # Take a slice of the training indices
                #     train_indices = train_master_indices[:n_train]
                #     # Pull those indices from X_full and y_full, offset by n_test
                #     X_train = X_full[n_test:][train_indices]  # Get data from after test set
                #     y_train = y_full[n_test:][train_indices]

                if n_train <= len(train_master_indices):
                    # Take a fixed slice of the training indices (first n_train elements)
                    # This ensures smaller datasets are consistent subsets of larger ones
                    train_indices = train_master_indices[:n_train]
                    X_train = X_full[train_indices]
                    y_train = y_full[train_indices]
                else:
                    X_train = X_full[train_master_indices]
                    y_train = y_full[train_master_indices]
                    #             else:
                    # # Use all available training data if n_train is larger than available
                    # X_train = X_full[n_test:]  # All data after test set
                    # y_train = y_full[n_test:]
                    # print(f"[GPU {gpu_id}] Warning: Requested n_train={n_train} exceeds available training samples ({len(train_master_indices)})")
                
                # Initialize model
                model_seed = hash(f"model_{config_item['model_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}")
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
            except Exception as e:
                print(f"[GPU {gpu_id}] Error initializing model for {unique_id}: {str(e)}")
                traceback.print_exc()
                
                # Log the error
                error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                with open(error_log_path, "a") as err_f:
                    fcntl.flock(err_f, fcntl.LOCK_EX)
                    err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error initializing model for {unique_id}: {str(e)}\n")
                    err_f.write(traceback.format_exc() + "\n")
                    fcntl.flock(err_f, fcntl.LOCK_UN)
        
        if not models:  # All experiments were already completed or failed to initialize
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
                    
                try:
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
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error during training step for model {i}: {str(e)}")
                    early_stop_flags[i] = True  # Mark this model as failed
                    
                    # Log the error
                    error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                    with open(error_log_path, "a") as err_f:
                        fcntl.flock(err_f, fcntl.LOCK_EX)
                        err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error during training for {unique_ids[i]}: {str(e)}\n")
                        err_f.write(traceback.format_exc() + "\n")
                        fcntl.flock(err_f, fcntl.LOCK_UN)
            
            # Evaluate periodically
            if (epoch + 1) % eval_interval == 0 or epoch == max_epochs - 1:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                        for i, model in enumerate(models):
                            if early_stop_flags[i]:
                                continue
                                
                            try:
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
                            except Exception as e:
                                print(f"[GPU {gpu_id}] Error during evaluation for model {i}: {str(e)}")
                                early_stop_flags[i] = True  # Mark this model as failed
                                
                                # Log the error
                                error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                                with open(error_log_path, "a") as err_f:
                                    fcntl.flock(err_f, fcntl.LOCK_EX)
                                    err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error during evaluation for {unique_ids[i]}: {str(e)}\n")
                                    err_f.write(traceback.format_exc() + "\n")
                                    fcntl.flock(err_f, fcntl.LOCK_UN)
        
        # Add fine tuning phase - always run full fine-tuning
        fine_tuning_epochs = base_config.get("fine_tuning_epochs", 500)
        
        # Do fine-tuning phase for all models
        for i, model in enumerate(models):
            if early_stop_flags[i]:
                continue
                
            try:
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
            except Exception as e:
                print(f"[GPU {gpu_id}] Error during fine-tuning for model {i}: {str(e)}")
                early_stop_flags[i] = True  # Mark this model as failed
                
                # Log the error
                error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                with open(error_log_path, "a") as err_f:
                    fcntl.flock(err_f, fcntl.LOCK_EX)
                    err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error during fine-tuning for {unique_ids[i]}: {str(e)}\n")
                    err_f.write(traceback.format_exc() + "\n")
                    fcntl.flock(err_f, fcntl.LOCK_UN)
        
        # Save results
        completed_count = 0
        for i in range(len(models)):
            try:
                model = models[i]
                config_item = config_items[i]
                X_train, y_train = train_data[i]
                unique_id = unique_ids[i]
                
                # Skip if early stopping flag is set (indicating failure)
                if early_stop_flags[i] and len(train_errors[i]) <= 1:
                    print(f"[GPU {gpu_id}] Skipping result for model {i} due to training failure")
                    continue
                
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
                
                # Save result - changed dataset references to model references
                result = {
                    'model_name': config_item['model_name'],
                    'model_path': config_item['model_path'],
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
                    'model_seed': hash(f"model_{config_item['model_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}"),
                    'experiment_num': config_item['experiment_num'],
                    'sweep_name': config_item['sweep_name'],
                    'parallel_trained': True,
                    'batch_size': batch_size,
                    'scaled_lr': scaled_lr,  # Include scaled LR in results
                    'batch_size_ratio': batch_size_ratio,
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                }
                
                # Save results with safe file handling
                results_file_path = os.path.join(full_results_dir, f"results_{timestamp}_gpu{gpu_id}.jsonl")
                try:
                    with open(results_file_path, "a") as f:
                        # Acquire an exclusive lock
                        fcntl.flock(f, fcntl.LOCK_EX)
                        
                        # Write the result
                        f.write(json.dumps(result) + "\n")
                        
                        # Ensure data is written to disk
                        f.flush()
                        os.fsync(f.fileno())
                        
                        # Release the lock
                        fcntl.flock(f, fcntl.LOCK_UN)
                    
                    # Mark as completed with safe checkpoint update
                    if safe_update_checkpoint(checkpoint_log_path, unique_id):
                        completed_configs.add(unique_id)
                        completed_count += 1
                    else:
                        print(f"[GPU {gpu_id}] Warning: Failed to update checkpoint for {unique_id}")
                
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error saving results for model {i}: {str(e)}")
                    traceback.print_exc()
                    
                    # Log the error
                    error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                    with open(error_log_path, "a") as err_f:
                        fcntl.flock(err_f, fcntl.LOCK_EX)
                        err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error saving results for {unique_id}: {str(e)}\n")
                        err_f.write(traceback.format_exc() + "\n")
                        fcntl.flock(err_f, fcntl.LOCK_UN)
            
            except Exception as e:
                print(f"[GPU {gpu_id}] Error finalizing model {i}: {str(e)}")
                traceback.print_exc()
                
                # Log the error
                if i < len(unique_ids):
                    id_info = f" ({unique_ids[i]})"
                else:
                    id_info = ""
                    
                error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
                with open(error_log_path, "a") as err_f:
                    fcntl.flock(err_f, fcntl.LOCK_EX)
                    err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error finalizing model {i}{id_info}: {str(e)}\n")
                    err_f.write(traceback.format_exc() + "\n")
                    fcntl.flock(err_f, fcntl.LOCK_UN)
        
        return completed_count
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error in fast_parallel_training: {str(e)}")
        traceback.print_exc()
        
        # Try to log the error
        try:
            error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
            with open(error_log_path, "a") as err_f:
                fcntl.flock(err_f, fcntl.LOCK_EX)
                err_f.write(f"{datetime.now()}: [GPU {gpu_id}] Error in fast_parallel_training: {str(e)}\n")
                err_f.write(traceback.format_exc() + "\n")
                fcntl.flock(err_f, fcntl.LOCK_UN)
        except:
            # If even the error logging fails, just continue
            pass
            
        return 0  # Return 0 completed experiments on error


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
            completed_configs = load_checkpoint_set(checkpoint_log_path)
            timestamp = os.path.basename(restart_checkpoint).replace("checkpoint_", "").replace(".txt", "")
            print(f"Restarting from checkpoint with {len(completed_configs)} completed configurations")
        else:
            if os.path.exists(checkpoint_log_path):
                completed_configs = load_checkpoint_set(checkpoint_log_path)
                print(f"Using existing checkpoint with {len(completed_configs)} completed configs")
            else:
                completed_configs = set()
                # Create an empty checkpoint file
                with open(checkpoint_log_path, "w") as f:
                    pass
                print(f"Starting new run")
            
            # Save hyperparameters
            hyperparams_path = os.path.join(full_results_dir, f"hyperparameters_{timestamp}.yaml")
            with open(hyperparams_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # Generate all combinations
        print("Generating parameter combinations...")
        all_combinations = generate_all_combinations(config)
        print(f"Generated {len(all_combinations)} combinations")
        
        # Initialize global stats
        global_stats["total_experiments"] = len(all_combinations)
        
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
        
        # Create error log file
        error_log_path = os.path.join(full_results_dir, f"error_log_{timestamp}.txt")
        with open(error_log_path, "w") as err_f:
            err_f.write(f"=== Error Log Started at {datetime.now()} ===\n")
            err_f.write(f"Processing {len(all_combinations)} total combinations\n")
            err_f.write(f"{len(remaining)} configurations remaining to process\n\n")
        
        # Launch one process per GPU
        mp.spawn(
            worker_process,
            args=(num_gpus, all_combinations, config, full_results_dir, timestamp, checkpoint_log_path, completed_configs),
            nprocs=num_gpus,
            join=True
        )
        
        # Final report after all GPUs finish
        try:
            # Count results files
            result_files = glob.glob(os.path.join(full_results_dir, f"results_{timestamp}_gpu*.jsonl"))
            result_count = 0
            for file_path in result_files:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            result_count += 1
            
            # Count completed experiments in checkpoint
            final_completed = load_checkpoint_set(checkpoint_log_path)
            
            # Generate summary report
            summary_path = os.path.join(full_results_dir, f"final_summary_{timestamp}.txt")
            with open(summary_path, "w") as f:
                f.write(f"=== Final Summary at {datetime.now()} ===\n")
                f.write(f"Total elapsed time: {time.time() - start_time:.2f} seconds\n")
                f.write(f"Total combinations: {len(all_combinations)}\n")
                f.write(f"Total completed (per checkpoint): {len(final_completed)}\n")
                f.write(f"Total results saved: {result_count}\n")
                f.write(f"Missing results: {len(final_completed) - result_count}\n")
                f.write(f"Results files: {len(result_files)}\n")
                
                # List all GPUs summaries
                gpu_summaries = glob.glob(os.path.join(full_results_dir, f"summary_{timestamp}_gpu*.json"))
                f.write(f"\nGPU Summaries ({len(gpu_summaries)}):\n")
                for gpu_sum in gpu_summaries:
                    try:
                        with open(gpu_sum, 'r') as gsf:
                            summary = json.load(gsf)
                            f.write(f"  GPU {summary.get('gpu_id')}: completed={summary.get('completed')}, "
                                  f"skipped={summary.get('skipped')}, failed={summary.get('failed')}\n")
                    except:
                        f.write(f"  Error reading {os.path.basename(gpu_sum)}\n")
            
            print(f"All processes completed in {time.time() - start_time:.2f} seconds")
            print(f"Final summary saved to {summary_path}")
            print(f"Results saved: {result_count}")
            print(f"Completed (per checkpoint): {len(final_completed)}")
            print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            print(f"Error generating final summary: {str(e)}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()