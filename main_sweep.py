#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Any
import random
from functools import partial
import json
import yaml
from datetime import datetime
import os
import sys
import glob
from mpi4py import MPI
import hashlib
from torch.utils.data import TensorDataset, DataLoader

# Import your model and helper functions.
from FFNN import DeepNN
from utils2 import save_dataset, save_results, save_model
from train2 import train_and_evaluate, shuffle_labels
# In both files, add:
from utils2 import GPUTensorDataset

# Optimize CUDA settings
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere+ GPUs

# Optimize memory allocation
if torch.cuda.is_available():
    # Reduce memory fragmentation
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster at the cost of reproducibility

# GPU dataset that stays on the device
class GPUTensorDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)

# Ensure prints flush immediately
print = partial(print, flush=True)

def load_yaml_config(config_path):
    """Load and return the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


import os

def extract_info_from_path(path):
    """
    Determine distribution_type by checking the top-level folder:
      - 'biggrid_0403_poly' => 'NP'
      - 'biggrid_0403_lin'  => 'NL'
      - 'biggrid_0403_exp'  => 'NE'

    Then parse tokens like d16, H4, D2, a0.5 from the path.
    """
    path_lower = path.lower()
    dist_type = None

    # 1) Set distribution_type from top-level folder
    if 'biggrid_0403_poly' in path_lower:
        dist_type = "NP"
    elif 'biggrid_0403_lin' in path_lower:
        dist_type = "NL"
    elif 'biggrid_0403_exp' in path_lower:
        dist_type = "NE"
    else:
        # If somehow not in poly/lin/exp, default to NP or raise an error.
        dist_type = "NP"
        # or raise ValueError(f"Cannot determine distribution from path: {path}")

    # 2) Parse dimension tokens from the final folder name
    basename = os.path.basename(path)
    parts = basename.split('_')

    info = {
        "distribution_type": dist_type
    }

    for i, part in enumerate(parts):
        # d16 => input_dim=16
        if part.startswith('d') and (i+1 < len(parts)) and parts[i+1].startswith('H'):
            try:
                info['input_dim'] = int(part[1:])
            except ValueError:
                pass
        
        # H8 => hidden_size=8
        if part.startswith('H') and (i+1 < len(parts)) and parts[i+1].startswith('D'):
            try:
                info['hidden_size'] = int(part[1:])
            except ValueError:
                pass
        
        # D2 => depth=2
        if part.startswith('D') and (i+1 < len(parts)) and parts[i+1].startswith('a'):
            try:
                info['depth'] = int(part[1:])
            except ValueError:
                pass
        
        # a0.5 => alpha=0.5
        if part.startswith('a'):
            try:
                alpha_str = part[1:]
                info['alpha'] = float(alpha_str)
            except ValueError:
                pass
    
    return info



def find_files_in_directory(directory, pattern):
    """Find files matching a pattern in a directory."""
    return glob.glob(os.path.join(directory, pattern))

def load_result_json(directory):
    """Load the results JSON file from a directory."""
    result_files = find_files_in_directory(directory, "results_*.json")
    if result_files:
        try:
            with open(result_files[0], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {result_files[0]}: {e}")
    return None



import os
import glob
import json

def load_dataset_info(directory):
    """
    1) Load results_*.json if present, else parse from path.
    2) Force distribution_type = NP|NL|NE from the top-level folder logic.
    3) Build ds_name with that distribution_type, plus d16, H8, a0.5, etc.
    """
    def find_files_in_directory(dirpath, pattern):
        return glob.glob(os.path.join(dirpath, pattern))
    
    def load_result_json(dirpath):
        result_files = find_files_in_directory(dirpath, "results_*.json")
        if result_files:
            try:
                with open(result_files[0], "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {result_files[0]}: {e}")
        return None

    # 1) Ensure directory is valid
    if not os.path.isdir(directory):
        print(f"Warning: {directory} is not a directory")
        return None
    
    # 2) Find a dataset_*.pt file
    dataset_files = find_files_in_directory(directory, "dataset_*.pt")
    if not dataset_files:
        print(f"Warning: No dataset files found in {directory}")
        return None
    
    dataset_path = dataset_files[0]  # Just grab the first
    results = load_result_json(directory)

    # 3) If results.json is missing or doesn't have hyperparameters, parse from path
    if not results or not isinstance(results, list) or (len(results) == 0) or ("hyperparameters" not in results[0]):
        params = extract_info_from_path(directory)
        if not params:
            # fallback to dataset filename
            params = extract_info_from_path(dataset_path)
    else:
        # load from JSON
        params = results[0]["hyperparameters"]
        # still unify with the top-level distribution from the path
        path_info = extract_info_from_path(directory)
        params["distribution_type"] = path_info["distribution_type"]
        
        # also fill in missing items from path if needed
        if "alpha" not in params and "alpha" in path_info:
            params["alpha"] = path_info["alpha"]
        if "depth" not in params and "depth" in path_info:
            params["depth"] = path_info["depth"]
        if "input_dim" not in params and "input_dim" in path_info:
            params["input_dim"] = path_info["input_dim"]
        if "hidden_size" not in params and "hidden_size" in path_info:
            params["hidden_size"] = path_info["hidden_size"]

    # 4) Guarantee input_dim is set
    if "input_dim" not in params:
        # try again from path
        path_params = extract_info_from_path(directory)
        if "input_dim" in path_params:
            params["input_dim"] = path_params["input_dim"]
        else:
            print(f"Error: Could not determine input_dim for {directory}")
            return None
    
    # 5) Build ds_name with distribution_type + dXX + HXX + aXX
    dist_type   = params.get("distribution_type", "")
    input_dim   = params.get("input_dim", "")
    hidden_size = params.get("hidden_size", "")
    alpha       = params.get("alpha", "")
    depth       = params.get("depth", None)

    if dist_type and input_dim and hidden_size != "":
        ds_name = f"{dist_type}_d{input_dim}_H{hidden_size}_a{alpha}"
    else:
        ds_name = os.path.basename(dataset_path).replace("dataset_", "").replace(".pt", "")

    print(f"[load_dataset_info] {directory} => {ds_name}, type={dist_type}, depth={depth}, alpha={alpha}")

    return {
        "path": dataset_path,
        "name": ds_name,
        "params": params,
        "directory": directory
    }






def load_dataset_with_cache(ds_path, rank, device):
    """
    Load dataset directly to GPU to avoid CPU-GPU transfers.
    """
    print(f"[Rank {rank}] Loading dataset from {ds_path} directly to GPU")
    try:
        # Load to CPU first, then immediately transfer to GPU
        data = torch.load(ds_path, map_location='cpu')
        
        # Move data to GPU right after loading
        if isinstance(data, dict) and 'X' in data and 'y' in data:
            data['X'] = data['X'].to(device, non_blocking=True)
            data['y'] = data['y'].to(device, non_blocking=True)
            print(f"[Rank {rank}] Dataset loaded and moved to GPU: X shape: {data['X'].shape}, y shape: {data['y'].shape}")
        
        return data
    except Exception as e:
        print(f"[Rank {rank}] Error loading dataset {ds_path}: {e}")
        raise

def generate_all_combinations(config):
    """Generate all parameter combinations from the config sweeps."""
    base_cfg = config["base_config"]
    sweeps = config["sweeps"]
    
    all_combinations = []
    
    # Process each sweep separately with its own datasets
    for sweep_name, sweep_info in sweeps.items():
        dataset_paths = sweep_info.get("dataset_paths", [])
        sweep_params = sweep_info.get("parameters", {})
        
        # Load dataset information for each path
        for ds_path in dataset_paths:
            ds_info = load_dataset_info(ds_path)
            if not ds_info:
                print(f"Warning: Could not load dataset info from {ds_path}")
                continue
                
            # Extract parameters from the dataset
            ds_params = ds_info['params']
            
            # CRITICAL: Check if input_dim is available
            if 'input_dim' not in ds_params:
                print(f"Error: No input_dim found in results.json for {ds_path}. This is required.")
                continue
                
            # Get the input dimension from the dataset's result.json
            input_dim = ds_params['input_dim']
            print(f"Using input_dim={input_dim} from dataset {ds_path}")
            
            # For each hyperparameter to sweep
            for n_train in sweep_params.get("n_train", [ds_params.get('train_size', 1024)]):
                for lr in sweep_params.get("learning_rates", [ds_params.get('learning_rate', 0.001)]):
                    # Use explicitly defined hidden_sizes from the sweep parameters
                    hidden_sizes = sweep_params.get("hidden_sizes", [256])
                        
                    for hidden_size in hidden_sizes:
                        # Use explicitly defined depths from the sweep parameters
                        depths = sweep_params.get("depths", [1])
                            
                        for depth in depths:
                            for mode in sweep_params.get("modes", [ds_params.get('mode', 'standard')]):
                                for alignment in sweep_params.get("alignment", [False]):
                                    for exp_num in range(1, base_cfg.get("num_experiments", 1) + 1):
                                        # Create a combination using dataset parameters and sweep parameters
                                        all_combinations.append({
                                            'ds_path': ds_info['path'],
                                            'ds_name': ds_info['name'],
                                            'ds_directory': ds_info['directory'],
                                            'hidden_size': hidden_size,
                                            'depth': depth,
                                            'input_dim': input_dim,  # From dataset's result.json
                                            'n_train': n_train,
                                            'lr': lr,
                                            'mode': mode,
                                            'gamma': ds_params.get('gamma', 1.0),
                                            'experiment_num': exp_num,
                                            'base_width': sweep_params.get('base_width', 10),
                                            'alignment': alignment,
                                            'sweep_name': sweep_name,
                                            'alpha': ds_params.get('alpha', 1.0)
                                        })
    
    # Sort by n_train to process smaller datasets first
    all_combinations.sort(key=lambda c: c['n_train'])
    
    return all_combinations


def generate_unique_id(config):
    import os
    
    ds_name = config['ds_name']  # e.g. "NP_d16_H4_a0.0"
    ds_path = config['ds_path']
    base_path = os.path.basename(ds_path)
    
    # Extract O_1 or O_2 or O_3, etc.
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



def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = load_yaml_config(config_path)
    
    # ────────────── Extract Base Config ──────────────
    base_cfg = config["base_config"]
    base_results_dir = base_cfg["base_results_dir"]
    restart_checkpoint = base_cfg.get("restart_checkpoint")
    
    # Extract other parameters
    epochs = base_cfg["epochs"]
    batch_size = base_cfg["batch_size"]
    checkpoint_epochs = base_cfg.get("checkpoint_epochs", [])
    weight_decay = base_cfg["weight_decay"]
    n_test = base_cfg["n_test"]
    save_model_flag = base_cfg.get("save_model", False)
    normalize_data = base_cfg.get("normalize_data", False)
    
    # ────────────── MPI and Device Setup ──────────────
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_gpus = torch.cuda.device_count()

    # H100 GPU optimizations
    if torch.cuda.is_available():
        # Calculate workers per GPU and adjust memory fraction
        if num_gpus > 0:
            workers_per_gpu = size // num_gpus
            mem_fraction = 0.95 / workers_per_gpu  # Scale down for more workers
            torch.cuda.memory.set_per_process_memory_fraction(mem_fraction)
        
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_mem_efficient_sdp = True  # H100-specific
        torch.backends.cudnn.benchmark = True
        
        # GPU assignment
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)  # Set before any tensor operations
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[Rank {rank}] Assigned to GPU {gpu_id}, {workers_per_gpu} workers per GPU, memory fraction: {mem_fraction:.3f}")
    else:
        device = torch.device('cpu')

    if rank == 0:
        print(f"[Rank 0] Number of available GPUs: {num_gpus}")
        print(f"[Rank 0] Total MPI processes: {size}")
        print(f"[Rank 0] Master process using device: {device}")

    # Enable benchmark mode for CuDNN (faster if input shapes are consistent)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_default_dtype(torch.float32)
    
    # ────────────── Generate Experiment Name Based on Sweeps ──────────────
    sweep_names = list(config["sweeps"].keys())
    experiment_name = f"{'_'.join(sweep_names)}_exp_{datetime.now().strftime('%Y%m%d')}"
    
    # ────────────── Set up Results Directory ──────────────
    full_results_dir = os.path.join(base_results_dir, experiment_name)
    if rank == 0:
        os.makedirs(full_results_dir, exist_ok=True)
    comm.Barrier()  # Ensure directory exists for all processes.

    # ────────────── Set up Checkpointing ──────────────
    if restart_checkpoint is not None:
        # Use the provided checkpoint file directly.
        checkpoint_log_path = restart_checkpoint
        with open(checkpoint_log_path, "r") as f:
            completed_configs = set(line.strip() for line in f if line.strip())
        # Extract timestamp from the checkpoint file name
        timestamp = os.path.basename(restart_checkpoint).replace("checkpoint_", "").replace(".txt", "")
    else:
        if rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = None
        timestamp = comm.bcast(timestamp, root=0)
        
        checkpoint_log_path = os.path.join(full_results_dir, f"checkpoint_{timestamp}.txt")
        if os.path.exists(checkpoint_log_path):
            with open(checkpoint_log_path, "r") as f:
                completed_configs = set(line.strip() for line in f if line.strip())
        else:
            completed_configs = set()

    # Save hyperparameters for a new run
    if restart_checkpoint is None and rank == 0:
        hyperparams_path = os.path.join(full_results_dir, f"hyperparameters_{timestamp}.yaml")
        with open(hyperparams_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # ────────────── Generate and Distribute Work ──────────────
    all_combinations = generate_all_combinations(config)
    
    # Each MPI worker processes a subset of configurations (round-robin distribution).
    worker_combinations = [
        config for idx, config in enumerate(all_combinations) if idx % size == rank
    ]
    print(f"[Rank {rank}] Total configurations to process: {len(worker_combinations)}")

    # A cache for GPU datasets
    dataset_cache = {}

    # File for partial results for this worker.
    results_file_path = os.path.join(full_results_dir, f"results_{timestamp}_rank{rank}.jsonl")
    # Only remove the results file if starting a fresh run (not a restart)
    if restart_checkpoint is None and os.path.exists(results_file_path):
        os.remove(results_file_path)
    worker_results = []

    # ────────────── Process Each Hyperparameter Configuration ──────────────
    for config in worker_combinations:
        # Generate a unique identifier for this configuration.
        unique_id = generate_unique_id(config)
        
        if unique_id in completed_configs:
            print(f"[Rank {rank}] Skipping completed configuration: {unique_id}")
            continue

        ds_path = config['ds_path']
        ds_name = config['ds_name']
        mode = config['mode']
        exp_num = config['experiment_num']
        d = config['input_dim']
        gamma = config.get('gamma', 1.0)
        base_width = config.get('base_width', 10)

        # Load dataset to GPU directly
        if ds_path not in dataset_cache:
            print(f"[Rank {rank}] Loading dataset '{ds_name}' from {ds_path}")
            try:
                # Load the dataset directly to GPU
                data = load_dataset_with_cache(ds_path, rank, device)
                
                if isinstance(data, dict) and 'X' in data and 'y' in data:
                    X_full = data['X']  # Already on GPU
                    y_full = data['y']  # Already on GPU
                    print(f"[Rank {rank}] Dataset loaded: X shape: {X_full.shape}, y shape: {y_full.shape}")
                else:
                    print(f"[Rank {rank}] WARNING: Unknown dataset format in {ds_path}")
                    continue

                # For a reproducible test/train split, use a fixed seed per dataset.
                fixed_seed = abs(hash(ds_path)) % (2**32)
                generator = torch.Generator(device=device)
                generator.manual_seed(fixed_seed)
                
                # Create indices on GPU
                indices = torch.randperm(len(X_full), device=device, generator=generator)
                test_indices = indices[:n_test]
                train_master_indices = indices[n_test:]
                
                # Keep everything on GPU
                X_test = X_full[test_indices]
                y_test = y_full[test_indices]
                X_train_master = X_full[train_master_indices]
                y_train_master = y_full[train_master_indices]
                
                # Store in GPU cache
                dataset_cache[ds_path] = {
                    'X_test': X_test,
                    'y_test': y_test,
                    'X_train_master': X_train_master,
                    'y_train_master': y_train_master
                }
                print(f"[Rank {rank}] Dataset '{ds_name}' loaded and cached on GPU.")
            except Exception as e:
                print(f"[Rank {rank}] ERROR loading dataset '{ds_name}': {str(e)}")
                continue
        else:
            X_test = dataset_cache[ds_path]['X_test']
            y_test = dataset_cache[ds_path]['y_test']
            X_train_master = dataset_cache[ds_path]['X_train_master']
            y_train_master = dataset_cache[ds_path]['y_train_master']

        # ───── Sample a Training Subset for This Configuration ─────
        sample_seed = hash(f"sample_{config['n_train']}_{ds_name}_{exp_num}")
        torch.manual_seed(sample_seed)
        
        # Sample training data directly on GPU
        if config['n_train'] < len(X_train_master):
            # Create indices on GPU
            indices = torch.randperm(len(X_train_master), device=device)[:config['n_train']]
            X_train = X_train_master[indices]
            y_train = y_train_master[indices]
        else:
            # Use all available training data if requested size exceeds available data
            X_train = X_train_master
            y_train = y_train_master

        # Optional normalization (staying on GPU)
        if normalize_data:
            X_mean = X_train.mean(dim=0)
            X_std = torch.clamp(X_train.std(dim=0), min=1e-8)
            y_mean = y_train.mean()
            y_std = torch.clamp(y_train.std(), min=1e-8)
            X_train_norm = (X_train - X_mean) / X_std
            X_test_norm = (X_test - X_mean) / X_std
            y_train_norm = (y_train - y_mean) / y_std
            y_test_norm = (y_test - y_mean) / y_std
        else:
            X_train_norm, X_test_norm = X_train, X_test
            y_train_norm, y_test_norm = y_train, y_test

        # Optional label shuffling (if specified in config)
        shuffled = base_cfg.get("shuffled", False)
        if shuffled:
            shuffle_seed = hash(f"shuffle_{config['n_train']}_{ds_name}_{timestamp}_{rank}_{exp_num}")
            # Shuffle directly on GPU
            y_train_norm = shuffle_labels(y_train_norm, seed=shuffle_seed)
            config['shuffled'] = True
            config['shuffle_seed'] = shuffle_seed

        # Create a prefix for naming files.
        align_tag = "_align" if config['alignment'] else ""
        model_prefix = (
            f"{ds_name}_h{config['hidden_size']}_d{config['depth']}_n{config['n_train']}"
            f"_lr{config['lr']}_g{gamma}_{mode}{align_tag}"
        )
        if shuffled:
            model_prefix += "_shuffled"

        # ───── Model Initialization ─────
        # Always initialize a fresh model with input_dim from dataset
        model_seed = hash(f"model_{ds_name}_{datetime.now()}_{rank}_{exp_num}")
        torch.manual_seed(model_seed)
        
        # Print model configuration details
        print(f"[Rank {rank}] Initializing fresh model: input_dim={d}, hidden_size={config['hidden_size']}, depth={config['depth']}, mode={mode}")
        
        # Initialize with appropriate dimensions from config
        model = DeepNN(d, config['hidden_size'], config['depth'], 
                      mode=mode, 
                      alignment=config['alignment'],
                      base_width=base_width,
                      gamma=gamma).to(device)
        
        # Apply torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, 'compile') and device.type == 'cuda':
            # Fix compilation issues with dynamic batches
            torch._dynamo.config.cache_size_limit = 64  # Increase from default 8
            try:
                # Use only mode parameter, not options
                model = torch.compile(
                    model,
                    mode="reduce-overhead"
                )
                print(f"[Rank {rank}] Model compiled with optimized settings")
            except Exception as e:
                # Fall back to basic compilation if advanced options fail
                model = torch.compile(model)
                print(f"[Rank {rank}] Using basic compilation: {str(e)}")
            
            print(f"[Rank {rank}] Model initialized with seed: {model_seed}")

        if save_model_flag:
            exp_results_dir = os.path.join(full_results_dir, f"experiment{exp_num}")
            os.makedirs(exp_results_dir, exist_ok=True)
            initial_model_path = os.path.join(exp_results_dir, f"initial_model_{model_prefix}_{timestamp}_rank{rank}.pt")
            save_model(model, initial_model_path)

            # Save the training dataset that the model is trained on.
            dataset_save_path = os.path.join(exp_results_dir, f"dataset_{model_prefix}_{timestamp}_rank{rank}.pt")
            save_dataset(X_train, y_train, dataset_save_path, rank)

        local_checkpoint_epochs = checkpoint_epochs if save_model_flag else []

        # Set dynamic batch size based on dataset size
        if config['n_train'] < 1000:
            actual_batch_size = min(batch_size, config['n_train'])
        elif config['n_train'] < 10000:
            actual_batch_size = min(2048, config['n_train'])
        else:
            actual_batch_size = min(16384, config['n_train'])
            
        # Calculate dynamic evaluation interval based on dataset size and epochs
        dynamic_eval_interval = max(10, min(100, epochs // 30))
        
        # ───── Train and Evaluate the Model ─────
        try:
            # Adjust early stop threshold based on dataset size
            early_stop_threshold = 1e-6 if config['n_train'] < 1000 else 1e-4
            
            test_error, initial_train_error, final_train_error, error_history, checkpoint_models = train_and_evaluate(
                model, X_train_norm, y_train_norm, X_test_norm, y_test_norm,
                actual_batch_size, epochs, local_checkpoint_epochs, config['lr'],
                weight_decay, mode, 
                alignment=config['alignment'],
                results_dir=full_results_dir, 
                timestamp=timestamp, 
                rank=rank,
                experiment_num=exp_num, 
                model_prefix=model_prefix,
                base_width=base_width,
                eval_interval=dynamic_eval_interval,
                eval_print_interval=dynamic_eval_interval * 3,
                early_stop_threshold=early_stop_threshold
            )
        except Exception as e:
            print(f"[Rank {rank}] ERROR during training for config {unique_id}: {str(e)}")
            continue

        if save_model_flag:
            exp_results_dir = os.path.join(full_results_dir, f"experiment{exp_num}")
            os.makedirs(exp_results_dir, exist_ok=True)
            final_model_path = os.path.join(
                exp_results_dir, f"final_model_{model_prefix}_{timestamp}_rank{rank}.pt"
            )
            save_model(model, final_model_path)

        # ───── Record Results ─────
        result = {
            'dataset_name': ds_name,
            'dataset_path': ds_path,
            'hidden_size': config['hidden_size'],
            'depth': config['depth'],
            'input_dim': d,
            'base_width': base_width,
            'n_train': config['n_train'],
            'learning_rate': config['lr'],
            'mode': mode,
            'alignment': config['alignment'],
            'gamma': gamma,
            'alpha': config.get('alpha', 1.0),
            'shuffled': shuffled,
            'shuffle_seed': config.get('shuffle_seed'),
            'test_error': test_error,
            'initial_train_error': initial_train_error,
            'final_train_error': final_train_error,
            'error_history': error_history,
            'worker_rank': rank,
            'sample_seed': sample_seed,
            'model_seed': model_seed,
            'experiment_num': exp_num,
            'checkpoint_epochs': checkpoint_epochs,
            'sweep_name': config['sweep_name']
        }
        worker_results.append(result)

        # Append new results in append mode.
        with open(results_file_path, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()
            os.fsync(f.fileno())

        # Append the unique configuration identifier to the shared checkpoint log.
        with open(checkpoint_log_path, "a") as cp_f:
            cp_f.write(unique_id + "\n")
        completed_configs.add(unique_id)

        print(f"[Rank {rank}] Completed configuration: {unique_id}")
        
        # Clear some memory if possible
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save final aggregated results for this worker.
    with open(os.path.join(full_results_dir, f"final_results_{timestamp}_rank{rank}.json"), "w") as f:
        json.dump(worker_results, f, indent=4)
    print(f"[Rank {rank}] Finished processing. Results saved to {results_file_path}")

if __name__ == "__main__":
    main()