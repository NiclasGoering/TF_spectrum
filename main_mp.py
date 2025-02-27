#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Set, Tuple
import random
from functools import partial
import json
from datetime import datetime
import os
import sys
from mpi4py import MPI

# Import your model and helper functions.
from FFNN import DeepNN
from utils2 import save_dataset, save_results, save_model
from train2 import train_and_evaluate, shuffle_labels

# Ensure prints flush immediately
print = partial(print, flush=True)

def main():
    # ────────────── Set Restart Checkpoint (None for new run) ──────────────
    # Set to a full path if you wish to restart from an old checkpoint,
    # otherwise set restart_checkpoint = None for a fresh run.
    restart_checkpoint = "/home/goring/TF_spectrum/results/results_correct_scaling/d10_hidden256_standard_2502/checkpoint_20250226_033859.txt"

    # ────────────── MPI and Device Setup ──────────────
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        device = torch.device('cpu')
    else:
        gpu_id = rank % num_gpus
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)

    if rank == 0:
        print(f"[Rank 0] Number of available GPUs: {num_gpus}")
        print(f"[Rank 0] Total MPI processes: {size}")
        print(f"[Rank 0] Master process using device: {device}")
    print(f"[Rank {rank}] Using device: {device}")

    # Enable benchmark mode for CuDNN (faster if input shapes are consistent)
    torch.backends.cudnn.benchmark = True

    # If you need deterministic, you can comment out the next line:
    torch.backends.cudnn.deterministic = False

    torch.set_default_dtype(torch.float32)

    # ────────────── Experiment and Hyperparameter Parameters ──────────────
    experiment_name = "d30_hidden256_ntk_al_2702"
    d = 30

    hidden_sizes = [256,512,1024,2**12,2**13]
    hidden_sizes.reverse()  # e.g. [512, 128, 256]
    depths = [1]
    depths.reverse()       # e.g. [1]

    # Base width for learning rate scaling - reference width for scaling formula
    base_width = 256  # New parameter for LR scaling

    n_test = 20000
    batch_size = 64
    epochs = 3000
    checkpoint_epochs = []  # e.g. [100, 1000]
    weight_decay = 0  # 1e-4
    
    # Mode options: 'standard_lr', 'ntk_lr', 'mup_lr', 'standard', 'ntk', 'mup'
    mode = 'ntk'
    
    # Alignment option for learning rate scaling
    alignment = True
    
    shuffled = False
    gamma = 1.0
    num_experiments = 1
    learning_rates = [0.001]

    n_train_sizes = [2**3, 2**7, 2**9, 2**10, 2**12, 2**14, 2**15, 2**16, 2**17, 2**18]
    n_train_sizes.reverse()

    # If using a pre-initialized model.
    model_init = ""
    init_hidden_size = 400
    init_depth = 1
    init_gamma = 1.0
    init_mode = 'mup_no_align'
    init_alignment = False  # Added for pre-initialized model
    save_model_flag = False   # Whether to save initial/final models.
    normalize_data = False    # Whether to normalize data.

    # ────────────── Dataset Paths and Names ──────────────
    dataset_paths = [
"/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha2.0_20250219_110709/dataset_model_d30_hidden256_depth1_alpha2.0_20250219_110709.pt",
"/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha1.0_20250219_105445/dataset_model_d30_hidden256_depth1_alpha1.0_20250219_105445.pt",
"/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha0.25_20250219_104955/dataset_model_d30_hidden256_depth1_alpha0.25_20250219_104955.pt",
"/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha0.5_20250219_105002/dataset_model_d30_hidden256_depth1_alpha0.5_20250219_105002.pt",
"/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha0.0_20250219_112539/dataset_model_d30_hidden256_depth1_alpha0.0_20250219_112539.pt",
"/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha5.0_20250219_105903/dataset_model_d30_hidden256_depth1_alpha5.0_20250219_105903.pt"
    ]
    dataset_names = ["a2", "a1","a025","a05","a0","a5"]

    # ────────────── Base Results Directory and Timestamp ──────────────
    base_results_dir = f"//home/goring/TF_spectrum/results/results_correct_scaling/{experiment_name}"
    if rank == 0:
        os.makedirs(base_results_dir, exist_ok=True)
    comm.Barrier()  # Ensure directory exists for all processes.

    # Use restart_checkpoint if provided, otherwise generate a new timestamp.
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

    # Save top-level hyperparameters only for a new run.
    if restart_checkpoint is None and rank == 0:
        hyperparams = {
            'd': d,
            'hidden_sizes': hidden_sizes,
            'depths': depths,
            'base_width': base_width,
            'n_test': n_test,
            'batch_size': batch_size,
            'epochs': epochs,
            'checkpoint_epochs': checkpoint_epochs,
            'learning_rates': learning_rates,
            'weight_decay': weight_decay,
            'mode': mode,
            'alignment': alignment,  # Add alignment to hyperparams
            'shuffled': shuffled,
            'n_train_sizes': n_train_sizes,
            'device': str(device),
            'num_workers': size,
            'gamma': gamma,
            'num_experiments': num_experiments,
            'model_init': model_init,
            'save_model': save_model_flag,
            'normalize_data': normalize_data,
            'dataset_paths': dataset_paths,
            'dataset_names': dataset_names
        }
        hyperparams_path = os.path.join(base_results_dir, f"hyperparameters_{timestamp}.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)

    # ────────────── Set up Shared Checkpoint Log ──────────────
    if restart_checkpoint is None:
        checkpoint_log_path = os.path.join(base_results_dir, f"checkpoint_{timestamp}.txt")
        if os.path.exists(checkpoint_log_path):
            with open(checkpoint_log_path, "r") as f:
                completed_configs = set(line.strip() for line in f if line.strip())
        else:
            completed_configs = set()

    # ────────────── Build All Configurations and Distribute Work ──────────────
    all_combinations = []
    for ds_path, ds_name in zip(dataset_paths, dataset_names):
        for hidden_size in hidden_sizes:
            for depth in depths:
                for n_train in n_train_sizes:
                    for lr in learning_rates:
                        for exp_num in range(1, num_experiments + 1):
                            all_combinations.append({
                                'ds_path': ds_path,
                                'ds_name': ds_name,
                                'hidden_size': hidden_size,
                                'depth': depth,
                                'n_train': n_train,
                                'lr': lr,
                                'gamma': gamma,
                                'experiment_num': exp_num,
                                'base_width': base_width,
                                'alignment': alignment  # Add alignment to each config
                            })

    # Each MPI worker processes a subset of configurations (round-robin distribution).
    worker_combinations = [
        config for idx, config in enumerate(all_combinations) if idx % size == rank
    ]
    print(f"[Rank {rank}] Total configurations to process: {len(worker_combinations)}")

    # A cache so that each dataset is loaded only once per worker.
    dataset_cache = {}

    # File for partial results for this worker.
    results_file_path = os.path.join(base_results_dir, f"results_{timestamp}_rank{rank}.jsonl")
    # Only remove the results file if starting a fresh run (not a restart)
    if restart_checkpoint is None and os.path.exists(results_file_path):
        os.remove(results_file_path)
    worker_results = []

    # ────────────── Process Each Hyperparameter Configuration ──────────────
    for config in worker_combinations:
        # Generate a unique identifier for this configuration.
        align_suffix = "_align" if config['alignment'] else ""
        unique_id = (f"{config['ds_name']}_h{config['hidden_size']}_d{config['depth']}_n"
                     f"{config['n_train']}_lr{config['lr']}_exp{config['experiment_num']}{align_suffix}")
        if unique_id in completed_configs:
            print(f"[Rank {rank}] Skipping completed configuration: {unique_id}")
            continue

        ds_path = config['ds_path']
        ds_name = config['ds_name']
        exp_num = config['experiment_num']

        # Load dataset from cache if not already loaded.
        if ds_path not in dataset_cache:
            print(f"[Rank {rank}] Loading dataset '{ds_name}' from {ds_path}")
            data = torch.load(ds_path)
            X_full = data['X']
            y_full = data['y']

            # For a reproducible test/train split, use a fixed seed per dataset.
            fixed_seed = abs(hash(ds_path)) % (2**32)
            generator = torch.Generator()
            generator.manual_seed(fixed_seed)
            indices = torch.randperm(len(X_full), generator=generator)
            test_indices = indices[:n_test]
            train_master_indices = indices[n_test:]
            X_test = X_full[test_indices].to(device)
            y_test = y_full[test_indices].to(device)
            X_train_master = X_full[train_master_indices].cpu()  # Keep on CPU initially.
            y_train_master = y_full[train_master_indices].cpu()
            dataset_cache[ds_path] = {
                'X_test': X_test,
                'y_test': y_test,
                'X_train_master': X_train_master,
                'y_train_master': y_train_master
            }
            print(f"[Rank {rank}] Dataset '{ds_name}' loaded and cached.")
        else:
            X_test = dataset_cache[ds_path]['X_test']
            y_test = dataset_cache[ds_path]['y_test']
            X_train_master = dataset_cache[ds_path]['X_train_master']
            y_train_master = dataset_cache[ds_path]['y_train_master']

        # ───── Sample a Training Subset for This Configuration ─────
        sample_seed = hash(f"sample_{config['n_train']}_{ds_name}_{exp_num}")
        torch.manual_seed(sample_seed)
        X_train_master_device = X_train_master.to(device)
        indices = torch.randperm(len(X_train_master_device))[:config['n_train']]
        X_train = X_train_master_device[indices]
        y_train = y_train_master.to(device)[indices]

        # Optional normalization.
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

        # Optional label shuffling.
        if shuffled:
            shuffle_seed = hash(f"shuffle_{config['n_train']}_{ds_name}_{timestamp}_{rank}_{exp_num}")
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
        model_seed = None
        if model_init != "":
            if config['hidden_size'] != init_hidden_size:
                raise ValueError("Mismatch in hidden_size for pre-initialized model.")
            if config['depth'] != init_depth:
                raise ValueError("Mismatch in depth for pre-initialized model.")
            if gamma != init_gamma:
                raise ValueError("Mismatch in gamma for pre-initialized model.")
            if mode != init_mode:
                raise ValueError("Mismatch in mode for pre-initialized model.")
            if config['alignment'] != init_alignment:
                raise ValueError("Mismatch in alignment for pre-initialized model.")
            print(f"[Rank {rank}] Loading initial model from {model_init}")
            loaded = torch.load(model_init, map_location=device)
            if isinstance(loaded, dict):
                model = DeepNN(d, init_hidden_size, init_depth, 
                              mode=init_mode, 
                              alignment=init_alignment,
                              base_width=base_width,
                              gamma=init_gamma).to(device)
                model.load_state_dict(loaded)
            else:
                model = loaded
        else:
            model_seed = hash(f"model_{ds_name}_{datetime.now()}_{rank}_{exp_num}")
            torch.manual_seed(model_seed)
            print(f"[Rank {rank}] Initializing model with seed: {model_seed}")
            model = DeepNN(d, config['hidden_size'], config['depth'], 
                          mode=mode, 
                          alignment=config['alignment'],
                          base_width=base_width,
                          gamma=gamma).to(device)

        if save_model_flag and model_init == "":
            exp_results_dir = os.path.join(base_results_dir, f"experiment{exp_num}")
            os.makedirs(exp_results_dir, exist_ok=True)
            initial_model_path = os.path.join(exp_results_dir, f"initial_model_{model_prefix}_{timestamp}_rank{rank}.pt")
            save_model(model, initial_model_path)

            # Save the training dataset that the model is trained on.
            dataset_save_path = os.path.join(exp_results_dir, f"dataset_{model_prefix}_{timestamp}_rank{rank}.pt")
            save_dataset(X_train, y_train, dataset_save_path, rank)

        local_checkpoint_epochs = checkpoint_epochs if save_model_flag else []

        # ───── Train and Evaluate the Model ─────
        test_error, initial_train_error, final_train_error, error_history, checkpoint_models = train_and_evaluate(
            model, X_train_norm, y_train_norm, X_test_norm, y_test_norm,
            batch_size, epochs, local_checkpoint_epochs, config['lr'],
            weight_decay, mode, 
            alignment=config['alignment'],  # Pass alignment to train_and_evaluate
            results_dir=base_results_dir, 
            timestamp=timestamp, 
            rank=rank,
            experiment_num=exp_num, 
            model_prefix=model_prefix,
            base_width=base_width
        )

        if save_model_flag:
            final_model_path = os.path.join(
                base_results_dir, f"experiment{exp_num}",
                f"final_model_{model_prefix}_{timestamp}_rank{rank}.pt"
            )
            save_model(model, final_model_path)

        # ───── Record Results ─────
        result = {
            'dataset_name': ds_name,
            'dataset_path': ds_path,
            'hidden_size': config['hidden_size'],
            'depth': config['depth'],
            'base_width': base_width,
            'n_train': config['n_train'],
            'learning_rate': config['lr'],
            'mode': mode,
            'alignment': config['alignment'],  # Add alignment to results
            'gamma': gamma,
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
            'checkpoint_epochs': checkpoint_epochs
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

        print(f"[Rank {rank}] Completed configuration: {config}")

    # Save final aggregated results for this worker.
    with open(os.path.join(base_results_dir, f"final_results_{timestamp}_rank{rank}.json"), "w") as f:
        json.dump(worker_results, f, indent=4)
    print(f"[Rank {rank}] Finished processing. Results saved to {results_file_path}")


if __name__ == "__main__":
    main()