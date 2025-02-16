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
from mpi4py import MPI

from FFNN import DeepNN
from utils2 import save_dataset, save_results, save_model
from train2 import train_and_evaluate, shuffle_labels

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set up the device: distribute processes over available GPUs.
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        device = torch.device('cpu')
    else:
        gpu_id = rank % num_gpus
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Number of available GPUs: {num_gpus}")
        print(f"Total processes: {size}")
        print(f"Master process using device: {device}")
    print(f"Process {rank} using device: {device}")

    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PARAMETERS
    experiment_name = "mup_bigtarger_initalpha=5_targetalpha=5.0_nonormalization"
    d = 30
    hidden_sizes = [400]  # You can use larger sizes than the teacher.
    hidden_sizes.reverse()  # (Reverse order if desired.)
    depths = [1]
    n_test = 10000  # Fixed test set size.
    batch_size = 64
    epochs = 5000
    checkpoint_epochs = []  # (If you want to checkpoint intermediate models.)
    weight_decay = 0.0  #1e-4
    mode = 'mup_no_align'  # mup_no_align
    shuffled = False
    n_train_sizes = [10, 100, 1000, 5000, 10000, 20000, 40000, 80000, 100000, 150000]
    n_train_sizes.reverse()
    gamma = 1.0
    num_experiments = 1  # For now, one experiment per configuration.
    learning_rates = [0.0005,0.005,0.05]

    # New hyperparameters for model initialization, saving, and data normalization.
    # If model_init is non-empty, then we load a pre-initialized model.
    # The following block specifies the hyperparameters for that model.
    model_init = "/home/goring/TF_setting/results/results_2_model_d30_hidden400_depth1_alpha5.0_20250211_005109/model_model_d30_hidden400_depth1_alpha5.0_20250211_005109.pt"  
    # Explicit hyperparameters for the pre-initialized model (do not extract from file name)
    init_hidden_size = 400
    init_depth = 1
    init_gamma = 1.0
    init_mode = 'mup_no_align'
    
    save_model_flag = False  # If True, models (initial, checkpoint, final) are saved; if False, only JSONL results are saved.
    
    # New hyperparameter: if True, training and test data will be normalized.
    normalize_data = False

    # Path to the teacher-generated dataset.
    dataset_path = "/home/goring/TF_setting/results/results_2_model_d30_hidden400_depth1_alpha5.0_20250211_005109/dataset_model_d30_hidden400_depth1_alpha5.0_20250211_005109.pt"

    # Create base results directory (only master process does this).
    base_results_dir = f"/home/goring/TF_setting/train_results/{experiment_name}"
    if rank == 0:
        os.makedirs(base_results_dir, exist_ok=True)
        # Create experiment-specific directories.
        for exp_num in range(1, num_experiments + 1):
            exp_dir = os.path.join(base_results_dir, f"experiment{exp_num}")
            os.makedirs(exp_dir, exist_ok=True)
        
        # Save hyperparameters.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hyperparams = {
            'd': d,
            'hidden_sizes': hidden_sizes,
            'depths': depths,
            'n_test': n_test,
            'batch_size': batch_size,
            'epochs': epochs,
            'checkpoint_epochs': checkpoint_epochs,
            'learning_rates': learning_rates,
            'weight_decay': weight_decay,
            'mode': mode,
            'shuffled': shuffled,
            'n_train_sizes': n_train_sizes,
            'device': str(device),
            'num_workers': size,
            'gamma': gamma,
            'num_experiments': num_experiments,
            'model_init': model_init,
            'save_model': save_model_flag,
            'normalize_data': normalize_data
        }
        hyperparams_path = os.path.join(base_results_dir, f"hyperparameters_{timestamp}.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None

    # Broadcast the timestamp to all workers.
    timestamp = comm.bcast(timestamp, root=0)

    # MASTER: Load the dataset.
    if rank == 0:
        print("Master process loading dataset...")
        data = torch.load(dataset_path)
        X = data['X']  # Assumes key 'X'
        y = data['y']  # Assumes key 'y'

        # Randomly split off a fixed test set.
        indices = torch.randperm(len(X))
        test_indices = indices[:n_test]
        train_master_indices = indices[n_test:]
        X_test = X[test_indices]
        y_test = y[test_indices]
        X_train_master = X[train_master_indices]
        y_train_master = y[train_master_indices]

        # Save the test dataset.
        test_dataset_path = os.path.join(base_results_dir, f"test_dataset_{timestamp}.pt")
        save_dataset(X_test, y_test, test_dataset_path, rank)
        print("Dataset loading complete.")
        
        # Move to CPU for broadcasting.
        X_train_master = X_train_master.cpu()
        y_train_master = y_train_master.cpu()
        X_test = X_test.cpu()
        y_test = y_test.cpu()
    else:
        X_train_master = None
        y_train_master = None
        X_test = None
        y_test = None

    # Broadcast the datasets to all workers.
    X_train_master = comm.bcast(X_train_master, root=0)
    y_train_master = comm.bcast(y_train_master, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)

    # Move datasets to this process's device.
    X_train_master = X_train_master.to(device)
    y_train_master = y_train_master.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    print(f"Process {rank} – Data is on device: {X_train_master.device}")

    # Generate all parameter combinations.
    all_combinations = []
    for hidden_size in hidden_sizes:
        for depth in depths:
            for n_train in n_train_sizes:
                for lr in learning_rates:
                    for exp_num in range(1, num_experiments + 1):
                        all_combinations.append({
                            'hidden_size': hidden_size,
                            'depth': depth,
                            'n_train': n_train,
                            'lr': lr,
                            'gamma': gamma,
                            'experiment_num': exp_num
                        })

    # Distribute the work among workers.
    worker_combinations = [all_combinations[i] for i in range(len(all_combinations)) if i % size == rank]

    # Create a file for saving results incrementally (JSON Lines format).
    results_file_path = os.path.join(base_results_dir, f"results_{timestamp}_rank{rank}.jsonl")
    if os.path.exists(results_file_path):
        os.remove(results_file_path)  # Clear previous results, if any.
    worker_results = []

    # Loop over the parameter combinations assigned to this worker.
    for params in worker_combinations:
        print(f"Worker {rank} processing: {params}")
        exp_num = params['experiment_num']
        exp_results_dir = os.path.join(base_results_dir, f"experiment{exp_num}")

        # Sample from the master training set with a fixed seed for reproducibility.
        sample_seed = hash(f"sample_{params['n_train']}_{exp_num}")
        torch.manual_seed(sample_seed)
        indices = torch.randperm(len(X_train_master), device=device)[:params['n_train']]
        X_train = X_train_master[indices]
        y_train = y_train_master[indices]

        # Normalize (standardize) the training and test data if normalize_data is True.
        if normalize_data:
            X_mean = X_train.mean(dim=0)
            X_std = X_train.std(dim=0)
            X_train_norm = (X_train - X_mean) / X_std
            X_test_norm = (X_test - X_mean) / X_std
        else:
            X_train_norm = X_train
            X_test_norm = X_test

        if shuffled:
            shuffle_seed = hash(f"shuffle_{params['n_train']}_{timestamp}_{rank}_{exp_num}")
            y_train = shuffle_labels(y_train, seed=shuffle_seed)
            params['shuffled'] = True
            params['shuffle_seed'] = shuffle_seed

        # Create a unique model prefix for file naming.
        model_prefix = f"h{params['hidden_size']}_d{params['depth']}_n{params['n_train']}_lr{params['lr']}_g{gamma}_{mode}"
        if shuffled:
            model_prefix += "_shuffled"

        # Save the training dataset for this configuration.
        train_dataset_path = os.path.join(exp_results_dir, f"train_dataset_{model_prefix}_{timestamp}_rank{rank}.pt")
        save_dataset(X_train_norm, y_train, train_dataset_path, rank)

        # Initialize the model.
        # If model_init is provided, load the model from the given path.
        # Use the explicit hyperparameters provided above for the pre-initialized model.
        model_seed = None  # default in case we load a model
        if model_init != "":
            # --- Check for consistency between the training parameters and pre-initialized model ---
            if params['hidden_size'] != init_hidden_size:
                raise ValueError("Mismatch in hidden_size: training parameter is {} but pre-initialized model expects {}."
                                 .format(params['hidden_size'], init_hidden_size))
            if params['depth'] != init_depth:
                raise ValueError("Mismatch in depth: training parameter is {} but pre-initialized model expects {}."
                                 .format(params['depth'], init_depth))
            if gamma != init_gamma:
                raise ValueError("Mismatch in gamma: training gamma is {} but pre-initialized model expects {}."
                                 .format(gamma, init_gamma))
            if mode != init_mode:
                raise ValueError("Mismatch in mode: training mode is {} but pre-initialized model expects {}."
                                 .format(mode, init_mode))
            # --- End of consistency checks ---
            print(f"Worker {rank} loading initial model from {model_init}")
            loaded = torch.load(model_init, map_location=device)
            if isinstance(loaded, dict):
                # Create a new model instance using the pre-initialized model hyperparameters.
                model = DeepNN(d, init_hidden_size, init_depth, mode=init_mode, gamma=init_gamma).to(device)
                model.load_state_dict(loaded)
            else:
                model = loaded
        else:
            model_seed = hash(f"model_{datetime.now()}_{rank}_{exp_num}")
            torch.manual_seed(model_seed)
            print(f"Worker {rank} – Model initialization seed: {model_seed}")
            model = DeepNN(d, params['hidden_size'], params['depth'], mode=mode, gamma=gamma).to(device)

        # Save the initial model if saving is enabled and if a new model was created.
        if save_model_flag and model_init == "":
            initial_model_path = os.path.join(exp_results_dir, f"initial_model_{model_prefix}_{timestamp}_rank{rank}.pt")
            save_model(model, initial_model_path)

        # If saving is disabled, ensure no checkpoint models are saved.
        if not save_model_flag:
            local_checkpoint_epochs = []
        else:
            local_checkpoint_epochs = checkpoint_epochs

        # Train the model.
        test_error, initial_train_error, final_train_error, error_history, checkpoint_models = train_and_evaluate(
            model, X_train_norm, y_train, X_test_norm, y_test,
            batch_size, epochs, local_checkpoint_epochs, params['lr'], weight_decay, mode,
            base_results_dir, timestamp, rank, exp_num, model_prefix, gamma
        )

        # Save the final model if saving is enabled.
        if save_model_flag:
            final_model_path = os.path.join(exp_results_dir, f"final_model_{model_prefix}_{timestamp}_rank{rank}.pt")
            save_model(model, final_model_path)

        # Prepare a result dictionary for this iteration.
        result = {
            'hidden_size': params['hidden_size'],
            'depth': params['depth'],
            'n_train': params['n_train'],
            'learning_rate': params['lr'],
            'mode': mode,
            'gamma': gamma,
            'shuffled': shuffled,
            'shuffle_seed': params.get('shuffle_seed'),
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

        # Append this result to the file (each line is one JSON object).
        with open(results_file_path, "a") as f:
            f.write(json.dumps(result, indent=4) + "\n")

        print(f"Worker {rank} completed configuration: {params}")

    # Wait for all workers.
    comm.Barrier()

    # Gather all results to the master process.
    all_results = comm.gather(worker_results, root=0)
    if rank == 0:
        combined_results = []
        for worker_res in all_results:
            combined_results.extend(worker_res)
        final_results_path = os.path.join(base_results_dir, f"final_results_{timestamp}.json")
        with open(final_results_path, "w") as f:
            json.dump(combined_results, f, indent=4)
        print("All workers completed. Combined results saved.")

if __name__ == "__main__":
    main()
