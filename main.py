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

    # ───────────────────── PARAMETERS ─────────────────────
    # Experiment name
    experiment_name = "d50_hidden512_NTK_depth_1"

    # Dimension of data (all datasets share the same d)
    d = 50

    # Potential model hidden sizes and depths
    hidden_sizes = [2**8, 2**9, 2**10]   # #[5000,10000]  #[2**8, 2**9, 2**10]  # [5000,10000]
    hidden_sizes.reverse()
    depths = [1,2,4]
    depths.reverse()
    # Basic training parameters
    n_test = 20000
    batch_size = 128  #64
    epochs = 5000
    checkpoint_epochs = []  # e.g. [100, 1000]
    weight_decay = 0.0
    mode = 'mup_no_align'  # or 'NTK', 'spectral', etc. mup_no_align
    shuffled = False
    gamma = 1.0
    num_experiments = 1
    learning_rates = [0.005, 0.0005, 0.05]  # [0.00001,0.000001]

    # Different training set sizes to loop over
    n_train_sizes = [2**3,2**7,2**9,2**10,2**12,2**14,2**15,2**16,2**17,2**18]
    n_train_sizes.reverse()

    # If using a pre-initialized model, specify path and relevant meta
    model_init = ""
    init_hidden_size = 400
    init_depth = 1
    init_gamma = 1.0
    init_mode = 'mup_no_align'
    save_model_flag = False  # Whether to save the final/initial model
    normalize_data = False    # Whether to normalize X_train/X_test and y_train/y_test

    # ─────────────────────────────────────────────────────
    #               ITERABLE DATASETS
    dataset_paths = [
        "/home/goring/TF_spectrum/results_pretrain_testgrid/results_2_model_d50_hidden512_depth1_alpha0.0_20250218_005030/dataset_model_d50_hidden512_depth1_alpha0.0_20250218_005030.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid/results_2_model_d50_hidden512_depth1_alpha0.25_20250218_004532/dataset_model_d50_hidden512_depth1_alpha0.25_20250218_004532.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid/results_2_model_d50_hidden512_depth1_alpha0.5_20250218_004702/dataset_model_d50_hidden512_depth1_alpha0.5_20250218_004702.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid/results_2_model_d50_hidden512_depth1_alpha1.0_20250218_010159/dataset_model_d50_hidden512_depth1_alpha1.0_20250218_010159.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid/results_2_model_d50_hidden512_depth1_alpha2.0_20250218_005200/dataset_model_d50_hidden512_depth1_alpha2.0_20250218_005200.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid/results_2_model_d50_hidden512_depth1_alpha5.0_20250218_005108/dataset_model_d50_hidden512_depth1_alpha5.0_20250218_005108.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid/results_2_model_d50_hidden512_depth1_alpha10.0_20250218_005135/dataset_model_d50_hidden512_depth1_alpha10.0_20250218_005135.pt"
    ]
    dataset_names = [
         "a0",
         "a025",
         "a05",
         "a1",
         "a2",
        "a5",
        "a10"
    ]
    # Make sure len(dataset_paths) == len(dataset_names).

    # Create base results directory (only master process does this).
    base_results_dir = f"/home/goring/TF_spectrum/results_testgrid/{experiment_name}"
    if rank == 0:
        os.makedirs(base_results_dir, exist_ok=True)

    # Timestamp for naming files
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save top-level hyperparameters info
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
            'normalize_data': normalize_data,
            'dataset_paths': dataset_paths,
            'dataset_names': dataset_names
        }
        hyperparams_path = os.path.join(base_results_dir, f"hyperparameters_{timestamp}.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None

    # Broadcast the timestamp to all workers
    timestamp = comm.bcast(timestamp, root=0)

    # ─────────────────────────────────────────────────────────
    #        MAIN LOOP OVER DATASETS (ITERABLE DATASETS)
    # ─────────────────────────────────────────────────────────
    for ds_path, ds_name in zip(dataset_paths, dataset_names):
        if rank == 0:
            print(f"\n=== Loading dataset: {ds_path} (name: {ds_name}) ===")

        # MASTER: Load dataset from ds_path
        if rank == 0:
            data = torch.load(ds_path)
            X_full = data['X']  # Expect shape [N, d]
            y_full = data['y']  # Expect shape [N]

            # Shuffle once, so test set is random
            indices = torch.randperm(len(X_full))
            test_indices = indices[:n_test]
            train_master_indices = indices[n_test:]
            X_test = X_full[test_indices]
            y_test = y_full[test_indices]
            X_train_master = X_full[train_master_indices]
            y_train_master = y_full[train_master_indices]

            # Save test dataset for reference
            test_dataset_path = os.path.join(
                base_results_dir,
                f"{ds_name}_test_dataset_{timestamp}.pt"
            )
            #save_dataset(X_test, y_test, test_dataset_path, rank)

            X_train_master = X_train_master.cpu()
            y_train_master = y_train_master.cpu()
            X_test = X_test.cpu()
            y_test = y_test.cpu()
        else:
            X_train_master = None
            y_train_master = None
            X_test = None
            y_test = None

        # Broadcast datasets to all ranks
        X_train_master = comm.bcast(X_train_master, root=0)
        y_train_master = comm.bcast(y_train_master, root=0)
        X_test = comm.bcast(X_test, root=0)
        y_test = comm.bcast(y_test, root=0)

        # Move to GPU (if available)
        X_train_master = X_train_master.to(device)
        y_train_master = y_train_master.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        print(f"Process {rank} – Dataset '{ds_name}' is on device: {X_train_master.device}")

        # Build the list of all (dataset, model hyperparams) combos
        all_combinations = []
        for hidden_size in hidden_sizes:
            for depth in depths:
                for n_train in n_train_sizes:
                    for lr in learning_rates:
                        for exp_num in range(1, num_experiments + 1):
                            all_combinations.append({
                                'ds_name': ds_name,
                                'ds_path': ds_path,
                                'hidden_size': hidden_size,
                                'depth': depth,
                                'n_train': n_train,
                                'lr': lr,
                                'gamma': gamma,
                                'experiment_num': exp_num
                            })

        # Distribute the work among workers
        worker_combinations = [
            all_combinations[i]
            for i in range(len(all_combinations))
            if i % size == rank
        ]

        # File for partial results (one file per rank, per dataset).
        results_file_path = os.path.join(
            base_results_dir,
            f"{ds_name}_results_{timestamp}_rank{rank}.jsonl"
        )
        if os.path.exists(results_file_path):
            os.remove(results_file_path)

        worker_results = []

        # ──────────────────────────────────────────────────────
        #         Train/Evaluate for each combination
        # ──────────────────────────────────────────────────────
        for params in worker_combinations:
            print(f"Worker {rank} processing: {params}")
            exp_num = params['experiment_num']
            # Create an experiment sub-directory for each experiment number
            exp_results_dir = os.path.join(base_results_dir, f"experiment{exp_num}")
            if rank == 0:
                os.makedirs(exp_results_dir, exist_ok=True)

            # Sample from the master training set
            sample_seed = hash(f"sample_{params['n_train']}_{ds_name}_{exp_num}")
            torch.manual_seed(sample_seed)
            indices = torch.randperm(len(X_train_master), device=device)[:params['n_train']]
            X_train = X_train_master[indices]
            y_train = y_train_master[indices]

            # Optional normalization (for both X and y)
            if normalize_data:
                # Normalize X
                X_mean = X_train.mean(dim=0)
                X_std = X_train.std(dim=0)
                X_std = torch.clamp(X_std, min=1e-8)  # Prevent division by zero

                # Normalize y
                y_mean = y_train.mean()
                y_std = y_train.std()
                y_std = torch.clamp(y_std, min=1e-8)  # Prevent division by zero

                # Normalize both train and test using training statistics
                X_train_norm = (X_train - X_mean) / X_std
                X_test_norm = (X_test - X_mean) / X_std

                y_train_norm = (y_train - y_mean) / y_std
                y_test_norm = (y_test - y_mean) / y_std
            else:
                X_train_norm = X_train
                X_test_norm = X_test
                y_train_norm = y_train
                y_test_norm = y_test

            # Optional label shuffling
            if shuffled:
                shuffle_seed = hash(f"shuffle_{params['n_train']}_{ds_name}_{timestamp}_{rank}_{exp_num}")
                y_train_norm = shuffle_labels(y_train_norm, seed=shuffle_seed)
                params['shuffled'] = True
                params['shuffle_seed'] = shuffle_seed

            # Prefix for naming files
            model_prefix = (
                f"{ds_name}_h{params['hidden_size']}"
                f"_d{params['depth']}"
                f"_n{params['n_train']}"
                f"_lr{params['lr']}"
                f"_g{gamma}_{mode}"
            )
            if shuffled:
                model_prefix += "_shuffled"

            # Save the local train subset
            train_dataset_path = os.path.join(
                exp_results_dir,
                f"train_dataset_{model_prefix}_{timestamp}_rank{rank}.pt"
            )
            #save_dataset(X_train_norm, y_train_norm, train_dataset_path, rank)

            # Initialize model
            model_seed = None
            if model_init != "":
                # Check for consistency if using a pre-initialized model
                if params['hidden_size'] != init_hidden_size:
                    raise ValueError("Mismatch in hidden_size for pre-initialized model.")
                if params['depth'] != init_depth:
                    raise ValueError("Mismatch in depth for pre-initialized model.")
                if gamma != init_gamma:
                    raise ValueError("Mismatch in gamma for pre-initialized model.")
                if mode != init_mode:
                    raise ValueError("Mismatch in mode for pre-initialized model.")
                print(f"Worker {rank} loading initial model from {model_init}")
                loaded = torch.load(model_init, map_location=device)
                if isinstance(loaded, dict):
                    model = DeepNN(d, init_hidden_size, init_depth, mode=init_mode, gamma=init_gamma).to(device)
                    model.load_state_dict(loaded)
                else:
                    model = loaded
            else:
                model_seed = hash(f"model_{ds_name}_{datetime.now()}_{rank}_{exp_num}")
                torch.manual_seed(model_seed)
                print(f"Worker {rank} – Model init seed: {model_seed}")
                model = DeepNN(d, params['hidden_size'], params['depth'], mode=mode, gamma=gamma).to(device)

            # Save the initial model if desired
            if save_model_flag and model_init == "":
                initial_model_path = os.path.join(
                    exp_results_dir,
                    f"initial_model_{model_prefix}_{timestamp}_rank{rank}.pt"
                )
                save_model(model, initial_model_path)

            if not save_model_flag:
                local_checkpoint_epochs = []
            else:
                local_checkpoint_epochs = checkpoint_epochs

            # Train the model using the normalized data
            test_error, initial_train_error, final_train_error, error_history, checkpoint_models = train_and_evaluate(
                model, X_train_norm, y_train_norm, X_test_norm, y_test_norm,
                batch_size, epochs, local_checkpoint_epochs, params['lr'],
                weight_decay, mode, base_results_dir, timestamp, rank,
                exp_num, model_prefix, gamma
            )

            # Save final model if desired
            if save_model_flag:
                final_model_path = os.path.join(
                    exp_results_dir,
                    f"final_model_{model_prefix}_{timestamp}_rank{rank}.pt"
                )
                save_model(model, final_model_path)

            # Prepare result dict
            result = {
                'dataset_name': ds_name,
                'dataset_path': ds_path,
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

            # Write partial result (JSON Lines)
            with open(results_file_path, "a") as f:
                line_str = json.dumps(result)
                f.write(line_str + "\n")
                f.flush()
                os.fsync(f.fileno())

            print(f"Worker {rank} completed configuration: {params}")

        # Barrier so that all ranks finish this dataset
        comm.Barrier()

        # Gather partial results to rank=0 (for this dataset)
        all_results = comm.gather(worker_results, root=0)
        if rank == 0:
            combined_results = []
            for worker_res in all_results:
                combined_results.extend(worker_res)

            # Save final combined results for this dataset
            final_results_path = os.path.join(
                base_results_dir,
                f"final_results_{ds_name}_{timestamp}.json"
            )
            with open(final_results_path, "w") as f:
                json.dump(combined_results, f, indent=4)

            print(f"All workers completed dataset '{ds_name}'. Results saved: {final_results_path}")

        # Next dataset (if any)

if __name__ == "__main__":
    main()
