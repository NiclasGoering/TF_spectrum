#!/usr/bin/env python3
import numpy as np
import jax
import jax.numpy as jnp
import json
from datetime import datetime
import os
from mpi4py import MPI
from functools import partial
import torch
from neural_tangents import stax  # to build the NTK network
import jax.scipy.sparse.linalg as sp_linalg

###############################################
# Helper functions for batched kernel operations
###############################################

def K_matvec(v, kernel_fn, X, batch_size, reg):
    """
    Compute the matrix–vector product: K(X,X) @ v, plus reg*v,
    without forming the full kernel matrix.
    
    Args:
      v: vector of shape (n_train,).
      kernel_fn: function to compute the NTK.
      X: training data of shape (n_train, d).
      batch_size: int, batch size.
      reg: regularization scalar.
    Returns:
      A vector of shape (n_train,).
    """
    v = v.reshape(-1)
    n = X.shape[0]
    out_batches = []
    for i in range(0, n, batch_size):
        X_batch = X[i:i+batch_size]
        K_batch = kernel_fn(X_batch, X, get='ntk')  # shape: [batch_size, n_train]
        out_batches.append(jnp.dot(K_batch, v))
    out = jnp.concatenate(out_batches, axis=0)
    return out + reg * v

def compute_diag(kernel_fn, X, batch_size, reg):
    """
    Compute the diagonal of the kernel matrix K(X, X) plus regularization.
    
    Args:
      kernel_fn: function to compute the NTK.
      X: training data of shape (n_train, d).
      batch_size: int, batch size.
      reg: regularization scalar.
      
    Returns:
      diag: vector of shape (n_train,) containing the diagonal elements.
    """
    n = X.shape[0]
    diag_list = []
    for i in range(0, n, batch_size):
        X_batch = X[i:i+batch_size]
        K_batch = kernel_fn(X_batch, X, get='ntk')
        # For each row j in the batch, the diagonal element is at column (i+j)
        diag_batch = [K_batch[j, i+j] for j in range(len(X_batch))]
        diag_list.extend(diag_batch)
    diag = jnp.array(diag_list)
    return diag + reg

def solve_kernel_system(kernel_fn, X, y, reg, batch_size, tol=1e-3, maxiter=20000, max_maxiter=500000):
    """
    Solve (K(X,X) + reg*I) * alpha = y using conjugate gradient in a batched manner,
    with a diagonal preconditioner.
    
    If CG does not converge with the initial maxiter, this function will double the
    maximum iterations and try again, up to max_maxiter.
    
    Args:
      kernel_fn: function returning the NTK.
      X: training data (n_train, d)
      y: training labels (n_train,)
      reg: regularization scalar.
      batch_size: int, batch size for kernel–vector products.
      tol: tolerance for CG.
      maxiter: initial maximum number of iterations.
      max_maxiter: maximum allowed number of iterations.
    Returns:
      alpha: solution vector (n_train,)
      info: CG info (0 if converged; else nonzero)
      residual: final residual norm.
    """
    n = X.shape[0]
    def matvec(v):
        return K_matvec(v, kernel_fn, X, batch_size, reg)
    
    # Compute a simple diagonal preconditioner: M^{-1} approximated by 1 / diag
    diag = compute_diag(kernel_fn, X, batch_size, reg)
    def preconditioner(v):
        return v / diag

    x0 = jnp.zeros(n)
    current_maxiter = maxiter
    while current_maxiter <= max_maxiter:
        alpha, info = sp_linalg.cg(matvec, y, x0=x0, tol=tol, maxiter=current_maxiter, M=preconditioner)
        residual = jnp.linalg.norm(matvec(alpha) - y)
        if info == 0:
            return alpha, info, residual
        else:
            print(f"Warning: CG did not converge with maxiter={current_maxiter}, residual={residual}")
            current_maxiter *= 2
            # Use the current alpha as the starting point for the next run.
            x0 = alpha
    return alpha, info, residual

def batched_kernel_mv(kernel_fn, X_test, X_train, alpha, batch_size):
    """
    Compute f_test = K(X_test, X_train) @ alpha in batches.
    
    Args:
      kernel_fn: function returning the NTK.
      X_test: test data of shape (n_test, d).
      X_train: training data of shape (n_train, d).
      alpha: solution vector (n_train,).
      batch_size: int, batch size.
    Returns:
      f_test: predictions of shape (n_test,).
    """
    n_test = X_test.shape[0]
    out_batches = []
    for i in range(0, n_test, batch_size):
        X_test_batch = X_test[i:i+batch_size]
        K_batch = kernel_fn(X_test_batch, X_train, get='ntk')
        out_batches.append(jnp.dot(K_batch, alpha))
    return jnp.concatenate(out_batches, axis=0)

###############################################
# Main script
###############################################

def main():
    # Initialize MPI.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # JAX device selection.
    device = jax.local_devices()[0]
    if rank == 0:
        print(f"Total MPI processes: {size}")
        print(f"Using device: {device}")
    print(f"Process {rank} using device: {device}")

    # ───────────────────── PARAMETERS ─────────────────────
    experiment_name = "d30_hidden256_NTK_2102_maxiter20k"
    d = 30  # Input dimension.
    hidden_sizes = [5000]
    hidden_sizes.reverse()  # (largest hidden size last)
    depths = [1, 4]
    #depths.reverse()       # e.g. [4, 2, 1]
    n_test = 20000
    pred_batch_size = 256  # Batch size for kernel–vector products.
    reg = 5e-5           # Regularization scalar increased to 1e-5.
    mode = 'NTK'
    shuffled = False
    gamma = 1.0
    num_experiments = 1
    learning_rates = [0.0005]  # For logging.
    n_train_sizes = [2**3, 2**7, 2**9, 2**10, 2**12, 2**14, 2**15, 2**16, 2**17, 2**18]
    #n_train_sizes.reverse()
    normalize_data = False

    model_init = ""
    save_model_flag = False

    # ───────────── ITERABLE DATASETS ─────────────
    dataset_paths = [
        "/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha0.0_20250219_112539/dataset_model_d30_hidden256_depth1_alpha0.0_20250219_112539.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha0.25_20250219_104955/dataset_model_d30_hidden256_depth1_alpha0.25_20250219_104955.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha0.5_20250219_105002/dataset_model_d30_hidden256_depth1_alpha0.5_20250219_105002.pt",
        "/home/goring/TF_spectrum/results_pretrain_testgrid_2/results_2_model_d30_hidden256_depth1_alpha1.0_20250219_105445/dataset_model_d30_hidden256_depth1_alpha1.0_20250219_105445.pt",
        "/home/goring/TF_spectrum/datasets/dataset_model_d30_hidden256_depth1_alpha2.0_20250219_110709.pt",
        "/home/goring/TF_spectrum/datasets/dataset_model_d30_hidden256_depth1_alpha5.0_20250219_105903.pt",
        "/home/goring/TF_spectrum/datasets/dataset_model_d30_hidden256_depth1_alpha10.0_20250219_112240.pt"  
    ]
    dataset_names = ["a0", "a025", "a05", "a1", "a2", "a5", "a10"]
    # ─────────────────────────────────────────────────────

    # Create base results directory.
    base_results_dir = f"/home/goring/TF_spectrum/results_testgrid/{experiment_name}"
    if rank == 0:
        os.makedirs(base_results_dir, exist_ok=True)

    # Timestamp for naming files.
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hyperparams = {
            'd': d,
            'hidden_sizes': hidden_sizes,
            'depths': depths,
            'n_test': n_test,
            'pred_batch_size': pred_batch_size,
            'reg': reg,
            'mode': mode,
            'shuffled': shuffled,
            'n_train_sizes': n_train_sizes,
            'gamma': gamma,
            'num_experiments': num_experiments,
            'dataset_paths': dataset_paths,
            'dataset_names': dataset_names,
            'normalize_data': normalize_data
        }
        hyperparams_path = os.path.join(base_results_dir, f"hyperparameters_{timestamp}.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None
    timestamp = comm.bcast(timestamp, root=0)

    # ───────────── MAIN LOOP OVER DATASETS ─────────────
    for ds_path, ds_name in zip(dataset_paths, dataset_names):
        if rank == 0:
            print(f"\n=== Loading dataset: {ds_path} (name: {ds_name}) ===")
        # MASTER: Load dataset using torch.load and convert to NumPy.
        if rank == 0:
            data = torch.load(ds_path, weights_only=True)
            X_full = data['X']
            y_full = data['y']
            indices = torch.randperm(len(X_full))
            test_indices = indices[:n_test]
            train_master_indices = indices[n_test:]
            X_test = X_full[test_indices]
            y_test = y_full[test_indices]
            X_train_master = X_full[train_master_indices]
            y_train_master = y_full[train_master_indices]
            # Convert to NumPy.
            X_test = X_test.numpy()
            y_test = y_test.numpy()
            X_train_master = X_train_master.numpy()
            y_train_master = y_train_master.numpy()
        else:
            X_train_master = None
            y_train_master = None
            X_test = None
            y_test = None

        X_train_master = comm.bcast(X_train_master, root=0)
        y_train_master = comm.bcast(y_train_master, root=0)
        X_test = comm.bcast(X_test, root=0)
        y_test = comm.bcast(y_test, root=0)

        X_train_master = jnp.array(X_train_master)
        y_train_master = jnp.array(y_train_master)
        X_test = jnp.array(X_test)
        y_test = jnp.array(y_test)
        if rank == 0:
            print(f"Process {rank} – Dataset '{ds_name}' loaded. X_train shape: {X_train_master.shape}, X_test shape: {X_test.shape}")

        # Build hyperparameter combinations.
        all_combinations = []
        for hidden_size in hidden_sizes:
            for depth in depths:
                for n_train in n_train_sizes:
                    for lr in learning_rates:  # lr is logged but not used.
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
        worker_combinations = [all_combinations[i] for i in range(len(all_combinations)) if i % size == rank]

        results_file_path = os.path.join(base_results_dir, f"{ds_name}_results_{timestamp}_rank{rank}.jsonl")
        if os.path.exists(results_file_path):
            os.remove(results_file_path)
        worker_results = []

        # ───────── LOOP OVER COMBINATIONS ─────────
        for params in worker_combinations:
            print(f"Worker {rank} processing: {params}")
            exp_num = params['experiment_num']
            exp_results_dir = os.path.join(base_results_dir, f"experiment{exp_num}")
            if rank == 0:
                os.makedirs(exp_results_dir, exist_ok=True)

            # Sample n_train examples.
            sample_seed = hash(f"sample_{params['n_train']}_{ds_name}_{exp_num}") % (2**32)
            rng = np.random.RandomState(sample_seed)
            indices = rng.permutation(X_train_master.shape[0])[:params['n_train']]
            X_train = X_train_master[indices]
            y_train = y_train_master[indices]

            # Optional normalization.
            if normalize_data:
                X_mean = jnp.mean(X_train, axis=0)
                X_std = jnp.std(X_train, axis=0) + 1e-8
                X_train_norm = (X_train - X_mean) / X_std
                X_test_norm = (X_test - X_mean) / X_std
                y_mean = jnp.mean(y_train)
                y_std = jnp.std(y_train) + 1e-8
                y_train = (y_train - y_mean) / y_std
                y_test_norm = (y_test - y_mean) / y_std
            else:
                X_train_norm = X_train
                X_test_norm = X_test

            if shuffled:
                shuffle_seed = hash(f"shuffle_{params['n_train']}_{ds_name}_{timestamp}_{rank}_{exp_num}") % (2**32)
                rng_shuffle = np.random.RandomState(shuffle_seed)
                perm = rng_shuffle.permutation(y_train.shape[0])
                y_train = y_train[perm]
                params['shuffled'] = True
                params['shuffle_seed'] = shuffle_seed

            # Build the Neural Tangents network.
            layers = []
            for _ in range(params['depth']):
                layers.append(stax.Dense(params['hidden_size']))
                layers.append(stax.Relu())
            layers.append(stax.Dense(1))
            init_fn, apply_fn, kernel_fn = stax.serial(*layers)

            # Solve (K(X_train, X_train) + reg I)*alpha = y_train adaptively.
            alpha, cg_info, cg_residual = solve_kernel_system(kernel_fn, X_train_norm, y_train, reg, pred_batch_size)
            if cg_info != 0:
                print(f"Warning: CG did not converge for params {params}, info={cg_info}, residual={cg_residual}")

            # Compute predictions on X_test in batches.
            y_test_pred = batched_kernel_mv(kernel_fn, X_test_norm, X_train_norm, alpha, pred_batch_size)
            test_error = float(jnp.mean((y_test_pred - y_test)**2))

            initial_train_error = None
            final_train_error = None
            error_history = {'test_error': test_error}

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
                'cg_info': cg_info,
                'cg_residual': float(cg_residual),
                'worker_rank': rank,
                'sample_seed': sample_seed,
                'experiment_num': exp_num,
                'checkpoint_epochs': []
            }
            worker_results.append(result)
            with open(results_file_path, "a") as f:
                line_str = json.dumps(result)
                f.write(line_str + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"Worker {rank} completed configuration: {params}")

        comm.Barrier()
        all_results = comm.gather(worker_results, root=0)
        if rank == 0:
            combined_results = []
            for worker_res in all_results:
                combined_results.extend(worker_res)
            final_results_path = os.path.join(base_results_dir, f"final_results_{ds_name}_{timestamp}.json")
            with open(final_results_path, "w") as f:
                json.dump(combined_results, f, indent=4)
            print(f"All workers completed dataset '{ds_name}'. Results saved: {final_results_path}")

if __name__ == "__main__":
    main()
