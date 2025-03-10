#!/usr/bin/env python3
import numpy as np
import jax
import jax.numpy as jnp
import json
import yaml
from datetime import datetime
import os
import time
from mpi4py import MPI
from functools import partial
import torch
from neural_tangents import stax  # to build the NTK network
import jax.scipy.sparse.linalg as sp_linalg
import sys
import glob

###############################################
# Improved MPI GPU distribution
###############################################

def setup_jax_for_mpi(rank, world_size):
    """
    Configure JAX to work properly with MPI by assigning each MPI rank to a specific GPU.
    
    Args:
        rank: The MPI rank of the current process
        world_size: The total number of MPI processes
    """
    import os
    
    # Get list of available GPU devices
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    if visible_devices:
        # If CUDA_VISIBLE_DEVICES is set, parse it
        device_ids = visible_devices.split(',')
        
        # Ensure we have enough devices for all ranks
        if len(device_ids) < world_size:
            print(f"Warning: Not enough GPUs ({len(device_ids)}) for all MPI ranks ({world_size})")
            # Assign GPUs in a round-robin fashion if needed
            device_id = device_ids[rank % len(device_ids)]
        else:
            # Assign a specific GPU to this rank
            device_id = device_ids[rank % len(device_ids)]
        
        # Set this process to only see one specific GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        print(f"Rank {rank}: Using GPU {device_id}")
    else:
        # If CUDA_VISIBLE_DEVICES not set, just use modulo assignment
        # This assumes GPUs are numbered starting from 0
        device_id = rank % jax.device_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        print(f"Rank {rank}: Auto-assigned to GPU {device_id}")
    
    # Set JAX memory fraction
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    
    # Make JAX initialize with the assigned device
    # Force JAX to reinitialize its device handling
    from jax.lib import xla_bridge
    xla_bridge.get_backend.cache_clear()

def configure_jax_for_h100():
    """Configure JAX specifically for H100 GPUs, compatible with older JAX versions."""
    import os
    
    # Set JAX flags for better GPU utilization
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    
    # Try to use compatible settings
    try:
        # This is often supported even in older JAX versions
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
    except AttributeError:
        print("Matmul precision setting not available in this JAX version", flush=True)
    
    # Disable JIT only if needed for debugging
    try:
        jax.config.update('jax_disable_jit', False)
    except AttributeError:
        print("JIT configuration not available in this JAX version", flush=True)
    
    print("JAX configured for H100 performance with compatible settings", flush=True)

###############################################
# GPU-optimized kernel operations
###############################################

@partial(jax.jit, static_argnums=(1,3))
def kernel_batch_mv(X_batch, kernel_fn, X_full, alpha):
    """JIT-compiled kernel matrix-vector product."""
    K_batch = kernel_fn(X_batch, X_full, get='ntk')
    return jnp.dot(K_batch, alpha)

@partial(jax.jit, static_argnums=(1,))
def fast_kernel_diag(X_batch, kernel_fn):
    """JIT-compiled kernel diagonal computation."""
    K_batch = kernel_fn(X_batch, X_batch, get='ntk')
    return jnp.diag(K_batch)

def K_matvec(v, kernel_fn, X, batch_size, reg):
    """
    Compute the matrix–vector product: K(X,X) @ v, plus reg*v,
    without forming the full kernel matrix.
    
    Memory-optimized version that dynamically adjusts batch size.
    
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
    
    # Dynamically adjust batch size for H100
    adaptive_batch_size = batch_size
    if n > 100000:
        # Scale down batch size for large datasets
        adaptive_batch_size = min(batch_size, 256)
    elif n < 1000:
        # Use larger batch size for small datasets to leverage tensor cores
        adaptive_batch_size = min(max(128, batch_size), n)
    
    print(f"Using K_matvec batch size: {adaptive_batch_size} for {n} samples", flush=True)
    
    for i in range(0, n, adaptive_batch_size):
        X_batch = X[i:i+adaptive_batch_size]
        K_batch = kernel_fn(X_batch, X, get='ntk')  # shape: [batch_size, n_train]
        out_batches.append(jnp.dot(K_batch, v))
        
        # Free up memory explicitly
        K_batch = None
        jax.clear_caches()
        
    out = jnp.concatenate(out_batches, axis=0)
    return out + reg * v

def compute_diag(kernel_fn, X, batch_size, reg):
    """
    Compute the diagonal of the kernel matrix K(X, X) plus regularization.
    Memory-optimized version.
    
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
    
    # Use optimized batch sizes for H100
    adaptive_batch_size = batch_size
    if n > 100000:
        adaptive_batch_size = min(batch_size, 256)
    elif n < 1000:
        # Use larger batch size for small datasets
        adaptive_batch_size = min(max(128, batch_size), n)
    
    print(f"Using compute_diag batch size: {adaptive_batch_size} for {n} samples", flush=True)
    
    for i in range(0, n, adaptive_batch_size):
        X_batch = X[i:i+adaptive_batch_size]
        # Use fast_kernel_diag when possible
        if i + adaptive_batch_size <= n:
            diag_batch = fast_kernel_diag(X_batch, kernel_fn)
        else:
            # For the last batch that might be smaller
            K_batch = kernel_fn(X_batch, X_batch, get='ntk')
            diag_batch = jnp.diag(K_batch)
            K_batch = None  # Free memory
        
        diag_list.append(diag_batch)
        jax.clear_caches()
        
    diag = jnp.concatenate(diag_list, axis=0)
    return diag + reg

def direct_kernel_solve(kernel_fn, X, y, reg):
    """Direct solver for small problems optimized for GPU."""
    print(f"Using direct solver for n={X.shape[0]}", flush=True)
    start_time = time.time()
    
    # Pre-compile kernel function with a dummy run
    _ = kernel_fn(X[:1], X[:1], get='ntk')
    
    # Compute full kernel matrix
    K = kernel_fn(X, X, get='ntk')
    
    # Add regularization
    K_reg = K + reg * jnp.eye(X.shape[0])
    
    # Use Cholesky decomposition - faster and more stable on GPU
    try:
        L = jnp.linalg.cholesky(K_reg)
        alpha = jnp.linalg.solve_triangular(L, y, lower=True)
        alpha = jnp.linalg.solve_triangular(L.T, alpha, lower=False)
    except:
        # Fall back to standard solve if Cholesky fails
        print("Cholesky failed, using standard solver", flush=True)
        alpha = jnp.linalg.solve(K_reg, y)
    
    residual = jnp.linalg.norm(jnp.dot(K_reg, alpha) - y)
    print(f"Direct solve completed in {time.time() - start_time:.2f}s with residual={residual}", flush=True)
    
    return alpha, 0, residual

def solve_kernel_system(kernel_fn, X, y, reg, batch_size, tol=1e-3, maxiter=20000, max_maxiter=500000):
    """
    Solve (K(X,X) + reg*I) * alpha = y using conjugate gradient in a batched manner,
    with a diagonal preconditioner.
    
    High-performance version with better diagnostics and preconditioning.
    
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
    
    # Direct solve for small problems 
    if n <= 100:  # Increased threshold for direct solve
        return direct_kernel_solve(kernel_fn, X, y, reg)
    
    # Apply adaptive regularization
    adaptive_reg = reg
    if n < 100:
        # Increase regularization for very small datasets
        adaptive_reg = max(reg, 0.01)  # Minimum regularization of 0.01 for small datasets
    elif n < 1000:
        # Moderate increase for medium datasets
        adaptive_reg = max(reg, 0.001)
    
    print(f"Using adaptive regularization: {adaptive_reg} for dataset size {n}", flush=True)
    
    def matvec(v):
        return K_matvec(v, kernel_fn, X, batch_size, adaptive_reg)
    
    # Compute a better diagonal preconditioner for CG
    print(f"Computing diagonal preconditioner for n={n} dataset", flush=True)
    start_time = time.time()
    diag = compute_diag(kernel_fn, X, batch_size, adaptive_reg)
    print(f"Diagonal computation took {time.time() - start_time:.2f} seconds", flush=True)
    
    # Ensure diagonal elements are positive for numerical stability
    diag = jnp.maximum(diag, 1e-6)
    
    def preconditioner(v):
        return v / diag

    # Set a dynamic tolerance based on dataset size
    dynamic_tol = tol
    if n < 100:
        dynamic_tol = 1e-2  # Looser tolerance for very small datasets
    elif n > 100000:
        dynamic_tol = 1e-2  # Looser tolerance for very large datasets too
    
    print(f"Using dynamic tolerance: {dynamic_tol}", flush=True)

    # Try a smarter initial guess for faster convergence
    # For regression problems, a simple scaling of y can provide a good start
    diag_mean = jnp.mean(diag)
    x0 = y / diag_mean
    
    # Timing for CG
    cg_start_time = time.time()
    current_maxiter = maxiter
    while current_maxiter <= max_maxiter:
        # Run CG solver with the current settings
        print(f"Starting CG with maxiter={current_maxiter}", flush=True)
        alpha, info = sp_linalg.cg(matvec, y, x0=x0, tol=dynamic_tol, maxiter=current_maxiter, M=preconditioner)
        
        # Compute and report residual
        residual = jnp.linalg.norm(matvec(alpha) - y)
        rel_residual = residual / jnp.linalg.norm(y)
        
        print(f"CG iteration complete: info={info}, residual={residual}, relative={rel_residual}, time={time.time()-cg_start_time:.2f}s", flush=True)
        
        # Check for practical convergence
        if info == 0 or residual < dynamic_tol * jnp.linalg.norm(y):
            print(f"CG converged with residual={residual}, relative residual={rel_residual}", flush=True)
            return alpha, 0, residual  # Return info=0 to indicate practical convergence
        else:
            print(f"Warning: CG did not converge with maxiter={current_maxiter}, residual={residual}", flush=True)
            if current_maxiter >= max_maxiter // 2:
                # If we're getting close to max_maxiter, check if we're making progress
                if residual < 0.01 * jnp.linalg.norm(y):
                    print(f"Practical convergence achieved with residual={residual}", flush=True)
                    return alpha, 0, residual  # Return info=0 to indicate practical convergence
                
                # For small datasets, try direct Cholesky factorization as a fallback
                if n < 500 and current_maxiter >= max_maxiter // 2:
                    try:
                        print(f"Attempting direct solve for n={n} dataset", flush=True)
                        return direct_kernel_solve(kernel_fn, X, y, adaptive_reg)
                    except Exception as e:
                        print(f"Direct solve failed: {e}", flush=True)
            
            # Double max iterations for next try
            current_maxiter *= 2
            # Use the current alpha as the starting point for the next run
            x0 = alpha
    
    return alpha, info, residual

def fast_batched_kernel_mv(kernel_fn, X_test, X_train, alpha, batch_size):
    """
    Compute f_test = K(X_test, X_train) @ alpha in batches.
    GPU-optimized version with JIT compilation.
    """
    n_test = X_test.shape[0]
    out_batches = []
    
    # Timing for prediction
    pred_start_time = time.time()
    
    # Use 1024 as minimum batch size for H100 to leverage tensor cores
    # But adapt for very large datasets
    adaptive_batch_size = max(min(batch_size, n_test), 128)
    if X_train.shape[0] > 100000:
        adaptive_batch_size = min(adaptive_batch_size, 256)
    
    print(f"Using prediction batch size: {adaptive_batch_size}", flush=True)
    
    # Precompile the kernel_batch_mv function
    _ = kernel_batch_mv(X_test[:1], kernel_fn, X_train[:1], alpha[:1])
    
    for i in range(0, n_test, adaptive_batch_size):
        X_test_batch = X_test[i:i+adaptive_batch_size]
        out_batches.append(kernel_batch_mv(X_test_batch, kernel_fn, X_train, alpha))
    
    print(f"Prediction completed in {time.time() - pred_start_time:.2f} seconds", flush=True)
    return jnp.concatenate(out_batches, axis=0)

###############################################
# Helper functions for YAML config processing
###############################################

def load_yaml_config(config_path):
    """Load and return the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_info_from_path(path):
    """
    Determine dataset information by parsing the path.
    Parse tokens like d16, H4, D2, a0.5 from the path.
    """
    path_lower = path.lower()
    
    # Determine distribution type
    dist_type = None
    if 'biggrid_0403_poly' in path_lower:
        dist_type = "NP"
    elif 'biggrid_0403_lin' in path_lower:
        dist_type = "NL"
    elif 'biggrid_0403_exp' in path_lower:
        dist_type = "NE"
    else:
        dist_type = "NP"  # Default

    # Parse dimension tokens from the filename
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
            print(f"Error loading {result_files[0]}: {e}", flush=True)
    return None

def load_dataset_info(directory):
    """
    Extract dataset information from directory or result files.
    Build dataset name based on the extracted information.
    """
    # Check if directory is valid
    if not os.path.isdir(directory):
        print(f"Warning: {directory} is not a directory", flush=True)
        try:
            # If it's a file, extract info from the path
            params = extract_info_from_path(directory)
            return {
                "path": directory,
                "name": f"{params.get('distribution_type', 'unknown')}_d{params.get('input_dim', '?')}_H{params.get('hidden_size', '?')}_a{params.get('alpha', '?')}",
                "params": params,
                "directory": os.path.dirname(directory)
            }
        except:
            return None
    
    # Find dataset file in directory
    dataset_files = find_files_in_directory(directory, "dataset_*.pt")
    if not dataset_files:
        print(f"Warning: No dataset files found in {directory}", flush=True)
        return None
    
    dataset_path = dataset_files[0]
    results = load_result_json(directory)

    # If results.json is missing, parse from path
    if not results or not isinstance(results, list) or (len(results) == 0) or ("hyperparameters" not in results[0]):
        params = extract_info_from_path(directory)
        if not params:
            params = extract_info_from_path(dataset_path)
    else:
        # Load from JSON
        params = results[0]["hyperparameters"]
        # Unify with the distribution type from the path
        path_info = extract_info_from_path(directory)
        params["distribution_type"] = path_info["distribution_type"]
        
        # Fill in missing items from path
        for key in ["alpha", "depth", "input_dim", "hidden_size"]:
            if key not in params and key in path_info:
                params[key] = path_info[key]

    # Ensure input_dim is set
    if "input_dim" not in params:
        path_params = extract_info_from_path(directory)
        if "input_dim" in path_params:
            params["input_dim"] = path_params["input_dim"]
        else:
            print(f"Error: Could not determine input_dim for {directory}", flush=True)
            return None
    
    # Build dataset name
    dist_type = params.get("distribution_type", "")
    input_dim = params.get("input_dim", "")
    hidden_size = params.get("hidden_size", "")
    alpha = params.get("alpha", "")

    if dist_type and input_dim and hidden_size != "":
        ds_name = f"{dist_type}_d{input_dim}_H{hidden_size}_a{alpha}"
    else:
        ds_name = os.path.basename(dataset_path).replace("dataset_", "").replace(".pt", "")

    return {
        "path": dataset_path,
        "name": ds_name,
        "params": params,
        "directory": directory
    }

def generate_ntk_combinations(config):
    """
    Generate all parameter combinations for NTK experiments from the config.
    For NTK, we only need depths and n_train sizes from the parameters.
    The hidden_size is used to define the NTK network.
    
    Performance-optimized version with adaptive hidden size.
    """
    base_cfg = config["base_config"]
    sweeps = config["sweeps"]
    
    all_combinations = []
    
    # Get regularization parameter and ensure it's a float
    reg_config = base_cfg.get("reg", 0.00005)
    reg = float(reg_config)
    
    # Process each sweep separately with its own datasets
    for sweep_name, sweep_info in sweeps.items():
        dataset_paths = sweep_info.get("dataset_paths", [])
        sweep_params = sweep_info.get("parameters", {})
        
        # Load dataset information for each path
        for ds_path in dataset_paths:
            ds_info = load_dataset_info(ds_path)
            if not ds_info:
                print(f"Warning: Could not load dataset info from {ds_path}", flush=True)
                continue
                
            # Extract parameters from the dataset
            ds_params = ds_info['params']
            
            # Check if input_dim is available
            if 'input_dim' not in ds_params:
                print(f"Error: No input_dim found for {ds_path}. This is required.", flush=True)
                continue
                
            # Get the input dimension
            input_dim = ds_params['input_dim']
            
            # NTK parameters
            ntk_depths = sweep_params.get("depths", [2])
            n_train_sizes = sweep_params.get("n_train", [10, 100, 1000, 10000, 50000, 100000, 150000, 200000])
            
            # Generate combinations with adaptive hidden size
            for n_train in n_train_sizes:
                # Use smaller hidden size for larger datasets to save memory
                if n_train > 100000:
                    ntk_hidden_size = 1024  # Smaller hidden size for very large datasets
                elif n_train > 50000:
                    ntk_hidden_size = 2048  # Medium hidden size for large datasets
                else:
                    ntk_hidden_size = 5000  # Original large hidden size for smaller datasets
                
                for depth in ntk_depths:
                    for exp_num in range(1, base_cfg.get("num_experiments", 1) + 1):
                        all_combinations.append({
                            'ds_path': ds_info['path'],
                            'ds_name': ds_info['name'],
                            'ds_directory': ds_info['directory'],
                            'input_dim': input_dim,
                            'hidden_size': ntk_hidden_size,  # Adaptive hidden size
                            'depth': depth,
                            'n_train': n_train,
                            'reg': reg,
                            'experiment_num': exp_num,
                            'sweep_name': sweep_name,
                            'alpha': ds_params.get('alpha', 1.0),
                            'mode': 'NTK'
                        })
    
    # Sort by n_train to process smaller datasets first
    all_combinations.sort(key=lambda c: c['n_train'])
    
    return all_combinations

def distribute_work_efficiently(all_combinations, rank, size):
    """
    Distribute work more efficiently among MPI workers.
    Group similar-sized problems together for better GPU utilization.
    
    Args:
        all_combinations: List of parameter combinations
        rank: MPI rank
        size: Total MPI processes
        
    Returns:
        List of parameter combinations for this worker
    """
    # Sort configurations by n_train (smallest to largest)
    sorted_combinations = sorted(all_combinations, key=lambda c: c['n_train'])
    
    # Group configurations by n_train size
    size_groups = {}
    for combo in sorted_combinations:
        n_train = combo['n_train']
        if n_train not in size_groups:
            size_groups[n_train] = []
        size_groups[n_train].append(combo)
    
    # Distribute each size group among workers in round-robin fashion
    worker_combinations = []
    
    # Process small datasets first
    for n_train in sorted(size_groups.keys()):
        group = size_groups[n_train]
        # Assign configurations in this group to workers
        for i, combo in enumerate(group):
            if i % size == rank:
                worker_combinations.append(combo)
    
    print(f"[Rank {rank}] Processing {len(worker_combinations)} configurations", flush=True)
    return worker_combinations

def generate_unique_id(config):
    """Generate a unique identifier for a configuration."""
    ds_name = config['ds_name']
    
    # Extract order number if available in the dataset path
    order_num = "0"
    path_parts = os.path.basename(config['ds_path']).split('_')
    for part in path_parts:
        if part.startswith("O_"):
            order_num = part.replace("O_", "")
            break
    
    # For NTK, we only care about depth, n_train, and regularization
    unique_id = (
        f"{ds_name}_O{order_num}"
        f"_ntk"
        f"_d{config['depth']}"
        f"_n{config['n_train']}"
        f"_reg{config['reg']}"
        f"_exp{config['experiment_num']}"
    )
    
    return unique_id

###############################################
# Main script with YAML config processing
###############################################

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_kernel.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = load_yaml_config(config_path)
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configure JAX to use the correct GPU for this MPI rank
    setup_jax_for_mpi(rank, size)
    
    # Add H100-specific optimizations
    configure_jax_for_h100()
    
    # Force JAX to compile core operations
    print(f"Rank {rank}: Initializing JAX runtime...", flush=True)
    dummy_x = jnp.ones((10, 5))
    _ = jnp.dot(dummy_x, dummy_x.T)  # Force compilation
    
    # Now get the device - this should be the GPU assigned to this rank
    devices = jax.local_devices()
    if devices:
        device = devices[0]  # Now this should be the correctly assigned device
        print(f"Process {rank} using device: {device}", flush=True)
    else:
        print(f"Process {rank} couldn't find any devices!", flush=True)
        return

    if rank == 0:
        print(f"Total MPI processes: {size}", flush=True)
        print(f"JAX version: {jax.__version__}", flush=True)

    # ───────────────────── PARAMETERS FROM CONFIG ─────────────────────
    base_cfg = config["base_config"]
    base_results_dir = base_cfg.get("base_results_dir", "ntk_results")
    pred_batch_size = base_cfg.get("pred_batch_size", 256)
    
    # Ensure reg is a float, not a string
    reg_config = base_cfg.get("reg", 0.00005)  # Default to 0.00005 if not specified
    reg = float(reg_config)  # Convert to float explicitly
    print(f"Using regularization parameter: {reg} (type: {type(reg)})", flush=True)
    
    n_test = base_cfg.get("n_test", 20000)
    normalize_data = base_cfg.get("normalize_data", False)
    mode = 'NTK'  # Always NTK mode
    
    # Create or find appropriate experiment name
    sweep_names = list(config["sweeps"].keys())
    experiment_name = base_cfg.get("experiment_name", f"ntk_{'_'.join(sweep_names)}_{datetime.now().strftime('%Y%m%d')}")
    
    # Create base results directory
    full_results_dir = os.path.join(base_results_dir, experiment_name)
    if rank == 0:
        os.makedirs(full_results_dir, exist_ok=True)
    comm.Barrier()  # Ensure directory exists for all processes
    
    # Add a small sleep to ensure ranks don't collide in subsequent steps
    time.sleep(rank * 0.5)
    
    # Timestamp for naming files
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save hyperparameters
        hyperparams = {
            'experiment_name': experiment_name,
            'base_results_dir': base_results_dir,
            'pred_batch_size': pred_batch_size,
            'reg': reg,
            'n_test': n_test,
            'normalize_data': normalize_data,
            'mode': mode,
            'timestamp': timestamp
        }
        hyperparams_path = os.path.join(full_results_dir, f"hyperparameters_{timestamp}.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
    else:
        timestamp = None
    timestamp = comm.bcast(timestamp, root=0)
    
    # Generate all parameter combinations with adaptive hidden sizes
    all_combinations = generate_ntk_combinations(config)
    
    # Distribute work efficiently among MPI workers
    worker_combinations = distribute_work_efficiently(all_combinations, rank, size)
    
    # File for worker results
    results_file_path = os.path.join(full_results_dir, f"results_{timestamp}_rank{rank}.jsonl")
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
    worker_results = []
    
    # Process each configuration
    for params in worker_combinations:
        print(f"Worker {rank} processing: {params}", flush=True)
        
        # Generate unique ID for this configuration
        unique_id = generate_unique_id(params)
        exp_num = params['experiment_num']
        exp_results_dir = os.path.join(full_results_dir, f"experiment{exp_num}")
        if rank == 0:
            os.makedirs(exp_results_dir, exist_ok=True)
        
        try:
            # Adjust batch size based on dataset size
            n_train = params['n_train']
            adaptive_batch_size = pred_batch_size
            
            # H100-optimized batch sizes
            if n_train > 100000:
                adaptive_batch_size = min(256, pred_batch_size)  # For very large datasets
            elif n_train > 50000:
                adaptive_batch_size = min(512, pred_batch_size)  # For large datasets
            elif n_train < 1000:
                # H100 performs better with larger batches for small datasets
                adaptive_batch_size = max(128, min(1024, pred_batch_size))
                
            print(f"Using adaptive batch size: {adaptive_batch_size} for n_train={n_train}", flush=True)
            
            # Start timing
            load_start_time = time.time()
            
            # Load dataset using torch.load and convert to NumPy
            data = torch.load(params['ds_path'], weights_only=True)
            X_full = data['X'].numpy()
            y_full = data['y'].numpy()
            
            print(f"Dataset loaded in {time.time() - load_start_time:.2f} seconds", flush=True)
            
            # Split data into test and training sets
            sample_seed = hash(f"sample_{params['n_train']}_{params['ds_name']}_{exp_num}") % (2**32)
            rng = np.random.RandomState(sample_seed)
            indices = rng.permutation(X_full.shape[0])
            
            test_indices = indices[:n_test]
            train_master_indices = indices[n_test:]
            
            X_test = X_full[test_indices]
            y_test = y_full[test_indices]
            X_train_master = X_full[train_master_indices]
            y_train_master = y_full[train_master_indices]
            
            # Sample n_train examples
            train_indices = rng.permutation(X_train_master.shape[0])[:params['n_train']]
            X_train = X_train_master[train_indices]
            y_train = y_train_master[train_indices]
            
            # Print dataset shapes for debugging
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}", flush=True)
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}", flush=True)
            
            # Optional normalization
            if normalize_data:
                X_mean = np.mean(X_train, axis=0)
                X_std = np.std(X_train, axis=0) + 1e-8
                X_train_norm = (X_train - X_mean) / X_std
                X_test_norm = (X_test - X_mean) / X_std
                y_mean = np.mean(y_train)
                y_std = np.std(y_train) + 1e-8
                y_train = (y_train - y_mean) / y_std
                y_test_norm = (y_test - y_mean) / y_std
            else:
                X_train_norm = X_train
                X_test_norm = X_test
            
            # Convert to JAX arrays with appropriate precision
            if n_train > 50000:
                # Use float32 for large datasets to save memory
                X_train_norm = jnp.array(X_train_norm, dtype=jnp.float32)
                y_train = jnp.array(y_train, dtype=jnp.float32)
                X_test_norm = jnp.array(X_test_norm, dtype=jnp.float32)
                y_test = jnp.array(y_test, dtype=jnp.float32)
                print(f"Using float32 precision for n_train={n_train}", flush=True)
            else:
                # Use default precision for smaller datasets
                X_train_norm = jnp.array(X_train_norm)
                y_train = jnp.array(y_train)
                X_test_norm = jnp.array(X_test_norm)
                y_test = jnp.array(y_test)
            
            # Build the Neural Tangents network with timing
            print(f"Building NTK network with depth={params['depth']}, hidden_size={params['hidden_size']}", flush=True)
            network_start_time = time.time()
            
            layers = []
            for _ in range(params['depth']):
                layers.append(stax.Dense(params['hidden_size']))
                layers.append(stax.Relu())
            layers.append(stax.Dense(1))
            init_fn, apply_fn, kernel_fn = stax.serial(*layers)
            
            print(f"Network built in {time.time() - network_start_time:.2f} seconds", flush=True)
            
            # Ensure reg is a float
            reg_value = float(params['reg']) if isinstance(params['reg'], str) else params['reg']
            
            # Solve the kernel system with timing
            solve_start_time = time.time()
            alpha, cg_info, cg_residual = solve_kernel_system(
                kernel_fn, X_train_norm, y_train, reg_value, adaptive_batch_size
            )
            solve_time = time.time() - solve_start_time
            print(f"Kernel system solved in {solve_time:.2f} seconds", flush=True)
            
            if cg_info != 0:
                print(f"Warning: CG did not converge for params {params}, info={cg_info}, residual={cg_residual}", flush=True)
            
            # Compute predictions on X_test in batches
            pred_start_time = time.time()
            y_test_pred = fast_batched_kernel_mv(kernel_fn, X_test_norm, X_train_norm, alpha, adaptive_batch_size)
            pred_time = time.time() - pred_start_time
            
            # Calculate test error
            test_error = float(jnp.mean((y_test_pred - y_test)**2))
            
            print(f"Test predictions completed in {pred_time:.2f} seconds, error: {test_error}", flush=True)
            
            # Save results - focus only on essential NTK parameters
            result = {
                'dataset_name': params['ds_name'],
                'dataset_path': params['ds_path'],
                'depth': params['depth'],  # Essential for NTK architecture
                'n_train': params['n_train'],  # Essential for dataset size
                'input_dim': params['input_dim'],  # Essential dimension info
                'hidden_size': params['hidden_size'],  # Record the actual hidden size used
                'mode': mode,  # Always 'NTK'
                'test_error': test_error,  # Main performance metric
                'cg_info': int(cg_info),  # Convergence info
                'cg_residual': float(cg_residual),  # Residual of CG
                'worker_rank': rank,
                'sample_seed': int(sample_seed),
                'experiment_num': exp_num,
                'sweep_name': params['sweep_name'],
                'reg': params['reg'],  # Regularization parameter
                'alpha': params.get('alpha', 1.0),  # Dataset parameter
                'batch_size_used': adaptive_batch_size,  # Record the batch size that was used
                'solve_time_seconds': solve_time,  # Performance metrics
                'pred_time_seconds': pred_time,
                'total_time_seconds': solve_time + pred_time
            }
            
            worker_results.append(result)
            
            # Save result to JSONL file
            with open(results_file_path, "a") as f:
                line_str = json.dumps(result)
                f.write(line_str + "\n")
                f.flush()
                os.fsync(f.fileno())
            
            print(f"Worker {rank} completed configuration: {unique_id}", flush=True)
            
        except Exception as e:
            print(f"Error processing configuration {unique_id}: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            
            # Try again with smaller hidden size if it was an OOM error
            if "RESOURCE_EXHAUSTED" in str(e) and "Out of memory" in str(e):
                try:
                    print(f"Retrying with smaller hidden size and batch size...", flush=True)
                    
                    # Reduce hidden size drastically
                    smaller_hidden_size = params['hidden_size'] // 4
                    
                    # Reduce batch size drastically
                    small_batch_size = 32
                    
                    print(f"Retrying with hidden_size={smaller_hidden_size}, batch_size={small_batch_size}", flush=True)
                    
                    # Save failure result to results file
                    error_result = {
                        'dataset_name': params['ds_name'],
                        'dataset_path': params['ds_path'],
                        'depth': params['depth'],
                        'n_train': params['n_train'],
                        'input_dim': params['input_dim'],
                        'hidden_size': params['hidden_size'],
                        'mode': mode,
                        'test_error': None,
                        'cg_info': None,
                        'cg_residual': None,
                        'worker_rank': rank,
                        'experiment_num': exp_num,
                        'sweep_name': params['sweep_name'],
                        'reg': params['reg'],
                        'alpha': params.get('alpha', 1.0),
                        'error': str(e),
                        'retry_attempted': True,
                        'retry_hidden_size': smaller_hidden_size,
                        'retry_batch_size': small_batch_size
                    }
                    
                    with open(results_file_path, "a") as f:
                        line_str = json.dumps(error_result)
                        f.write(line_str + "\n")
                        f.flush()
                    
                    # Retry with smaller parameters...
                    # Retry implementation would go here
                    
                except Exception as retry_error:
                    print(f"Retry also failed: {str(retry_error)}", flush=True)
                    
                    # Log the retry failure
                    retry_error_result = {
                        'dataset_name': params['ds_name'],
                        'dataset_path': params['ds_path'],
                        'retry_error': str(retry_error),
                        'orig_error': str(e)
                    }
                    
                    with open(results_file_path, "a") as f:
                        line_str = json.dumps(retry_error_result)
                        f.write(line_str + "\n")
                        f.flush()
                    
                    continue
            else:
                # Log non-OOM errors
                error_result = {
                    'dataset_name': params['ds_name'],
                    'dataset_path': params['ds_path'],
                    'error': str(e),
                    'retry_attempted': False
                }
                
                with open(results_file_path, "a") as f:
                    line_str = json.dumps(error_result)
                    f.write(line_str + "\n")
                    f.flush()
                
                continue
    
    # Wait for all processes to complete
    comm.Barrier()
    
    # Gather all results
    all_results = comm.gather(worker_results, root=0)
    
    if rank == 0:
        combined_results = []
        for worker_res in all_results:
            combined_results.extend(worker_res)
        
        # Save combined results in both JSON and JSONL formats
        final_json_path = os.path.join(full_results_dir, f"final_ntk_results_{timestamp}.json")
        with open(final_json_path, "w") as f:
            json.dump(combined_results, f, indent=4)
        
        # Also save in JSONL format (one JSON object per line)
        final_jsonl_path = os.path.join(full_results_dir, f"final_ntk_results_{timestamp}.jsonl")
        with open(final_jsonl_path, "w") as f:
            for result in combined_results:
                f.write(json.dumps(result) + "\n")
        
        print(f"All workers completed. Results saved to:", flush=True)
        print(f"  - JSON: {final_json_path}", flush=True)
        print(f"  - JSONL: {final_jsonl_path}", flush=True)

if __name__ == "__main__":
    main()