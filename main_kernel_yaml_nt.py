#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import neural_tangents as nt
from neural_tangents import stax
import torch
import time
import os
import json
import yaml
from datetime import datetime
import glob

def configure_jax_for_h100():
    """Configure JAX for H100 GPU."""
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    jax.config.update('jax_default_matmul_precision', 'bfloat16')
    print("JAX configured for H100", flush=True)

def build_ntk_network(input_dim, hidden_size, depth):
    """Build a simple NTK network."""
    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(hidden_size))  # Uses NTK parameterization by default
        layers.append(stax.Relu())
    layers.append(stax.Dense(1))
    
    return stax.serial(*layers)

def load_yaml_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_info_from_path(path):
    """Extract dataset parameters from path."""
    # Determine distribution type
    if 'biggrid_0403_poly' in path.lower():
        dist_type = "NP"
    elif 'biggrid_0403_lin' in path.lower():
        dist_type = "NL"
    elif 'biggrid_0403_exp' in path.lower():
        dist_type = "NE"
    else:
        dist_type = "NP"  # Default

    # Parse dimension tokens
    basename = os.path.basename(path)
    parts = basename.split('_')
    info = {"distribution_type": dist_type}

    for i, part in enumerate(parts):
        if part.startswith('d') and part[1:].isdigit():
            info['input_dim'] = int(part[1:])
        elif part.startswith('H') and part[1:].isdigit():
            info['hidden_size'] = int(part[1:])
        elif part.startswith('D') and part[1:].isdigit():
            info['depth'] = int(part[1:])
        elif part.startswith('a') and i+1 < len(parts):
            try:
                info['alpha'] = float(part[1:])
            except ValueError:
                pass
    
    return info

def process_dataset(params, n_test=20000):
    """Process a single dataset with NTK."""
    print(f"Processing: {params}", flush=True)
    
    try:
        # Load dataset
        start_time = time.time()
        data = torch.load(params['ds_path'], weights_only=True)
        X_full = data['X'].numpy()
        y_full = data['y'].numpy()
        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds", flush=True)
        
        # Split into train/test
        sample_seed = hash(f"sample_{params['n_train']}_{params['ds_name']}_{params['experiment_num']}") % (2**32)
        rng = np.random.RandomState(sample_seed)
        indices = rng.permutation(X_full.shape[0])
        
        test_indices = indices[:n_test]
        train_master_indices = indices[n_test:]
        
        X_test = X_full[test_indices]
        y_test = y_full[test_indices]
        
        # Sample n_train examples
        train_indices = rng.permutation(train_master_indices.shape[0])[:params['n_train']]
        X_train = X_full[train_master_indices[train_indices]]
        y_train = y_full[train_master_indices[train_indices]]
        
        # Print shapes
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}", flush=True)
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}", flush=True)
        
        # Convert to JAX arrays
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_test = jnp.array(X_test)
        y_test = jnp.array(y_test)
        
        # Determine optimal batch size for H100
        if params['n_train'] > 10000:
            batch_size = 256  # For larger datasets
        else:
            batch_size = 1024  # H100 can handle larger batches for small datasets
            
        print(f"Using batch size: {batch_size}", flush=True)
        
        # Build NTK network
        _, _, kernel_fn = build_ntk_network(
            input_dim=params['input_dim'],
            hidden_size=params['hidden_size'],
            depth=params['depth']
        )
        
        # Time the prediction
        pred_start_time = time.time()
        
        # Use NT's gradient_descent_mse_ensemble directly
        predictor = nt.predict.gradient_descent_mse_ensemble(
            kernel_fn=kernel_fn,
            x_train=X_train,
            y_train=y_train,
            diag_reg=params['reg'],
            batch_size=batch_size  # This handles batching automatically
        )
        
        # Get predictions
        y_test_pred = predictor(x_test=X_test, get='ntk')
        pred_time = time.time() - pred_start_time
        
        # Calculate error
        test_error = float(jnp.mean((y_test_pred - y_test)**2))
        print(f"Test error: {test_error}, time: {pred_time:.2f}s", flush=True)
        
        return {
            'dataset_name': params['ds_name'],
            'n_train': params['n_train'],
            'hidden_size': params['hidden_size'],
            'depth': params['depth'],
            'test_error': test_error,
            'compute_time': pred_time,
            'experiment_num': params['experiment_num']
        }
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        
        return {
            'dataset_name': params['ds_name'],
            'n_train': params['n_train'],
            'error': str(e)
        }

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = load_yaml_config(config_path)
    
    # Configure JAX for H100
    configure_jax_for_h100()
    
    # Extract parameters
    base_cfg = config["base_config"]
    base_results_dir = base_cfg.get("base_results_dir", "ntk_results")
    reg = float(base_cfg.get("reg", 0.0005))
    n_test = base_cfg.get("n_test", 20000)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_results_dir, f"ntk_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    # Process each dataset in the config
    for sweep_name, sweep_info in config["sweeps"].items():
        dataset_paths = sweep_info.get("dataset_paths", [])
        depths = sweep_info.get("parameters", {}).get("depths", [2])
        n_train_sizes = sweep_info.get("parameters", {}).get("n_train", [10, 100, 1000, 10000])
        
        for ds_path in dataset_paths:
            # Extract dataset info
            ds_info = extract_info_from_path(ds_path)
            ds_name = f"{ds_info.get('distribution_type', 'unknown')}_d{ds_info.get('input_dim', '?')}_H{ds_info.get('hidden_size', '?')}_a{ds_info.get('alpha', '?')}"
            
            # Process with different depths and training sizes
            for depth in depths:
                for n_train in n_train_sizes:
                    # Determine appropriate hidden size for dataset size
                    if n_train > 10000:
                        hidden_size = 1024  # Smaller for large datasets
                    else:
                        hidden_size = 5000  # Larger for small datasets
                    
                    params = {
                        'ds_path': ds_path,
                        'ds_name': ds_name,
                        'input_dim': ds_info.get('input_dim', 16),
                        'hidden_size': hidden_size,
                        'depth': depth,
                        'n_train': n_train,
                        'reg': reg,
                        'experiment_num': 1,
                        'sweep_name': sweep_name
                    }
                    
                    # Process the dataset
                    result = process_dataset(params, n_test)
                    results.append(result)
                    
                    # Save intermediate result
                    with open(os.path.join(results_dir, "results.jsonl"), "a") as f:
                        f.write(json.dumps(result) + "\n")
    
    # Save final results
    with open(os.path.join(results_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"All results saved to {results_dir}")

if __name__ == "__main__":
    main()