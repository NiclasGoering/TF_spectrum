import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import your modules (ensure these files exist in your project)
from FFNN import DeepNN
from utils2 import save_results, save_model, save_dataset


# def create_target_kernel(feature_dim: int, alpha: float, device: torch.device):
#     """
#     Creates a target kernel matrix T ∈ ℝ^(feature_dim×feature_dim) with eigenvalues decaying as 1/(i+1)^α.
#     The eigenvalues are scaled so that trace(T) = feature_dim.
#     """
#     eig_vals = np.array([1.0 / ((i + 1) ** alpha) for i in range(feature_dim)], dtype=np.float32)
#     scale = feature_dim / eig_vals.sum()
#     eig_vals_scaled = eig_vals * scale
#     D = torch.diag(torch.tensor(eig_vals_scaled, device=device))
    
#     # Create a random orthogonal matrix Q via QR decomposition.
#     A = torch.randn(feature_dim, feature_dim, device=device)
#     Q, _ = torch.linalg.qr(A)
    
#     target_kernel = Q @ D @ Q.T
#     target_kernel = (target_kernel + target_kernel.T) / 2  # Force symmetry.
    
#     # Shift T if necessary to be positive semidefinite.
#     eigvals = torch.linalg.eigvalsh(target_kernel)
#     if eigvals[0] < 0:
#         target_kernel = target_kernel - eigvals[0] * torch.eye(feature_dim, device=device)
#     target_eigenvals = torch.sort(torch.linalg.eigvalsh(target_kernel))[0]
#     return target_kernel, target_eigenvals



def create_target_kernel(feature_dim: int, alpha: float, device: torch.device, effective_rank: int = None, eps: float = 1e-8):
    """
    Creates a target kernel matrix T ∈ ℝ^(feature_dim×feature_dim) such that:
      - For the first 'effective_rank' eigenvalues, they decay as 1/(i+1)^alpha.
      - For the remaining eigenvalues, they are set to a small value (eps).
    The eigenvalues are then scaled so that trace(T) = effective_rank.
    
    Args:
        feature_dim: Total dimension of the kernel.
        alpha: Decay exponent for the eigenvalues.
        device: Torch device.
        effective_rank: The number of eigenvalues to have non-negligible values. If None, use feature_dim.
        eps: A small constant for the eigenvalues beyond the effective rank.
        
    Returns:
        target_kernel: The constructed target kernel matrix.
        target_eigenvals: The sorted eigenvalues of the target kernel.
    """
    if effective_rank is None:
        effective_rank = feature_dim

    # Initialize eigenvalues: for indices < effective_rank use 1/(i+1)^alpha; for the rest, use eps.
    eig_vals = np.zeros(feature_dim, dtype=np.float32)
    for i in range(effective_rank):
        eig_vals[i] = 1.0 / ((i + 1) ** alpha)
    for i in range(effective_rank, feature_dim):
        eig_vals[i] = eps

    # Scale eigenvalues so that the trace equals effective_rank.
    scale = effective_rank / eig_vals.sum()
    eig_vals_scaled = eig_vals * scale
    D = torch.diag(torch.tensor(eig_vals_scaled, device=device))
    
    # Create a random orthogonal matrix Q via QR decomposition.
    A = torch.randn(feature_dim, feature_dim, device=device)
    Q, _ = torch.linalg.qr(A)
    
    target_kernel = Q @ D @ Q.T
    target_kernel = (target_kernel + target_kernel.T) / 2  # Force symmetry.
    
    # Ensure the matrix is positive semidefinite.
    eigvals = torch.linalg.eigvalsh(target_kernel)
    if eigvals[0] < 0:
        target_kernel = target_kernel - eigvals[0] * torch.eye(feature_dim, device=device)
    
    target_eigenvals = torch.sort(torch.linalg.eigvalsh(target_kernel))[0]
    return target_kernel, target_eigenvals


def smart_initialize_with_input_stats_modified(model: DeepNN, target_kernel: torch.Tensor, X: torch.Tensor, small_bias: float = 1e-2):
    """
    Adjusts the penultimate layer weights based on input statistics and the square-root of the target kernel.
    
    Note: Unlike the previous version, we do not reinitialize the earlier layers with an identity-like pattern.
    This way, the earlier layers remain at their default (normal) initialization.
    """
    # For the penultimate layer, adjust weights using input statistics.
    penultimate_layer = model.network[-2]
    if isinstance(penultimate_layer, nn.Linear):
        N = X.shape[0]
        input_cov = (X.T @ X) / N
        eps = 1e-6
        input_cov = input_cov + eps * torch.eye(input_cov.shape[0], device=input_cov.device)
        
        # Compute the square root and inverse square root of the input covariance.
        input_cov_eigvals, input_cov_eigvecs = torch.linalg.eigh(input_cov)
        input_cov_sqrt = input_cov_eigvecs @ torch.diag(torch.sqrt(input_cov_eigvals)) @ input_cov_eigvecs.T
        input_cov_sqrt_inv = input_cov_eigvecs @ torch.diag(1.0 / torch.sqrt(input_cov_eigvals)) @ input_cov_eigvecs.T
        
        # Compute the square root of the target kernel.
        target_eigvals, target_eigvecs = torch.linalg.eigh(target_kernel)
        target_eigvals = torch.clamp(target_eigvals, min=1e-10)
        target_sqrt = target_eigvecs @ torch.diag(torch.sqrt(target_eigvals)) @ target_eigvecs.T
        
        # Set the weight for the penultimate layer.
        W_target = target_sqrt @ input_cov_sqrt_inv
        # Add a small identity term to avoid rank-deficiency.
        W_target = W_target + 1e-2 * torch.eye(W_target.size(0), device=W_target.device)
        
        penultimate_layer.weight.data.copy_(W_target.to(penultimate_layer.weight.device))
        if penultimate_layer.bias is not None:
            penultimate_layer.bias.data.zero_()
    
    return model


def lsuv_init(model: DeepNN, X: torch.Tensor, needed_std: float = 1.0, tol: float = 0.1, max_iter: int = 10):
    """
    Applies the LSUV (Layer-sequential Unit-Variance) initialization.
    For each linear layer in the network, a forward hook is used to measure the output standard deviation.
    The weights are then scaled until the output standard deviation is approximately 'needed_std'.
    """
    model.eval()
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            hook_handle = layer.register_forward_hook(hook)
            
            # Run a forward pass to capture the activation.
            _ = model(X)
            if len(outputs) == 0:
                hook_handle.remove()
                continue
            act = outputs[0]
            std = act.std().item()
            count = 0
            # Adjust weights until the output std is near the desired value.
            while abs(std - needed_std) > tol and count < max_iter:
                scaling = needed_std / (std + 1e-8)
                layer.weight.data.mul_(scaling)
                outputs.clear()
                _ = model(X)
                act = outputs[0]
                std = act.std().item()
                count += 1
            print(f"LSUV init for layer {i}: final std = {std:.4f} after {count} iterations")
            hook_handle.remove()
    model.train()
    return model


def kernel_loss(model: DeepNN, X: torch.Tensor, target_kernel: torch.Tensor,
                lambda_eig: float = 1.0, use_log: bool = True,
                top_k: int = 20, lambda_top: float = 1e4,
                frob_scale: float = 100.0, eig_scale: float = 1.0,
                rank_preservation_weight: float = 0.0):
    """
    Computes a composite loss between the normalized covariance of the penultimate features and the target kernel.
    Includes an optional rank preservation penalty.
    
    Args:
        frob_scale: multiplier for the Frobenius norm term.
        eig_scale: multiplier for the eigenvalue loss term.
        rank_preservation_weight: weight for the rank preservation penalty (if > 0, penalizes low-rank features).
    """
    # Compute penultimate features.
    features = X
    for layer in model.network[:-1]:
        features = layer(features)
    N = features.shape[0]
    C_norm = features.T @ features / N

    # Regularize for numerical stability.
    I = torch.eye(C_norm.shape[0], device=C_norm.device)
    C_norm_reg = C_norm + 1e-6 * I

    # Compute Frobenius norm difference (relative difference).
    frob_diff = torch.norm(C_norm_reg - target_kernel, p='fro')**2
    target_norm_sq = torch.norm(target_kernel, p='fro')**2 + 1e-8
    loss_frob = frob_diff / target_norm_sq
    loss_frob = frob_scale * loss_frob

    # Compute eigenvalue-based loss.
    eig_C = torch.linalg.eigvalsh(C_norm_reg)
    eig_T = torch.linalg.eigvalsh(target_kernel)
    
    if use_log:
        eps = 1e-8
        eig_C = torch.clamp(eig_C, min=eps)
        eig_T = torch.clamp(eig_T, min=eps)
        loss_eig = torch.sum((torch.log(eig_C) - torch.log(eig_T))**2)
    else:
        loss_eig = torch.sum((eig_C - eig_T)**2)
    loss_eig = eig_scale * loss_eig

    # Compute top-k eigenvalue loss.
    top_eig_loss = torch.mean((eig_C[-top_k:] - eig_T[-top_k:])**2)

    total_loss = loss_frob + lambda_eig * loss_eig + lambda_top * top_eig_loss

    # Add rank preservation penalty if specified.
    if rank_preservation_weight > 0.0:
        eig_C_clamped = torch.clamp(eig_C, min=1e-12)
        rank_penalty = -torch.sum(torch.log(eig_C_clamped))
        total_loss += rank_preservation_weight * rank_penalty

    return total_loss, loss_frob, loss_eig, top_eig_loss


def train_kernel_modified(model: DeepNN, X: torch.Tensor, target_kernel: torch.Tensor, epochs: int = 5000,
                          lr: float = 1e-3, lambda_eig: float = 10.0, use_log: bool = True,
                          top_k: int = 5, lambda_top: float = 100.0):
    """
    Trains the network using Adam as the optimizer with a cosine annealing scheduler.
    The loss is based on the difference between the kernel of the penultimate features and the target kernel.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_frob, loss_eig, top_eig_loss = kernel_loss(
            model, X, target_kernel,
            lambda_eig=lambda_eig,
            use_log=use_log,
            top_k=top_k,
            lambda_top=lambda_top
        )
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + 1)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.4e} | Frobenius: {loss_frob.item():.4e} | "
                  f"Eig Loss: {loss_eig.item():.4e} | Top-{top_k} Loss: {top_eig_loss.item():.4e}")
    return model


def compute_last_hidden_kernel_unnormalized_spectrum(model: DeepNN, X: torch.Tensor):
    """
    Computes the eigenvalues of the unnormalized kernel (H^T H) of the penultimate layer.
    """
    with torch.no_grad():
        features = X
        for layer in model.network[:-1]:
            features = layer(features)
        K_unnorm = features.T @ features
        eigenvalues = torch.linalg.eigvalsh(K_unnorm)
    return eigenvalues


def main():
    # --- Hyperparameters ---
    d = 30                      # Input dimension.
    hidden_size = 256           # Hidden layer (penultimate) size.
    depth = 1                   # Total network depth.
    train_size = 300000         # Number of training samples.
    mode = 'mup_no_align'       # Network mode.
    alpha = 0.0                 # Use a milder decay exponent for a smoother target spectrum.
    lambda_eig = 50.0          # Adjusted eigenvalue loss weight.
    use_log = False             # Use logarithmic eigenvalue loss.
    epochs = 100000              # Training epochs.
    lr = 5e-2                   # Learning rate.
    top_k = 5                   # Number of top eigenvalues to match.
    lambda_top = 10.0          # Adjusted top-k loss weight.
    # Set the rank preservation weight (nonzero to add the penalty; adjust as needed)
    rank_preservation_weight = 0.01  
    rank = 0                    # Rank for saving dataset.
    effective_rank= 256 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Create the network and target kernel ---
    model = DeepNN(d, hidden_size, depth, mode=mode).to(device)
    target_kernel_norm, target_eigenvalues_norm = create_target_kernel(hidden_size, alpha, device, effective_rank=effective_rank)
    #target_kernel_norm, target_eigenvalues_norm = create_target_kernel(hidden_size, alpha, device)
    print("\nTarget kernel eigenvalues (ascending):")
    print(target_eigenvalues_norm.detach().cpu().numpy())
    
    target_kernel_used = target_kernel_norm
    
    # --- Generate the input dataset X ~ N(0, I) ---
    X = torch.randn(train_size, d, device=device)
    
    # --- Print input covariance statistics ---
    input_stats = (X.T @ X) / train_size
    input_eigvals = torch.linalg.eigvalsh(input_stats)
    print("\nInput covariance eigenvalues (ascending):")
    print(input_eigvals.detach().cpu().numpy())
    
    # --- Smart initialization using input statistics ---
    print("\nPerforming smart initialization with input statistics...")
    model = smart_initialize_with_input_stats_modified(model, target_kernel_used, X, small_bias=1e-2)
    
    # --- Data-dependent LSUV initialization on all layers ---
    print("\nApplying LSUV initialization for data-dependent preconditioning...")
    model = lsuv_init(model, X, needed_std=1.0, tol=0.1, max_iter=10)
    
    # --- Verify the initial kernel spectrum ---
    with torch.no_grad():
        features = X
        for layer in model.network[:-1]:
            features = layer(features)
        initial_kernel = (features.T @ features) / train_size
        initial_eigvals = torch.linalg.eigvalsh(initial_kernel)
    print("\nInitial kernel eigenvalues after initialization (ascending):")
    print(initial_eigvals.detach().cpu().numpy())
    
    # --- Fine-tune the network ---
    print("\nStarting training...")
    model = train_kernel_modified(model, X, target_kernel_used, epochs=epochs, lr=lr,
                                  lambda_eig=lambda_eig, use_log=use_log,
                                  top_k=top_k, lambda_top=lambda_top)
    
    # --- Compute the final (unnormalized) kernel spectrum ---
    target_eigenvalues_unnorm = target_eigenvalues_norm * train_size
    last_hidden_eigenvalues_unnorm = compute_last_hidden_kernel_unnormalized_spectrum(model, X)
    
    target_spec = np.sort(target_eigenvalues_unnorm.detach().cpu().numpy())[::-1]
    last_hidden_spec = np.sort(last_hidden_eigenvalues_unnorm.detach().cpu().numpy())[::-1]
    
    # --- Save results and plots ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    smart_name = f"model_d{d}_hidden{hidden_size}_depth{depth}_alpha{alpha}_{timestamp}"
    save_dir = os.path.join("/home/goring/TF_spectrum/results_pretrain_testgrid_2/",
                            f"results_2_{smart_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(np.arange(1, len(target_spec) + 1), target_spec, 'o-', 
              label='Target Kernel Spectrum', markersize=4)
    plt.loglog(np.arange(1, len(last_hidden_spec) + 1), last_hidden_spec, 's-',
              label='Last Hidden Kernel Spectrum', markersize=4)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Kernel Spectrum Comparison (Log-Log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plot_path = os.path.join(save_dir, f'spectrum_plot_{smart_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved log-log spectrum plot to {plot_path}")
    
    results = {
        'hyperparameters': {
            'input_dim': d,
            'hidden_size': hidden_size,
            'depth': depth,
            'train_size': train_size,
            'mode': mode,
            'alpha': alpha,
            'lambda_eig': lambda_eig,
            'use_log': use_log,
            'epochs': epochs,
            'learning_rate': lr,
            'top_k': top_k,
            'lambda_top': lambda_top,
            'rank_preservation_weight': rank_preservation_weight
        },
        'unnormalized_target_spectrum': target_spec.tolist(),
        'unnormalized_last_hidden_spectrum': last_hidden_spec.tolist(),
        'initial_kernel_spectrum': initial_eigvals.detach().cpu().numpy().tolist(),
        'input_covariance_spectrum': input_eigvals.detach().cpu().numpy().tolist()
    }
    save_results([results], save_dir, smart_name)
    print("Saved results (hyperparameters and spectra).")
    
    with torch.no_grad():
        y = model(X)
    dataset_path = os.path.join(save_dir, f"dataset_{smart_name}.pt")
    save_dataset(X, y, dataset_path, rank)
    
    target_kernel_path = os.path.join(save_dir, f"target_kernel_{smart_name}.pt")
    torch.save(target_kernel_used.detach().cpu(), target_kernel_path)
    print(f"Saved target kernel to {target_kernel_path}")
    
    model_path = os.path.join(save_dir, f"model_{smart_name}.pt")
    save_model(model, model_path)
    print(f"Saved model to {model_path}")
    
    print("\nFinal spectrum comparison summary:")
    print(f"Target spectrum range: [{target_spec[-1]:.2e}, {target_spec[0]:.2e}]")
    print(f"Achieved spectrum range: [{last_hidden_spec[-1]:.2e}, {last_hidden_spec[0]:.2e}]")
    print(f"Condition numbers - Target: {target_spec[0]/(target_spec[-1]+1e-12):.2e}, "
          f"Achieved: {last_hidden_spec[0]/(last_hidden_spec[-1]+1e-12):.2e}")


if __name__ == "__main__":
    main()
