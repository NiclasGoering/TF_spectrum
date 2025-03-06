import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
from FFNN import DeepNN  # Import DeepNN from your FFNN module


def generate_data(distribution_type, train_size, d, device, r=0.5):
    """
    Generate data from specified distribution.
    
    Args:
        distribution_type: String indicating the distribution ('normal', 'uniform', or 'spiked_normal')
        train_size: Number of samples
        d: Input dimension
        device: Torch device
        r: Exponent for spiked normal (only used if distribution_type='spiked_normal')
        
    Returns:
        X: Generated data as a torch tensor of shape (train_size, d)
    """
    if distribution_type == 'normal':
        # Standard normal distribution
        X = torch.randn(train_size, d, device=device)
        
    elif distribution_type == 'uniform':
        # Uniform distribution in [-1, 1]
        X = 2 * torch.rand(train_size, d, device=device) - 1
        
    elif distribution_type == 'spiked_normal':
        # Spiked normal: N(0, I_d + θθ^T * d^r)
        # First, create a random unit vector θ
        theta = torch.randn(d, device=device)
        theta = theta / torch.norm(theta)  # Normalize to unit vector
        
        # Create base normal distribution
        X = torch.randn(train_size, d, device=device)
        
        # Add the spike component: X += Z * θ where Z ~ N(0, d^r)
        spike_scale = d**r
        Z = torch.randn(train_size, 1, device=device) * torch.sqrt(torch.tensor(spike_scale))
        X = X + Z * theta
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return X


def create_hierarchical_kernel(feature_dim: int, device: torch.device,
                              k1: int, k2: int, p1: int, p2: int,
                              alpha1: float = 2.0, alpha2: float = 0.5, alpha3: float = 0.1,
                              eps: float = 1e-8):
    """
    Creates a target kernel matrix with a hierarchical eigenvalue spectrum that includes plateaus.
    
    The eigenvalue distribution follows this pattern:
    λᵢ ∝ i^(-alpha1)                 for i = 1 to k₁
    λᵢ ∝ (k₁)^(-alpha1)              for i = k₁+1 to k₁+p₁     [PLATEAU 1]
    λᵢ ∝ (i-p₁)^(-alpha2)            for i = k₁+p₁+1 to k₂
    λᵢ ∝ (k₂-p₁)^(-alpha2)           for i = k₂+1 to k₂+p₂     [PLATEAU 2]
    λᵢ ∝ (i-p₁-p₂)^(-alpha3)         for i = k₂+p₂+1 to feature_dim
    
    Args:
        feature_dim: Total dimension of the kernel.
        device: Torch device.
        k1: First transition point.
        k2: Second transition point (must be > k1 + p1).
        p1: Width of the first plateau.
        p2: Width of the second plateau.
        alpha1: Decay exponent for the first regime (default: 2.0).
        alpha2: Decay exponent for the second regime (default: 0.5).
        alpha3: Decay exponent for the third regime (default: 0.1).
        eps: A small constant for numerical stability.
        
    Returns:
        target_kernel: The constructed target kernel matrix.
        target_eigenvals: The sorted eigenvalues of the target kernel.
    """
    # Input validation
    if k1 <= 0 or p1 <= 0 or k2 <= k1 + p1 or p2 <= 0 or feature_dim <= k2 + p2:
        raise ValueError(f"Invalid hierarchical parameters: k1={k1}, p1={p1}, k2={k2}, p2={p2}, feature_dim={feature_dim}")
    
    # Initialize eigenvalues
    eig_vals = np.zeros(feature_dim, dtype=np.float32)
    
    # First regime: λᵢ ∝ i^(-alpha1) for i = 1 to k₁
    for i in range(k1):
        eig_vals[i] = 1.0 / ((i + 1) ** alpha1)
    
    # First plateau: λᵢ ∝ (k₁)^(-alpha1) for i = k₁+1 to k₁+p₁
    plateau1_value = 1.0 / (k1 ** alpha1)
    for i in range(k1, k1 + p1):
        eig_vals[i] = plateau1_value
    
    # Second regime: λᵢ ∝ (i-p₁)^(-alpha2) for i = k₁+p₁+1 to k₂
    for i in range(k1 + p1, k2):
        adjusted_i = i - p1
        eig_vals[i] = 1.0 / ((adjusted_i + 1) ** alpha2)
    
    # Second plateau: λᵢ ∝ (k₂-p₁)^(-alpha2) for i = k₂+1 to k₂+p₂
    plateau2_value = 1.0 / ((k2 - p1) ** alpha2)
    for i in range(k2, k2 + p2):
        eig_vals[i] = plateau2_value
    
    # Third regime: λᵢ ∝ (i-p₁-p₂)^(-alpha3) for i = k₂+p₂+1 to feature_dim
    for i in range(k2 + p2, feature_dim):
        adjusted_i = i - p1 - p2
        eig_vals[i] = 1.0 / ((adjusted_i + 1) ** alpha3)
    
    # Sort eigenvalues in ascending order
    eig_vals = np.sort(eig_vals)
    
    # Add a small epsilon to avoid numerical issues
    eig_vals = eig_vals + eps
    
    # Scale eigenvalues so that the trace equals feature_dim
    scale = feature_dim / (np.sum(eig_vals) + 1e-12)
    eig_vals_scaled = eig_vals * scale
    
    # Convert to torch tensor and create diagonal matrix
    D = torch.diag(torch.tensor(eig_vals_scaled, device=device))
    
    # Create a random orthogonal matrix Q via QR decomposition
    A = torch.randn(feature_dim, feature_dim, device=device)
    Q, _ = torch.linalg.qr(A)
    
    # Create the kernel matrix
    target_kernel = Q @ D @ Q.T
    
    # Force symmetry
    target_kernel = (target_kernel + target_kernel.T) / 2
    
    # Ensure the matrix is positive semidefinite
    eigvals = torch.linalg.eigvalsh(target_kernel)
    if eigvals[0] < 0:
        target_kernel = target_kernel - eigvals[0] * torch.eye(feature_dim, device=device)
    
    # Get the sorted eigenvalues
    target_eigenvals = torch.sort(torch.linalg.eigvalsh(target_kernel))[0]
    
    return target_kernel, target_eigenvals


def visualize_hierarchical_spectrum(feature_dim, k1, k2, p1, p2, 
                                   alpha1, alpha2, alpha3, 
                                   show_regimes=True, save_path=None):
    """
    Visualize the eigenvalue spectrum of a hierarchical kernel with plateaus.
    
    Args:
        feature_dim: Total dimension of the kernel
        k1, k2: Transition points
        p1, p2: Plateau widths
        alpha1, alpha2, alpha3: Decay exponents for each regime
        show_regimes: Whether to highlight different regimes with colors
        save_path: Path to save the figure (if None, just displays it)
    """
    # Initialize eigenvalues
    eig_vals = np.zeros(feature_dim)
    
    # First regime: λᵢ ∝ i^(-alpha1) for i = 1 to k₁
    for i in range(k1):
        eig_vals[i] = 1.0 / ((i + 1) ** alpha1)
    
    # First plateau: λᵢ ∝ (k₁)^(-alpha1) for i = k₁+1 to k₁+p₁
    plateau1_value = 1.0 / (k1 ** alpha1)
    for i in range(k1, k1 + p1):
        eig_vals[i] = plateau1_value
    
    # Second regime: λᵢ ∝ (i-p₁)^(-alpha2) for i = k₁+p₁+1 to k₂
    for i in range(k1 + p1, k2):
        adjusted_i = i - p1
        eig_vals[i] = 1.0 / ((adjusted_i + 1) ** alpha2)
    
    # Second plateau: λᵢ ∝ (k₂-p₁)^(-alpha2) for i = k₂+1 to k₂+p₂
    plateau2_value = 1.0 / ((k2 - p1) ** alpha2)
    for i in range(k2, k2 + p2):
        eig_vals[i] = plateau2_value
    
    # Third regime: λᵢ ∝ (i-p₁-p₂)^(-alpha3) for i = k₂+p₂+1 to feature_dim
    for i in range(k2 + p2, feature_dim):
        adjusted_i = i - p1 - p2
        eig_vals[i] = 1.0 / ((adjusted_i + 1) ** alpha3)
    
    # Scale eigenvalues so that the trace equals feature_dim
    scale = feature_dim / (np.sum(eig_vals) + 1e-12)
    eig_vals_scaled = eig_vals * scale
    
    # Sort in descending order for visualization
    eig_vals_descending = np.sort(eig_vals_scaled)[::-1]
    
    # Create indices for plotting
    indices = np.arange(1, feature_dim + 1)
    
    # Set up figure with multiple subplots for different view scales
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle(f'Hierarchical Kernel Spectrum with Plateaus\n' +
                f'k1={k1}, k2={k2}, p1={p1}, p2={p2}, ' +
                f'α1={alpha1}, α2={alpha2}, α3={alpha3}', 
                fontsize=16)
    
    # Linear Scale
    ax1 = axs[0]
    if show_regimes:
        # Highlight different regimes with colors
        ax1.plot(indices[:k1], eig_vals_descending[:k1], 'r-', label='Regime 1 (α1)')
        ax1.plot(indices[k1:k1+p1], eig_vals_descending[k1:k1+p1], 'g-', label='Plateau 1')
        ax1.plot(indices[k1+p1:k2], eig_vals_descending[k1+p1:k2], 'b-', label='Regime 2 (α2)')
        ax1.plot(indices[k2:k2+p2], eig_vals_descending[k2:k2+p2], 'c-', label='Plateau 2')
        ax1.plot(indices[k2+p2:], eig_vals_descending[k2+p2:], 'm-', label='Regime 3 (α3)')
    else:
        ax1.plot(indices, eig_vals_descending, 'b-')
    
    ax1.set_title('Linear Scale', fontsize=14)
    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.grid(True, alpha=0.3)
    if show_regimes:
        ax1.legend()
    
    # Semi-log Scale (log y-axis)
    ax2 = axs[1]
    if show_regimes:
        # Highlight different regimes with colors
        ax2.semilogy(indices[:k1], eig_vals_descending[:k1], 'r-', label='Regime 1 (α1)')
        ax2.semilogy(indices[k1:k1+p1], eig_vals_descending[k1:k1+p1], 'g-', label='Plateau 1')
        ax2.semilogy(indices[k1+p1:k2], eig_vals_descending[k1+p1:k2], 'b-', label='Regime 2 (α2)')
        ax2.semilogy(indices[k2:k2+p2], eig_vals_descending[k2:k2+p2], 'c-', label='Plateau 2')
        ax2.semilogy(indices[k2+p2:], eig_vals_descending[k2+p2:], 'm-', label='Regime 3 (α3)')
    else:
        ax2.semilogy(indices, eig_vals_descending, 'b-')
    
    ax2.set_title('Semi-log Scale (Log Y-axis)', fontsize=14)
    ax2.set_xlabel('Eigenvalue Index', fontsize=12)
    ax2.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    if show_regimes:
        ax2.legend()
    
    # Log-log Scale
    ax3 = axs[2]
    if show_regimes:
        # Highlight different regimes with colors
        ax3.loglog(indices[:k1], eig_vals_descending[:k1], 'r-', label='Regime 1 (α1)')
        ax3.loglog(indices[k1:k1+p1], eig_vals_descending[k1:k1+p1], 'g-', label='Plateau 1')
        ax3.loglog(indices[k1+p1:k2], eig_vals_descending[k1+p1:k2], 'b-', label='Regime 2 (α2)')
        ax3.loglog(indices[k2:k2+p2], eig_vals_descending[k2:k2+p2], 'c-', label='Plateau 2')
        ax3.loglog(indices[k2+p2:], eig_vals_descending[k2+p2:], 'm-', label='Regime 3 (α3)')
    else:
        ax3.loglog(indices, eig_vals_descending, 'b-')
    
    ax3.set_title('Log-log Scale', fontsize=14)
    ax3.set_xlabel('Eigenvalue Index (log scale)', fontsize=12)
    ax3.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    if show_regimes:
        ax3.legend()
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    return eig_vals_descending


def smart_initialize_with_input_stats_modified(model, target_kernel, X, small_bias=1e-2):
    """
    Adjusts the penultimate layer weights based on input statistics and the square-root of the target kernel.
    
    This version handles the dimension mismatch between input features and target kernel.
    """
    # Find all linear layers in the model
    linear_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            linear_layers.append(i)
    
    if len(linear_layers) < 2:
        print("Warning: Not enough linear layers to identify penultimate layer")
        return model
    
    penultimate_layer_idx = linear_layers[-2]  # Second-to-last linear layer
    penultimate_layer = model.layers[penultimate_layer_idx]
    
    # We need to get the actual input to the penultimate layer, not the raw X
    # First, get features up to the layer before the penultimate layer
    features = X
    for i in range(penultimate_layer_idx):
        features = model.layers[i](features)
    
    if isinstance(penultimate_layer, nn.Linear):
        N = features.shape[0]
        feature_dim = features.shape[1]  # Dimension of features entering the penultimate layer
        target_dim = target_kernel.shape[0]  # Dimension of the target kernel
        
        # Compute input covariance matrix using the actual input to the penultimate layer
        input_cov = (features.T @ features) / N
        eps = 1e-6
        input_cov = input_cov + eps * torch.eye(input_cov.shape[0], device=input_cov.device)
        
        # Compute the square root and inverse square root of the input covariance
        input_cov_eigvals, input_cov_eigvecs = torch.linalg.eigh(input_cov)
        input_cov_sqrt = input_cov_eigvecs @ torch.diag(torch.sqrt(input_cov_eigvals)) @ input_cov_eigvecs.T
        input_cov_sqrt_inv = input_cov_eigvecs @ torch.diag(1.0 / torch.sqrt(input_cov_eigvals)) @ input_cov_eigvecs.T
        
        # Compute the square root of the target kernel
        target_eigvals, target_eigvecs = torch.linalg.eigh(target_kernel)
        target_eigvals = torch.clamp(target_eigvals, min=1e-10)
        target_sqrt = target_eigvecs @ torch.diag(torch.sqrt(target_eigvals)) @ target_eigvecs.T
        
        # Check if dimensions match for matrix multiplication
        if feature_dim == target_dim:
            # Direct multiplication is possible
            W_target = target_sqrt @ input_cov_sqrt_inv
        else:
            print(f"Dimension mismatch: feature_dim={feature_dim}, target_dim={target_dim}")
            print("Using an alternative initialization approach...")
            
            # Option 1: Random orthogonal initialization scaled by target eigenvalues
            penultimate_out_dim = penultimate_layer.weight.shape[0]
            penultimate_in_dim = penultimate_layer.weight.shape[1]
            
            # Create a random orthogonal weight matrix
            W_random = torch.randn(penultimate_out_dim, penultimate_in_dim, device=penultimate_layer.weight.device)
            W_random, _ = torch.linalg.qr(W_random)
            
            # Scale based on the average of target eigenvalues
            target_eigenval_avg = target_eigvals.mean()
            scale_factor = torch.sqrt(target_eigenval_avg)
            W_target = scale_factor * W_random
            
            print(f"Initialized with random orthogonal matrix scaled by √{target_eigenval_avg:.4f}")
        
        # Add a small identity term to avoid rank-deficiency (if shapes allow)
        if W_target.size(0) == W_target.size(1):
            W_target = W_target + small_bias * torch.eye(W_target.size(0), device=W_target.device)
        
        # Ensure the weight matrix has the correct shape before assigning
        if W_target.shape == penultimate_layer.weight.shape:
            penultimate_layer.weight.data.copy_(W_target)
        else:
            print(f"Warning: W_target shape {W_target.shape} doesn't match penultimate layer weight shape {penultimate_layer.weight.shape}")
            print("Keeping original weights for penultimate layer")
        
        # Set bias to zero
        if penultimate_layer.bias is not None:
            penultimate_layer.bias.data.zero_()
    
    return model


def apply_lsuv_init(model, X, needed_std=1.0, tol=0.1, max_iter=10):
    """
    Applies the LSUV (Layer-sequential Unit-Variance) initialization.
    For each linear layer in the network, a forward hook is used to measure the output standard deviation.
    The weights are then scaled until the output standard deviation is approximately 'needed_std'.
    
    Changed function name from lsuv_init to apply_lsuv_init to avoid naming conflict.
    """
    model.eval()
    for i, layer in enumerate(model.layers):
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


def make_orthogonal(layer, verbose=False):
    """
    Makes the weight matrix of a linear layer orthogonal and sets bias to zero.
    For vector outputs (where out_features=1), normalizes the vector to unit length.
    """
    with torch.no_grad():
        weight = layer.weight.data
        
        # Special case: if output is 1-dimensional (vector output)
        if weight.shape[0] == 1:
            # Normalize the vector to unit length
            norm = torch.norm(weight)
            if norm > 0:  # Avoid division by zero
                layer.weight.data.copy_(weight / norm)
            if verbose:
                print("Normalized readout vector to unit length")
        # Check if the weight matrix is fat (more columns than rows)
        elif weight.shape[0] <= weight.shape[1]:
            # For fat matrices, use QR decomposition
            q, r = torch.linalg.qr(weight)
            # Set weights to Q (orthogonal matrix)
            layer.weight.data.copy_(q)
            if verbose:
                print(f"Applied orthogonal constraint to readout layer ({weight.shape[0]}×{weight.shape[1]} matrix)")
        else:
            # For tall matrices, use QR decomposition on the transpose
            q, r = torch.linalg.qr(weight.T)
            # Set weights to Q.T (orthogonal matrix)
            layer.weight.data.copy_(q.T)
            if verbose:
                print(f"Applied orthogonal constraint to readout layer ({weight.shape[0]}×{weight.shape[1]} matrix)")
        
        # Set bias to zero
        if layer.bias is not None:
            layer.bias.data.zero_()
            
        # Make layer non-trainable
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
    
    return layer


def apply_orthogonal_constraint(model, verbose=False):
    """
    Makes the readout layer (last linear layer) of the model orthogonal with zero bias
    and freezes its parameters.
    """
    # Find all linear layers
    linear_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            linear_layers.append(i)
    
    if not linear_layers:
        print("ERROR: No linear layers found in the model")
        return model
        
    # Get the last linear layer index
    last_linear_idx = linear_layers[-1]
    
    # Make the last linear layer orthogonal with zero bias and freeze it
    model.layers[last_linear_idx] = make_orthogonal(model.layers[last_linear_idx], verbose=verbose)
    
    if verbose:
        print("Readout layer weights and bias are now frozen (non-trainable)")
    
    return model


def compute_features_before_last_linear(model, X):
    """
    Helper function to compute features before the last linear layer.
    """
    features = X
    linear_layers = []
    
    # Find all linear layers
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            linear_layers.append(i)
    
    if not linear_layers:
        return features
    
    last_linear_idx = linear_layers[-1]
    
    # Process through all layers up to (but not including) the last linear layer
    for i, layer in enumerate(model.layers):
        if i < last_linear_idx:
            features = layer(features)
    
    return features


def kernel_loss(model, X, target_kernel,
                lambda_eig=1.0, use_log=True,
                top_k=20, lambda_top=1e4,
                frob_scale=100.0, eig_scale=1.0,
                rank_preservation_weight=0.0):
    """
    Computes a composite loss between the normalized covariance of the penultimate features and the target kernel.
    Includes an optional rank preservation penalty.
    """
    # Compute penultimate features
    features = compute_features_before_last_linear(model, X)
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


def train_kernel_modified(model, X, target_kernel, epochs=5000,
                          lr=1e-3, lambda_eig=10.0, use_log=True,
                          top_k=5, lambda_top=100.0, orthogonal=False):
    """
    Trains the network using Adam as the optimizer with a cosine annealing scheduler.
    The loss is based on the difference between the kernel of the penultimate features and the target kernel.
    
    If orthogonal is True, the readout layer is made orthogonal/unit-length and frozen before training.
    """
    # Apply orthogonal constraint initially if required
    if orthogonal:
        print("Applying orthogonal constraint to readout layer and freezing it...")
        model = apply_orthogonal_constraint(model, verbose=True)
    
    # Only create optimizer for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)
    
    losses = []
    
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
        
        # Store losses for analysis
        losses.append({
            'epoch': epoch, 
            'total_loss': loss.item(),
            'frob_loss': loss_frob.item(),
            'eig_loss': loss_eig.item(),
            'top_k_loss': top_eig_loss.item()
        })
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.4e} | Frobenius: {loss_frob.item():.4e} | "
                  f"Eig Loss: {loss_eig.item():.4e} | Top-{top_k} Loss: {top_eig_loss.item():.4e}")
    
    return model, losses


def compute_last_hidden_kernel_unnormalized_spectrum(model, X):
    """
    Computes the eigenvalues of the unnormalized kernel (H^T H) of the penultimate layer.
    """
    with torch.no_grad():
        # Get all linear layers
        linear_layers = []
        for i, layer in enumerate(model.layers):
            if isinstance(layer, nn.Linear):
                linear_layers.append(i)
        
        if len(linear_layers) < 2:
            print("WARNING: Not enough linear layers to compute penultimate features")
            return torch.zeros(1, device=X.device)  # Return dummy value
            
        # Get penultimate linear layer index
        penultimate_linear_idx = linear_layers[-2]
        
        # Process through layers up to the penultimate linear + activation
        features = X
        for i, layer in enumerate(model.layers):
            if i <= penultimate_linear_idx + 1:  # +1 to include the activation after the linear layer
                features = layer(features)
        
        # Compute kernel matrix
        K_unnorm = features.T @ features
        
        # Debug: Check kernel matrix properties
        diag_mean = torch.diagonal(K_unnorm).mean().item()
        print(f"Kernel matrix stats: Shape={K_unnorm.shape}, Mean diagonal={diag_mean:.4e}")
        
        # Compute eigenvalues
        try:
            eigenvalues = torch.linalg.eigvalsh(K_unnorm)
            
            # Debug: Check eigenvalue properties
            min_eig = eigenvalues.min().item()
            max_eig = eigenvalues.max().item()
            num_negative = (eigenvalues < 0).sum().item()
            print(f"Eigenvalue range: [{min_eig:.4e}, {max_eig:.4e}], Negative eigenvalues: {num_negative}")
            
            return eigenvalues
        except Exception as e:
            print(f"Error computing eigenvalues: {e}")
            return torch.zeros(K_unnorm.shape[0], device=X.device)


def save_model(model, path):
    """Save the model state dictionary to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def save_results(results, save_dir, name_prefix):
    """Save experimental results to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f"{name_prefix}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def save_dataset(X, y, path, rank=0):
    """Save dataset tensors to a file."""
    data = {'X': X.cpu(), 'y': y.cpu()}
    torch.save(data, path)
    print(f"Process {rank}: Dataset saved to {path}")


def main():
    # ===== CONFIGURATION PARAMETERS =====
    # Set all parameters directly here
    
    # Hierarchical spectrum parameters
    k1 = 15             # First transition point
    k2 = 80            # Second transition point
    p1 = 50            # Width of first plateau
    p2 = 40            # Width of second plateau
    alpha1 = 5.0       # Decay exponent for first regime
    alpha2 = 2.0       # Decay exponent for second regime
    alpha3 = 0.1       # Decay exponent for third regime
    
    # Model and data parameters
    input_dim = 128     # Input dimension
    hidden_dim = 128   # Hidden dimension (must be > k2+p2)
    depth = 2          # Network depth
    distribution = 'normal'  # Input distribution: 'normal', 'uniform', or 'spiked_normal'
    r_value = 0.5      # Exponent for spiked normal distribution (only used if distribution='spiked_normal')
    train_size = 3000000  # Number of training samples
    
    # Training parameters
    epochs = 8000      # Number of training epochs
    learning_rate = 5e-4  # Learning rate
    lambda_eig = 50.0  # Weight for eigenvalue loss
    use_log = True     # Use logarithmic eigenvalue loss
    top_k = 15         # Number of top eigenvalues to match
    lambda_top = 5.0  # Weight for top-k eigenvalue loss
    
    # Initialization and constraint parameters
    use_smart_init = True   # Use smart initialization
    use_lsuv_init = True    # Use LSUV initialization (renamed from lsuv_init to avoid conflict)
    orthogonal = True       # Use orthogonal constraint on readout layer
    
    # Other parameters
    seed = 42          # Random seed
    output_dir = '/home/goring/TF_spectrum/pretrain/staircase'  # Output directory
    use_cuda = True    # Use CUDA if available
    
    # ===== END OF CONFIGURATION PARAMETERS =====
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Validate hierarchical parameters
    if not (0 < k1 < k2 and k1 + p1 < k2 and k2 + p2 < hidden_dim):
        raise ValueError(f"Invalid hierarchical parameters: k1={k1}, k2={k2}, p1={p1}, p2={p2}, hidden_dim={hidden_dim}")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hier_k1{k1}_k2{k2}_p1{p1}_p2{p2}_H{hidden_dim}_{timestamp}"
    save_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to {save_dir}")
    
    # Create and visualize the target hierarchical spectrum
    print(f"Creating hierarchical kernel with parameters:")
    print(f"  k1={k1}, k2={k2}, p1={p1}, p2={p2}")
    print(f"  alpha1={alpha1}, alpha2={alpha2}, alpha3={alpha3}")
    
    # Generate visualization of target spectrum
    visualize_hierarchical_spectrum(
        feature_dim=hidden_dim,
        k1=k1, k2=k2, p1=p1, p2=p2,
        alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
        show_regimes=True,
        save_path=os.path.join(save_dir, "target_spectrum_visualization.png")
    )
    
    # Create the target kernel matrix
    target_kernel, target_eigenvalues = create_hierarchical_kernel(
        feature_dim=hidden_dim,
        device=device,
        k1=k1, k2=k2, p1=p1, p2=p2,
        alpha1=alpha1, alpha2=alpha2, alpha3=alpha3
    )
    
    print(f"Target kernel created with shape {target_kernel.shape}")
    
    # Generate input data
    print(f"Generating {distribution} data with dimension {input_dim}...")
    X = generate_data(
        distribution_type=distribution,
        train_size=train_size,
        d=input_dim,
        device=device,
        r=r_value
    )
    
    print(f"Generated data with shape {X.shape}")
    
    # Create the neural network model - Using the parameter names expected by your FFNN.DeepNN class
    model = DeepNN(
        d=input_dim,                # Changed from input_dim to d
        hidden_size=hidden_dim,     # Changed from hidden_dim to hidden_size
        depth=depth,
        mode='standard_lr'
    ).to(device)
    
    print(f"Created model with d={input_dim}, hidden_size={hidden_dim}, depth={depth}")
    
    # Apply initialization techniques
    if use_smart_init:
        print("Applying smart initialization...")
        model = smart_initialize_with_input_stats_modified(model, target_kernel, X, small_bias=1e-2)
    
    if use_lsuv_init:
        print("Applying LSUV initialization...")
        model = apply_lsuv_init(model, X, needed_std=1.0, tol=0.1, max_iter=10)  # Changed function name
    
    # Check initial kernel spectrum
    with torch.no_grad():
        features = compute_features_before_last_linear(model, X)
        initial_kernel = (features.T @ features) / train_size
        initial_eigvals = torch.linalg.eigvalsh(initial_kernel)
    
    print("Initial kernel eigenvalues (ascending):")
    print(initial_eigvals.detach().cpu().numpy())
    
    # Train the model to match the target kernel
    print(f"Training model for {epochs} epochs (orthogonal={orthogonal})...")
    model, training_losses = train_kernel_modified(
        model=model,
        X=X,
        target_kernel=target_kernel,
        epochs=epochs,
        lr=learning_rate,
        lambda_eig=lambda_eig,
        use_log=use_log,
        top_k=top_k,
        lambda_top=lambda_top,
        orthogonal=orthogonal
    )
    
    # Compute final spectrum
    print("Computing final kernel spectrum...")
    last_hidden_eigenvalues = compute_last_hidden_kernel_unnormalized_spectrum(model, X)
    
    # Create visualization of results
    target_spec = np.sort(target_eigenvalues.detach().cpu().numpy() * train_size)[::-1]
    achieved_spec = np.sort(last_hidden_eigenvalues.detach().cpu().numpy())[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(np.arange(1, len(target_spec) + 1), target_spec, 'o-', 
              label='Target Kernel Spectrum', markersize=4)
    plt.loglog(np.arange(1, len(achieved_spec) + 1), achieved_spec, 's-',
              label='Achieved Kernel Spectrum', markersize=4)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title('Kernel Spectrum Comparison (Log-Log)', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(save_dir, "spectrum_comparison_loglog.png"), dpi=300, bbox_inches='tight')
    
    # Semi-log plot which better shows plateaus
    plt.figure(figsize=(10, 6))
    plt.semilogy(np.arange(1, len(target_spec) + 1), target_spec, 'o-', 
                label='Target Kernel Spectrum', markersize=4)
    plt.semilogy(np.arange(1, len(achieved_spec) + 1), achieved_spec, 's-',
                label='Achieved Kernel Spectrum', markersize=4)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Eigenvalue (log scale)', fontsize=12)
    plt.title('Kernel Spectrum Comparison (Semi-Log)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(save_dir, "spectrum_comparison_semilog.png"), dpi=300, bbox_inches='tight')
    
    # Save training loss curve
    epoch_numbers = [loss['epoch'] for loss in training_losses]
    total_losses = [loss['total_loss'] for loss in training_losses]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, total_losses)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
    
    # Store results
    results = {
        'hyperparameters': {
            'distribution_type': distribution,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'depth': depth,
            'train_size': train_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'orthogonal': orthogonal,
            'use_smart_init': use_smart_init,
            'use_lsuv_init': use_lsuv_init,
            'lambda_eig': lambda_eig,
            'use_log': use_log,
            'top_k': top_k,
            'lambda_top': lambda_top,
            'seed': seed,
            'hierarchical_params': {
                'k1': k1,
                'k2': k2,
                'p1': p1,
                'p2': p2,
                'alpha1': alpha1,
                'alpha2': alpha2,
                'alpha3': alpha3
            }
        },
        'target_spectrum': target_spec.tolist(),
        'achieved_spectrum': achieved_spec.tolist(),
        'initial_spectrum': initial_eigvals.detach().cpu().numpy().tolist(),
        'training_losses': training_losses
    }
    
    # Save model and results
    save_model(model, os.path.join(save_dir, "trained_model.pt"))
    save_results([results], save_dir, "experiment")
    
    # Generate model output and save dataset
    with torch.no_grad():
        y = model(X)
    save_dataset(X, y, os.path.join(save_dir, "dataset.pt"))
    
    print(f"\nExperiment completed. Results saved to {save_dir}")
    print(f"Target spectrum range: [{target_spec[-1]:.2e}, {target_spec[0]:.2e}]")
    print(f"Achieved spectrum range: [{achieved_spec[-1]:.2e}, {achieved_spec[0]:.2e}]")
    print(f"Condition numbers - Target: {target_spec[0]/(target_spec[-1]+1e-12):.2e}, "
          f"Achieved: {achieved_spec[0]/(achieved_spec[-1]+1e-12):.2e}")


if __name__ == "__main__":
    main()