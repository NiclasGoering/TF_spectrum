import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import matplotlib.pyplot as plt

################################################################################
# 1) Network Architecture with Nonlinearities
#
# The network is built as follows:
#   - If hidden_depth == 0: a single linear mapping.
#   - If hidden_depth >= 1: a block of hidden layers and a final readout.
#
# For example:
#   hidden_depth = 1: [Linear(input_dim, hidden_dim) + ReLU] --> Linear(hidden_dim, 1)
#   hidden_depth = 2: [Linear(input_dim, hidden_dim) + ReLU,
#                     Linear(hidden_dim, hidden_dim) + ReLU] --> Linear(hidden_dim, 1)
#
# The kernel we wish to “precondition” is defined by the covariance of the post-ReLU
# output of the last hidden (penultimate) layer.
################################################################################
class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, mode='mup_no_align'):
        super().__init__()
        self.hidden_depth = hidden_depth
        self.mode = mode
        if hidden_depth < 0:
            raise ValueError("hidden_depth must be >= 0")
        if hidden_depth == 0:
            # No hidden layer: simple linear regression.
            self.network = nn.Sequential(
                nn.Linear(input_dim, 1)
            )
        else:
            layers = []
            # First hidden layer.
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Additional hidden layers.
            for i in range(1, hidden_depth):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            # Final readout layer.
            layers.append(nn.Linear(hidden_dim, 1))
            self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

################################################################################
# 2) Create Target Kernel with Effective Rank
#
# For a given feature_dim (typically equal to hidden_dim), the first effective_rank
# eigenvalues decay as 1/(i+1)^alpha and the remaining are set to a small value.
# They are then scaled so that trace(T) = effective_rank.
################################################################################
def create_target_kernel(feature_dim: int, alpha: float, device: torch.device,
                           effective_rank: int = None, eps: float = 1e-8):
    if effective_rank is None:
        effective_rank = feature_dim
    eig_vals = np.zeros(feature_dim, dtype=np.float32)
    for i in range(effective_rank):
        eig_vals[i] = 1.0 / ((i + 1) ** alpha)
    for i in range(effective_rank, feature_dim):
        eig_vals[i] = eps
    scale = effective_rank / eig_vals.sum()
    eig_vals_scaled = eig_vals * scale
    D = torch.diag(torch.tensor(eig_vals_scaled, device=device))
    A = torch.randn(feature_dim, feature_dim, device=device)
    Q, _ = torch.linalg.qr(A)
    target_kernel = Q @ D @ Q.T
    target_kernel = 0.5 * (target_kernel + target_kernel.T)
    eigvals = torch.linalg.eigvalsh(target_kernel)
    if eigvals[0] < 0:
        target_kernel = target_kernel - eigvals[0] * torch.eye(feature_dim, device=device)
    target_eigs = torch.sort(torch.linalg.eigvalsh(target_kernel))[0]
    return target_kernel, target_eigs

################################################################################
# 3) Smart Initialization of the Penultimate Kernel (Post-ReLU)
#
# We wish that the covariance kernel K_post = (1/N) * [ReLU(penultimate_output)]^T [ReLU(penultimate_output)]
# matches the target kernel.
#
# Steps:
#   1. Feed inputs X through all layers _up to_ (but not including) the penultimate layer.
#   2. Compute the penultimate layer output: pre = W * h_in + b.
#   3. Compute the post-activation output: h_post = ReLU(pre).
#   4. Compute the empirical covariance C_post = (h_post^T h_post)/N.
#   5. Compute its eigen-decomposition and invert its square root.
#   6. Compute Tsqrt = sqrt(target_kernel) (via eigen-decomposition).
#   7. Define a transformation S = Tsqrt @ inv_sqrt(C_post).
#   8. Update the penultimate layer’s weight as: W_new = S @ W_old and zero its bias.
#
# In our Sequential model, for hidden_depth>=1:
#   The hidden layers and ReLUs come first, then the final readout.
#   We choose the penultimate layer to be the last hidden linear layer.
################################################################################
def smart_init_penultimate_postrelu(model: DeepNN, target_kernel: torch.Tensor, X: torch.Tensor, small_bias: float = 1e-2):
    if model.hidden_depth < 1:
        print("No hidden layer present; skipping smart initialization.")
        return model
    # In our Sequential, when hidden_depth>=1:
    # For hidden_depth==1, network = [Linear (hidden), ReLU, Linear (readout)]
    # For hidden_depth==2, network = [Linear, ReLU, Linear, ReLU, Linear (readout)], etc.
    # The penultimate layer is the last hidden linear layer. Its index in the sequential is:
    #   penultimate_index = 0 for hidden_depth==1,
    #   penultimate_index = 2 for hidden_depth==2, etc.
    penultimate_index = 2 * (model.hidden_depth - 1)
    penultimate_layer = model.network[penultimate_index]
    
    # First, compute h_in: output of layers before the penultimate layer.
    h_in = X
    for i in range(penultimate_index):
        h_in = model.network[i](h_in)
    # Now compute the current post-activation output of the penultimate layer:
    pre = penultimate_layer(h_in)
    h_post = torch.relu(pre)
    device = X.device
    N = h_post.shape[0]
    # Compute the empirical covariance of the post-activation features.
    C_post = (h_post.T @ h_post) / N
    # Ensure numerical stability.
    eps = 1e-6
    C_post = C_post + eps * torch.eye(C_post.shape[0], device=device)
    # Compute eigen-decomposition of C_post.
    evals_C, evecs_C = torch.linalg.eigh(C_post)
    # Clamp eigenvalues for stability.
    evals_C = torch.clamp(evals_C, min=1e-2)
    inv_sqrt_C = evecs_C @ torch.diag(1.0 / torch.sqrt(evals_C)) @ evecs_C.T
    # Compute square root of the target kernel.
    evals_T, evecs_T = torch.linalg.eigh(target_kernel)
    evals_T = torch.clamp(evals_T, min=1e-2)
    sqrt_T = evecs_T @ torch.diag(torch.sqrt(evals_T)) @ evecs_T.T
    # Define the transformation.
    S = sqrt_T @ inv_sqrt_C
    # Update penultimate layer's weight.
    W_old = penultimate_layer.weight.data
    W_new = S @ W_old
    penultimate_layer.weight.data.copy_(W_new)
    # Zero out the bias.
    if penultimate_layer.bias is not None:
        penultimate_layer.bias.data.zero_()
    print("Smart initialization of penultimate (post-ReLU kernel) complete.")
    return model

################################################################################
# 4b) LSUV Initialization (Optional)
#
# Applies LSUV to all linear layers.
################################################################################
def lsuv_init(model: nn.Module, X: torch.Tensor, needed_std: float = 1.0, tol: float = 0.1, max_iter: int = 10):
    model.eval()
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            outputs = []
            def hook(module, input, output):
                outputs.append(output)
            h = layer.register_forward_hook(hook)
            _ = model(X)
            if len(outputs) == 0:
                h.remove()
                continue
            act = outputs[0]
            std = act.std().item()
            count = 0
            while abs(std - needed_std) > tol and count < max_iter:
                scaling = needed_std / (std + 1e-8)
                layer.weight.data.mul_(scaling)
                outputs.clear()
                _ = model(X)
                act = outputs[0]
                std = act.std().item()
                count += 1
            print(f"LSUV init for layer {i}: final std = {std:.4f} after {count} iterations")
            h.remove()
    model.train()
    return model

################################################################################
# 5) Kernel Loss with Rank Preservation Penalty
#
# This loss is computed on the post-activation outputs of all hidden layers
# (i.e. the penultimate features). It compares the normalized covariance of these
# features with the target kernel.
################################################################################
def kernel_loss(model: DeepNN, X: torch.Tensor, target_kernel: torch.Tensor,
                lambda_eig: float = 1.0, use_log: bool = True,
                top_k: int = 20, lambda_top: float = 1e4,
                frob_scale: float = 1.0, eig_scale: float = 1.0,
                rank_preservation_weight: float = 0.0):
    # For networks with hidden layers, use the output of all layers except the final readout.
    if model.hidden_depth >= 1:
        # Create a module that contains all hidden layers (i.e. up to final linear).
        hidden_module = model.network[:-1]
        feats = hidden_module(X)
    else:
        feats = X
    N = feats.shape[0]
    C_norm = feats.T @ feats / N
    I = torch.eye(C_norm.shape[0], device=C_norm.device)
    C_reg = C_norm + 1e-6 * I
    frob_diff = torch.norm(C_reg - target_kernel, p='fro')**2
    target_norm_sq = torch.norm(target_kernel, p='fro')**2 + 1e-8
    loss_frob = frob_scale * (frob_diff / target_norm_sq)
    eig_C = torch.linalg.eigvalsh(C_reg)
    eig_T = torch.linalg.eigvalsh(target_kernel)
    if use_log:
        eps = 1e-8
        eig_C = torch.clamp(eig_C, min=eps)
        eig_T = torch.clamp(eig_T, min=eps)
        loss_eig = torch.sum((torch.log(eig_C) - torch.log(eig_T))**2)
    else:
        loss_eig = torch.sum((eig_C - eig_T)**2)
    loss_eig = eig_scale * loss_eig
    top_eig_loss = torch.mean((eig_C[-top_k:] - eig_T[-top_k:])**2)
    total_loss = loss_frob + lambda_eig * loss_eig + lambda_top * top_eig_loss
    if rank_preservation_weight > 0.0:
        eig_C = torch.clamp(eig_C, min=1e-12)
        rank_penalty = -torch.sum(torch.log(eig_C))
        total_loss += rank_preservation_weight * rank_penalty
    return total_loss, loss_frob, loss_eig, top_eig_loss

################################################################################
# 6) Training Procedure
#
# Uses AdamW with a cosine-annealing-with-warm-restarts scheduler.
################################################################################
def train_kernel_modified(model: DeepNN, X: torch.Tensor, target_kernel: torch.Tensor,
                          epochs: int = 5000, lr: float = 1e-3, lambda_eig: float = 10.0,
                          use_log: bool = True, top_k: int = 5, lambda_top: float = 100.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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

################################################################################
# 7) Compute Final (Unnormalized) Penultimate Kernel Spectrum
################################################################################
def compute_last_hidden_kernel_unnormalized_spectrum(model: DeepNN, X: torch.Tensor):
    with torch.no_grad():
        if model.hidden_depth >= 1:
            hidden_module = model.network[:-1]
            feats = hidden_module(X)
        else:
            feats = X
        K_unnorm = feats.T @ feats
        eigenvalues = torch.linalg.eigvalsh(K_unnorm)
    return eigenvalues

################################################################################
# 8) Utility Functions for Saving Results, Model, and Dataset
################################################################################
def save_results(results, save_dir, smart_name):
    import json
    results_path = os.path.join(save_dir, f"results_{smart_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_dataset(X, y, path, rank):
    torch.save({'X': X.cpu(), 'y': y.cpu(), 'rank': rank}, path)

################################################################################
# 9) Main Script
################################################################################
def main():
    # --- Hyperparameters ---
    d = 10                      # Input dimension.
    hidden_size = 256           # Hidden layer size.
    hidden_depth = 1            # Number of hidden layers (e.g., 1 means 1 hidden layer+readout; 2 means 2 hidden layers+readout).
    train_size = 300000         # Number of training samples.
    mode = 'mup_no_align'       # Network mode.
    alpha = 1.0                 # Decay exponent for target kernel (set to 0 for milder decay).
    lambda_eig = 20.0           # Eigenvalue loss weight.
    use_log = True             # Use log-based eigenvalue loss.
    epochs = 5000              # Training epochs.
    lr = 5e-3                   # Learning rate.
    top_k = 10                 # Top-k eigenvalue matching.
    lambda_top = 50.0          # Top-k loss weight.
    rank = 0                  # For saving.
    effective_rank = d        # Effective rank for target kernel.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Build the network and create the target kernel ---
    model = DeepNN(d, hidden_size, hidden_depth, mode=mode).to(device)
    target_kernel_norm, target_eigenvalues_norm = create_target_kernel(hidden_size, alpha, device,
                                                                       effective_rank=effective_rank, eps=1e-8)
    print("\nTarget kernel eigenvalues (ascending):")
    print(target_eigenvalues_norm.detach().cpu().numpy())
    target_kernel_used = target_kernel_norm
    
    # --- Generate input data X ~ N(0, I) ---
    X = torch.randn(train_size, d, device=device)
    input_cov = (X.T @ X) / train_size
    input_eigs = torch.linalg.eigvalsh(input_cov)
    print("\nInput covariance eigenvalues (ascending):")
    print(input_eigs.detach().cpu().numpy())
    
    # --- Smart initialization using the post-ReLU kernel of the penultimate layer ---
    print("\nPerforming smart initialization (targeting the post-ReLU kernel)...")
    model = smart_init_penultimate_postrelu(model, target_kernel_used, X, small_bias=1e-2)
    
    # --- Optionally, apply LSUV initialization ---
    print("\nApplying LSUV initialization for data-dependent preconditioning...")
    model = lsuv_init(model, X, needed_std=1.0, tol=0.1, max_iter=10)
    
    # --- Verify initial penultimate kernel spectrum ---
    with torch.no_grad():
        if hidden_depth >= 1:
            hidden_module = model.network[:-1]
            feats = hidden_module(X)
        else:
            feats = X
        initial_kernel = (feats.T @ feats) / train_size
        initial_eigvals = torch.linalg.eigvalsh(initial_kernel)
    print("\nInitial kernel eigenvalues after initialization (ascending):")
    print(initial_eigvals.detach().cpu().numpy())
    
    # --- Fine-tune the network ---
    print("\nStarting training...")
    model = train_kernel_modified(model, X, target_kernel_used, epochs=epochs, lr=lr,
                                  lambda_eig=lambda_eig, use_log=use_log,
                                  top_k=top_k, lambda_top=lambda_top)
    
    # --- Compute final (unnormalized) penultimate kernel spectrum ---
    target_eigenvalues_unnorm = target_eigenvalues_norm * train_size
    penult_eigenvalues_unnorm = compute_last_hidden_kernel_unnormalized_spectrum(model, X)
    target_spec = np.sort(target_eigenvalues_unnorm.detach().cpu().numpy())[::-1]
    penult_spec = np.sort(penult_eigenvalues_unnorm.detach().cpu().numpy())[::-1]
    
    # --- Save results and plot the spectrum ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    smart_name = f"model_d{d}_hidden{hidden_size}_depth{hidden_depth}_alpha{alpha}_{timestamp}"
    save_dir = os.path.join("/home/goring/TF_spectrum/results_pretrain_testgrid",
                            f"results_{smart_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(np.arange(1, len(target_spec) + 1), target_spec, 'o-', 
              label='Target Kernel Spectrum', markersize=4)
    plt.loglog(np.arange(1, len(penult_spec) + 1), penult_spec, 's-',
              label='Penultimate Spectrum', markersize=4)
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
            'hidden_dim': hidden_size,
            'hidden_depth': hidden_depth,
            'train_size': train_size,
            'mode': mode,
            'alpha': alpha,
            'lambda_eig': lambda_eig,
            'use_log': use_log,
            'epochs': epochs,
            'learning_rate': lr,
            'top_k': top_k,
            'lambda_top': lambda_top,
            'effective_rank': effective_rank
        },
        'unnormalized_target_spectrum': target_spec.tolist(),
        'unnormalized_penultimate_spectrum': penult_spec.tolist(),
        'initial_kernel_spectrum': initial_eigvals.detach().cpu().numpy().tolist(),
        'input_covariance_spectrum': input_eigs.detach().cpu().numpy().tolist()
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
    print(f"Achieved spectrum range: [{penult_spec[-1]:.2e}, {penult_spec[0]:.2e}]")
    print(f"Condition numbers - Target: {target_spec[0]/(target_spec[-1]+1e-12):.2e}, "
          f"Achieved: {penult_spec[0]/(penult_spec[-1]+1e-12):.2e}")

if __name__ == "__main__":
    main()
