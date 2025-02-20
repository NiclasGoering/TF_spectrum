import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import matplotlib.pyplot as plt

################################################################################
# 1) Network Architecture
#    - First layer: (input_dim -> hidden_dim)
#    - (depth-2) hidden layers: (hidden_dim -> hidden_dim)
#    - If depth>1, penultimate layer: (hidden_dim -> hidden_dim)
#    - Final readout: (hidden_dim -> 1)
################################################################################
class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, mode='mup_no_align'):
        super().__init__()
        layers = []
        if depth < 1:
            raise ValueError("depth must be >= 1")

        # First hidden layer: d -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim, bias=True))

        # (depth-2) middle layers: hidden_dim -> hidden_dim
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

        # Penultimate hidden layer (if depth>1): hidden_dim -> hidden_dim
        if depth > 1:
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

        # Final readout layer: hidden_dim -> 1
        layers.append(nn.Linear(hidden_dim, 1, bias=True))

        self.network = nn.ModuleList(layers)
        self.mode = mode

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x

################################################################################
# 2) Create Target Kernel with Effective Rank
#    This function now accepts an 'effective_rank' parameter.
#    For indices i < effective_rank, eigenvalue = 1/(i+1)^alpha.
#    For the remaining indices, eigenvalue = eps (a very small number).
#    Then we scale so that trace(T) equals effective_rank.
################################################################################
def create_target_kernel(feature_dim: int, alpha: float, device: torch.device,
                           effective_rank: int = None, eps: float = 1e-8):
    """
    Creates a target kernel T ∈ ℝ^(feature_dim×feature_dim).
    
    If effective_rank is provided (< feature_dim), then the first
    effective_rank eigenvalues decay as 1/(i+1)^alpha and the remaining
    eigenvalues are set to eps.
    
    The eigenvalues are then scaled such that trace(T) equals effective_rank.
    """
    if effective_rank is None:
        effective_rank = feature_dim

    eig_vals = np.zeros(feature_dim, dtype=np.float32)
    # For the effective directions, decay as 1/(i+1)^alpha.
    for i in range(effective_rank):
        eig_vals[i] = 1.0 / ((i + 1) ** alpha)
    # For the remaining directions, set them to a very small value.
    for i in range(effective_rank, feature_dim):
        eig_vals[i] = eps

    # Scale so that the trace equals effective_rank.
    scale = effective_rank / eig_vals.sum()
    eig_vals_scaled = eig_vals * scale

    D = torch.diag(torch.tensor(eig_vals_scaled, device=device))
    # Create a random orthogonal matrix Q via QR decomposition.
    A = torch.randn(feature_dim, feature_dim, device=device)
    Q, _ = torch.linalg.qr(A)
    target_kernel = Q @ D @ Q.T
    target_kernel = 0.5 * (target_kernel + target_kernel.T)  # Ensure symmetry

    # Optionally shift up if any eigenvalue is negative
    eigvals = torch.linalg.eigvalsh(target_kernel)
    if eigvals[0] < 0:
        target_kernel = target_kernel - eigvals[0] * torch.eye(feature_dim, device=device)
    target_eigs = torch.sort(torch.linalg.eigvalsh(target_kernel))[0]
    return target_kernel, target_eigs

################################################################################
# 3) Smart Initialization for the Penultimate Layer Only
################################################################################
def smart_init_penultimate_only(model: DeepNN, target_kernel: torch.Tensor,
                                X: torch.Tensor, smart_eps: float = 1e-2, cov_eps: float = 1e-6):
    """
    Passes X through all but the last two layers to get h_in.
    Then computes Cov(h_in) and its eigen-decomposition (with clamping),
    and sets the penultimate layer weight to:
        W = sqrt(T)*inv_sqrt(Cov) + smart_eps * I.
    """
    device = X.device
    # Forward pass through all but last two layers.
    h_in = X
    for layer in model.network[:-2]:
        h_in = layer(h_in)
        
    N = h_in.shape[0]
    dim_h = h_in.shape[1]
    cov_h = (h_in.T @ h_in) / N + cov_eps * torch.eye(dim_h, device=device)

    # Eigen-decomposition with clamping.
    evalsC, evecsC = torch.linalg.eigh(cov_h)
    evalsC = torch.clamp(evalsC, min=smart_eps)
    Csqrt_inv = evecsC @ torch.diag(1.0 / torch.sqrt(evalsC)) @ evecsC.T

    # For the target kernel.
    evalsT, evecsT = torch.linalg.eigh(target_kernel)
    evalsT = torch.clamp(evalsT, min=smart_eps)
    Tsqrt = evecsT @ torch.diag(torch.sqrt(evalsT)) @ evecsT.T

    penultimate = model.network[-2]
    if not isinstance(penultimate, nn.Linear):
        print("Warning: penultimate layer is not Linear; skipping smart init.")
        return model

    out_f, in_f = penultimate.weight.shape  # weight shape: [out_features, in_features]
    if out_f != in_f:
        print(f"Warning: penultimate layer is not square ({out_f} vs {in_f}); skipping smart init.")
        return model
    if out_f != target_kernel.shape[0]:
        print("Warning: penultimate dimension does not match target kernel dimension.")
        return model
    if in_f != dim_h:
        print(f"Warning: mismatch between penultimate in_features={in_f} and h_in dimension={dim_h}.")
        return model

    W_target = Tsqrt @ Csqrt_inv + smart_eps * torch.eye(out_f, device=device)
    penultimate.weight.data.copy_(W_target)
    if penultimate.bias is not None:
        penultimate.bias.data.zero_()

    return model

################################################################################
# 4) LSUV Initialization (Optional)
################################################################################
def lsuv_init(model: nn.Module, X: torch.Tensor, needed_std: float = 1.0,
              tol: float = 0.1, max_iter: int = 10):
    model.eval()
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            outputs = []
            def hook(_, __, outp):
                outputs.append(outp)
            h = layer.register_forward_hook(hook)

            _ = model(X)
            if len(outputs) == 0:
                h.remove()
                continue
            std_now = outputs[0].std().item()
            count = 0
            while abs(std_now - needed_std) > tol and count < max_iter:
                scale = needed_std / (std_now + 1e-8)
                layer.weight.data.mul_(scale)
                outputs.clear()
                _ = model(X)
                std_now = outputs[0].std().item()
                count += 1

            print(f"LSUV init for layer {i}: final std = {std_now:.4f} after {count} iterations")
            h.remove()
    model.train()
    return model

################################################################################
# 5) Kernel Loss with Rank Preservation Penalty
################################################################################
def kernel_loss(model: nn.Module, X: torch.Tensor, target_kernel: torch.Tensor,
                lambda_eig: float = 1.0, use_log: bool = True,
                top_k: int = 20, lambda_top: float = 1e4,
                frob_scale: float = 1.0, eig_scale: float = 1.0,
                rank_preservation_weight: float = 0.0):
    """
    Computes loss comparing the penultimate feature covariance to target_kernel.
    In addition to Frobenius and eigenvalue (and top-k) losses, a rank preservation
    penalty is added: -sum(log(eigenvalues)) (to maximize the determinant).
    """
    feats = X
    for layer in model.network[:-1]:
        feats = layer(feats)
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
        # Clamp eigenvalues to avoid log(0)
        eig_C = torch.clamp(eig_C, min=1e-12)
        rank_penalty = -torch.sum(torch.log(eig_C))
        total_loss += rank_preservation_weight * rank_penalty

    return total_loss, loss_frob, loss_eig, top_eig_loss

################################################################################
# 6) Training Procedure
################################################################################
def train_kernel_modified(model: nn.Module, X: torch.Tensor, target_kernel: torch.Tensor,
                          epochs: int = 5000, lr: float = 1e-3,
                          lambda_eig: float = 10.0, use_log: bool = True,
                          top_k: int = 5, lambda_top: float = 100.0,
                          rank_preservation_weight: float = 0.0):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, l_frob, l_eig, l_top = kernel_loss(
            model, X, target_kernel,
            lambda_eig=lambda_eig,
            use_log=use_log,
            top_k=top_k,
            lambda_top=lambda_top,
            rank_preservation_weight=rank_preservation_weight
        )
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + 1)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4e} | "
                  f"Frob: {l_frob.item():.4e} | Eig: {l_eig.item():.4e} | Top-{top_k}: {l_top.item():.4e}")
    return model

################################################################################
# 7) Compute the Unnormalized Penultimate Kernel Spectrum
################################################################################
def compute_last_hidden_kernel_unnormalized_spectrum(model: nn.Module, X: torch.Tensor):
    with torch.no_grad():
        feats = X
        for layer in model.network[:-1]:
            feats = layer(feats)
        K_unnorm = feats.T @ feats
        eivals = torch.linalg.eigvalsh(K_unnorm)
    return eivals

################################################################################
# 8) Utility Functions to Save Results, Model, and Dataset
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
    # Hyperparameters
    d = 30                      # Input dimension
    hidden_size = 256           # Hidden dimension (for square penultimate layer)
    depth = 1                   # Architecture: (d->hidden), then (hidden->hidden), then (hidden->1)
    train_size = 300000         # Number of training samples
    mode = 'mup_no_align'
    alpha = 5.0                 # Target kernel decay exponent
    lambda_eig = 5.0
    use_log = True
    epochs = 5000
    lr = 1e-2
    top_k = 25
    lambda_top = 50.0
    rank = 0

    # Additional hyperparameters
    smart_eps = 1e-6            # Damping epsilon for smart initialization and eigenvalue clamping
    do_lsuv = False              # Whether to apply LSUV initialization
    rank_preservation_weight = 1.0  # Weight for rank preservation penalty
    # Set effective_rank for the target kernel to d (so d eigenvalues decay, rest nearly zero)
    effective_rank = d

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) Build the network (using default PyTorch initialization)
    model = DeepNN(d, hidden_size, depth, mode=mode).to(device)
    print("\n=== Layer Shapes ===")
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            print(f" Layer {i}: weight shape = {layer.weight.shape}")

    # 2) Create target kernel for dimension 'hidden_size' with effective_rank = d
    target_kernel_norm, target_eigs_norm = create_target_kernel(hidden_size, alpha, device,
                                                                 effective_rank=effective_rank,
                                                                 eps=1e-8)
    print("\nTarget kernel eigenvalues (ascending):")
    print(target_eigs_norm.detach().cpu().numpy())

    # 3) Generate input data X ~ N(0, I)
    X = torch.randn(train_size, d, device=device)
    input_cov = (X.T @ X) / train_size
    input_eigs = torch.linalg.eigvalsh(input_cov)
    print("\nInput covariance eigenvalues (ascending):")
    print(input_eigs.detach().cpu().numpy())

    # 4) Apply smart initialization for the penultimate layer
    print("\nSmart-initializing the penultimate layer ...")
    model = smart_init_penultimate_only(model, target_kernel_norm, X, smart_eps=smart_eps)

    # 5) Optionally apply LSUV initialization
    if do_lsuv:
        print("\nApplying LSUV initialization ...")
        model = lsuv_init(model, X, needed_std=1.0, tol=0.1, max_iter=10)
    else:
        print("\nSkipping LSUV initialization.")

    # 6) Inspect initial penultimate kernel spectrum
    with torch.no_grad():
        feats_init = X
        for layer in model.network[:-1]:
            feats_init = layer(feats_init)
        init_kernel = (feats_init.T @ feats_init) / train_size
        init_eigs = torch.linalg.eigvalsh(init_kernel)
    print("\nInitial kernel eigenvalues (ascending):")
    print(init_eigs.detach().cpu().numpy())

    # 7) Train the network with rank preservation penalty
    print("\nStarting training ...")
    model = train_kernel_modified(
        model, X, target_kernel_norm,
        epochs=epochs, lr=lr,
        lambda_eig=lambda_eig,
        use_log=use_log,
        top_k=top_k,
        lambda_top=lambda_top,
        rank_preservation_weight=rank_preservation_weight
    )

    # 8) Compute final (unnormalized) penultimate kernel spectrum
    targ_eigs_unnorm = target_eigs_norm * train_size
    penult_eigs_unnorm = compute_last_hidden_kernel_unnormalized_spectrum(model, X)

    targ_spec = np.sort(targ_eigs_unnorm.detach().cpu().numpy())[::-1]
    penult_spec = np.sort(penult_eigs_unnorm.detach().cpu().numpy())[::-1]

    # 9) Save results and plot the spectrum
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    smart_name = f"model_d{d}_hid{hidden_size}_depth{depth}_alpha{alpha}_{timestamp}"
    save_dir = os.path.join("/home/goring/TF_spectrum/results_pretrain_testgrid", f"results_{smart_name}")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.loglog(range(1, len(targ_spec)+1), targ_spec, 'o-', label='Target Spectrum', markersize=4)
    plt.loglog(range(1, len(penult_spec)+1), penult_spec, 's-', label='Penultimate Spectrum', markersize=4)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Penultimate Kernel Spectrum (Log-Log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plot_path = os.path.join(save_dir, f"spectrum_plot_{smart_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved spectrum plot to {plot_path}")

    results = {
        'hyperparams': {
            'input_dim': d,
            'hidden_dim': hidden_size,
            'depth': depth,
            'train_size': train_size,
            'alpha': alpha,
            'lambda_eig': lambda_eig,
            'use_log': use_log,
            'epochs': epochs,
            'learning_rate': lr,
            'top_k': top_k,
            'lambda_top': lambda_top,
            'smart_eps': smart_eps,
            'do_lsuv': do_lsuv,
            'rank_preservation_weight': rank_preservation_weight,
            'effective_rank': effective_rank
        },
        'target_spectrum_unnorm': targ_spec.tolist(),
        'penult_spectrum_unnorm': penult_spec.tolist(),
        'initial_kernel_spectrum': init_eigs.detach().cpu().numpy().tolist(),
        'input_cov_eigs': input_eigs.detach().cpu().numpy().tolist()
    }
    save_results([results], save_dir, smart_name)
    print("Saved JSON results.")

    with torch.no_grad():
        y_out = model(X)
    dataset_path = os.path.join(save_dir, f"dataset_{smart_name}.pt")
    save_dataset(X, y_out, dataset_path, rank)

    tk_path = os.path.join(save_dir, f"target_kernel_{smart_name}.pt")
    torch.save(target_kernel_norm.detach().cpu(), tk_path)
    print(f"Saved target kernel to {tk_path}")

    model_path = os.path.join(save_dir, f"model_{smart_name}.pt")
    save_model(model, model_path)
    print(f"Saved model to {model_path}")

    print("\nFinal spectrum comparison summary:")
    print(f"Target range: [{targ_spec[-1]:.2e}, {targ_spec[0]:.2e}]")
    print(f"Penultimate range: [{penult_spec[-1]:.2e}, {penult_spec[0]:.2e}]")
    cond_target = targ_spec[0] / (targ_spec[-1] + 1e-12)
    cond_achieved = penult_spec[0] / (penult_spec[-1] + 1e-12)
    print(f"Condition numbers => Target: {cond_target:.2e}, Penultimate: {cond_achieved:.2e}")

if __name__ == "__main__":
    main()
