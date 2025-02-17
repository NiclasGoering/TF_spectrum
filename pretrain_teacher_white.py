import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ===============  Your MLP Class (FFNN.py)  ===============
class DeepNN(nn.Module):
    """
    A very simple feed-forward network with 'depth' layers (not counting final output).
    Penultimate dimension is 'hidden_size'. 
    'mode' can be used if you have variants, but you can ignore that for now.
    """
    def __init__(self, input_dim, hidden_size, depth=1, mode='mup_no_align'):
        super().__init__()
        self.mode = mode
        
        layers = []
        in_dim = input_dim

        # If depth=1, we have a single hidden layer -> penultimate -> final
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_size, bias=True))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        
        # Suppose your final output layer is also dimension 'hidden_size -> 1' or something:
        # For "white penultimate kernel," the last layer is not relevant if we only measure penultimate features.
        # But let's just put an output dimension=1 for demonstration:
        layers.append(nn.Linear(hidden_size, 1, bias=True))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ===============  Your Utility Functions (utils2.py)  ===============
def save_results(results, save_dir, smart_name):
    import json
    path = os.path.join(save_dir, f"results_{smart_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results JSON to {path}")


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Saved model parameters to {model_path}")


def save_dataset(X, y, path, rank=0):
    # Just saving a sample dataset
    data_dict = {
        "X": X.cpu(),
        "y": y.cpu(),
        "rank": rank
    }
    torch.save(data_dict, path)
    print(f"Saved dataset to {path}")


# ===============  Modified create_target_kernel  ===============
def create_target_kernel(feature_dim: int, alpha: float, device: torch.device):
    """
    Creates a target kernel matrix T ∈ ℝ^(feature_dim×feature_dim) whose eigenvalues decay as 1/(i+1)^alpha.
    If alpha=0, we skip the random-orthogonal step and produce exactly the identity matrix.

    The code ensures trace(T) = feature_dim by scaling if alpha!=0.
    """
    if alpha == 0:
        # Just produce the identity matrix
        print("[INFO] alpha=0 -> Using pure Identity kernel.")
        target_kernel = torch.eye(feature_dim, device=device)
        return target_kernel, torch.sort(torch.linalg.eigvalsh(target_kernel))[0]

    # Otherwise, do the usual method with Q, D, Q^T
    eig_vals = np.array([1.0 / ((i + 1) ** alpha) for i in range(feature_dim)], dtype=np.float32)
    scale = feature_dim / eig_vals.sum()
    eig_vals_scaled = eig_vals * scale
    D = torch.diag(torch.tensor(eig_vals_scaled, device=device))

    # Create a random orthogonal matrix Q
    A = torch.randn(feature_dim, feature_dim, device=device)
    Q, _ = torch.linalg.qr(A)
    
    target_kernel = Q @ D @ Q.T
    target_kernel = 0.5 * (target_kernel + target_kernel.T)  # symmetrize

    # Shift if needed to ensure PSD
    eigvals = torch.linalg.eigvalsh(target_kernel)
    if eigvals[0] < 0:
        target_kernel = target_kernel - eigvals[0] * torch.eye(feature_dim, device=device)
    target_eigenvals = torch.sort(torch.linalg.eigvalsh(target_kernel))[0]
    return target_kernel, target_eigenvals


# ===============  Smart Init  ===============
def smart_initialize_with_input_stats_modified(
    model: DeepNN, 
    target_kernel: torch.Tensor, 
    X: torch.Tensor, 
    small_bias: float = 1e-2
):
    """
    Initializes the network as follows:
      1. For all layers up to (but not including) the penultimate layer, use identity-like pattern.
      2. For the penultimate layer, compute weights using input covariance and the sqrt of the target kernel
         (or do a partial identity if you prefer).
    """
    # Identity-like on earlier layers
    for layer in model.network[:-2]:
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
            weight = torch.zeros(out_features, in_features, device=layer.weight.device)
            if out_features >= in_features:
                weight[:in_features, :] = torch.eye(in_features, device=layer.weight.device)
            else:
                weight[:, :out_features] = torch.eye(out_features, device=layer.weight.device)
            layer.weight.data.copy_(weight)
            if layer.bias is not None:
                layer.bias.data.fill_(small_bias)
    
    # The "penultimate" layer is the last Linear before the final.
    penultimate_layer = model.network[-2]
    if isinstance(penultimate_layer, nn.Linear):
        N = X.shape[0]
        input_cov = (X.T @ X) / N
        eps = 1e-6
        input_cov = input_cov + eps * torch.eye(input_cov.shape[0], device=input_cov.device)
        
        # sqrt of input cov
        vals_x, vecs_x = torch.linalg.eigh(input_cov)
        x_sqrt = vecs_x @ torch.diag(torch.sqrt(vals_x)) @ vecs_x.T
        x_sqrt_inv = vecs_x @ torch.diag(1.0 / torch.sqrt(vals_x)) @ vecs_x.T
        
        # sqrt of target kernel
        vals_t, vecs_t = torch.linalg.eigh(target_kernel)
        vals_t = torch.clamp(vals_t, min=1e-10)
        t_sqrt = vecs_t @ torch.diag(torch.sqrt(vals_t)) @ vecs_t.T

        # W_target
        W_target = t_sqrt @ x_sqrt_inv

        # add small identity
        W_target = W_target + 1e-2 * torch.eye(W_target.size(0), device=W_target.device)
        
        # copy to penultimate
        penultimate_layer.weight.data.copy_(W_target.to(penultimate_layer.weight.device))

        if penultimate_layer.bias is not None:
            penultimate_layer.bias.data.zero_()
    
    return model


# ===============  LSUV  ===============
def lsuv_init(model: DeepNN, X: torch.Tensor, needed_std: float = 1.0, tol: float = 0.1, max_iter: int = 10):
    model.eval()
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            outputs = []
            def hook(module, inp, outp):
                outputs.append(outp)
            hook_handle = layer.register_forward_hook(hook)
            
            # forward pass
            _ = model(X)
            if len(outputs) == 0:
                hook_handle.remove()
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
            hook_handle.remove()
    model.train()
    return model


# ===============  Kernel Loss  ===============
def kernel_loss(
    model: DeepNN, X: torch.Tensor, target_kernel: torch.Tensor,
    lambda_eig: float = 1.0, use_log: bool = True,
    top_k: int = 20, lambda_top: float = 1e4,
    frob_scale: float = 10.0, eig_scale: float = 1.0
):
    # penultimate features
    features = X
    for layer in model.network[:-1]:
        features = layer(features)
    N = features.shape[0]
    C_norm = features.T @ features / N

    # small reg
    I = torch.eye(C_norm.shape[0], device=C_norm.device)
    C_norm_reg = C_norm + 1e-6 * I

    # Frobenius difference
    frob_diff = torch.norm(C_norm_reg - target_kernel, p='fro')**2
    target_norm_sq = torch.norm(target_kernel, p='fro')**2 + 1e-8
    loss_frob = frob_diff / target_norm_sq
    loss_frob = frob_scale * loss_frob

    # eigenvalue difference
    eig_C = torch.linalg.eigvalsh(C_norm)
    eig_T = torch.linalg.eigvalsh(target_kernel)
    
    if use_log:
        eps = 1e-8
        eig_C = torch.clamp(eig_C, min=eps)
        eig_T = torch.clamp(eig_T, min=eps)
        loss_eig = torch.sum((torch.log(eig_C) - torch.log(eig_T))**2)
    else:
        loss_eig = torch.sum((eig_C - eig_T)**2)

    loss_eig = eig_scale * loss_eig
    
    # top-k penalty
    top_eig_loss = torch.mean((eig_C[-top_k:] - eig_T[-top_k:])**2)

    total_loss = loss_frob + lambda_eig * loss_eig + lambda_top * top_eig_loss
    return total_loss, loss_frob, loss_eig, top_eig_loss


# ===============  Train Kernel  ===============
def train_kernel_modified(
    model: DeepNN, X: torch.Tensor, target_kernel: torch.Tensor, 
    epochs: int = 5000, lr: float = 1e-3, 
    lambda_eig: float = 10.0, use_log: bool = True,
    top_k: int = 5, lambda_top: float = 100.0
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_frob, loss_eig_val, top_eig_loss = kernel_loss(
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
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.4e} "
                  f"| Frobenius: {loss_frob.item():.4e} "
                  f"| Eig Loss: {loss_eig_val.item():.4e} "
                  f"| Top-{top_k} Loss: {top_eig_loss.item():.4e}")
    return model


# ===============  Compute last hidden kernel (unnormalized)  ===============
def compute_last_hidden_kernel_unnormalized_spectrum(model: DeepNN, X: torch.Tensor):
    with torch.no_grad():
        features = X
        for layer in model.network[:-1]:
            features = layer(features)
        K_unnorm = features.T @ features
        eigenvalues = torch.linalg.eigvalsh(K_unnorm)
    return eigenvalues


# ===============  MAIN  ===============
def main():
    # --- Hyperparameters ---
    d = 10                       # Input dimension
    hidden_size = 512           # Penultimate dimension
    depth = 1                   # Just one hidden layer
    train_size = 300000         # Number of training samples
    mode = 'mup_no_align'
    alpha = 0.0                 # <-- If 0.0, we do a pure identity kernel in create_target_kernel
    lambda_eig = 10.0
    use_log = False
    epochs = 40000
    lr = 1e-2
    top_k = 10
    lambda_top = 2.0  # set to 0 if you don't want top-k penalty

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the network
    model = DeepNN(d, hidden_size, depth, mode=mode).to(device)

    # Create target kernel
    target_kernel_norm, target_eigenvalues_norm = create_target_kernel(hidden_size, alpha, device)
    print("\n[Target kernel eigenvalues (ascending)]")
    print(target_eigenvalues_norm.detach().cpu().numpy())

    # Generate input data
    X = torch.randn(train_size, d, device=device)
    print(f"\nX shape: {X.shape}, e.g. {train_size} samples in {d}-dim")

    # Inspect input covariance
    input_stats = (X.T @ X) / train_size
    input_eigvals = torch.linalg.eigvalsh(input_stats)
    print("\n[Input covariance eigenvalues (ascending)]")
    print(input_eigvals.detach().cpu().numpy())

    # Smart initialization
    print("\nPerforming smart initialization with input statistics...")
    model = smart_initialize_with_input_stats_modified(model, target_kernel_norm, X, small_bias=1e-2)

    # LSUV init
    print("\nApplying LSUV initialization...")
    model = lsuv_init(model, X, needed_std=1.0, tol=0.1, max_iter=10)

    # Check initial kernel
    with torch.no_grad():
        features = X
        for layer in model.network[:-1]:
            features = layer(features)
        initial_kernel = (features.T @ features) / train_size
        initial_eigvals = torch.linalg.eigvalsh(initial_kernel)

    print("\n[Initial kernel eigenvalues (after init, ascending)]")
    print(initial_eigvals.detach().cpu().numpy())

    # Train
    print("\n[Starting training ...]")
    model = train_kernel_modified(
        model, X, target_kernel_norm, 
        epochs=epochs, lr=lr,
        lambda_eig=lambda_eig, use_log=use_log,
        top_k=top_k, lambda_top=lambda_top
    )

    # Final spectrum
    # (For alpha=0, target is literally Identity => unnormalized target = I * train_size)
    target_eigenvalues_unnorm = target_eigenvalues_norm * train_size
    last_hidden_eigenvalues_unnorm = compute_last_hidden_kernel_unnormalized_spectrum(model, X)

    target_spec = np.sort(target_eigenvalues_unnorm.detach().cpu().numpy())[::-1]
    last_hidden_spec = np.sort(last_hidden_eigenvalues_unnorm.detach().cpu().numpy())[::-1]

    # Save plots, etc.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    smart_name = f"model_d{d}_hidden{hidden_size}_depth{depth}_alpha{alpha}_{timestamp}"
    save_dir = os.path.join("/home/goring/TF_spectrum/results_pretrain_testgrid", f"results_{smart_name}")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.loglog(np.arange(1, len(target_spec) + 1), target_spec, 'o-', label='Target Kernel Spectrum', markersize=4)
    plt.loglog(np.arange(1, len(last_hidden_spec) + 1), last_hidden_spec, 's-', label='Last Hidden Kernel Spectrum', markersize=4)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Kernel Spectrum Comparison (Log-Log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plot_path = os.path.join(save_dir, f"spectrum_plot_{smart_name}.png")
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
            'lambda_top': lambda_top
        },
        'unnormalized_target_spectrum': target_spec.tolist(),
        'unnormalized_last_hidden_spectrum': last_hidden_spec.tolist(),
        'initial_kernel_spectrum': initial_eigvals.detach().cpu().numpy().tolist(),
        'input_covariance_spectrum': input_eigvals.detach().cpu().numpy().tolist()
    }
    save_results([results], save_dir, smart_name)
    print("Saved results (hyperparameters and spectra).")

    # Save dataset
    with torch.no_grad():
        y = model(X)
    dataset_path = os.path.join(save_dir, f"dataset_{smart_name}.pt")
    save_dataset(X, y, dataset_path, rank=0)

    # Save target kernel
    target_kernel_path = os.path.join(save_dir, f"target_kernel_{smart_name}.pt")
    torch.save(target_kernel_norm.detach().cpu(), target_kernel_path)
    print(f"Saved target kernel to {target_kernel_path}")

    # Save model
    model_path = os.path.join(save_dir, f"model_{smart_name}.pt")
    save_model(model, model_path)
    print(f"Saved model to {model_path}")

    # Final summary
    print("\n[Final spectrum comparison summary]")
    print(f"Target spectrum range: [{target_spec[-1]:.2e}, {target_spec[0]:.2e}]")
    print(f"Achieved spectrum range: [{last_hidden_spec[-1]:.2e}, {last_hidden_spec[0]:.2e}]")
    cond_target = (target_spec[0] / target_spec[-1]) if target_spec[-1]!=0 else float('inf')
    cond_achieved = (last_hidden_spec[0] / last_hidden_spec[-1]) if last_hidden_spec[-1]!=0 else float('inf')
    print(f"Condition numbers - Target: {cond_target:.2e}, Achieved: {cond_achieved:.2e}")


if __name__ == "__main__":
    main()
