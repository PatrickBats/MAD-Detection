"""
Simplified local Jacobian spectrum computation for MADness detection.
Core functionality only - no bloat.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

# Import model architecture
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import DDPM, ContextUnet


def load_model_and_anchors(generation, device='cuda'):
    """Load DDPM model and get local anchor points."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load model
    n_T = 500
    nn_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)
    ddpm = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

    checkpoint_path = os.path.join(script_dir, 'data', 'diffusion_outputs10', f'model_{generation}_w0.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract nn_model weights (they're prefixed with 'nn_model.')
    nn_state = {k[9:]: v for k, v in checkpoint.items() if k.startswith('nn_model.')}
    ddpm.nn_model.load_state_dict(nn_state)
    ddpm.nn_model.eval()

    # Load anchor points from previous generation's data
    prev_gen = generation - 1 if generation > 0 else -1
    anchor_file = 'gen_data_with_w_initial_w0' if prev_gen == -1 else f'gen_data_with_w{prev_gen}_w0'
    data_path = os.path.join(script_dir, 'data', 'diffusion_outputs10', anchor_file)

    # Try zipped version if unzipped doesn't exist
    if not os.path.exists(data_path):
        data_path = data_path + '.zip'

    if os.path.exists(data_path):
        data = torch.load(data_path, map_location=device)
        if len(data.shape) == 3:
            data = data.unsqueeze(1)
        # Sample 50 random anchors, normalize to [-1, 1]
        indices = torch.randperm(len(data))[:50]
        anchors = data[indices].float() / 127.5 - 1.0
    else:
        # Fallback to random anchors
        anchors = torch.randn(50, 1, 28, 28, device=device)

    return ddpm, anchors


def compute_jacobian_eigenvalues(model, x, t=1, epsilon=0.01, k=10, n_iters=30):
    """
    Compute top-k eigenvalues of J^T J where J is the Jacobian of the denoiser.
    Uses power iteration with deflation.
    """
    x = x.detach()
    eigenvalues = []
    eigenvectors = []

    # Add small perturbation for local analysis
    x_local = x + epsilon * torch.randn_like(x)

    for i in range(k):
        # Random initialization
        v = torch.randn_like(x_local)

        # Orthogonalize against previous eigenvectors
        for prev_v in eigenvectors:
            v = v - (v * prev_v).sum() * prev_v
        v = v / (v.norm() + 1e-10)

        # Power iteration
        for _ in range(n_iters):
            # Forward: J @ v
            x_local_grad = x_local.detach().requires_grad_(True)

            # Prepare inputs for the model
            c = torch.zeros(1, dtype=torch.long, device=x.device)  # Class label (integer)
            t_norm = torch.tensor([t], dtype=torch.float32, device=x.device) / model.n_T
            context_mask = torch.zeros(1).to(x.device)

            # Get model output
            with torch.enable_grad():
                eps_pred = model.nn_model(x_local_grad, c, t_norm, context_mask)

                # J @ v
                jv = torch.autograd.grad(eps_pred, x_local_grad, v, create_graph=False)[0]

                # J^T @ (J @ v)
                x_local_grad2 = x_local.detach().requires_grad_(True)
                # Need to recompute inputs for second forward pass
                c2 = torch.zeros(1, dtype=torch.long, device=x.device)  # Class label (integer)
                t_norm2 = torch.tensor([t], dtype=torch.float32, device=x.device) / model.n_T
                context_mask2 = torch.zeros(1).to(x.device)
                eps_pred2 = model.nn_model(x_local_grad2, c2, t_norm2, context_mask2)
                eps_pred2.backward(jv, retain_graph=False)
                w = x_local_grad2.grad

            # Compute eigenvalue BEFORE deflation
            lambda_curr = (v * w).sum() / (v * v).sum()

            # Deflate
            for prev_v in eigenvectors:
                w = w - (w * prev_v).sum() * prev_v

            # Update v
            v = w / (w.norm() + 1e-10)

        # Store converged eigenvalue (use last computed value)
        eigenvalues.append(lambda_curr.item())
        eigenvectors.append(v.detach())

        # Stop if eigenvalue is too small
        if abs(lambda_curr.item()) < 1e-8:
            break

    return eigenvalues


def main():
    """Compute and visualize eigenvalue evolution across generations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find available generations
    generations = []
    for gen in range(20):
        if os.path.exists(os.path.join(script_dir, 'data', 'diffusion_outputs10', f'model_{gen}_w0.pth')):
            generations.append(gen)

    print(f"Found {len(generations)} generations: {generations}")

    # Compute spectra
    all_eigenvalues = {}

    for gen in tqdm(generations, desc="Computing spectra"):
        model, anchors = load_model_and_anchors(gen, device)

        gen_eigenvalues = []
        for idx, anchor in enumerate(tqdm(anchors[:20], desc=f"Gen {gen}", leave=False)):  # Use 20 anchors for speed
            anchor = anchor.unsqueeze(0)
            try:
                eigs = compute_jacobian_eigenvalues(model, anchor, t=1, k=5)  # Top 5 eigenvalues
                gen_eigenvalues.extend(eigs)
            except Exception as e:
                if idx == 0:  # Only print error once per generation
                    print(f"Error in gen {gen}: {e}")
                continue

        all_eigenvalues[gen] = gen_eigenvalues
        print(f"Gen {gen}: collected {len(gen_eigenvalues)} eigenvalues")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(generations)))

    for i, gen in enumerate(generations):
        if gen in all_eigenvalues and len(all_eigenvalues[gen]) > 0:
            log_eigs = np.log10(np.abs(all_eigenvalues[gen]) + 1e-10)
            plt.hist(log_eigs, bins=20, alpha=0.5, color=colors[i],
                    label=f'Gen {gen}', density=True, edgecolor='black', linewidth=0.5)

    plt.xlabel('log10(|eigenvalue|)')
    plt.ylabel('Density')
    plt.title('Local Jacobian Eigenvalue Evolution (MADness Detection)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('eigenvalue_evolution_simple.png', dpi=150)
    plt.show()

    # Save results
    with open('eigenvalue_results_simple.pkl', 'wb') as f:
        pickle.dump(all_eigenvalues, f)

    # Print summary statistics
    print("\n=== Summary ===")
    for gen in generations:
        if gen in all_eigenvalues and len(all_eigenvalues[gen]) > 0:
            eigs = all_eigenvalues[gen]
            print(f"Gen {gen}: max={max(eigs):.4f}, mean={np.mean(eigs):.4f}, "
                  f"condition={max(eigs)/min(eigs) if min(eigs)>0 else np.inf:.1f}")


if __name__ == "__main__":
    main()