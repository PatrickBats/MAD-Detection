#!/usr/bin/env python3
"""
Visualize amplified singular vectors from DDIM Jacobian to understand MADness artifacts.

1. Compute Jacobian J of DDIM for each generation
2. SVD: J = U @ S @ V^T
3. Find small eigenvalues that grew most under MADness
4. Visualize their U vectors (potential artifact directions)
"""

import torch
import torch.autograd.functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from metrics import DDPM, ContextUnet


def load_model(generation, device='cuda'):
    n_T = 500
    nn_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)
    ddpm = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

    if generation == 'initial':
        path = os.path.join(parent_dir, 'data', 'diffusion_outputs10', 'model_initial.pth')
    else:
        path = os.path.join(parent_dir, 'data', 'diffusion_outputs10', f'model_{generation}_w0.pth')

    checkpoint = torch.load(path, map_location=device)
    nn_state = {k[9:]: v for k, v in checkpoint.items() if k.startswith('nn_model.')}
    ddpm.nn_model.load_state_dict(nn_state)
    ddpm.nn_model.eval()
    ddpm.eval()
    return ddpm


def ddim_generation(z, model, class_label=7):
    device = z.device
    c = torch.tensor([class_label], device=device, dtype=torch.long)
    context_mask = torch.zeros(1, device=device, dtype=torch.float32)
    x = z.clone().float()

    for t in range(500, 0, -1):
        t_norm = torch.tensor([t / model.n_T], device=device, dtype=torch.float32).view(1, 1, 1, 1)
        eps = model.nn_model(x, c, t_norm, context_mask)
        if t > 1:
            x_0_pred = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]
            x = model.sqrtab[t-1] * x_0_pred + model.sqrtmab[t-1] * eps
        else:
            x = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]
    return x


def compute_jacobian_svd(model, z_fixed, class_label=7):
    z_flat = z_fixed.flatten().detach().float().requires_grad_(True)

    def gen_func(z_input):
        z_shaped = z_input.reshape(1, 1, 28, 28)
        return ddim_generation(z_shaped, model, class_label).flatten()

    jacobian = F.jacobian(gen_func, z_flat)
    U, S, Vh = torch.linalg.svd(jacobian, full_matrices=False)
    return U.cpu().numpy(), S.cpu().numpy()


def process_generation(args):
    """Process a single generation on a specific GPU."""
    gen, gpu_id, z_fixed_cpu, class_label = args
    device = f'cuda:{gpu_id}'

    print(f"[GPU {gpu_id}] Processing {gen}...")
    model = load_model(gen, device)
    z_fixed = z_fixed_cpu.to(device)
    U, S = compute_jacobian_svd(model, z_fixed, class_label)

    del model
    torch.cuda.empty_cache()

    return gen, {'U': U, 'S': S}


def find_amplified_eigenvalues(results, top_k=10, small_percentile=50):
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    S0 = results[generations[0]]['S']**2
    ST = results[generations[-1]]['S']**2

    threshold = np.percentile(S0, small_percentile)
    mask = S0 < threshold

    ratios = np.zeros_like(S0)
    ratios[mask] = ST[mask] / (S0[mask] + 1e-15)

    indices = np.argsort(-ratios)[:top_k]
    return indices, ratios[indices], S0[indices], ST[indices]


def plot_amplified_vectors(results, top_k=10):
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    indices, ratios, _, _ = find_amplified_eigenvalues(results, top_k)
    gen_labels = ['Initial' if g == 'initial' else f'Gen {g}' for g in generations]

    n_gens = len(generations)
    fig, axes = plt.subplots(n_gens, top_k, figsize=(2*top_k, 2.5*n_gens))

    for row, gen in enumerate(generations):
        U, S = results[gen]['U'], results[gen]['S']
        for col, idx in enumerate(indices):
            ax = axes[row, col] if n_gens > 1 else axes[col]
            ax.imshow(U[:, idx].reshape(28, 28), cmap='RdBu', vmin=-0.2, vmax=0.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'idx={idx}\nλ={S[idx]**2:.1e}', fontsize=9)

        # Y-axis label for generation
        ax_left = axes[row, 0] if n_gens > 1 else axes[0]
        ax_left.set_ylabel(gen_labels[row], fontsize=11, fontweight='bold')

    fig.suptitle('Most Amplified Small Eigenvalues (Artifact Directions)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'amplified_singular_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_largest_eigenvectors(results, top_k=10):
    """Plot U vectors for largest eigenvalues across all generations."""
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    indices = list(range(top_k))  # Largest eigenvalues are first
    gen_labels = ['Initial' if g == 'initial' else f'Gen {g}' for g in generations]

    n_gens = len(generations)
    fig, axes = plt.subplots(n_gens, top_k, figsize=(2*top_k, 2.5*n_gens))

    for row, gen in enumerate(generations):
        U, S = results[gen]['U'], results[gen]['S']
        for col, idx in enumerate(indices):
            ax = axes[row, col] if n_gens > 1 else axes[col]
            ax.imshow(U[:, idx].reshape(28, 28), cmap='RdBu', vmin=-0.2, vmax=0.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'idx={idx}\nλ={S[idx]**2:.1e}', fontsize=9)

        # Y-axis label for generation
        ax_left = axes[row, 0] if n_gens > 1 else axes[0]
        ax_left.set_ylabel(gen_labels[row], fontsize=11, fontweight='bold')

    fig.suptitle('Largest Eigenvalues (Main Image Directions)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'largest_singular_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_sampled_eigenvectors(results, step=50):
    """Sample every Nth eigenvector across the spectrum."""
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    gen_labels = ['Initial' if g == 'initial' else f'Gen {g}' for g in generations]
    n_total = results[generations[0]]['S'].shape[0]
    indices = list(range(0, n_total, step))

    n_gens = len(generations)
    n_cols = len(indices)
    fig, axes = plt.subplots(n_gens, n_cols, figsize=(1.8*n_cols, 2.5*n_gens))

    for row, gen in enumerate(generations):
        U, S = results[gen]['U'], results[gen]['S']
        for col, idx in enumerate(indices):
            ax = axes[row, col] if n_gens > 1 else axes[col]
            ax.imshow(U[:, idx].reshape(28, 28), cmap='RdBu', vmin=-0.2, vmax=0.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'idx={idx}', fontsize=10)

        # Y-axis label for generation
        ax_left = axes[row, 0] if n_gens > 1 else axes[0]
        ax_left.set_ylabel(gen_labels[row], fontsize=11, fontweight='bold')

    # Add x-axis label
    fig.text(0.5, 0.02, 'Eigenvalue Index (sorted by magnitude, largest first)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Generation', va='center', rotation='vertical', fontsize=12)
    fig.suptitle(f'Sampled Eigenvectors (every {step}th)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    plt.savefig(os.path.join(script_dir, 'sampled_singular_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compute_eigenvector_variance(results):
    """Compute variance of each U vector as sparsity metric."""
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)

    variances = {}
    for gen in generations:
        U = results[gen]['U']
        # Variance of each column (eigenvector) reshaped as image
        variances[gen] = np.array([np.var(U[:, i]) for i in range(U.shape[1])])

    return variances


def plot_variance_metric(results):
    """Plot variance of eigenvectors as collapse metric."""
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    variances = compute_eigenvector_variance(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Variance across spectrum for each generation
    for gen in generations:
        label = 'Initial' if gen == 'initial' else f'Gen {gen}'
        axes[0].semilogy(variances[gen], label=label, alpha=0.7)
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('Variance of U vector')
    axes[0].set_title('Eigenvector Variance Across Spectrum')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Mean variance per generation (overall sparsity trend)
    mean_vars = [np.mean(variances[gen]) for gen in generations]
    gen_labels = ['Initial' if g == 'initial' else f'Gen {g}' for g in generations]
    axes[1].bar(gen_labels, mean_vars)
    axes[1].set_ylabel('Mean Variance')
    axes[1].set_title('Mean Eigenvector Variance per Generation')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Variance ratio (last/first) across spectrum
    first_var = variances[generations[0]]
    last_var = variances[generations[-1]]
    ratio = last_var / (first_var + 1e-15)
    axes[2].semilogy(ratio)
    axes[2].axhline(y=1, linestyle='--', alpha=0.5, color='red')
    axes[2].set_xlabel('Eigenvalue Index')
    axes[2].set_ylabel('Variance Ratio (Last/First)')
    axes[2].set_title('Variance Change Across Generations')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'eigenvector_variance.png'), dpi=150)
    plt.close()


def plot_eigenvalue_changes(results):
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    S_first, S_last = results[generations[0]]['S'], results[generations[-1]]['S']
    ratio = (S_last**2) / (S_first**2 + 1e-15)

    indices, ratios, _, _ = find_amplified_eigenvalues(results, top_k=20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(ratio)
    axes[0].axhline(y=1, linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('Ratio (Last/First)')
    axes[0].set_title('Eigenvalue Amplification')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(indices)), ratios)
    axes[1].set_xticks(range(len(indices)))
    axes[1].set_xticklabels([str(i) for i in indices], rotation=45, fontsize=8)
    axes[1].set_xlabel('Eigenvalue Index')
    axes[1].set_ylabel('Amplification Ratio')
    axes[1].set_title('Top 20 Most Amplified')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'eigenvalue_trajectories.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', nargs='+', default=['initial', '0', '1', '2', '3', '4', '5'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generations = [g if g == 'initial' else int(g) for g in args.generations]
    num_gpus = torch.cuda.device_count()

    print(f"Using {num_gpus} GPUs for {len(generations)} generations")

    torch.manual_seed(args.seed)
    z_fixed_cpu = torch.randn(1, 1, 28, 28)

    # Prepare args: (gen, gpu_id, z_fixed_cpu, class_label)
    task_args = [(gen, i % num_gpus, z_fixed_cpu, 7) for i, gen in enumerate(generations)]

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=min(num_gpus, len(generations))) as pool:
        results_list = pool.map(process_generation, task_args)

    results = {gen: data for gen, data in results_list}

    # Save results to pkl for replotting later
    pkl_path = os.path.join(script_dir, 'singular_vector_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {pkl_path}")

    plot_eigenvalue_changes(results)
    plot_amplified_vectors(results)
    plot_largest_eigenvectors(results)
    plot_sampled_eigenvectors(results, step=50)
    plot_variance_metric(results)
    print("Done!")


if __name__ == '__main__':
    main()
