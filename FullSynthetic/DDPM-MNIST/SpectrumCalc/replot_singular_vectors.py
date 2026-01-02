#!/usr/bin/env python3
"""
Replot singular vector visualizations from existing pkl file.
No Jacobian/SVD recomputation - just loads results and generates plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
PKL_FILE = os.path.join(script_dir, 'singular_vector_results.pkl')


def load_results():
    with open(PKL_FILE, 'rb') as f:
        return pickle.load(f)


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

        ax_left = axes[row, 0] if n_gens > 1 else axes[0]
        ax_left.set_ylabel(gen_labels[row], fontsize=11, fontweight='bold')

    fig.suptitle('Most Amplified Small Eigenvalues (Artifact Directions)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'amplified_singular_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_largest_eigenvectors(results, top_k=10):
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    indices = list(range(top_k))
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

        ax_left = axes[row, 0] if n_gens > 1 else axes[0]
        ax_left.set_ylabel(gen_labels[row], fontsize=11, fontweight='bold')

    fig.suptitle('Largest Eigenvalues (Main Image Directions)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'largest_singular_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_sampled_eigenvectors(results, step=50):
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

        ax_left = axes[row, 0] if n_gens > 1 else axes[0]
        ax_left.set_ylabel(gen_labels[row], fontsize=11, fontweight='bold')

    fig.text(0.5, 0.02, 'Eigenvalue Index (sorted by magnitude, largest first)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Generation', va='center', rotation='vertical', fontsize=12)
    fig.suptitle(f'Sampled Eigenvectors (every {step}th)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    plt.savefig(os.path.join(script_dir, 'sampled_singular_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compute_eigenvector_variance(results):
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    variances = {}
    for gen in generations:
        U = results[gen]['U']
        variances[gen] = np.array([np.var(U[:, i]) for i in range(U.shape[1])])
    return variances


def plot_variance_metric(results):
    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    variances = compute_eigenvector_variance(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for gen in generations:
        label = 'Initial' if gen == 'initial' else f'Gen {gen}'
        axes[0].semilogy(variances[gen], label=label, alpha=0.7)
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('Variance of U vector')
    axes[0].set_title('Eigenvector Variance Across Spectrum')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    mean_vars = [np.mean(variances[gen]) for gen in generations]
    gen_labels = ['Initial' if g == 'initial' else f'Gen {g}' for g in generations]
    axes[1].bar(gen_labels, mean_vars)
    axes[1].set_ylabel('Mean Variance')
    axes[1].set_title('Mean Eigenvector Variance per Generation')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

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
    plt.savefig(os.path.join(script_dir, 'eigenvector_variance.png'), dpi=150, bbox_inches='tight')
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
    plt.savefig(os.path.join(script_dir, 'eigenvalue_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print(f"Loading results from {PKL_FILE}...")
    results = load_results()

    generations = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    print(f"Found generations: {generations}")

    print("Plotting eigenvalue changes...")
    plot_eigenvalue_changes(results)

    print("Plotting amplified vectors...")
    plot_amplified_vectors(results)

    print("Plotting largest eigenvectors...")
    plot_largest_eigenvectors(results)

    print("Plotting sampled eigenvectors...")
    plot_sampled_eigenvectors(results, step=50)

    print("Plotting variance metric...")
    plot_variance_metric(results)

    print("Done!")


if __name__ == '__main__':
    main()
