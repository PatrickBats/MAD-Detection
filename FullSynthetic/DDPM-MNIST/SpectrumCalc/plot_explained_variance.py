#!/usr/bin/env python3
"""
Plot cumulative explained variance across generations.
Shows how variance is distributed across principal components.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_results(filename='jacobian_ddim_results.pkl'):
    """Load DDIM Jacobian results."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def compute_explained_variance(eigenvalues):
    """
    Compute cumulative explained variance from eigenvalues.

    Args:
        eigenvalues: Array of eigenvalues (sorted descending)

    Returns:
        cumulative_variance: Cumulative explained variance (0 to 1)
    """
    total_variance = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return cumulative_variance


def find_components_for_variance(cumulative_variance, threshold):
    """
    Find number of components needed to explain threshold variance.

    Args:
        cumulative_variance: Cumulative variance array
        threshold: Variance threshold (e.g., 0.90 for 90%)

    Returns:
        Number of components needed
    """
    idx = np.argmax(cumulative_variance >= threshold)
    return idx + 1  


def plot_cumulative_variance():
    """Plot cumulative explained variance for all generations."""

    results = load_results()
    sorted_keys = sorted(results.keys(), key=lambda x: (-1 if x == 'initial' else x))

    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    stats = []

    for idx, gen in enumerate(sorted_keys):
        eigenvalues = results[gen]['eigenvalues']

        # Sort eigenvalues in descending order
        sorted_eigs = np.sort(eigenvalues)[::-1]

        # Compute cumulative variance
        cumulative_var = compute_explained_variance(sorted_eigs)

        # Number of components (1 to 784)
        n_components = np.arange(1, len(cumulative_var) + 1)

        # Plot cumulative variance
        color = colors[idx % len(colors)]
        # Format label
        label = 'Initial' if gen == 'initial' else f'Gen {gen}'
        ax1.plot(n_components, cumulative_var,
                label=label,
                color=color,
                linewidth=2,
                alpha=0.8)

        # Compute statistics
        n_90 = find_components_for_variance(cumulative_var, 0.90)
        n_95 = find_components_for_variance(cumulative_var, 0.95)
        n_99 = find_components_for_variance(cumulative_var, 0.99)

        stats.append({
            'gen': gen,
            'n_90': n_90,
            'n_95': n_95,
            'n_99': n_99,
            'color': color
        })

    ax1.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax1.text(650, 0.91, '90%', fontsize=10, color='gray')
    ax1.text(650, 0.96, '95%', fontsize=10, color='gray')
    ax1.text(650, 1.00, '99%', fontsize=10, color='gray')

    ax1.set_xlabel('Number of Principal Components', fontsize=12)
    ax1.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax1.set_title('Cumulative Explained Variance by Principal Components', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 784])
    ax1.set_ylim([0, 1.05])

    x_pos = np.arange(len(sorted_keys))
    width = 0.25

    n_90_vals = [s['n_90'] for s in stats]
    n_95_vals = [s['n_95'] for s in stats]
    n_99_vals = [s['n_99'] for s in stats]

    ax2.bar(x_pos - width, n_90_vals, width, label='90% variance', alpha=0.8)
    ax2.bar(x_pos, n_95_vals, width, label='95% variance', alpha=0.8)
    ax2.bar(x_pos + width, n_99_vals, width, label='99% variance', alpha=0.8)

    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Number of Components', fontsize=12)
    ax2.set_title('Components Needed to Explain Variance', fontsize=14)
    ax2.set_xticks(x_pos)
    # Format x-axis labels
    ax2.set_xticklabels(['Initial' if s["gen"] == 'initial' else f'Gen {s["gen"]}' for s in stats])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('explained_variance_ddim.png', dpi=150)
    print("\n✓ Plot saved to explained_variance_ddim.png")


  

    for s in stats:
        print(f"Gen {s['gen']:<8} {s['n_90']:<12} {s['n_95']:<12} {s['n_99']:<12}")


    plt.show()


if __name__ == "__main__":
    plot_cumulative_variance()
