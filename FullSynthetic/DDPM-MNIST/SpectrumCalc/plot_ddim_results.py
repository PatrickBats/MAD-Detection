#!/usr/bin/env python3
"""
Plot eigenvalue spectra and images from precomputed DDIM Jacobian results.
Reads from jacobian_ddim_results.pkl and creates the same plots as jacobian_ddim.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_results(filename='jacobian_ddim_results.pkl'):
    """Load DDIM Jacobian results from pickle file."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_eigenvalue_comparison(results):
    """Plot eigenvalue spectra comparison."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan']

    sorted_keys = sorted(results.keys(), key=lambda x: (-1 if x == 'initial' else x))

    for idx, gen in enumerate(sorted_keys):
        eigs = results[gen]['eigenvalues']
        color = colors[idx % len(colors)]

        label = 'Initial' if gen == 'initial' else f'Gen {gen}'

        ax1.semilogy(range(1, len(eigs)+1), eigs,
                    label=label, color=color, linewidth=2)

        
        ax2.hist(np.log10(eigs + 1e-10), bins=50, alpha=0.5,
                label=label, color=color)

    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title('Eigenvalue Spectrum\n(Full 500-step DDIM Jacobian)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('log10(Eigenvalue)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of log(Eigenvalues)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jacobian_ddim_spectrum.png', dpi=150)
    plt.close()

    print("\n✓ Plot saved to jacobian_ddim_spectrum.png")


def plot_generated_images(results):
    """Plot generated images from each generation."""

    n_gens = len(results)
    if n_gens == 0:
        return

    fig, axes = plt.subplots(1, n_gens, figsize=(3*n_gens, 3))

    if n_gens == 1:
        axes = [axes]

    # Sort keys with 'initial' first
    sorted_keys = sorted(results.keys(), key=lambda x: (-1 if x == 'initial' else x))

    for idx, gen in enumerate(sorted_keys):
        image = results[gen]['image']

        # Reshape and display
        if len(image.shape) == 4:
            image = image[0, 0]
        elif len(image.shape) == 3:
            image = image[0]
        elif len(image.shape) == 1:
            image = image.reshape(28, 28)

        # Format title
        title = 'Initial' if gen == 'initial' else f'Gen {gen}'

        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')

    plt.suptitle('Images Generated with DDIM (Deterministic, 500 steps)', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_ddim_images.png', dpi=150)
    plt.close()



def print_statistics(results):
    """Print eigenvalue statistics for all generations."""

    # Sort keys with 'initial' first
    sorted_keys = sorted(results.keys(), key=lambda x: (-1 if x == 'initial' else x))


    for gen in sorted_keys:
        eigs = results[gen]['eigenvalues']

        # Compute effective rank
        eigs_normalized = eigs / eigs.sum()
        eff_rank = np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))

        label = 'Initial' if gen == 'initial' else f'Gen {gen}'

        print(f"\n{label}:")
        print(f"  Total eigenvalues: {len(eigs)}")
        print(f"  Max eigenvalue: {eigs[0]:.4f}")
        print(f"  Min eigenvalue: {eigs[-1]:.4e}")
        print(f"  Median eigenvalue: {np.median(eigs):.4e}")
        print(f"  Effective rank: {eff_rank:.2f} / 784")

    print("\n" + "="*70)


def main():
    """Load results and create all plots."""

    results = load_results()

    print(f"Loaded results for generations: {sorted(results.keys(), key=lambda x: (-1 if x == 'initial' else x))}")

    # Print statistics
    print_statistics(results)

    # Create plots
    print("\nCreating plots...")
    plot_eigenvalue_comparison(results)
    plot_generated_images(results)


if __name__ == "__main__":
    main()
