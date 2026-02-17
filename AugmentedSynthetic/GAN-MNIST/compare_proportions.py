#!/usr/bin/env python3
"""
Compare results across all constant proportion experiments.

Creates a comprehensive visualization showing:
1. FID scores across generations for each proportion
2. Effective rank across generations for each proportion
3. Critical threshold analysis
4. Sample quality visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from torchvision.utils import make_grid
import matplotlib.gridspec as gridspec

# Proportions to compare
PROPORTIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


def load_fid_results(script_dir):
    """Load or compute FID results."""
    # This would load from saved FID computation
    # For now, return placeholder structure
    return None


def load_jacobian_results(script_dir, class_label=7):
    """Load Jacobian results from pickle file."""
    pkl_path = os.path.join(script_dir, f'jacobian_proportions_class{class_label}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_sample_images(script_dir, proportion, generation, n_samples=16):
    """Load sample images for visualization."""
    p_int = int(proportion * 100)
    data_dir = os.path.join(script_dir, 'data', f'gan_outputs_p{p_int}')
    gen_path = os.path.join(data_dir, f'gen_data_{generation}.pt')

    if os.path.exists(gen_path):
        images = torch.load(gen_path).float()
        if images.max() > 1:
            images = images / 255.0
        return images[:n_samples]
    return None


def create_comprehensive_comparison(script_dir):
    """Create a comprehensive comparison figure."""
    jacobian_results = load_jacobian_results(script_dir)

    if jacobian_results is None:
        print("No Jacobian results found. Run jacobian_constant_proportion.py first.")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, len(jacobian_results)))

    # Plot 1: Effective Rank vs Generation (all proportions)
    ax1 = fig.add_subplot(gs[0, 0])

    for idx, p in enumerate(sorted(jacobian_results.keys())):
        gens = sorted(jacobian_results[p].keys())
        mean_ers = []
        std_ers = []

        for gen in gens:
            eff_ranks = [r['effective_rank'] for r in jacobian_results[p][gen]]
            mean_ers.append(np.mean(eff_ranks))
            std_ers.append(np.std(eff_ranks))

        ax1.errorbar(gens, mean_ers, yerr=std_ers, fmt='o-',
                     label=f'p={p:.2f}', color=colors[idx],
                     linewidth=2, markersize=6, capsize=2)

    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Effective Rank', fontsize=11)
    ax1.set_title('Effective Rank vs Generation', fontsize=12, fontweight='bold')
    ax1.legend(title='Proportion', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final ER vs Proportion (with trend)
    ax2 = fig.add_subplot(gs[0, 1])

    proportions = sorted(jacobian_results.keys())
    initial_ers = []
    final_ers = []
    initial_stds = []
    final_stds = []

    for p in proportions:
        gens = sorted(jacobian_results[p].keys())
        initial_gen = min(gens)
        final_gen = max(gens)

        initial_eff_ranks = [r['effective_rank'] for r in jacobian_results[p][initial_gen]]
        final_eff_ranks = [r['effective_rank'] for r in jacobian_results[p][final_gen]]

        initial_ers.append(np.mean(initial_eff_ranks))
        initial_stds.append(np.std(initial_eff_ranks))
        final_ers.append(np.mean(final_eff_ranks))
        final_stds.append(np.std(final_eff_ranks))

    ax2.errorbar(proportions, initial_ers, yerr=initial_stds, fmt='s-',
                 color='green', linewidth=2, markersize=8, capsize=4, label='Initial (Gen 0)')
    ax2.errorbar(proportions, final_ers, yerr=final_stds, fmt='o-',
                 color='red', linewidth=2, markersize=8, capsize=4, label='Final Generation')

    ax2.set_xlabel('Synthetic Proportion', fontsize=11)
    ax2.set_ylabel('Effective Rank', fontsize=11)
    ax2.set_title('Initial vs Final Effective Rank', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: ER Degradation (Final - Initial) vs Proportion
    ax3 = fig.add_subplot(gs[1, 0])

    degradation = np.array(final_ers) - np.array(initial_ers)

    bars = ax3.bar(proportions, degradation, width=0.08, color=colors, edgecolor='black')

    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('Synthetic Proportion', fontsize=11)
    ax3.set_ylabel('ER Change (Final - Initial)', fontsize=11)
    ax3.set_title('Effective Rank Degradation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, deg in zip(bars, degradation):
        height = bar.get_height()
        ax3.annotate(f'{deg:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height >= 0 else -10),
                     textcoords="offset points",
                     ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=9)

    # Plot 4: Eigenvalue spectrum comparison (Gen 0 vs Final for extreme proportions)
    ax4 = fig.add_subplot(gs[1, 1])

    # Compare p=0.0 and p=1.0 if available
    if 0.0 in jacobian_results and 1.0 in jacobian_results:
        # p=0.0 final generation
        p0_gens = sorted(jacobian_results[0.0].keys())
        p0_final = max(p0_gens)
        p0_eigs = np.array([r['eigenvalues'] for r in jacobian_results[0.0][p0_final]])
        p0_mean = np.mean(p0_eigs, axis=0)
        p0_std = np.std(p0_eigs, axis=0)

        # p=1.0 final generation
        p1_gens = sorted(jacobian_results[1.0].keys())
        p1_final = max(p1_gens)
        p1_eigs = np.array([r['eigenvalues'] for r in jacobian_results[1.0][p1_final]])
        p1_mean = np.mean(p1_eigs, axis=0)
        p1_std = np.std(p1_eigs, axis=0)

        x = range(1, len(p0_mean) + 1)

        ax4.semilogy(x, p0_mean, label=f'p=0.0 (Gen {p0_final})', color='green', linewidth=2)
        ax4.fill_between(x, p0_mean - p0_std, p0_mean + p0_std, color='green', alpha=0.2)

        ax4.semilogy(x, p1_mean, label=f'p=1.0 (Gen {p1_final})', color='red', linewidth=2)
        ax4.fill_between(x, p1_mean - p1_std, p1_mean + p1_std, color='red', alpha=0.2)

        ax4.set_xlabel('Eigenvalue Index', fontsize=11)
        ax4.set_ylabel('Eigenvalue (log scale)', fontsize=11)
        ax4.set_title('Eigenvalue Spectrum: 0% vs 100% Synthetic', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

    plt.suptitle('Constant Proportion Experiment Results\n'
                 'Effect of Synthetic Data Proportion on Model Autophagy',
                 fontsize=14, fontweight='bold', y=1.02)

    save_path = os.path.join(script_dir, 'constant_proportion_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive comparison saved to: {save_path}")


def create_sample_grid(script_dir):
    """Create a grid showing samples from different proportions at final generation."""
    fig, axes = plt.subplots(len(PROPORTIONS), 4, figsize=(8, 2*len(PROPORTIONS)))

    for i, proportion in enumerate(PROPORTIONS):
        p_int = int(proportion * 100)
        data_dir = os.path.join(script_dir, 'data', f'gan_outputs_p{p_int}')

        # Find final generation
        final_gen = None
        for gen in range(20, -1, -1):
            gen_path = os.path.join(data_dir, f'gen_data_{gen}.pt')
            if os.path.exists(gen_path):
                final_gen = gen
                break

        if final_gen is not None:
            images = load_sample_images(script_dir, proportion, final_gen, n_samples=4)
            if images is not None:
                for j in range(4):
                    ax = axes[i, j] if len(PROPORTIONS) > 1 else axes[j]
                    ax.imshow(images[j, 0].numpy(), cmap='gray')
                    ax.axis('off')
                    if j == 0:
                        ax.set_ylabel(f'p={proportion}', fontsize=10, rotation=0, labelpad=30)
        else:
            for j in range(4):
                ax = axes[i, j] if len(PROPORTIONS) > 1 else axes[j]
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                ax.axis('off')
                if j == 0:
                    ax.set_ylabel(f'p={proportion}', fontsize=10, rotation=0, labelpad=30)

    plt.suptitle('Final Generation Samples by Proportion', fontsize=12, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(script_dir, 'sample_comparison_proportions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample comparison saved to: {save_path}")


def print_summary_table(script_dir):
    """Print a summary table of key metrics."""
    jacobian_results = load_jacobian_results(script_dir)

    if jacobian_results is None:
        print("No results to summarize.")
        return

    print("\n" + "="*80)
    print("CONSTANT PROPORTION EXPERIMENT SUMMARY")
    print("="*80)

    print(f"\n{'Proportion':<12} {'Initial ER':<14} {'Final ER':<14} {'Change':<12} {'% Change':<12}")
    print("-"*80)

    for p in sorted(jacobian_results.keys()):
        gens = sorted(jacobian_results[p].keys())
        initial_gen = min(gens)
        final_gen = max(gens)

        initial_ers = [r['effective_rank'] for r in jacobian_results[p][initial_gen]]
        final_ers = [r['effective_rank'] for r in jacobian_results[p][final_gen]]

        initial_mean = np.mean(initial_ers)
        final_mean = np.mean(final_ers)
        change = final_mean - initial_mean
        pct_change = 100 * change / initial_mean

        sign = "+" if change > 0 else ""
        print(f"{p:<12.2f} {initial_mean:<14.2f} {final_mean:<14.2f} {sign}{change:<11.2f} {sign}{pct_change:<11.1f}%")

    print("-"*80)

    # Find critical threshold
    print("\nCRITICAL THRESHOLD ANALYSIS:")
    print("-"*40)

    for p in sorted(jacobian_results.keys()):
        gens = sorted(jacobian_results[p].keys())
        final_gen = max(gens)
        final_ers = [r['effective_rank'] for r in jacobian_results[p][final_gen]]
        final_mean = np.mean(final_ers)

        if final_mean < 2.0:  # Arbitrary threshold for "severe" degradation
            print(f"  p={p}: Final ER = {final_mean:.2f} (SEVERE DEGRADATION)")
        elif final_mean < 3.0:
            print(f"  p={p}: Final ER = {final_mean:.2f} (MODERATE DEGRADATION)")
        else:
            print(f"  p={p}: Final ER = {final_mean:.2f} (STABLE)")

    print("="*80)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Creating comprehensive comparison...")
    create_comprehensive_comparison(script_dir)

    print("\nCreating sample grid...")
    create_sample_grid(script_dir)

    print("\n")
    print_summary_table(script_dir)


if __name__ == "__main__":
    main()
