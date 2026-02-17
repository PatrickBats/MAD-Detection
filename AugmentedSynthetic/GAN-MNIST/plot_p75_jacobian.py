#!/usr/bin/env python3
"""
Plot eigenvalue spectrum and effective rank for p=0.75 constant proportion experiment.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'font.family': 'serif',
})

PROPORTION = 0.75
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(SCRIPT_DIR, 'jacobian_proportions_class7.pkl')
OUTPUT_DIR = SCRIPT_DIR


def main():
    # Load data
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    # Get p=0.75 data
    results_dict = data[PROPORTION]
    sorted_gens = sorted(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    print(f"Plotting p={PROPORTION} (75% synthetic, 25% real)")
    print(f"Generations: {sorted_gens}")
    print()

    # Plot eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results_dict[gen]
        if len(samples) == 0:
            continue

        all_eigs = np.array([s['eigenvalues'] for s in samples])
        mean_eigs = np.mean(all_eigs, axis=0)
        std_eigs = np.std(all_eigs, axis=0)

        x = range(1, len(mean_eigs) + 1)
        color = colors[gen_idx]

        ax.semilogy(x, mean_eigs, label=f'Gen {gen}', color=color, linewidth=2)
        ax.fill_between(x,
                        np.maximum(mean_eigs - std_eigs, 1e-10),
                        mean_eigs + std_eigs,
                        color=color, alpha=0.2)

    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    spectrum_path = os.path.join(OUTPUT_DIR, 'gan_mnist_p75_spectrum.pdf')
    plt.savefig(spectrum_path, bbox_inches='tight')
    plt.close()
    print(f'Saved spectrum plot to {spectrum_path}')

    # Plot effective rank
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results_dict[gen]
        if len(samples) == 0:
            continue

        eff_ranks = [s['effective_rank'] for s in samples]
        color = colors[gen_idx]

        jitter = np.random.uniform(-0.2, 0.2, len(eff_ranks))
        ax.scatter([gen_idx] * len(eff_ranks) + jitter, eff_ranks, color=color, alpha=0.3, s=15)
        ax.errorbar([gen_idx], [np.mean(eff_ranks)], yerr=[np.std(eff_ranks)],
                    fmt='o', color='black', capsize=5, markersize=10)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Effective Rank')
    ax.set_xticks(range(len(sorted_gens)))
    ax.set_xticklabels([str(g) for g in sorted_gens])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    eff_rank_path = os.path.join(OUTPUT_DIR, 'gan_mnist_p75_effective_rank.pdf')
    plt.savefig(eff_rank_path, bbox_inches='tight')
    plt.close()
    print(f'Saved effective rank plot to {eff_rank_path}')

    # Print statistics
    print()
    print(f"p={PROPORTION} Statistics:")
    print("="*50)
    print(f"{'Gen':<6} {'Eff Rank (mean±std)':<25} {'n':<5}")
    print("-"*50)
    for gen in sorted_gens:
        samples = results_dict[gen]
        eff_ranks = [s['effective_rank'] for s in samples]
        print(f"{gen:<6} {np.mean(eff_ranks):.2f} ± {np.std(eff_ranks):.2f}          {len(samples):<5}")
    print("="*50)


if __name__ == "__main__":
    main()
