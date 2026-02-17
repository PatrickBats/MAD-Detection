#!/usr/bin/env python3
"""
Plot effective rank with individual sample scatter for 75% synthetic data.
Shows mean with error bars plus jittered individual points to reveal distribution shape.
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

CLASS_LABEL = 7
PROPORTION = 0.75  # 75% synthetic

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load results
    pkl_path = os.path.join(script_dir, f'jacobian_proportions_class{CLASS_LABEL}.pkl')
    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)

    if PROPORTION not in all_results:
        print(f"No results for proportion {PROPORTION}")
        return

    results = all_results[PROPORTION]
    sorted_gens = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results[gen]
        if len(samples) == 0:
            continue

        eff_ranks = [s['effective_rank'] for s in samples]
        color = colors[gen_idx]

        # Scatter with jitter
        jitter = np.random.uniform(-0.2, 0.2, len(eff_ranks))
        ax.scatter([gen_idx] * len(eff_ranks) + jitter, eff_ranks,
                   color=color, alpha=0.5, s=30)

        # Mean with error bar
        ax.errorbar([gen_idx], [np.mean(eff_ranks)], yerr=[np.std(eff_ranks)],
                    fmt='o', color='black', capsize=5, markersize=10)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Effective Rank')
    ax.set_xticks(range(len(sorted_gens)))
    ax.set_xticklabels([str(g) for g in sorted_gens])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(script_dir, 'effective_rank_75pct_scatter.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
