#!/usr/bin/env python3
"""
Create clean effective rank vs generation plot for constant proportion experiments.
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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load results
    pkl_path = os.path.join(script_dir, f'jacobian_proportions_class{CLASS_LABEL}.pkl')
    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)

    print(f"Loaded results for proportions: {sorted(all_results.keys())}")

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4.5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

    for idx, p in enumerate(sorted(all_results.keys())):
        gens = sorted(all_results[p].keys())
        mean_ers = []

        for gen in gens:
            eff_ranks = [r['effective_rank'] for r in all_results[p][gen]]
            mean_ers.append(np.mean(eff_ranks))

        ax.plot(gens, mean_ers, 'o-', label=f'{int(p*100)}%', color=colors[idx],
                linewidth=2, markersize=6)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Effective Rank')
    ax.legend(title='Synthetic Data', loc='lower left', fontsize=8, title_fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(script_dir, f'effective_rank_proportions.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
