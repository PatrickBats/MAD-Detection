"""
Analyze spectrum trends from the randomized SVD results.
Focus on mean eigenvalue shift across generations.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load results
with open(os.path.join(script_dir, 'spectrum_results_randomsvd.pkl'), 'rb') as f:
    results = pickle.load(f)

# Extract generations
generations = sorted([k for k in results.keys() if isinstance(k, int)])
timesteps = [50, 100, 250, 400]

print("Analyzing spectrum shift across generations...\n")

# Create comprehensive plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for t_idx, t in enumerate(timesteps):
    ax = axes[t_idx // 2, t_idx % 2]

    # Extract metrics
    gens_plot = []
    mean_eigenvalues = []
    max_eigenvalues = []
    median_eigenvalues = []

    for gen in generations:
        if gen in results and 'timesteps' in results[gen]:
            if t in results[gen]['timesteps']:
                metrics = results[gen]['timesteps'][t]['avg_metrics']
                if metrics:
                    gens_plot.append(gen)
                    mean_eigenvalues.append(metrics['mean'])
                    max_eigenvalues.append(metrics['max'])
                    median_eigenvalues.append(metrics['median'])

    # Plot on log scale
    if mean_eigenvalues:
        ax.plot(gens_plot, mean_eigenvalues, 'o-', linewidth=2, markersize=8, label='Mean')
        ax.plot(gens_plot, median_eigenvalues, 's-', linewidth=2, markersize=6, label='Median', alpha=0.7)

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Eigenvalue Magnitude', fontsize=12)
        ax.set_title(f'Timestep t={t}', fontsize=14)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.suptitle('Spectrum Magnitude Evolution (MADness Detection)\nRightward shift = Increased sensitivity', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'spectrum_magnitude_trend.png'), dpi=150)
print(f"Saved: spectrum_magnitude_trend.png")
plt.close()

# Compute percent change from generation 0
print("\n" + "="*80)
print("SPECTRUM SHIFT ANALYSIS (% change from Generation 0)")
print("="*80)

for t in timesteps:
    print(f"\nTimestep t={t}:")

    # Get gen 0 baseline
    if 0 in results and 'timesteps' in results[0]:
        if t in results[0]['timesteps']:
            baseline_mean = results[0]['timesteps'][t]['avg_metrics']['mean']
            baseline_max = results[0]['timesteps'][t]['avg_metrics']['max']

            print(f"  Gen 0 baseline: mean={baseline_mean:.2f}, max={baseline_max:.2f}")

            # Track key generations
            for gen in [1, 5, 10, 15, 19]:
                if gen in results and 'timesteps' in results[gen]:
                    if t in results[gen]['timesteps']:
                        metrics = results[gen]['timesteps'][t]['avg_metrics']

                        mean_change = ((metrics['mean'] - baseline_mean) / baseline_mean) * 100
                        max_change = ((metrics['max'] - baseline_max) / baseline_max) * 100

                        arrow = "↑" if mean_change > 0 else "↓"
                        print(f"  Gen {gen:2d}: mean {mean_change:+6.1f}% {arrow}, max {max_change:+6.1f}%")

# Plot distribution shift for specific generations
fig, axes = plt.subplots(4, 5, figsize=(20, 12))

key_gens = [0, 4, 9, 14, 19]

for t_idx, t in enumerate(timesteps):
    for gen_idx, gen in enumerate(key_gens):
        ax = axes[t_idx, gen_idx]

        if gen in results and 'timesteps' in results[gen]:
            if t in results[gen]['timesteps']:
                all_eigs = results[gen]['timesteps'][t]['all_eigenvalues']

                if all_eigs:
                    # Flatten
                    flat_eigs = np.concatenate(all_eigs)
                    log_eigs = np.log10(flat_eigs + 1e-10)

                    ax.hist(log_eigs, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)
                    ax.axvline(np.mean(log_eigs), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(log_eigs):.2f}')

                    if t_idx == 0:
                        ax.set_title(f'Gen {gen}', fontsize=12, fontweight='bold')
                    if gen_idx == 0:
                        ax.set_ylabel(f't={t}', fontsize=11)

                    ax.set_xlim(-2, 3)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

plt.suptitle('Eigenvalue Distribution Shift Across Generations\n(Notice rightward drift in later generations)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'distribution_shift_analysis.png'), dpi=150)
print(f"\nSaved: distribution_shift_analysis.png")
print("\nAnalysis complete!")
