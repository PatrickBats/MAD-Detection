#!/usr/bin/env python3
"""
Jacobian Effective Rank Analysis for Constant Proportion Experiments.

Computes the Jacobian eigenvalue spectrum and effective rank for generators
trained with different synthetic data proportions.

Uses single-class analysis (digit 7) to control for class complexity.
"""

import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import argparse

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

# Generator architecture (must match training)
latent_dim = 100
n_classes = 10
img_size = 28
channels = 1

# Analysis settings
CLASS_LABEL = 7
CLASS_NAME = "Seven"
N_SAMPLES = 50  # Samples per generation
SEEDS = list(range(1000, 1000 + N_SAMPLES))

# Proportions to analyze
PROPORTIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


class Generator(nn.Module):
    """Conditional Generator for MNIST"""
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, img_size * img_size * channels),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img


def compute_effective_rank(eigenvalues):
    """Compute effective rank from eigenvalues using entropy formula."""
    eigs_normalized = eigenvalues / eigenvalues.sum()
    return np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))


def compute_jacobian_for_sample(args):
    """Compute Jacobian for a single sample."""
    gen_path, sample_idx, seed, gpu_id = args

    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    torch.manual_seed(seed)
    z_fixed = torch.randn(1, latent_dim, dtype=torch.float32).to(device)

    try:
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        generator.eval()

        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)
        c = torch.tensor([CLASS_LABEL], device=device, dtype=torch.long)

        def gen_func(z_input):
            z_shaped = z_input.reshape(1, latent_dim)
            output = generator(z_shaped, c)
            return output.flatten()

        jacobian = F.jacobian(gen_func, z_flat)

        # Compute SVD
        U, S, V = torch.linalg.svd(jacobian)
        eigenvalues = S.pow(2).cpu().numpy()

        # Effective rank
        eff_rank = compute_effective_rank(eigenvalues)

        # Free memory
        del generator, jacobian, U, S, V
        torch.cuda.empty_cache()

        return {
            'effective_rank': eff_rank,
            'eigenvalues': eigenvalues,
            'seed': seed
        }

    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        return None


def analyze_proportion(proportion, script_dir, device, num_gpus):
    """Analyze all generations for a single proportion."""
    p_int = int(proportion * 100)
    data_dir = os.path.join(script_dir, 'data', f'gan_outputs_p{p_int}')

    if not os.path.exists(data_dir):
        print(f"  Directory not found: {data_dir}")
        return None

    # Find available generations
    generations = []
    for i in range(20):
        gen_path = os.path.join(data_dir, f'generator_{i}.pth')
        if os.path.exists(gen_path):
            generations.append(i)

    if not generations:
        print(f"  No generators found in {data_dir}")
        return None

    print(f"  Found generations: {generations}")

    results_dict = {}

    for gen in generations:
        gen_path = os.path.join(data_dir, f'generator_{gen}.pth')
        print(f"    Processing Gen {gen}...", end=" ", flush=True)

        # Build tasks for multiprocessing
        tasks = []
        for sample_idx, seed in enumerate(SEEDS):
            gpu_id = sample_idx % num_gpus
            tasks.append((gen_path, sample_idx, seed, gpu_id))

        # Process in parallel
        with mp.Pool(processes=num_gpus) as pool:
            results = pool.map(compute_jacobian_for_sample, tasks)

        # Filter successful results
        valid_results = [r for r in results if r is not None]
        results_dict[gen] = valid_results

        if valid_results:
            eff_ranks = [r['effective_rank'] for r in valid_results]
            print(f"ER = {np.mean(eff_ranks):.2f} ± {np.std(eff_ranks):.2f}")
        else:
            print("No valid results")

    return results_dict


def main():
    parser = argparse.ArgumentParser(description='Jacobian analysis for constant proportion experiments')
    parser.add_argument('--proportions', type=float, nargs='+', default=PROPORTIONS,
                        help=f'Proportions to analyze (default: {PROPORTIONS})')
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES,
                        help=f'Samples per generation (default: {N_SAMPLES})')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}, GPUs: {num_gpus}")

    mp.set_start_method('spawn', force=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    all_results = {}

    for proportion in args.proportions:
        print(f"\n{'='*60}")
        print(f"ANALYZING p={proportion} ({proportion*100:.0f}% synthetic)")
        print(f"{'='*60}")

        results = analyze_proportion(proportion, script_dir, device, num_gpus)
        if results:
            all_results[proportion] = results

    # Save all results
    save_path = os.path.join(script_dir, f'jacobian_proportions_class{CLASS_LABEL}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {save_path}")

    # Print summary table
    print("\n" + "="*80)
    print(f"EFFECTIVE RANK SUMMARY (Class {CLASS_LABEL} - {CLASS_NAME})")
    print("="*80)

    # Find max generations
    max_gens = 0
    for p_results in all_results.values():
        max_gens = max(max_gens, max(p_results.keys()) + 1)

    # Header
    header = f"{'Gen':<6}"
    for p in sorted(all_results.keys()):
        header += f"{'p='+str(p):<14}"
    print(header)
    print("-"*80)

    # Data rows
    for gen in range(max_gens):
        row = f"{gen:<6}"
        for p in sorted(all_results.keys()):
            if gen in all_results[p]:
                eff_ranks = [r['effective_rank'] for r in all_results[p][gen]]
                mean_er = np.mean(eff_ranks)
                std_er = np.std(eff_ranks)
                row += f"{mean_er:.2f}±{std_er:.2f}".ljust(14)
            else:
                row += f"{'-':<14}"
        print(row)

    print("="*80)

    # Create comparison plot
    create_comparison_plot(all_results, script_dir)


def create_comparison_plot(all_results, save_dir):
    """Create comparison plot of effective rank across proportions."""
    if not all_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

    # Plot 1: Effective rank vs generation
    for idx, p in enumerate(sorted(all_results.keys())):
        gens = sorted(all_results[p].keys())
        mean_ers = []
        std_ers = []

        for gen in gens:
            eff_ranks = [r['effective_rank'] for r in all_results[p][gen]]
            mean_ers.append(np.mean(eff_ranks))
            std_ers.append(np.std(eff_ranks))

        ax1.errorbar(gens, mean_ers, yerr=std_ers, fmt='o-',
                     label=f'p={p:.2f}', color=colors[idx],
                     linewidth=2, markersize=8, capsize=3)

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Effective Rank')
    ax1.legend(title='Proportion')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final ER vs proportion
    proportions = sorted(all_results.keys())
    final_ers = []
    final_stds = []

    for p in proportions:
        final_gen = max(all_results[p].keys())
        eff_ranks = [r['effective_rank'] for r in all_results[p][final_gen]]
        final_ers.append(np.mean(eff_ranks))
        final_stds.append(np.std(eff_ranks))

    ax2.errorbar(proportions, final_ers, yerr=final_stds, fmt='o-',
                 color='darkblue', linewidth=2, markersize=10, capsize=5)
    ax2.fill_between(proportions,
                     np.array(final_ers) - np.array(final_stds),
                     np.array(final_ers) + np.array(final_stds),
                     alpha=0.2, color='blue')

    ax2.set_xlabel('Synthetic Proportion')
    ax2.set_ylabel('Final Effective Rank')
    ax2.set_xticks(proportions)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'jacobian_comparison_proportions_class{CLASS_LABEL}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
