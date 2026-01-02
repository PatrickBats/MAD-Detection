#!/usr/bin/env python3
"""
Multi-latent vector Jacobian analysis for KMNIST GAN generators with averaging.
Samples multiple latent vectors per generation and computes mean ± std of metrics.
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


# Generator architecture (must match main.py)
latent_dim = 100
n_classes = 10
img_size = 28
channels = 1


class Generator(nn.Module):
    """Conditional Generator for KMNIST"""
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


# Configuration
N_SAMPLES = 100  # Number of latent vectors to sample per generation
SEEDS = list(range(100, 100 + N_SAMPLES))  # Seeds 100-199


def load_generator(generation, device='cuda'):
    """Load Generator model for a specific generation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, 'data', 'gan_outputs', f'generator_{generation}.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.eval()

    return generator


def compute_jacobian_for_task(task_args):
    """
    Compute Jacobian for one (generation, sample_idx) pair.

    Args:
        task_args: tuple of (gen, sample_idx, seed, gpu_id)
    """
    gen, sample_idx, seed, gpu_id = task_args

    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    # Generate random noise with this seed
    torch.manual_seed(seed)
    z_fixed = torch.randn(1, latent_dim, dtype=torch.float32).to(device)

    # Fixed class label for all samples (reduces variance)
    class_label = 7  # Use class 7 for all samples

    try:
        generator = load_generator(gen, device)

        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)
        c = torch.tensor([class_label], device=device, dtype=torch.long)

        print(f"[GPU {gpu_id}] Gen {gen}, sample {sample_idx}: Computing Jacobian...")

        def gen_func(z_input):
            z_shaped = z_input.reshape(1, latent_dim)
            output = generator(z_shaped, c)
            return output.flatten()

        start_time = time.time()
        jacobian = F.jacobian(gen_func, z_flat)
        elapsed = time.time() - start_time

        # Compute SVD
        U, S, V = torch.linalg.svd(jacobian)
        eigenvalues = S.pow(2).cpu().numpy()

        with torch.no_grad():
            image = generator(z_fixed, c)

        # Compute metrics
        max_eig = eigenvalues[0]
        median_eig = np.median(eigenvalues)

        # Effective rank
        eigs_normalized = eigenvalues / eigenvalues.sum()
        eff_rank = np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))

        # Condition number
        condition_number = eigenvalues[0] / (eigenvalues[-1] + 1e-10)

        print(f"[GPU {gpu_id}] Gen {gen}, sample {sample_idx}: MaxEig={max_eig:.4f}, EffRank={eff_rank:.2f}, Time={elapsed:.1f}s")

        result = {
            'eigenvalues': eigenvalues,
            'singular_values': S.cpu().numpy(),
            'U': U.cpu().numpy(),
            'V': V.cpu().numpy(),
            'jacobian': jacobian.cpu().numpy(),
            'image': image.cpu().numpy(),
            'max_eigenvalue': max_eig,
            'median_eigenvalue': median_eig,
            'effective_rank': eff_rank,
            'condition_number': condition_number,
            'time': elapsed,
            'class_label': class_label,
            'seed': seed,
            'sample_idx': sample_idx
        }

        # Save to disk immediately
        save_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(save_dir, 'svd_results_avg'), exist_ok=True)
        save_path = os.path.join(save_dir, 'svd_results_avg', f'gen{gen}_sample{sample_idx}_seed{seed}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)

        # Free memory
        del generator, jacobian, U, S, V
        torch.cuda.empty_cache()

        # Return lightweight data
        return (gen, sample_idx), {
            'max_eigenvalue': max_eig,
            'median_eigenvalue': median_eig,
            'effective_rank': eff_rank,
            'condition_number': condition_number,
            'eigenvalues': eigenvalues,
            'time': elapsed,
            'class_label': class_label,
            'seed': seed,
            'save_path': save_path
        }

    except Exception as e:
        print(f"[GPU {gpu_id}] Gen {gen}, sample {sample_idx}: ERROR: {e}")
        import traceback
        traceback.print_exc()
        return (gen, sample_idx), None


def main():
    """Main function with multi-GPU parallel processing."""

    generations = list(range(8))  # 0-7
    num_gpus = torch.cuda.device_count()

    print(f"\nKMNIST GAN Jacobian Analysis with Averaging")
    print(f"Available GPUs: {num_gpus}")
    print(f"Generations: {generations}")
    print(f"Samples per generation: {N_SAMPLES}")
    print(f"Total computations: {len(generations) * N_SAMPLES}")
    print()

    # Build task list: all (generation, sample) combinations
    tasks = []
    task_idx = 0
    for gen in generations:
        for sample_idx, seed in enumerate(SEEDS):
            gpu_id = task_idx % num_gpus
            tasks.append((gen, sample_idx, seed, gpu_id))
            task_idx += 1

    print(f"Distributing {len(tasks)} tasks across {num_gpus} GPUs")
    print()

    # Process in parallel
    mp.set_start_method('spawn', force=True)
    start_total = time.time()

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.map(compute_jacobian_for_task, tasks)

    total_time = time.time() - start_total

    # Organize results: {generation: [list of sample results]}
    results_dict = {gen: [] for gen in generations}
    for (gen, sample_idx), result in results:
        if result is not None:
            results_dict[gen].append(result)

    # Count successes
    success_count = sum(len(samples) for samples in results_dict.values())
    print(f"\n✓ Completed {success_count}/{len(tasks)} computations")
    print(f"Total time: {total_time:.1f}s")

    # Compute statistics per generation
    stats = compute_statistics(results_dict)

    # Save results
    print("Saving results...")
    with open('jacobian_gan_avg_results.pkl', 'wb') as f:
        pickle.dump({'results': results_dict, 'stats': stats}, f)
    print("Saved to jacobian_gan_avg_results.pkl")

    # Plot results
    if success_count > 0:
        plot_metrics_with_error_bars(stats)
        plot_spectra_with_bands(results_dict)
        print_statistics_table(stats)


def compute_statistics(results_dict):
    """Compute mean and std of metrics for each generation."""
    stats = {}

    for gen, samples in results_dict.items():
        if len(samples) == 0:
            continue

        max_eigs = [s['max_eigenvalue'] for s in samples]
        median_eigs = [s['median_eigenvalue'] for s in samples]
        eff_ranks = [s['effective_rank'] for s in samples]
        cond_nums = [s['condition_number'] for s in samples]

        stats[gen] = {
            'max_eigenvalue': {'mean': np.mean(max_eigs), 'std': np.std(max_eigs)},
            'median_eigenvalue': {'mean': np.mean(median_eigs), 'std': np.std(median_eigs)},
            'effective_rank': {'mean': np.mean(eff_ranks), 'std': np.std(eff_ranks)},
            'condition_number': {'mean': np.mean(cond_nums), 'std': np.std(cond_nums)},
            'n_samples': len(samples)
        }

    return stats


def plot_metrics_with_error_bars(stats):
    """Plot metrics with mean ± std error bars."""

    sorted_gens = sorted(stats.keys())
    x_pos = range(len(sorted_gens))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('max_eigenvalue', 'Max Eigenvalue', True),
        ('median_eigenvalue', 'Median Eigenvalue', True),
        ('effective_rank', 'Effective Rank', False),
        ('condition_number', 'Condition Number', True)
    ]

    for ax, (metric, title, use_log) in zip(axes.flatten(), metrics):
        means = [stats[gen][metric]['mean'] for gen in sorted_gens]
        stds = [stats[gen][metric]['std'] for gen in sorted_gens]

        ax.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5,
                   linewidth=2, markersize=8, color='blue', ecolor='lightblue')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} (mean ± std, n={N_SAMPLES})', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(g) for g in sorted_gens])
        ax.grid(True, alpha=0.3)

        if use_log:
            ax.set_yscale('log')

    plt.suptitle(f'KMNIST GAN Jacobian Metrics Averaged Across {N_SAMPLES} Latent Vectors', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_gan_avg_metrics.png', dpi=150)
    plt.close()

    print("Metrics plot saved to jacobian_gan_avg_metrics.png")


def plot_spectra_with_bands(results_dict):
    """Plot eigenvalue spectra with mean and std bands."""

    sorted_gens = sorted(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results_dict[gen]
        if len(samples) == 0:
            continue

        # Stack all eigenvalue arrays
        all_eigs = np.array([s['eigenvalues'] for s in samples])

        mean_eigs = np.mean(all_eigs, axis=0)
        std_eigs = np.std(all_eigs, axis=0)

        x = range(1, len(mean_eigs) + 1)
        color = colors[gen_idx]

        # Log scale plot with bands
        ax1.semilogy(x, mean_eigs, label=f'Gen {gen}', color=color, linewidth=2)
        ax1.fill_between(x, mean_eigs - std_eigs, mean_eigs + std_eigs,
                        color=color, alpha=0.2)

    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title(f'Eigenvalue Spectrum (mean ± std, n={N_SAMPLES})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot effective rank distribution per generation
    for gen_idx, gen in enumerate(sorted_gens):
        samples = results_dict[gen]
        if len(samples) == 0:
            continue

        eff_ranks = [s['effective_rank'] for s in samples]
        color = colors[gen_idx]

        # Violin/box style - use scatter with jitter
        jitter = np.random.uniform(-0.2, 0.2, len(eff_ranks))
        ax2.scatter([gen_idx] * len(eff_ranks) + jitter, eff_ranks,
                   color=color, alpha=0.5, s=30)
        ax2.errorbar([gen_idx], [np.mean(eff_ranks)], yerr=[np.std(eff_ranks)],
                    fmt='o', color='black', capsize=5, markersize=10)

    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Effective Rank', fontsize=12)
    ax2.set_title(f'Effective Rank Distribution (n={N_SAMPLES})', fontsize=12)
    ax2.set_xticks(range(len(sorted_gens)))
    ax2.set_xticklabels([str(g) for g in sorted_gens])
    ax2.grid(True, alpha=0.3)

    plt.suptitle('KMNIST GAN Eigenvalue Analysis with Averaging', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_gan_avg_spectra.png', dpi=150)
    plt.close()

    print("Spectra plot saved to jacobian_gan_avg_spectra.png")


def print_statistics_table(stats):
    """Print a table of statistics."""

    print("\n" + "="*80)
    print("KMNIST STATISTICS TABLE")
    print("="*80)
    print(f"{'Gen':<5} {'Max Eig (mean±std)':<25} {'Eff Rank (mean±std)':<25} {'n':<5}")
    print("-"*80)

    for gen in sorted(stats.keys()):
        s = stats[gen]
        max_eig = f"{s['max_eigenvalue']['mean']:.4f} ± {s['max_eigenvalue']['std']:.4f}"
        eff_rank = f"{s['effective_rank']['mean']:.2f} ± {s['effective_rank']['std']:.2f}"
        print(f"{gen:<5} {max_eig:<25} {eff_rank:<25} {s['n_samples']:<5}")

    print("="*80)


if __name__ == "__main__":
    main()
