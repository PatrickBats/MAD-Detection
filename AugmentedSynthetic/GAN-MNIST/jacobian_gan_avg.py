#!/usr/bin/env python3
"""
Jacobian analysis for Augmented Synthetic GAN-MNIST with random classes (averaged).
Samples 100 latent vectors per generation with random class labels (matching fully synthetic approach).
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


# Configuration
N_SAMPLES = 100
SEEDS = list(range(1000, 1000 + N_SAMPLES))


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
    """Compute Jacobian for one (generation, sample_idx) pair with random class."""
    gen, sample_idx, seed, gpu_id = task_args

    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    # Generate random noise with this seed
    torch.manual_seed(seed)
    z_fixed = torch.randn(1, latent_dim, dtype=torch.float32).to(device)

    # Random class label (also seeded) - same as fully synthetic approach
    class_label = seed % 10

    try:
        generator = load_generator(gen, device)

        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)
        c = torch.tensor([class_label], device=device, dtype=torch.long)

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

        # Compute metrics
        max_eig = eigenvalues[0]
        median_eig = np.median(eigenvalues)

        # Effective rank
        eigs_normalized = eigenvalues / eigenvalues.sum()
        eff_rank = np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))

        print(f"[GPU {gpu_id}] Gen {gen}, sample {sample_idx}, class {class_label}: EffRank={eff_rank:.2f}")

        # Free memory
        del generator, jacobian, U, S, V
        torch.cuda.empty_cache()

        return (gen, sample_idx), {
            'max_eigenvalue': max_eig,
            'median_eigenvalue': median_eig,
            'effective_rank': eff_rank,
            'eigenvalues': eigenvalues,
            'seed': seed,
            'class_label': class_label
        }

    except Exception as e:
        print(f"[GPU {gpu_id}] Gen {gen}, sample {sample_idx}: ERROR: {e}")
        return (gen, sample_idx), None


def main():
    # Generations 1-8 for augmented synthetic loop
    generations = list(range(1, 9))
    num_gpus = torch.cuda.device_count()

    print(f"\nAugmented Synthetic GAN-MNIST Jacobian Analysis (Random Classes, Averaged)")
    print(f"Samples per generation: {N_SAMPLES}")
    print(f"Class labels: Random (seed % 10)")
    print(f"Available GPUs: {num_gpus}")
    print()

    # Build task list
    tasks = []
    task_idx = 0
    for gen in generations:
        for sample_idx, seed in enumerate(SEEDS):
            gpu_id = task_idx % num_gpus
            tasks.append((gen, sample_idx, seed, gpu_id))
            task_idx += 1

    print(f"Total computations: {len(tasks)}")
    print()

    mp.set_start_method('spawn', force=True)
    start_total = time.time()

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.map(compute_jacobian_for_task, tasks)

    total_time = time.time() - start_total

    # Organize results
    results_dict = {gen: [] for gen in generations}
    for (gen, sample_idx), result in results:
        if result is not None:
            results_dict[gen].append(result)

    success_count = sum(len(samples) for samples in results_dict.values())
    print(f"\n✓ Completed {success_count}/{len(tasks)} computations")
    print(f"Total time: {total_time:.1f}s")

    # Compute statistics
    stats = {}
    for gen, samples in results_dict.items():
        if len(samples) == 0:
            continue
        eff_ranks = [s['effective_rank'] for s in samples]
        max_eigs = [s['max_eigenvalue'] for s in samples]
        stats[gen] = {
            'effective_rank': {'mean': np.mean(eff_ranks), 'std': np.std(eff_ranks)},
            'max_eigenvalue': {'mean': np.mean(max_eigs), 'std': np.std(max_eigs)},
            'n_samples': len(samples)
        }

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'jacobian_gan_avg_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'results': results_dict, 'stats': stats}, f)
    print(f"Saved to {save_path}")

    # Plot
    plot_results(results_dict, stats, script_dir)
    print_table(stats)


def plot_results(results_dict, stats, save_dir):
    sorted_gens = sorted(stats.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Eigenvalue spectra
    for gen_idx, gen in enumerate(sorted_gens):
        samples = results_dict[gen]
        if len(samples) == 0:
            continue

        all_eigs = np.array([s['eigenvalues'] for s in samples])
        mean_eigs = np.mean(all_eigs, axis=0)
        std_eigs = np.std(all_eigs, axis=0)

        x = range(1, len(mean_eigs) + 1)
        color = colors[gen_idx]

        ax1.semilogy(x, mean_eigs, label=f'Gen {gen}', color=color, linewidth=2)
        ax1.fill_between(x,
                         np.maximum(mean_eigs - std_eigs, 1e-10),
                         mean_eigs + std_eigs,
                         color=color, alpha=0.2)

    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title(f'Eigenvalue Spectrum (n={N_SAMPLES}, random classes)', fontsize=12)
    ax1.legend(ncol=2)
    ax1.grid(True, alpha=0.3)

    # Effective rank distribution
    for gen_idx, gen in enumerate(sorted_gens):
        samples = results_dict[gen]
        if len(samples) == 0:
            continue

        eff_ranks = [s['effective_rank'] for s in samples]
        color = colors[gen_idx]

        jitter = np.random.uniform(-0.2, 0.2, len(eff_ranks))
        ax2.scatter([gen_idx] * len(eff_ranks) + jitter, eff_ranks, color=color, alpha=0.3, s=15)
        ax2.errorbar([gen_idx], [np.mean(eff_ranks)], yerr=[np.std(eff_ranks)],
                    fmt='o', color='black', capsize=5, markersize=10)

    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Effective Rank', fontsize=12)
    ax2.set_title(f'Effective Rank Distribution (n={N_SAMPLES}, random classes)', fontsize=12)
    ax2.set_xticks(range(len(sorted_gens)))
    ax2.set_xticklabels([str(g) for g in sorted_gens])
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Augmented Synthetic GAN-MNIST - Jacobian Analysis (Averaged)', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'jacobian_gan_avg_spectra.png')
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Plot saved to {save_path}")


def print_table(stats):
    print("\n" + "="*60)
    print("AUGMENTED SYNTHETIC GAN-MNIST - STATISTICS (Random Classes)")
    print("="*60)
    print(f"{'Gen':<5} {'Eff Rank (mean±std)':<25} {'n':<5}")
    print("-"*60)

    for gen in sorted(stats.keys()):
        s = stats[gen]
        eff_rank = f"{s['effective_rank']['mean']:.2f} ± {s['effective_rank']['std']:.2f}"
        print(f"{gen:<5} {eff_rank:<25} {s['n_samples']:<5}")

    print("="*60)


if __name__ == "__main__":
    main()
