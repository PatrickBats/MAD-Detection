#!/usr/bin/env python3
"""
Multi-seed Jacobian analysis for DDPM using DDIM sampling with averaging.
Samples multiple noise vectors per generation and computes mean ± std of metrics.
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

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from metrics import DDPM, ContextUnet


# Configuration
N_SAMPLES = 20  # Number of noise vectors per generation
CLASS_LABEL = 7  # Fixed class for all samples
SEEDS = list(range(100, 100 + N_SAMPLES))


def load_model(generation, device='cuda'):
    """Load DDPM model for a specific generation."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    n_T = 500
    nn_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)
    ddpm = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

    if generation == 'initial':
        checkpoint_path = os.path.join(parent_dir, 'data', 'diffusion_outputs10', 'model_initial.pth')
    else:
        checkpoint_path = os.path.join(parent_dir, 'data', 'diffusion_outputs10', f'model_{generation}_w0.pth')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    nn_state = {k[9:]: v for k, v in checkpoint.items() if k.startswith('nn_model.')}
    ddpm.nn_model.load_state_dict(nn_state)

    ddpm.nn_model.eval()
    ddpm.eval()

    return ddpm


def generation_full_500steps_ddim(z, model, class_label=7):
    """Full 500-step DDIM generation (deterministic)."""
    device = z.device
    timesteps = range(500, 0, -1)

    c = torch.tensor([class_label], device=device, dtype=torch.long)
    context_mask = torch.zeros(1, device=device, dtype=torch.float32)

    x = z.clone().float()

    for t in timesteps:
        t_norm = torch.tensor([t / model.n_T], device=device, dtype=torch.float32).view(1, 1, 1, 1)
        eps = model.nn_model(x, c, t_norm, context_mask)

        if t > 1:
            x_0_pred = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]
            x = model.sqrtab[t-1] * x_0_pred + model.sqrtmab[t-1] * eps
        else:
            x = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]

    return x


def compute_jacobian_for_task(task_args):
    """Compute Jacobian for one (generation, sample_idx) pair."""
    gen, sample_idx, seed, gpu_id = task_args

    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    torch.manual_seed(seed)
    z_fixed = torch.randn(1, 1, 28, 28, dtype=torch.float32).to(device)

    try:
        model = load_model(gen, device)

        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)

        gen_label = 'initial' if gen == 'initial' else f'gen{gen}'
        print(f"[GPU {gpu_id}] {gen_label}, sample {sample_idx}: Computing Jacobian...")

        def gen_func(z_input):
            z_shaped = z_input.reshape(1, 1, 28, 28)
            output = generation_full_500steps_ddim(z_shaped, model, CLASS_LABEL)
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

        print(f"[GPU {gpu_id}] {gen_label}, sample {sample_idx}: MaxEig={max_eig:.4f}, EffRank={eff_rank:.2f}, Time={elapsed:.1f}s")

        # Save to disk immediately
        save_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(save_dir, 'svd_results_avg'), exist_ok=True)
        gen_str = 'initial' if gen == 'initial' else f'gen{gen}'
        save_path = os.path.join(save_dir, 'svd_results_avg', f'{gen_str}_sample{sample_idx}_seed{seed}.pkl')

        result = {
            'eigenvalues': eigenvalues,
            'singular_values': S.cpu().numpy(),
            'max_eigenvalue': max_eig,
            'median_eigenvalue': median_eig,
            'effective_rank': eff_rank,
            'time': elapsed,
            'seed': seed
        }

        with open(save_path, 'wb') as f:
            pickle.dump(result, f)

        # Free memory
        del model, jacobian, U, S, V
        torch.cuda.empty_cache()

        return (gen, sample_idx), {
            'max_eigenvalue': max_eig,
            'median_eigenvalue': median_eig,
            'effective_rank': eff_rank,
            'eigenvalues': eigenvalues,
            'time': elapsed,
            'seed': seed,
            'save_path': save_path
        }

    except Exception as e:
        print(f"[GPU {gpu_id}] {gen}, sample {sample_idx}: ERROR: {e}")
        import traceback
        traceback.print_exc()
        return (gen, sample_idx), None


def main():
    generations = ['initial', 0, 1, 2, 3, 4, 5]
    num_gpus = torch.cuda.device_count()

    print(f"\nDDIM Jacobian Analysis with Averaging")
    print(f"Class: {CLASS_LABEL}")
    print(f"Samples per generation: {N_SAMPLES}")
    print(f"Available GPUs: {num_gpus}")
    print(f"Total computations: {len(generations) * N_SAMPLES}")
    print()

    # Build task list
    tasks = []
    task_idx = 0
    for gen in generations:
        for sample_idx, seed in enumerate(SEEDS):
            gpu_id = task_idx % num_gpus
            tasks.append((gen, sample_idx, seed, gpu_id))
            task_idx += 1

    # Run sequentially to avoid OOM (DDPM Jacobian is memory intensive)
    mp.set_start_method('spawn', force=True)
    start_total = time.time()

    results = []
    for task in tasks:
        result = compute_jacobian_for_task(task)
        results.append(result)

    total_time = time.time() - start_total

    # Organize results
    results_dict = {gen: [] for gen in generations}
    for (gen, sample_idx), result in results:
        if result is not None:
            results_dict[gen].append(result)

    success_count = sum(len(samples) for samples in results_dict.values())
    print(f"\n✓ Completed {success_count}/{len(tasks)} computations")
    print(f"Total time: {total_time/60:.1f} minutes")

    # Compute statistics
    stats = compute_statistics(results_dict)

    # Save results
    with open('jacobian_ddim_avg_results.pkl', 'wb') as f:
        pickle.dump({'results': results_dict, 'stats': stats}, f)
    print("Saved to jacobian_ddim_avg_results.pkl")

    # Plot
    if success_count > 0:
        plot_results(results_dict, stats)
        print_table(stats)


def compute_statistics(results_dict):
    stats = {}
    for gen, samples in results_dict.items():
        if len(samples) == 0:
            continue

        max_eigs = [s['max_eigenvalue'] for s in samples]
        median_eigs = [s['median_eigenvalue'] for s in samples]
        eff_ranks = [s['effective_rank'] for s in samples]

        stats[gen] = {
            'max_eigenvalue': {'mean': np.mean(max_eigs), 'std': np.std(max_eigs)},
            'median_eigenvalue': {'mean': np.mean(median_eigs), 'std': np.std(median_eigs)},
            'effective_rank': {'mean': np.mean(eff_ranks), 'std': np.std(eff_ranks)},
            'n_samples': len(samples)
        }

    return stats


def plot_results(results_dict, stats):
    sorted_gens = sorted(stats.keys(), key=lambda x: -1 if x == 'initial' else x)
    gen_labels = ['Init' if g == 'initial' else str(g) for g in sorted_gens]
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
        label = 'Initial' if gen == 'initial' else f'Gen {gen}'

        ax1.semilogy(x, mean_eigs, label=label, color=color, linewidth=2)
        ax1.fill_between(x, mean_eigs - std_eigs, mean_eigs + std_eigs, color=color, alpha=0.2)

    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title(f'Eigenvalue Spectrum (mean ± std, n={N_SAMPLES})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Effective rank distribution
    for gen_idx, gen in enumerate(sorted_gens):
        samples = results_dict[gen]
        if len(samples) == 0:
            continue

        eff_ranks = [s['effective_rank'] for s in samples]
        color = colors[gen_idx]

        jitter = np.random.uniform(-0.2, 0.2, len(eff_ranks))
        ax2.scatter([gen_idx] * len(eff_ranks) + jitter, eff_ranks, color=color, alpha=0.5, s=30)
        ax2.errorbar([gen_idx], [np.mean(eff_ranks)], yerr=[np.std(eff_ranks)],
                    fmt='o', color='black', capsize=5, markersize=10)

    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Effective Rank', fontsize=12)
    ax2.set_title(f'Effective Rank Distribution (n={N_SAMPLES})', fontsize=12)
    ax2.set_xticks(range(len(sorted_gens)))
    ax2.set_xticklabels(gen_labels)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('DDIM Eigenvalue Analysis with Averaging', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_ddim_avg_spectra.png', dpi=150)
    plt.close()

    print("Plot saved to jacobian_ddim_avg_spectra.png")


def print_table(stats):
    print("\n" + "="*60)
    print("DDIM STATISTICS TABLE")
    print("="*60)
    print(f"{'Gen':<8} {'Eff Rank (mean±std)':<25} {'n':<5}")
    print("-"*60)

    sorted_gens = sorted(stats.keys(), key=lambda x: -1 if x == 'initial' else x)
    for gen in sorted_gens:
        s = stats[gen]
        gen_label = 'Initial' if gen == 'initial' else str(gen)
        eff_rank = f"{s['effective_rank']['mean']:.2f} ± {s['effective_rank']['std']:.2f}"
        print(f"{gen_label:<8} {eff_rank:<25} {s['n_samples']:<5}")

    print("="*60)


if __name__ == "__main__":
    main()
