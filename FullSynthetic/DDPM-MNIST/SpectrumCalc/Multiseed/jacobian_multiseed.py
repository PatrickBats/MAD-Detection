#!/usr/bin/env python3
"""
Test Jacobian computation across multiple random seeds.
Runs the same computation as jacobian.py but for 10 different seeds.
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


def load_model(generation, device='cuda'):
    """Load DDPM model for a specific generation."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    n_T = 500
    nn_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)
    ddpm = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

    checkpoint_path = os.path.join(parent_dir, 'data', 'diffusion_outputs10', f'model_{generation}_w0.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    nn_state = {k[9:]: v for k, v in checkpoint.items() if k.startswith('nn_model.')}
    ddpm.nn_model.load_state_dict(nn_state)

    ddpm.nn_model.eval()
    ddpm.eval()

    return ddpm


def generation_full_500steps(z, model, fixed_noise_sequence, class_label=7):
    """
    Full 500-step DDPM generation with fixed noise sequence.
    """
    device = z.device
    timesteps = range(500, 0, -1)

    c = torch.tensor([class_label], device=device, dtype=torch.long)
    context_mask = torch.zeros(1, device=device, dtype=torch.float32)

    x = z.clone().float()

    for t in timesteps:
        t_norm = torch.tensor([t / model.n_T], device=device, dtype=torch.float32).view(1, 1, 1, 1)
        eps = model.nn_model(x, c, t_norm, context_mask)

        if t > 1:
            x = model.oneover_sqrta[t] * (x - eps * model.mab_over_sqrtmab[t])
            noise = fixed_noise_sequence[t].to(device)
            x = x + model.sqrt_beta_t[t] * noise
        else:
            x = model.oneover_sqrta[t] * (x - eps * model.mab_over_sqrtmab[t])

    return x


def compute_jacobian_for_generation(gen, z_fixed_cpu, fixed_noise_cpu, gpu_id, class_label=7):
    """Compute Jacobian for one generation on one GPU."""
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    print(f"[GPU {gpu_id}] Processing Generation {gen}")

    try:
        model = load_model(gen, device)

        z_fixed = z_fixed_cpu.to(device)
        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)

        def gen_func(z_input):
            z_shaped = z_input.reshape(1, 1, 28, 28)
            output = generation_full_500steps(z_shaped, model, fixed_noise_cpu, class_label)
            return output.flatten()

        start_time = time.time()
        jacobian = F.jacobian(gen_func, z_flat)
        elapsed = time.time() - start_time

        U, S, V = torch.linalg.svd(jacobian)
        eigenvalues = S.pow(2).cpu().numpy()

        with torch.no_grad():
            image = generation_full_500steps(z_fixed, model, fixed_noise_cpu, class_label)

        print(f"[GPU {gpu_id}] Gen {gen}: Max eig={eigenvalues[0]:.4f}, time={elapsed:.1f}s")

        result = {
            'eigenvalues': eigenvalues,
            'image': image.cpu().numpy(),
            'time': elapsed
        }

        return gen, result

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return gen, None


def run_single_seed(seed, z_fixed_cpu):
    """Run Jacobian computation for a single seed.

    Args:
        seed: Random seed for denoising noise sequence
        z_fixed_cpu: Fixed initial noise (same for all seeds)
    """
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}\n")

    generations = [0, 1, 2, 3, 4, 5]
    num_gpus = torch.cuda.device_count()

    # Generate denoising noise sequence with this seed
    # Initial noise z_fixed_cpu is the SAME for all seeds
    torch.manual_seed(seed)
    fixed_noise_sequence = {}
    for t in range(500, 1, -1):
        fixed_noise_sequence[t] = torch.randn(1, 1, 28, 28, dtype=torch.float32)

    # Process all generations in parallel
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=len(generations)) as pool:
        results = pool.starmap(
            compute_jacobian_for_generation,
            [(gen, z_fixed_cpu, fixed_noise_sequence, idx % num_gpus, 7) for idx, gen in enumerate(generations)]
        )

    # Collect results
    results_dict = {}
    for gen, result in results:
        if result is not None:
            results_dict[gen] = result

    return results_dict


def main():
    """Run Jacobian computation for 10 different seeds."""
    seeds = [0, 1, 42, 123, 256, 512, 777, 999, 1234, 5678]

    # FIXED initial noise - same for ALL seeds
    torch.manual_seed(42)
    z_fixed_cpu = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    print(f"Fixed initial noise: mean={z_fixed_cpu.mean():.4f}, std={z_fixed_cpu.std():.4f}")
    print(f"This initial noise is the SAME for all {len(seeds)} seeds\n")

    all_results = {}

    for seed in seeds:
        results = run_single_seed(seed, z_fixed_cpu)
        all_results[seed] = results

    # Save all results
    output_file = 'jacobian_multiseed_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n{'='*60}")
    print(f"All seeds completed!")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}\n")

    # Compute statistics across seeds
    print("\nEffective Rank Across Seeds:")
    print(f"{'Seed':<10} {'Gen 0':<10} {'Gen 5':<10} {'Gen 10':<10} {'Gen 15':<10} {'Gen 19':<10}")
    print("-" * 60)

    for seed in seeds:
        ranks = []
        for gen in [0, 5, 10, 15, 19]:
            if seed in all_results and gen in all_results[seed]:
                eigs = all_results[seed][gen]['eigenvalues']
                eigs_norm = eigs / eigs.sum()
                eff_rank = np.exp(-np.sum(eigs_norm * np.log(eigs_norm + 1e-10)))
                ranks.append(f"{eff_rank:.2f}")
            else:
                ranks.append("N/A")
        print(f"{seed:<10} {ranks[0]:<10} {ranks[1]:<10} {ranks[2]:<10} {ranks[3]:<10} {ranks[4]:<10}")


if __name__ == "__main__":
    main()
