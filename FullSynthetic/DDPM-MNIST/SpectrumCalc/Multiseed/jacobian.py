#!/usr/bin/env python3
"""
Full 500-step generation Jacobian with multi-GPU support.
Each generation runs on a separate GPU for parallel processing.
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

    Args:
        z: Initial noise [1, 1, 28, 28]
        model: DDPM model
        fixed_noise_sequence: Pre-generated noise for each timestep (dict: t -> noise tensor)
        class_label: Class to generate
    """
    device = z.device

    timesteps = range(500, 0, -1)

    # Setup class and context
    c = torch.tensor([class_label], device=device, dtype=torch.long)
    context_mask = torch.zeros(1, device=device, dtype=torch.float32)

    # Start with noise
    x = z.clone().float()

    # Full denoising loop (500 steps)
    for t in timesteps:
        t_norm = torch.tensor([t / model.n_T], device=device, dtype=torch.float32).view(1, 1, 1, 1)

        # Predict noise (no torch.no_grad() for Jacobian)
        eps = model.nn_model(x, c, t_norm, context_mask)

        # Apply DDPM denoising formula
        if t > 1:
            # Standard denoising step with FIXED noise (same across all generations)
            x = model.oneover_sqrta[t] * (x - eps * model.mab_over_sqrtmab[t])
            noise = fixed_noise_sequence[t].to(device)  # Use pre-generated fixed noise
            x = x + model.sqrt_beta_t[t] * noise
        else:
            # Final step without noise
            x = model.oneover_sqrta[t] * (x - eps * model.mab_over_sqrtmab[t])

    return x


def compute_jacobian_for_generation(gen, z_fixed_cpu, fixed_noise_cpu, gpu_id, class_label=7):
    """
    Compute Jacobian for one generation on one GPU.

    Args:
        gen: Generation number
        z_fixed_cpu: Fixed noise on CPU (to be moved to GPU)
        fixed_noise_cpu: Fixed noise sequence on CPU (dict: t -> noise tensor)
        gpu_id: GPU ID to use
        class_label: Class to generate
    """

    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    print(f"\n[GPU {gpu_id}] Processing Generation {gen}")
    print(f"[GPU {gpu_id}] Loading model...")

    try:
        model = load_model(gen, device)

        # Move fixed noise to this GPU
        z_fixed = z_fixed_cpu.to(device)
        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)

        print(f"[GPU {gpu_id}] Computing Jacobian (500 steps)...")
        print(f"[GPU {gpu_id}]   This will take ~10-20 minutes...")

        # Define generation function
        def gen_func(z_input):
            z_shaped = z_input.reshape(1, 1, 28, 28)
            output = generation_full_500steps(z_shaped, model, fixed_noise_cpu, class_label)
            return output.flatten()

        # Compute full Jacobian matrix
        start_time = time.time()
        jacobian = F.jacobian(gen_func, z_flat)
        elapsed = time.time() - start_time

        print(f"[GPU {gpu_id}] Jacobian computed in {elapsed:.2f} seconds ({elapsed/60:.1f} min)")
        print(f"[GPU {gpu_id}] Jacobian shape: {jacobian.shape}")

        # Compute SVD for eigenvalues
        print(f"[GPU {gpu_id}] Computing SVD...")
        U, S, V = torch.linalg.svd(jacobian)

        # Eigenvalues of J^T J are singular values squared
        eigenvalues = S.pow(2).cpu().numpy()

        # Also generate the image for visualization
        with torch.no_grad():
            image = generation_full_500steps(z_fixed, model, fixed_noise_cpu, class_label)

        # Print statistics
        print(f"[GPU {gpu_id}] Eigenvalue Statistics:")
        print(f"[GPU {gpu_id}]   Total: {len(eigenvalues)}")
        print(f"[GPU {gpu_id}]   Max: {eigenvalues[0]:.4f}")
        print(f"[GPU {gpu_id}]   Min: {eigenvalues[-1]:.4e}")
        print(f"[GPU {gpu_id}]   Median: {np.median(eigenvalues):.4f}")

        # Effective rank
        eigs_normalized = eigenvalues / eigenvalues.sum()
        eff_rank = np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))
        print(f"[GPU {gpu_id}]   Effective rank: {eff_rank:.2f} / 784")

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


def main():
    """Main function with multi-GPU parallel processing."""

    generations = [0, 5, 10, 15, 19]
    num_gpus = torch.cuda.device_count()

    print(f"\nAvailable GPUs: {num_gpus}")
    print(f"Generations to process: {generations}")
    print(f"Each generation will run on a separate GPU in parallel")
    print()

    # SINGLE FIXED POINT - same for all generations
    torch.manual_seed(42)
    z_fixed_cpu = torch.randn(1, 1, 28, 28, dtype=torch.float32)

    # Generate FIXED noise sequence for all 500 timesteps (same for all generations)
    print("Generating fixed noise sequence for 500 timesteps...")
    fixed_noise_sequence = {}
    for t in range(500, 1, -1):  # t=500 down to t=2
        fixed_noise_sequence[t] = torch.randn(1, 1, 28, 28, dtype=torch.float32)

    # Save the fixed noise sequence
    with open('fixed_noise_sequence.pkl', 'wb') as f:
        pickle.dump(fixed_noise_sequence, f)
    print("✓ Fixed noise sequence saved to fixed_noise_sequence.pkl")

    print(f"Initial noise statistics: mean={z_fixed_cpu.mean():.4f}, std={z_fixed_cpu.std():.4f}")
    print(f"Full 500-step denoising per generation with FIXED stochastic noise")
    print()
    print()

    # Process all generations in parallel
    mp.set_start_method('spawn', force=True)

    print("Starting parallel computation...")
    start_total = time.time()

    with mp.Pool(processes=len(generations)) as pool:
        # Each generation gets its own GPU
        results = pool.starmap(
            compute_jacobian_for_generation,
            [(gen, z_fixed_cpu, fixed_noise_sequence, idx % num_gpus, 7) for idx, gen in enumerate(generations)]
        )

    total_time = time.time() - start_total
    print(f"All generations completed in {total_time:.2f} seconds ({total_time/60:.1f} min)")

    # Collect results
    results_dict = {}
    for gen, result in results:
        if result is not None:
            results_dict[gen] = result
            print(f"✓ Generation {gen} completed successfully")
        else:
            print(f"✗ Generation {gen} failed")

    # Save results
    print("Saving results...")
    with open('jacobian_500steps_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print("Saved to jacobian_500steps_results.pkl")

    # Plot comparison
    if len(results_dict) > 0:
        plot_eigenvalue_comparison(results_dict)
        plot_generated_images(results_dict)


def plot_eigenvalue_comparison(results):
    """Plot eigenvalue spectra comparison."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for idx, gen in enumerate(sorted(results.keys())):
        eigs = results[gen]['eigenvalues']
        color = colors[idx % len(colors)]

        # Log scale plot
        ax1.semilogy(range(1, len(eigs)+1), eigs,
                    label=f'Gen {gen}', color=color, linewidth=2)

        # Histogram
        ax2.hist(np.log10(eigs + 1e-10), bins=50, alpha=0.5,
                label=f'Gen {gen}', color=color)

    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title('Eigenvalue Spectrum\n(Full 500-step Generation Jacobian)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('log10(Eigenvalue)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of log(Eigenvalues)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jacobian_500steps_spectrum.png', dpi=150)
    plt.close()

    print("\nPlot saved to jacobian_500steps_spectrum.png")


def plot_generated_images(results):

    n_gens = len(results)
    if n_gens == 0:
        return

    fig, axes = plt.subplots(1, n_gens, figsize=(3*n_gens, 3))

    if n_gens == 1:
        axes = [axes]

    for idx, gen in enumerate(sorted(results.keys())):
        image = results[gen]['image']

        # Reshape and display
        if len(image.shape) == 4:
            image = image[0, 0]
        elif len(image.shape) == 3:
            image = image[0]
        elif len(image.shape) == 1:
            image = image.reshape(28, 28)

        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(f'Gen {gen}', fontsize=12)
        axes[idx].axis('off')

    plt.suptitle('Images Generated from Same Fixed Noise (500 steps)', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_500steps_images.png', dpi=150)
    plt.close()

    print("Images saved to jacobian_500steps_images.png")


if __name__ == "__main__":
    main()
