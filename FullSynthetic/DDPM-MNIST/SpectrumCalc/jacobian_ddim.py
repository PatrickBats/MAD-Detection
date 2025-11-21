#!/usr/bin/env python3
"""
Full 500-step generation Jacobian using DDIM (deterministic) sampling.
Uses DDIM formula instead of DDPM for reproducible, deterministic Jacobians.
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
    """Load DDPM model for a specific generation.

    Args:
        generation: Generation number, or 'initial' for model_initial.pth
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    n_T = 500
    nn_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)
    ddpm = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

    # Handle initial model checkpoint
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
    """
    Full 500-step DDIM generation (deterministic).

    Args:
        z: Initial noise [1, 1, 28, 28]
        model: DDPM model (using DDIM sampling)
        class_label: Class to generate
    """
    device = z.device

    timesteps = range(500, 0, -1)

    # Setup class and context
    c = torch.tensor([class_label], device=device, dtype=torch.long)
    context_mask = torch.zeros(1, device=device, dtype=torch.float32)

    # Start with noise
    x = z.clone().float()

    # Full DDIM denoising loop 
    for t in timesteps:
        t_norm = torch.tensor([t / model.n_T], device=device, dtype=torch.float32).view(1, 1, 1, 1)

        # Predict noise (no torch.no_grad() for Jacobian)
        eps = model.nn_model(x, c, t_norm, context_mask)

        # Apply DDIM denoising formula 
        if t > 1:
            # Predict x_0 from current x_t
            x_0_pred = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]

            # Compute x_{t-1} using DDIM formula
            x = model.sqrtab[t-1] * x_0_pred + model.sqrtmab[t-1] * eps
        else:
            #predict x_0
            x = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]

    return x


def compute_jacobian_for_generation(gen, z_fixed_cpu, gpu_id, class_label=7):
    """
    Compute Jacobian for one generation on one GPU using DDIM.

    Args:
        gen: Generation number
        z_fixed_cpu: Fixed noise on CPU (to be moved to GPU)
        gpu_id: GPU ID to use
        class_label: Class to generate
    """

    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'



    try:
        model = load_model(gen, device)

        z_fixed = z_fixed_cpu.to(device)
        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)

        print(f"[GPU {gpu_id}] Computing Jacobian (500 steps DDIM)...")
        print(f"[GPU {gpu_id}]   This will take ~10-20 minutes...")

        # Define generation function
        def gen_func(z_input):
            z_shaped = z_input.reshape(1, 1, 28, 28)
            output = generation_full_500steps_ddim(z_shaped, model, class_label)
            return output.flatten()

        start_time = time.time()
        jacobian = F.jacobian(gen_func, z_flat)
        elapsed = time.time() - start_time

        print(f"[GPU {gpu_id}] Jacobian shape: {jacobian.shape}")

        # Compute SVD for eigenvalues
        print(f"[GPU {gpu_id}] Computing SVD...")
        U, S, V = torch.linalg.svd(jacobian)

        eigenvalues = S.pow(2).cpu().numpy()

        with torch.no_grad():
            image = generation_full_500steps_ddim(z_fixed, model, class_label)

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
    """Main function with multi-GPU parallel processing using DDIM."""

    generations = ['initial', 0, 1, 2, 3, 4, 5]
    num_gpus = torch.cuda.device_count()

    print(f"\nDDIM Jacobian Computation")
    print(f"Available GPUs: {num_gpus}")
    print(f"Generations to process: {generations}")
    print(f"Each generation will run on a separate GPU in parallel")
    print()

    # SINGLE FIXED POINT - same for all generations
    torch.manual_seed(42)
    z_fixed_cpu = torch.randn(1, 1, 28, 28, dtype=torch.float32)

    print(f"Initial noise statistics: mean={z_fixed_cpu.mean():.4f}, std={z_fixed_cpu.std():.4f}")
    print(f"Full 500-step DDIM (deterministic) denoising per generation")
    print()
    print()

    # Process all generations in parallel
    mp.set_start_method('spawn', force=True)
    start_total = time.time()

    with mp.Pool(processes=len(generations)) as pool:
        # Each generation gets its own GPU
        results = pool.starmap(
            compute_jacobian_for_generation,
            [(gen, z_fixed_cpu, idx % num_gpus, 7) for idx, gen in enumerate(generations)]
        )

    total_time = time.time() - start_total

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
    with open('jacobian_ddim_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print("Saved to jacobian_ddim_results.pkl")

    # Plot comparison
    if len(results_dict) > 0:
        plot_eigenvalue_comparison(results_dict)
        plot_generated_images(results_dict)


def plot_eigenvalue_comparison(results):
    """Plot eigenvalue spectra comparison."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan']

    # Sort keys with 'initial' first
    sorted_keys = sorted(results.keys(), key=lambda x: (-1 if x == 'initial' else x))

    for idx, gen in enumerate(sorted_keys):
        eigs = results[gen]['eigenvalues']
        color = colors[idx % len(colors)]

        # Format label
        label = 'Initial' if gen == 'initial' else f'Gen {gen}'

        # Log scale plot
        ax1.semilogy(range(1, len(eigs)+1), eigs,
                    label=label, color=color, linewidth=2)

        # Histogram
        ax2.hist(np.log10(eigs + 1e-10), bins=50, alpha=0.5,
                label=label, color=color)

    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title('Eigenvalue Spectrum\n(Full 500-step DDIM Jacobian)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('log10(Eigenvalue)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of log(Eigenvalues)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jacobian_ddim_spectrum.png', dpi=150)
    plt.close()

    print("\nPlot saved to jacobian_ddim_spectrum.png")


def plot_generated_images(results):

    n_gens = len(results)
    if n_gens == 0:
        return

    fig, axes = plt.subplots(1, n_gens, figsize=(3*n_gens, 3))

    if n_gens == 1:
        axes = [axes]

    # Sort keys with 'initial' first
    sorted_keys = sorted(results.keys(), key=lambda x: (-1 if x == 'initial' else x))

    for idx, gen in enumerate(sorted_keys):
        image = results[gen]['image']

        # Reshape and display
        if len(image.shape) == 4:
            image = image[0, 0]
        elif len(image.shape) == 3:
            image = image[0]
        elif len(image.shape) == 1:
            image = image.reshape(28, 28)

        # Format title
        title = 'Initial' if gen == 'initial' else f'Gen {gen}'

        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')

    plt.suptitle('Images Generated with DDIM (Deterministic, 500 steps)', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_ddim_images.png', dpi=150)
    plt.close()

    print("Images saved to jacobian_ddim_images.png")


if __name__ == "__main__":
    main()
