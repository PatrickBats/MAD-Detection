#!/usr/bin/env python3
"""
Jacobian spectrum analysis for GAN generators across generations.
Computes eigenvalues of the Jacobian J for the mapping: noise -> image.
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
class Generator(nn.Module):
    """Conditional Generator for MNIST"""
    def __init__(self):
        super(Generator, self).__init__()

        latent_dim = 100
        n_classes = 10
        img_size = 28
        channels = 1

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
        img = img.view(img.size(0), 1, 28, 28)
        return img


def load_generator(generation, device='cuda'):
    """Load Generator model for a specific generation.

    Args:
        generation: Generation number (0-7)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, 'data', 'gan_outputs', f'generator_{generation}.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.eval()

    return generator


def compute_jacobian_for_generation(gen, z_fixed_cpu, gpu_id, class_label=7):
    """
    Compute Jacobian for one GAN generation on one GPU.

    Args:
        gen: Generation number
        z_fixed_cpu: Fixed noise on CPU (to be moved to GPU) [1, 100]
        gpu_id: GPU ID to use
        class_label: Class to generate
    """

    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    try:
        generator = load_generator(gen, device)

        z_fixed = z_fixed_cpu.to(device)
        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)

        # Class label
        c = torch.tensor([class_label], device=device, dtype=torch.long)

        print(f"[GPU {gpu_id}] Gen {gen}: Computing Jacobian...")

        # Define generation function
        def gen_func(z_input):
            z_shaped = z_input.reshape(1, 100)
            output = generator(z_shaped, c)
            return output.flatten()

        start_time = time.time()
        jacobian = F.jacobian(gen_func, z_flat)
        elapsed = time.time() - start_time

        print(f"[GPU {gpu_id}] Gen {gen}: Jacobian shape: {jacobian.shape}")

        # Compute SVD for eigenvalues
        print(f"[GPU {gpu_id}] Gen {gen}: Computing SVD...")
        U, S, V = torch.linalg.svd(jacobian)

        eigenvalues = S.pow(2).cpu().numpy()

        # Generate image for visualization
        with torch.no_grad():
            image = generator(z_fixed.reshape(1, 100), c)

        # Print statistics
        print(f"[GPU {gpu_id}] Gen {gen}: Eigenvalue Statistics:")
        print(f"[GPU {gpu_id}]   Total: {len(eigenvalues)}")
        print(f"[GPU {gpu_id}]   Max: {eigenvalues[0]:.4f}")
        print(f"[GPU {gpu_id}]   Min: {eigenvalues[-1]:.4e}")
        print(f"[GPU {gpu_id}]   Median: {np.median(eigenvalues):.4f}")

        # Effective rank
        eigs_normalized = eigenvalues / eigenvalues.sum()
        eff_rank = np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))
        print(f"[GPU {gpu_id}]   Effective rank: {eff_rank:.2f} / 100")
        print(f"[GPU {gpu_id}]   Time: {elapsed:.2f}s")

        result = {
            'eigenvalues': eigenvalues,
            'image': image.cpu().numpy(),
            'time': elapsed
        }

        return gen, result

    except Exception as e:
        print(f"[GPU {gpu_id}] Gen {gen}: ERROR: {e}")
        import traceback
        traceback.print_exc()
        return gen, None


def main():
    """Main function with multi-GPU parallel processing."""

    generations = list(range(8))  # 0-7
    num_gpus = torch.cuda.device_count()

    print(f"\nGAN Jacobian Spectrum Analysis")
    print(f"Available GPUs: {num_gpus}")
    print(f"Generations to process: {generations}")
    print(f"Processing in parallel across GPUs")
    print()

    # SINGLE FIXED NOISE VECTOR - same for all generations
    torch.manual_seed(42)
    z_fixed_cpu = torch.randn(1, 100, dtype=torch.float32)

    print(f"Fixed noise statistics: mean={z_fixed_cpu.mean():.4f}, std={z_fixed_cpu.std():.4f}")
    print(f"Class label: 7 (same anchor point for all generations)")
    print()

    # Process all generations in parallel
    mp.set_start_method('spawn', force=True)
    start_total = time.time()

    with mp.Pool(processes=min(len(generations), num_gpus)) as pool:
        # Distribute generations across available GPUs
        results = pool.starmap(
            compute_jacobian_for_generation,
            [(gen, z_fixed_cpu, gen % num_gpus, 7) for gen in generations]
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

    print(f"\nTotal time: {total_time:.2f}s")

    # Save results
    print("Saving results...")
    with open('jacobian_gan_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print("Saved to jacobian_gan_results.pkl")

    # Plot comparison
    if len(results_dict) > 0:
        plot_eigenvalue_comparison(results_dict)
        plot_generated_images(results_dict)
        plot_eigenvalue_trends(results_dict)


def plot_eigenvalue_comparison(results):
    """Plot eigenvalue spectra comparison across generations."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    sorted_gens = sorted(results.keys())

    for idx, gen in enumerate(sorted_gens):
        eigs = results[gen]['eigenvalues']
        color = colors[idx]

        # Log scale plot
        ax1.semilogy(range(1, len(eigs)+1), eigs,
                    label=f'Gen {gen}', color=color, linewidth=2)

        # Histogram
        ax2.hist(np.log10(eigs + 1e-10), bins=50, alpha=0.5,
                label=f'Gen {gen}', color=color)

    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax1.set_title('Eigenvalue Spectrum\n(GAN Generator Jacobian)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('log10(Eigenvalue)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of log(Eigenvalues)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jacobian_gan_spectrum.png', dpi=150)
    plt.close()

    print("\nSpectrum plot saved to jacobian_gan_spectrum.png")


def plot_generated_images(results):
    """Plot generated images from fixed noise across generations."""

    n_gens = len(results)
    if n_gens == 0:
        return

    fig, axes = plt.subplots(1, n_gens, figsize=(3*n_gens, 3))

    if n_gens == 1:
        axes = [axes]

    sorted_gens = sorted(results.keys())

    for idx, gen in enumerate(sorted_gens):
        image = results[gen]['image']

        # Reshape and display
        if len(image.shape) == 4:
            image = image[0, 0]
        elif len(image.shape) == 3:
            image = image[0]
        elif len(image.shape) == 1:
            image = image.reshape(28, 28)

        # Convert from [-1, 1] to [0, 1] for display
        image = (image + 1) / 2

        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(f'Gen {gen}', fontsize=12)
        axes[idx].axis('off')

    plt.suptitle('Images Generated from Fixed Noise (Class 7)', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_gan_images.png', dpi=150)
    plt.close()

    print("Images saved to jacobian_gan_images.png")


def plot_eigenvalue_trends(results):
    """Plot how key eigenvalue metrics evolve across generations."""

    sorted_gens = sorted(results.keys())

    max_eigs = []
    median_eigs = []
    eff_ranks = []

    for gen in sorted_gens:
        eigs = results[gen]['eigenvalues']
        max_eigs.append(eigs[0])
        median_eigs.append(np.median(eigs))

        # Effective rank
        eigs_normalized = eigs / eigs.sum()
        eff_rank = np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))
        eff_ranks.append(eff_rank)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Max and median eigenvalue trends
    ax1.plot(sorted_gens, max_eigs, 'o-', label='Max Eigenvalue', linewidth=2, markersize=8)
    ax1.plot(sorted_gens, median_eigs, 's-', label='Median Eigenvalue', linewidth=2, markersize=8)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Eigenvalue Trends Across Generations', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Effective rank trend
    ax2.plot(sorted_gens, eff_ranks, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Effective Rank', fontsize=12)
    ax2.set_title('Effective Rank Across Generations', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=100, color='red', linestyle='--', label='Max Rank (100)', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('jacobian_gan_trends.png', dpi=150)
    plt.close()

    print("Trends plot saved to jacobian_gan_trends.png")


if __name__ == "__main__":
    main()
