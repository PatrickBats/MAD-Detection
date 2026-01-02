#!/usr/bin/env python3
"""
Multi-anchor Jacobian analysis for GAN generators.
Tests 6 different anchor points: 2 noise vectors each for classes 7, 3, and 5.
Verifies if eigenvalue trends are consistent across different inputs.
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


# Define the 6 anchor points: (class_label, seed)
ANCHOR_POINTS = [
    (7, 42),   # 7_a
    (7, 123),  # 7_b
    (3, 42),   # 3_a
    (3, 123),  # 3_b
    (5, 42),   # 5_a
    (5, 123),  # 5_b
]


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
    Compute Jacobian for one (generation, anchor) pair.

    Args:
        task_args: tuple of (gen, anchor_idx, class_label, seed, gpu_id)
    """
    gen, anchor_idx, class_label, seed, gpu_id = task_args

    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    # Generate fixed noise for this anchor
    torch.manual_seed(seed)
    z_fixed = torch.randn(1, latent_dim, dtype=torch.float32).to(device)

    anchor_name = f"{class_label}_{chr(ord('a') + anchor_idx % 2)}"

    try:
        generator = load_generator(gen, device)

        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)
        c = torch.tensor([class_label], device=device, dtype=torch.long)

        print(f"[GPU {gpu_id}] Gen {gen}, anchor {anchor_name}: Computing Jacobian...")

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

        # Effective rank
        eigs_normalized = eigenvalues / eigenvalues.sum()
        eff_rank = np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))

        print(f"[GPU {gpu_id}] Gen {gen}, anchor {anchor_name}: Max={eigenvalues[0]:.4f}, EffRank={eff_rank:.2f}, Time={elapsed:.1f}s")

        result = {
            'eigenvalues': eigenvalues,
            'singular_values': S.cpu().numpy(),
            'U': U.cpu().numpy(),
            'V': V.cpu().numpy(),
            'jacobian': jacobian.cpu().numpy(),
            'image': image.cpu().numpy(),
            'time': elapsed,
            'class_label': class_label,
            'seed': seed,
            'anchor_name': anchor_name
        }

        # Save to disk immediately
        save_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(save_dir, 'svd_results'), exist_ok=True)
        save_path = os.path.join(save_dir, 'svd_results', f'gen{gen}_anchor{anchor_idx}_{anchor_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"[GPU {gpu_id}] Saved to {save_path}")

        # Free memory
        del generator, jacobian, U, S, V
        torch.cuda.empty_cache()

        # Return lightweight data
        return (gen, anchor_idx), {
            'eigenvalues': eigenvalues,
            'time': elapsed,
            'anchor_name': anchor_name,
            'class_label': class_label,
            'seed': seed,
            'save_path': save_path
        }

    except Exception as e:
        print(f"[GPU {gpu_id}] Gen {gen}, anchor {anchor_name}: ERROR: {e}")
        import traceback
        traceback.print_exc()
        return (gen, anchor_idx), None


def main():
    """Main function with multi-GPU parallel processing."""

    generations = list(range(8))  # 0-7
    num_gpus = torch.cuda.device_count()

    print(f"\nMulti-Anchor GAN Jacobian Analysis")
    print(f"Available GPUs: {num_gpus}")
    print(f"Generations: {generations}")
    print(f"Anchor points: 6 (2 each for classes 7, 3, 5)")
    print(f"Total computations: {len(generations) * len(ANCHOR_POINTS)} = {len(generations)} gens × 6 anchors")
    print()

    # Build task list: all (generation, anchor) combinations
    tasks = []
    task_idx = 0
    for gen in generations:
        for anchor_idx, (class_label, seed) in enumerate(ANCHOR_POINTS):
            gpu_id = task_idx % num_gpus
            tasks.append((gen, anchor_idx, class_label, seed, gpu_id))
            task_idx += 1

    print(f"Distributing {len(tasks)} tasks across {num_gpus} GPUs")
    print()

    # Process in parallel - GAN is lightweight, can run many in parallel
    mp.set_start_method('spawn', force=True)
    start_total = time.time()

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.map(compute_jacobian_for_task, tasks)

    total_time = time.time() - start_total

    # Organize results: {generation: {anchor_idx: result}}
    results_dict = {}
    for (gen, anchor_idx), result in results:
        if gen not in results_dict:
            results_dict[gen] = {}
        if result is not None:
            results_dict[gen][anchor_idx] = result

    # Count successes
    success_count = sum(len(anchors) for anchors in results_dict.values())
    print(f"\n✓ Completed {success_count}/{len(tasks)} computations")
    print(f"Total time: {total_time:.1f}s")

    # Save summary results
    print("Saving summary results...")
    with open('jacobian_gan_multi_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print("Saved to jacobian_gan_multi_results.pkl")

    # Plot results
    if success_count > 0:
        plot_eigenvalue_trends(results_dict)
        plot_all_spectra(results_dict)
        plot_generated_images(results_dict)


def plot_eigenvalue_trends(results):
    """Plot how eigenvalue metrics evolve across generations for each anchor."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    sorted_gens = sorted(results.keys())
    gen_labels = [str(g) for g in sorted_gens]
    x_pos = range(len(sorted_gens))

    colors = ['blue', 'lightblue', 'red', 'lightcoral', 'green', 'lightgreen']
    anchor_labels = ['7_a', '7_b', '3_a', '3_b', '5_a', '5_b']

    # Collect metrics for each anchor across generations
    for anchor_idx in range(6):
        max_eigs = []
        eff_ranks = []

        for gen in sorted_gens:
            if gen in results and anchor_idx in results[gen]:
                eigs = results[gen][anchor_idx]['eigenvalues']
                max_eigs.append(eigs[0])

                eigs_norm = eigs / eigs.sum()
                eff_rank = np.exp(-np.sum(eigs_norm * np.log(eigs_norm + 1e-10)))
                eff_ranks.append(eff_rank)
            else:
                max_eigs.append(np.nan)
                eff_ranks.append(np.nan)

        # Plot max eigenvalue
        axes[0, anchor_idx // 2].plot(x_pos, max_eigs, 'o-',
                                       color=colors[anchor_idx],
                                       label=anchor_labels[anchor_idx],
                                       linewidth=2, markersize=6)

        # Plot effective rank
        axes[1, anchor_idx // 2].plot(x_pos, eff_ranks, 's-',
                                       color=colors[anchor_idx],
                                       label=anchor_labels[anchor_idx],
                                       linewidth=2, markersize=6)

    # Format plots
    class_names = ['Class 7', 'Class 3', 'Class 5']
    for i in range(3):
        axes[0, i].set_title(f'{class_names[i]} - Max Eigenvalue', fontsize=12)
        axes[0, i].set_xlabel('Generation')
        axes[0, i].set_ylabel('Max Eigenvalue')
        axes[0, i].set_xticks(x_pos)
        axes[0, i].set_xticklabels(gen_labels)
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].set_title(f'{class_names[i]} - Effective Rank', fontsize=12)
        axes[1, i].set_xlabel('Generation')
        axes[1, i].set_ylabel('Effective Rank')
        axes[1, i].set_xticks(x_pos)
        axes[1, i].set_xticklabels(gen_labels)
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle('GAN Eigenvalue Trends Across Generations\n(2 anchor points per class)', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_gan_multi_trends.png', dpi=150)
    plt.close()

    print("Trends plot saved to jacobian_gan_multi_trends.png")


def plot_all_spectra(results):
    """Plot all eigenvalue spectra, grouped by anchor point."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    sorted_gens = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    anchor_labels = ['7_a', '7_b', '3_a', '3_b', '5_a', '5_b']

    for anchor_idx in range(6):
        ax = axes[anchor_idx // 3, anchor_idx % 3]

        for gen_idx, gen in enumerate(sorted_gens):
            if gen in results and anchor_idx in results[gen]:
                eigs = results[gen][anchor_idx]['eigenvalues']
                label = f'Gen {gen}'
                ax.semilogy(range(1, len(eigs)+1), eigs,
                           label=label, color=colors[gen_idx], linewidth=1.5)

        ax.set_title(f'Anchor {anchor_labels[anchor_idx]}', fontsize=12)
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('GAN Eigenvalue Spectra by Anchor Point', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_gan_multi_spectra.png', dpi=150)
    plt.close()

    print("Spectra plot saved to jacobian_gan_multi_spectra.png")


def plot_generated_images(results):
    """Plot all generated images in a grid: anchors × generations."""

    sorted_gens = sorted(results.keys())
    n_anchors = 6
    n_gens = len(sorted_gens)

    fig, axes = plt.subplots(n_anchors, n_gens, figsize=(2.5*n_gens, 2.5*n_anchors))

    anchor_labels = ['7_a', '7_b', '3_a', '3_b', '5_a', '5_b']

    for anchor_idx in range(n_anchors):
        for gen_idx, gen in enumerate(sorted_gens):
            ax = axes[anchor_idx, gen_idx]

            if gen in results and anchor_idx in results[gen]:
                # Load image from saved pkl file
                save_path = results[gen][anchor_idx].get('save_path')
                if save_path and os.path.exists(save_path):
                    with open(save_path, 'rb') as f:
                        full_result = pickle.load(f)
                    image = full_result['image']
                else:
                    image = results[gen][anchor_idx].get('image')

                if image is not None:
                    if len(image.shape) == 4:
                        image = image[0, 0]
                    elif len(image.shape) == 3:
                        image = image[0]
                    elif len(image.shape) == 1:
                        image = image.reshape(28, 28)

                    # Convert from [-1, 1] to [0, 1]
                    image = (image + 1) / 2

                    ax.imshow(image, cmap='gray')
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            # Row labels (anchor names)
            if gen_idx == 0:
                ax.set_ylabel(anchor_labels[anchor_idx], fontsize=12, rotation=0, labelpad=30)

            # Column labels (generation)
            if anchor_idx == 0:
                ax.set_title(f'Gen {gen}', fontsize=12)

    plt.suptitle('GAN Generated Images: Anchor Points × Generations', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('jacobian_gan_multi_images.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Images saved to jacobian_gan_multi_images.png")


if __name__ == "__main__":
    main()
