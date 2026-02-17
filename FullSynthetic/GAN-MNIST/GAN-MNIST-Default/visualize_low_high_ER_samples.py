#!/usr/bin/env python3
"""
Visualize samples with lowest vs highest effective rank Jacobians.
Uses a FIXED CLASS to control for digit complexity differences.

This script:
1. Computes Jacobian eigenvalues for N_SAMPLES latent vectors per generation (single class)
2. Identifies samples with lowest and highest effective rank
3. Generates and visualizes those samples to look for artifacts

Key insight: Lower effective rank → more mode collapse → potentially more artifacts
"""

import torch
import torch.nn as nn
import torch.autograd.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse

# Generator architecture (must match training)
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


def compute_effective_rank(eigenvalues):
    """Compute effective rank from eigenvalues using entropy formula."""
    eigs_normalized = eigenvalues / eigenvalues.sum()
    return np.exp(-np.sum(eigs_normalized * np.log(eigs_normalized + 1e-10)))


def compute_jacobian_samples(generator, class_label, n_samples, device, seed_start=1000):
    """Compute Jacobian effective rank for multiple samples of a single class."""
    generator.eval()
    results = []

    for sample_idx in range(n_samples):
        seed = seed_start + sample_idx
        torch.manual_seed(seed)
        z_fixed = torch.randn(1, latent_dim).to(device)

        z_flat = z_fixed.flatten().detach().float().requires_grad_(True)
        c = torch.tensor([class_label], device=device, dtype=torch.long)

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

        results.append({
            'effective_rank': eff_rank,
            'eigenvalues': eigenvalues,
            'seed': seed,
            'class_label': class_label
        })

    return results


def generate_image(generator, seed, class_label, device):
    """Generate a single image from seed and class."""
    torch.manual_seed(seed)
    z = torch.randn(1, latent_dim).to(device)
    c = torch.tensor([class_label]).to(device)

    with torch.no_grad():
        img = generator(z, c)
        img = (img + 1) / 2  # Convert from [-1,1] to [0,1]

    return img[0, 0].cpu().numpy()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Class: {args.class_label}, Samples per gen: {args.n_samples}")
    print()

    # Determine generations based on available files
    data_dir = args.data_dir
    generations = []
    for i in range(20):  # Check up to gen 20
        gen_path = os.path.join(data_dir, f'generator_{i}.pth')
        if os.path.exists(gen_path):
            generations.append(i)

    print(f"Found generations: {generations}")

    # Compute or load results
    results_path = os.path.join(args.output_dir, f'jacobian_class{args.class_label}_results.pkl')

    if os.path.exists(results_path) and not args.recompute:
        print(f"Loading existing results from {results_path}")
        with open(results_path, 'rb') as f:
            results_dict = pickle.load(f)['results']
    else:
        print("Computing Jacobian effective ranks...")
        results_dict = {}

        for gen in generations:
            print(f"  Processing Gen {gen}...")

            gen_path = os.path.join(data_dir, f'generator_{gen}.pth')
            generator = Generator().to(device)
            generator.load_state_dict(torch.load(gen_path, map_location=device))
            generator.eval()

            results_dict[gen] = compute_jacobian_samples(
                generator, args.class_label, args.n_samples, device
            )

            eff_ranks = [s['effective_rank'] for s in results_dict[gen]]
            print(f"    ER = {np.mean(eff_ranks):.2f} +/- {np.std(eff_ranks):.2f}")

        # Save results
        with open(results_path, 'wb') as f:
            pickle.dump({'results': results_dict}, f)
        print(f"Saved results to {results_path}")

    # Create visualization
    print("\nCreating visualization...")

    n_show = min(3, args.n_samples // 2)  # Show top/bottom 3

    fig, axes = plt.subplots(len(generations), n_show * 2,
                              figsize=(2 * n_show * 2, 2 * len(generations)))
    fig.suptitle(f'Digit {args.class_label}: Lowest ER (left {n_show}) vs Highest ER (right {n_show})\n'
                 f'Lower ER = more mode collapse', fontsize=12, y=1.02)

    for gen_idx, gen in enumerate(generations):
        samples = results_dict[gen]
        sorted_samples = sorted(samples, key=lambda x: x['effective_rank'])

        lowest = sorted_samples[:n_show]
        highest = sorted_samples[-n_show:]

        # Load generator
        gen_path = os.path.join(data_dir, f'generator_{gen}.pth')
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        generator.eval()

        # Lowest ER samples
        for i, sample in enumerate(lowest):
            img = generate_image(generator, sample['seed'], args.class_label, device)

            ax = axes[gen_idx, i] if len(generations) > 1 else axes[i]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'ER={sample["effective_rank"]:.2f}', fontsize=8)
            ax.axis('off')

        # Highest ER samples
        for i, sample in enumerate(highest):
            img = generate_image(generator, sample['seed'], args.class_label, device)

            ax = axes[gen_idx, n_show + i] if len(generations) > 1 else axes[n_show + i]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'ER={sample["effective_rank"]:.2f}', fontsize=8)
            ax.axis('off')

        # Add generation label
        first_ax = axes[gen_idx, 0] if len(generations) > 1 else axes[0]
        first_ax.text(-0.3, 0.5, f'Gen {gen}', transform=first_ax.transAxes,
                      fontsize=11, fontweight='bold', va='center', ha='right')

    # Add column headers to first row
    first_row = axes[0] if len(generations) > 1 else axes
    for i in range(n_show):
        title = first_row[i].get_title()
        rank_label = ['Lowest', '2nd Low', '3rd Low'][i] if i < 3 else f'{i+1}th Low'
        first_row[i].set_title(f'{rank_label}\n{title}', fontsize=8)
    for i in range(n_show):
        title = first_row[n_show + i].get_title()
        rank_label = ['3rd High', '2nd High', 'Highest'][i] if i < 3 else f'{n_show-i}th High'
        first_row[n_show + i].set_title(f'{rank_label}\n{title}', fontsize=8)

    plt.tight_layout()

    save_path = os.path.join(args.output_dir, f'class{args.class_label}_low_vs_high_ER.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")

    # Print summary table
    print("\n" + "="*50)
    print(f"EFFECTIVE RANK SUMMARY (Class {args.class_label})")
    print("="*50)
    print(f"{'Gen':<5} {'Mean ER':<12} {'Min ER':<12} {'Max ER':<12}")
    print("-"*50)

    for gen in generations:
        eff_ranks = [s['effective_rank'] for s in results_dict[gen]]
        print(f"{gen:<5} {np.mean(eff_ranks):<12.2f} {np.min(eff_ranks):<12.2f} {np.max(eff_ranks):<12.2f}")

    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize low vs high effective rank samples')
    parser.add_argument('--class_label', type=int, default=7,
                        help='Digit class to analyze (default: 7)')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of samples per generation (default: 50)')
    parser.add_argument('--data_dir', type=str, default='./data/gan_outputs/',
                        help='Directory containing generator checkpoints')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to save outputs')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute Jacobians even if results file exists')

    args = parser.parse_args()
    main(args)
