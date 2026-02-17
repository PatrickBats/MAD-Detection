#!/usr/bin/env python3


import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

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

OUTPUT_DIR = '/home/patrick/MAD-Detection/FullSynthetic/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gan_results():
    """Load GAN MNIST averaging results."""
    path = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/jacobian_gan_avg_results.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_ddpm_results():
    """Load DDPM MNIST averaging results."""
    path = '/home/patrick/MAD-Detection/FullSynthetic/DDPM-MNIST/SpectrumCalc/jacobian_ddim_avg_results.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_fashion_results():
    """Load FashionMNIST GAN class7 results (100 samples per gen)."""
    path = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/jacobian_gan_class7_results.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_gan_spectrum_plot(data):
    """Create singular value spectrum plot for GAN MNIST."""
    results = data['results']
    sorted_gens = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results[gen]
        if len(samples) == 0:
            continue

        # Convert squared singular values back to singular values
        all_svs = np.array([np.sqrt(s['eigenvalues']) for s in samples])
        mean_svs = np.mean(all_svs, axis=0)
        std_svs = np.std(all_svs, axis=0)

        x = range(1, len(mean_svs) + 1)
        color = colors[gen_idx]

        ax.semilogy(x, mean_svs, label=f'Gen {gen}', color=color, linewidth=1.5)
        ax.fill_between(x,
                        np.maximum(mean_svs - std_svs, 1e-10),
                        mean_svs + std_svs,
                        color=color, alpha=0.2)

    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'gan_mnist_spectrum.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def create_gan_effective_rank_plot(data):
    """Create effective rank scatter plot with mean±std for GAN MNIST."""
    results = data['results']
    sorted_gens = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results[gen]
        if len(samples) == 0:
            continue

        eff_ranks = [s['effective_rank'] for s in samples]
        color = colors[gen_idx]

        # Scatter with jitter
        jitter = np.random.uniform(-0.2, 0.2, len(eff_ranks))
        ax.scatter([gen_idx] * len(eff_ranks) + jitter, eff_ranks,
                   color=color, alpha=0.5, s=30)

        # Mean with error bar
        ax.errorbar([gen_idx], [np.mean(eff_ranks)], yerr=[np.std(eff_ranks)],
                    fmt='o', color='black', capsize=5, markersize=10)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Effective Rank')
    ax.set_xticks(range(len(sorted_gens)))
    ax.set_xticklabels([str(g) for g in sorted_gens])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'gan_mnist_effective_rank.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def create_ddpm_spectrum_plot(data):
    """Create singular value spectrum plot for DDPM MNIST."""
    results = data['results']
    # Sort with 'initial' first, then numeric generations
    sorted_gens = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results[gen]
        if len(samples) == 0:
            continue

        # Convert squared singular values back to singular values
        all_svs = np.array([np.sqrt(s['eigenvalues']) for s in samples])
        mean_svs = np.mean(all_svs, axis=0)
        std_svs = np.std(all_svs, axis=0)

        x = range(1, len(mean_svs) + 1)
        color = colors[gen_idx]
        # Convert: initial -> Gen 0, gen N -> Gen N+1
        display_gen = 0 if gen == 'initial' else gen + 1
        label = f'Gen {display_gen}'

        ax.semilogy(x, mean_svs, label=label, color=color, linewidth=1.5)
        ax.fill_between(x,
                        np.maximum(mean_svs - std_svs, 1e-10),
                        mean_svs + std_svs,
                        color=color, alpha=0.2)

    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'ddpm_mnist_spectrum.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def create_ddpm_effective_rank_plot(data):
    """Create effective rank scatter plot with mean±std for DDPM MNIST."""
    results = data['results']
    # Sort with 'initial' first, then numeric generations
    sorted_gens = sorted(results.keys(), key=lambda x: -1 if x == 'initial' else x)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    gen_labels = []
    for gen_idx, gen in enumerate(sorted_gens):
        samples = results[gen]
        if len(samples) == 0:
            continue

        eff_ranks = [s['effective_rank'] for s in samples]
        color = colors[gen_idx]
        # Convert: initial -> 0, gen N -> N+1
        display_gen = 0 if gen == 'initial' else gen + 1
        gen_labels.append(str(display_gen))

        # Scatter with jitter
        jitter = np.random.uniform(-0.2, 0.2, len(eff_ranks))
        ax.scatter([gen_idx] * len(eff_ranks) + jitter, eff_ranks,
                   color=color, alpha=0.5, s=30)

        # Mean with error bar
        ax.errorbar([gen_idx], [np.mean(eff_ranks)], yerr=[np.std(eff_ranks)],
                    fmt='o', color='black', capsize=5, markersize=10)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Effective Rank')
    ax.set_xticks(range(len(sorted_gens)))
    ax.set_xticklabels(gen_labels)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'ddpm_mnist_effective_rank.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def create_fashion_spectrum_plot(data):
    """Create singular value spectrum plot for FashionMNIST GAN."""
    results = data['results']
    sorted_gens = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results[gen]
        if len(samples) == 0:
            continue

        # Convert squared singular values back to singular values
        all_svs = np.array([np.sqrt(s['eigenvalues']) for s in samples])
        mean_svs = np.mean(all_svs, axis=0)
        std_svs = np.std(all_svs, axis=0)

        x = range(1, len(mean_svs) + 1)
        color = colors[gen_idx]

        ax.semilogy(x, mean_svs, label=f'Gen {gen}', color=color, linewidth=1.5)
        ax.fill_between(x,
                        np.maximum(mean_svs - std_svs, 1e-10),
                        mean_svs + std_svs,
                        color=color, alpha=0.2)

    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'gan_fashion_spectrum.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def create_fashion_effective_rank_plot(data):
    """Create effective rank scatter plot with mean±std for FashionMNIST GAN."""
    results = data['results']
    sorted_gens = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for gen_idx, gen in enumerate(sorted_gens):
        samples = results[gen]
        if len(samples) == 0:
            continue

        eff_ranks = [s['effective_rank'] for s in samples]
        color = colors[gen_idx]

        # Scatter with jitter
        jitter = np.random.uniform(-0.2, 0.2, len(eff_ranks))
        ax.scatter([gen_idx] * len(eff_ranks) + jitter, eff_ranks,
                   color=color, alpha=0.5, s=30)

        # Mean with error bar
        ax.errorbar([gen_idx], [np.mean(eff_ranks)], yerr=[np.std(eff_ranks)],
                    fmt='o', color='black', capsize=5, markersize=10)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Effective Rank')
    ax.set_xticks(range(len(sorted_gens)))
    ax.set_xticklabels([str(g) for g in sorted_gens])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'gan_fashion_effective_rank.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Creating publication-quality figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load data
    gan_data = load_gan_results()
    ddpm_data = load_ddpm_results()
    fashion_data = load_fashion_results()

    print()

    # Create figures
    print("Creating GAN MNIST singular value spectrum...")
    create_gan_spectrum_plot(gan_data)

    create_gan_effective_rank_plot(gan_data)

    print("Creating DDPM MNIST singular value spectrum...")
    create_ddpm_spectrum_plot(ddpm_data)

    create_ddpm_effective_rank_plot(ddpm_data)

    print("Creating FashionMNIST GAN singular value spectrum...")
    create_fashion_spectrum_plot(fashion_data)

    create_fashion_effective_rank_plot(fashion_data)

    print()

if __name__ == "__main__":
    main()
