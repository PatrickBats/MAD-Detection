"""
t-SNE visualization for GAN MADness experiments.
Compares real data vs synthetic data from each generation.
Supports KMNIST and FashionMNIST datasets.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.datasets import MNIST, KMNIST, FashionMNIST
from torchvision import transforms
import os
import argparse


def run_tsne(dataset_name, n_samples=10000):
    """Run t-SNE for specified dataset"""

    if dataset_name == 'mnist':
        RealDataset = MNIST
        data_dir = './GAN-MNIST-Default/data/gan_outputs/'
        dataset_label = 'MNIST'
    elif dataset_name == 'kmnist':
        RealDataset = KMNIST
        data_dir = './GAN-MNIST-Kuzushiji/data/gan_outputs/'
        dataset_label = 'KMNIST'
    elif dataset_name == 'fashionmnist':
        RealDataset = FashionMNIST
        data_dir = './GAN-MNIST-Fashion/data/gan_outputs/'
        dataset_label = 'FashionMNIST'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load real data
    print(f"Loading real {dataset_label} data...")
    tf = transforms.Compose([transforms.ToTensor()])
    real_dataset = RealDataset("./data", train=True, download=True, transform=tf)
    real_data = torch.stack([real_dataset[i][0] for i in range(n_samples)])
    real_data = real_data.view(n_samples, -1).numpy()  # Flatten to (N, 784)

    # Get list of available generations
    gen_files = sorted([f for f in os.listdir(data_dir) if f.startswith('gen_data_') and f.endswith('.pt')])
    generations = [int(f.split('_')[2].split('.')[0]) for f in gen_files]
    print(f"Found generations: {generations}")

    # Create t-SNE plot for each generation
    for gen in generations:
        print(f"\nProcessing generation {gen}...")

        # Load synthetic data
        syn_path = data_dir + f"gen_data_{gen}.pt"
        syn_data = torch.load(syn_path).float()

        # Convert from [0, 255] to [0, 1] if needed
        if syn_data.max() > 1:
            syn_data = syn_data / 255.0

        # Take subset and flatten
        syn_data = syn_data[:n_samples].view(n_samples, -1).numpy()

        # Concatenate real and synthetic
        combined = np.concatenate([real_data, syn_data], axis=0)

        # Run t-SNE
        print(f"Running t-SNE on {combined.shape[0]} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embedded = tsne.fit_transform(combined)

        # Split back
        real_embedded = embedded[:n_samples]
        syn_embedded = embedded[n_samples:]

        # Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(real_embedded[:, 0], real_embedded[:, 1], c='red', s=0.5, alpha=0.5, label=f'Real {dataset_label}')
        plt.scatter(syn_embedded[:, 0], syn_embedded[:, 1], c='blue', s=0.5, alpha=0.5, label=f'Synthetic (Gen {gen})')
        plt.legend(markerscale=10)
        plt.title(f'{dataset_label} GAN - Generation {gen}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')

        # Save
        out_path = data_dir + f'tsne_generation_{gen}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    print(f"\nDone! t-SNE plots saved to {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['mnist', 'kmnist', 'fashionmnist', 'all'],
                        help='Dataset to run t-SNE for')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples for t-SNE')
    args = parser.parse_args()

    if args.dataset in ['mnist', 'all']:
        print("\n" + "="*50)
        print("Running t-SNE for MNIST")
        print("="*50)
        run_tsne('mnist', n_samples=args.n_samples)

    if args.dataset in ['kmnist', 'all']:
        print("\n" + "="*50)
        print("Running t-SNE for KMNIST")
        print("="*50)
        run_tsne('kmnist', n_samples=args.n_samples)

    if args.dataset in ['fashionmnist', 'all']:
        print("\n" + "="*50)
        print("Running t-SNE for FashionMNIST")
        print("="*50)
        run_tsne('fashionmnist', n_samples=args.n_samples)

    print("\nAll done!")
