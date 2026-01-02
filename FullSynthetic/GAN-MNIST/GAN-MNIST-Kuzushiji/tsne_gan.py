"""
t-SNE visualization for GAN-KMNIST MADness experiment.
Compares real KMNIST data vs synthetic data from each generation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.datasets import KMNIST
from torchvision import transforms
import os

save_dir = './data/gan_outputs/'
n_samples = 10000  # Number of samples for t-SNE (more = better but slower)

# Load real KMNIST data
print("Loading real KMNIST data...")
tf = transforms.Compose([transforms.ToTensor()])
real_dataset = KMNIST("../data", train=True, download=True, transform=tf)
real_data = torch.stack([real_dataset[i][0] for i in range(n_samples)])
real_data = real_data.view(n_samples, -1).numpy()  # Flatten to (N, 784)

# Get list of available generations
gen_files = sorted([f for f in os.listdir(save_dir) if f.startswith('gen_data_') and f.endswith('.pt')])
generations = [int(f.split('_')[2].split('.')[0]) for f in gen_files]
print(f"Found generations: {generations}")

# Create t-SNE plot for each generation
for gen in generations:
    print(f"\nProcessing generation {gen}...")

    # Load synthetic data
    syn_path = save_dir + f"gen_data_{gen}.pt"
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
    plt.scatter(real_embedded[:, 0], real_embedded[:, 1], c='red', s=0.5, alpha=0.5, label='Real KMNIST')
    plt.scatter(syn_embedded[:, 0], syn_embedded[:, 1], c='blue', s=0.5, alpha=0.5, label=f'Synthetic (Gen {gen})')
    plt.legend(markerscale=10)
    plt.title(f'KMNIST GAN - Generation {gen}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # Save
    out_path = save_dir + f'tsne_generation_{gen}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

print("\nDone! t-SNE plots saved to", save_dir)
