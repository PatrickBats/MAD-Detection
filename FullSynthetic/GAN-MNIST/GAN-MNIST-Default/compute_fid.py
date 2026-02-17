#!/usr/bin/env python3
"""
Compute FID scores for GAN-MNIST-Default across generations.
Uses LeNet as feature extractor (84-dim features from penultimate layer).
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from scipy import linalg
import os
import matplotlib.pyplot as plt

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

# Paths
DATA_DIR = './data/gan_outputs/'
OUTPUT_DIR = '/home/patrick/MAD-Detection/FullSynthetic/figures'
LENET_MODEL_PATH = './data/gan_outputs/lenet_mnist.pth'


class LeNet(nn.Module):
    """LeNet for MNIST feature extraction (84-dim from penultimate layer)."""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x  # Return 84-dim features


def extract_features(images, model, device, batch_size=256):
    """Extract features from images using the model."""
    model.eval()
    features = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            if batch.dim() == 3:
                batch = batch.unsqueeze(1)
            feats = model(batch)
            features.append(feats.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_fid(real_features, gen_features):
    """Compute FID between two sets of features."""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    diff = mu_real - mu_gen

    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load real MNIST
    tf = transforms.Compose([transforms.ToTensor()])
    real_dataset = MNIST('../../data', train=True, download=True, transform=tf)
    real_images = real_dataset.data.float() / 255.0
    print(f"Loaded {len(real_images)} real MNIST images")

    # Initialize feature extractor with pretrained weights
    model = LeNet().to(device)
    if os.path.exists(LENET_MODEL_PATH):
        model.load_state_dict(torch.load(LENET_MODEL_PATH, map_location=device))
        print(f"Loaded pretrained LeNet from {LENET_MODEL_PATH}")
    else:
        print(f"WARNING: No pretrained model found at {LENET_MODEL_PATH}")
        print("FID scores may be unreliable with random weights!")
    model.eval()

    # Extract real features once
    print("Extracting real MNIST features...")
    real_features = extract_features(real_images, model, device)
    print(f"Real features shape: {real_features.shape}")

    # Find available generations
    generations = []
    for i in range(20):
        gen_path = os.path.join(DATA_DIR, f'gen_data_{i}.pt')
        if os.path.exists(gen_path):
            generations.append(i)

    print(f"Found generations: {generations}")

    # Compute FID for each generation
    fid_scores = {}

    for gen in generations:
        gen_path = os.path.join(DATA_DIR, f'gen_data_{gen}.pt')
        gen_images = torch.load(gen_path).float()

        # Normalize to [0, 1]
        if gen_images.max() > 1:
            gen_images = gen_images / 255.0

        # Extract features
        gen_features = extract_features(gen_images, model, device)

        # Compute FID
        fid = compute_fid(real_features, gen_features)
        fid_scores[gen] = fid

        print(f"Gen {gen}: FID = {fid:.2f}")

    # Print summary
    print("\n" + "="*40)
    print("FID SCORES SUMMARY")
    print("="*40)
    print(f"{'Generation':<12} {'FID Score':<12}")
    print("-"*40)
    for gen in sorted(fid_scores.keys()):
        print(f"{gen:<12} {fid_scores[gen]:<12.2f}")
    print("-"*40)

    initial_fid = fid_scores[min(fid_scores.keys())]
    final_fid = fid_scores[max(fid_scores.keys())]
    print(f"Degradation: {initial_fid:.2f} -> {final_fid:.2f} ({final_fid/initial_fid:.1f}x)")
    print("="*40)

    # Save results
    results = {
        'fid_scores': fid_scores,
        'generations': generations
    }
    save_path = os.path.join(DATA_DIR, 'fid_scores.pkl')
    torch.save(results, save_path)
    print(f"\nResults saved to: {save_path}")

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4.5))

    gens = sorted(fid_scores.keys())
    fids = [fid_scores[g] for g in gens]

    ax.plot(gens, fids, 'o-', color='darkblue', linewidth=2, markersize=8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('FID Score')
    ax.set_xticks(gens)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, 'gan_mnist_fid.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
