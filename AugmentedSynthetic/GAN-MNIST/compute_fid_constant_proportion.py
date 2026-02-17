#!/usr/bin/env python3
"""
Compute FID scores for all constant proportion experiments.

Compares generated samples from each generation against real MNIST
for each proportion experiment (p=0.0, 0.25, 0.5, 0.75, 0.9, 1.0).
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

# Proportions to analyze
PROPORTIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

# Pretrained LeNet model path
LENET_MODEL_PATH = './models/lenet_mnist.pth'


class LeNet(nn.Module):
    """LeNet for MNIST feature extraction (84-dim from penultimate layer)."""
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  # Return 84-dim features


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
    real_dataset = MNIST('../data', train=True, download=True, transform=tf)
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

    # Results storage
    all_results = {}

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for proportion in PROPORTIONS:
        p_int = int(proportion * 100)
        data_dir = os.path.join(script_dir, 'data', f'gan_outputs_p{p_int}')

        if not os.path.exists(data_dir):
            print(f"\n⚠ Skipping p={proportion}: Directory not found: {data_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"PROPORTION p={proportion} ({p_int}% synthetic)")
        print(f"{'='*60}")

        # Find available generations
        generations = []
        for i in range(20):
            gen_path = os.path.join(data_dir, f'gen_data_{i}.pt')
            if os.path.exists(gen_path):
                generations.append(i)

        if not generations:
            print(f"No generation data found in {data_dir}")
            continue

        print(f"Found generations: {generations}")

        fid_scores = []
        for gen in generations:
            gen_path = os.path.join(data_dir, f'gen_data_{gen}.pt')
            gen_images = torch.load(gen_path).float()

            # Normalize to [0, 1]
            if gen_images.max() > 1:
                gen_images = gen_images / 255.0

            # Extract features
            gen_features = extract_features(gen_images, model, device)

            # Compute FID
            fid = compute_fid(real_features, gen_features)
            fid_scores.append(fid)

            print(f"  Gen {gen}: FID = {fid:.2f}")

        all_results[proportion] = {
            'generations': generations,
            'fid_scores': fid_scores
        }

    # Print summary table
    print("\n" + "="*80)
    print("FID SCORE SUMMARY ACROSS PROPORTIONS")
    print("="*80)

    # Find max generations across all experiments
    max_gens = max(len(r['generations']) for r in all_results.values()) if all_results else 0

    # Header
    header = f"{'Gen':<6}"
    for p in sorted(all_results.keys()):
        header += f"{'p='+str(p):<12}"
    print(header)
    print("-"*80)

    # Data rows
    for gen_idx in range(max_gens):
        row = f"{gen_idx:<6}"
        for p in sorted(all_results.keys()):
            if gen_idx < len(all_results[p]['fid_scores']):
                fid = all_results[p]['fid_scores'][gen_idx]
                row += f"{fid:<12.2f}"
            else:
                row += f"{'-':<12}"
        print(row)

    print("="*80)

    # Compute degradation (FID change from Gen 0 to final)
    print("\nDEGRADATION SUMMARY:")
    print("-"*60)
    print(f"{'Proportion':<12} {'Initial FID':<14} {'Final FID':<14} {'Change':<14}")
    print("-"*60)

    for p in sorted(all_results.keys()):
        fid_scores = all_results[p]['fid_scores']
        if len(fid_scores) >= 2:
            initial = fid_scores[0]
            final = fid_scores[-1]
            change = final - initial
            sign = "+" if change > 0 else ""
            print(f"{p:<12.2f} {initial:<14.2f} {final:<14.2f} {sign}{change:<13.2f}")
    print("-"*60)

    # Create comparison plot
    if len(all_results) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

        for idx, p in enumerate(sorted(all_results.keys())):
            gens = all_results[p]['generations']
            fids = all_results[p]['fid_scores']
            ax.plot(gens, fids, 'o-', label=f'p={p:.2f}', color=colors[idx], linewidth=2, markersize=8)

        ax.set_xlabel('Generation')
        ax.set_ylabel('FID Score')
        ax.legend(title='Proportion')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(script_dir, 'fid_comparison_proportions.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    main()
