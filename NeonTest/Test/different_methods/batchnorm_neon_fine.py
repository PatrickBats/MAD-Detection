#!/usr/bin/env python3
"""
Fine-grained sweep of BatchNorm-only NEON for GAN-MNIST

Based on results from selective_layer_neon.py, BatchNorm-only showed the most
promise with optimal w=0.05 for Generation 5. This script does a finer sweep
of w values specifically for the batchnorm_only strategy.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from scipy import linalg
import os
import copy
import pickle
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

# Hyperparameters
latent_dim = 100
img_size = 28
channels = 1
n_classes = 10
N_eval = 10000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
GAN_DIR = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/'
SAVE_DIR = '/home/patrick/MAD-Detection/NeonTest/Test/different_methods/results_batchnorm_fine/'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + "samples/", exist_ok=True)


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


class LeNet(nn.Module):
    """LeNet for feature extraction"""
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def extract_features(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


def load_generator(generation):
    """Load generator checkpoint"""
    checkpoint_path = os.path.join(GAN_DIR, f'generator_{generation}.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.eval()
    return generator


def merge_batchnorm_only(base_model: nn.Module, collapsed_model: nn.Module, w: float) -> nn.Module:
    """
    Apply NEON only to BatchNorm parameters (gamma and beta).

    BatchNorm layers in Generator:
    - model.3: BatchNorm1d(256) after first hidden layer
    - model.6: BatchNorm1d(512) after second hidden layer
    - model.9: BatchNorm1d(1024) after third hidden layer
    """
    merged = copy.deepcopy(base_model)

    # BatchNorm parameter names
    bn_params = ['model.3.weight', 'model.3.bias',   # 256-dim
                 'model.6.weight', 'model.6.bias',   # 512-dim
                 'model.9.weight', 'model.9.bias']   # 1024-dim

    with torch.no_grad():
        base_dict = dict(base_model.named_parameters())
        collapsed_dict = dict(collapsed_model.named_parameters())

        for name, param in merged.named_parameters():
            if name in bn_params:
                # Apply NEON: (1+w)*base - w*collapsed
                param.copy_((1.0 + w) * base_dict[name] - w * collapsed_dict[name])

    return merged


def generate_samples(generator, N):
    """Generate N samples"""
    generator.eval()
    all_imgs = []
    batch_size = 500
    n_batches = N // batch_size

    with torch.no_grad():
        for _ in range(n_batches):
            z = torch.randn(batch_size, latent_dim).to(device)
            labels = torch.randint(0, n_classes, (batch_size,)).to(device)
            imgs = generator(z, labels)
            imgs = (imgs + 1) / 2
            all_imgs.append(imgs.cpu())

    return torch.cat(all_imgs, dim=0)


def extract_features_batch(model, images, batch_size=256):
    """Extract features from images"""
    model.eval()
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = model.extract_features(batch)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(real_features, fake_features):
    """Compute FID"""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_fake += eps * np.eye(sigma_fake.shape[0])

    diff = mu_real - mu_fake
    try:
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    except:
        fid = float('inf')

    return fid


def main():
    print("="*70)
    print("BatchNorm-Only NEON Fine Sweep for GAN-MNIST")
    print("="*70)

    # Load LeNet
    lenet_path = os.path.join(GAN_DIR, 'lenet_mnist.pth')
    lenet = LeNet().to(device)
    if os.path.exists(lenet_path):
        lenet.load_state_dict(torch.load(lenet_path, map_location=device))
    else:
        raise FileNotFoundError(f"LeNet not found at {lenet_path}")
    lenet.eval()

    # Get real features
    print("\nExtracting real MNIST features...")
    tf = transforms.Compose([transforms.ToTensor()])
    mnist_data = MNIST("../../data", train=True, download=True, transform=tf)
    real_loader = DataLoader(mnist_data, batch_size=256, shuffle=True)

    real_images = []
    for imgs, _ in real_loader:
        real_images.append(imgs)
        if len(real_images) * 256 >= N_eval:
            break
    real_images = torch.cat(real_images)[:N_eval]
    real_features = extract_features_batch(lenet, real_images)

    # Load base model
    print("\nLoading base model (generation 0)...")
    gen_base = load_generator(0)

    # Compute base FID
    base_imgs = generate_samples(gen_base, N_eval)
    base_features = extract_features_batch(lenet, base_imgs)
    fid_base = compute_fid(real_features, base_features)
    print(f"Base model (gen 0) FID: {fid_base:.2f}")

    save_image(base_imgs[:100], SAVE_DIR + "samples/gen0_base.png", nrow=10)

    # Fine-grained w values around the promising region
    w_values = [-0.1, -0.075, -0.05, -0.025, -0.01,
                0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5]

    # Test all generations
    test_generations = list(range(1, 8))

    results = {
        'fid_base': fid_base,
        'w_values': w_values,
        'generations': {}
    }

    for collapsed_gen in test_generations:
        print(f"\n{'='*60}")
        print(f"Testing with collapsed generation {collapsed_gen}")
        print(f"{'='*60}")

        gen_collapsed = load_generator(collapsed_gen)

        # Compute collapsed FID
        collapsed_imgs = generate_samples(gen_collapsed, N_eval)
        collapsed_features = extract_features_batch(lenet, collapsed_imgs)
        fid_collapsed = compute_fid(real_features, collapsed_features)
        print(f"Collapsed model (gen {collapsed_gen}) FID: {fid_collapsed:.2f}")

        save_image(collapsed_imgs[:100],
                   SAVE_DIR + f"samples/gen{collapsed_gen}_collapsed.png", nrow=10)

        gen_results = {
            'fid_collapsed': fid_collapsed,
            'fid_at_w': {}
        }

        best_fid = float('inf')
        best_w = None

        for w in w_values:
            gen_neon = merge_batchnorm_only(gen_base, gen_collapsed, w)
            neon_imgs = generate_samples(gen_neon, N_eval)

            if torch.isnan(neon_imgs).any() or torch.isinf(neon_imgs).any():
                print(f"  w={w:+.3f}: UNSTABLE")
                gen_results['fid_at_w'][w] = float('inf')
                continue

            neon_features = extract_features_batch(lenet, neon_imgs)
            fid = compute_fid(real_features, neon_features)
            gen_results['fid_at_w'][w] = fid

            marker = " <-- BEST" if fid < best_fid else ""
            print(f"  w={w:+.3f}: FID={fid:.2f}{marker}")

            if fid < best_fid:
                best_fid = fid
                best_w = w

            # Save samples at key w values
            if w in [0.0, 0.05, 0.1]:
                save_image(neon_imgs[:100],
                          SAVE_DIR + f"samples/gen{collapsed_gen}_neon_w{w}.png", nrow=10)

        gen_results['best_w'] = best_w
        gen_results['best_fid'] = best_fid
        results['generations'][collapsed_gen] = gen_results

        improvement_base = fid_base - best_fid
        improvement_collapsed = fid_collapsed - best_fid
        print(f"\n  Best: w={best_w:.3f}, FID={best_fid:.2f}")
        print(f"  Improvement over base: {improvement_base:.2f}")
        print(f"  Improvement over collapsed: {improvement_collapsed:.2f}")

    # Save results
    with open(SAVE_DIR + 'batchnorm_fine_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {SAVE_DIR}batchnorm_fine_results.pkl")

    # Create plots
    plot_results(results, test_generations)
    print_summary(results, test_generations)


def plot_results(results, test_generations):
    """Create visualization plots"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: FID vs w for all generations
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_generations)))

    for idx, gen in enumerate(test_generations):
        fid_at_w = results['generations'][gen]['fid_at_w']
        ws = sorted(fid_at_w.keys())
        fids = [fid_at_w[w] for w in ws]

        valid = [(w, f) for w, f in zip(ws, fids) if f != float('inf')]
        if valid:
            w_plot, fid_plot = zip(*valid)
            ax1.plot(w_plot, fid_plot, 'o-', color=colors[idx],
                    label=f'Gen {gen}', linewidth=1.5, markersize=3)

    ax1.axhline(y=results['fid_base'], color='green', linestyle='--',
                label=f'Base: {results["fid_base"]:.1f}', linewidth=2)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('NEON Weight (w)')
    ax1.set_ylabel('FID Score')
    ax1.set_title('BatchNorm-Only NEON: FID vs Weight')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.15, 0.55)

    # Plot 2: Zoomed in around w=0
    ax2 = axes[0, 1]

    for idx, gen in enumerate(test_generations):
        fid_at_w = results['generations'][gen]['fid_at_w']
        ws = sorted(fid_at_w.keys())
        fids = [fid_at_w[w] for w in ws]

        valid = [(w, f) for w, f in zip(ws, fids) if f != float('inf') and -0.1 <= w <= 0.15]
        if valid:
            w_plot, fid_plot = zip(*valid)
            ax2.plot(w_plot, fid_plot, 'o-', color=colors[idx],
                    label=f'Gen {gen}', linewidth=1.5, markersize=4)

    ax2.axhline(y=results['fid_base'], color='green', linestyle='--',
                label=f'Base: {results["fid_base"]:.1f}', linewidth=2)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('NEON Weight (w)')
    ax2.set_ylabel('FID Score')
    ax2.set_title('Zoomed: w ∈ [-0.1, 0.15]')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best w vs generation
    ax3 = axes[1, 0]

    gens = list(results['generations'].keys())
    best_ws = [results['generations'][g]['best_w'] for g in gens]

    ax3.bar(range(len(gens)), best_ws, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(gens)))
    ax3.set_xticklabels([f'Gen {g}' for g in gens])
    ax3.set_xlabel('Collapsed Generation')
    ax3.set_ylabel('Optimal NEON Weight (w)')
    ax3.set_title('Optimal w vs Collapse Severity')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: FID improvement over base
    ax4 = axes[1, 1]

    best_fids = [results['generations'][g]['best_fid'] for g in gens]
    improvements = [results['fid_base'] - bf for bf in best_fids]

    colors_bar = ['green' if imp > 0 else 'red' for imp in improvements]
    ax4.bar(range(len(gens)), improvements, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(gens)))
    ax4.set_xticklabels([f'Gen {g}' for g in gens])
    ax4.set_xlabel('Collapsed Generation Used')
    ax4.set_ylabel('FID Improvement over Base')
    ax4.set_title('NEON Improvement (positive = better than base)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('BatchNorm-Only NEON Fine Sweep', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + 'batchnorm_fine_analysis.pdf', bbox_inches='tight')
    plt.savefig(SAVE_DIR + 'batchnorm_fine_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {SAVE_DIR}batchnorm_fine_analysis.pdf")


def print_summary(results, test_generations):
    """Print summary table"""
    print("\n" + "="*80)
    print("SUMMARY: BatchNorm-Only NEON Fine Sweep")
    print("="*80)
    print(f"Base model FID: {results['fid_base']:.2f}")
    print()
    print(f"{'Gen':<6} {'Collapsed FID':<15} {'Best w':<10} {'Best FID':<12} {'Improvement':<12}")
    print("-"*80)

    for gen in test_generations:
        r = results['generations'][gen]
        improvement = results['fid_base'] - r['best_fid']
        sign = "+" if improvement > 0 else ""
        print(f"{gen:<6} {r['fid_collapsed']:<15.2f} {r['best_w']:<10.3f} {r['best_fid']:<12.2f} {sign}{improvement:<11.2f}")

    print("="*80)

    # Find if any generation shows consistent improvement with w > 0
    positive_w_improvements = []
    for gen in test_generations:
        r = results['generations'][gen]
        if r['best_w'] > 0 and results['fid_base'] - r['best_fid'] > 0.1:
            positive_w_improvements.append((gen, r['best_w'], results['fid_base'] - r['best_fid']))

    if positive_w_improvements:
        print("\nGenerations showing improvement with positive w:")
        for gen, w, imp in positive_w_improvements:
            print(f"  Gen {gen}: w={w:.3f}, improvement={imp:.2f}")
    else:
        print("\nNo generation shows consistent improvement with positive w > 0")

    print("="*80)


if __name__ == "__main__":
    main()
