#!/usr/bin/env python3
"""
Selective Layer NEON for GAN-MNIST

Instead of merging ALL parameters, we selectively merge only certain layers.
The hypothesis is that embedding and output layers are most sensitive to
extrapolation, while middle layers (feature transformation) may benefit.

Layers in Generator:
- label_emb: Embedding(10, 10) - maps class labels to embeddings
- model.0: Linear(110, 128) - first layer (no BatchNorm)
- model.2: Linear(128, 256) + BatchNorm + LeakyReLU
- model.5: Linear(256, 512) + BatchNorm + LeakyReLU
- model.8: Linear(512, 1024) + BatchNorm + LeakyReLU
- model.11: Linear(1024, 784) - output layer
- model.12: Tanh

We'll test different merge strategies:
1. Merge only middle layers (256->512, 512->1024)
2. Merge all except embedding
3. Merge all except output layer
4. Merge only BatchNorm parameters
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

# Hyperparameters (must match GAN training)
latent_dim = 100
img_size = 28
channels = 1
n_classes = 10
N_eval = 10000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
GAN_DIR = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/'
SAVE_DIR = '/home/patrick/MAD-Detection/NeonTest/Test/different_methods/results_selective/'

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


# Define merge strategies
MERGE_STRATEGIES = {
    'all': {
        'description': 'Merge all parameters (original NEON)',
        'include': lambda name: True
    },
    'no_embedding': {
        'description': 'Merge all except embedding layer',
        'include': lambda name: 'label_emb' not in name
    },
    'no_output': {
        'description': 'Merge all except output layer (model.11)',
        'include': lambda name: 'model.11' not in name
    },
    'no_emb_no_output': {
        'description': 'Merge all except embedding and output',
        'include': lambda name: 'label_emb' not in name and 'model.11' not in name
    },
    'middle_only': {
        'description': 'Merge only middle layers (model.2 to model.8)',
        'include': lambda name: any(x in name for x in ['model.2', 'model.3', 'model.4',
                                                         'model.5', 'model.6', 'model.7',
                                                         'model.8', 'model.9', 'model.10'])
    },
    'linear_only': {
        'description': 'Merge only Linear layer weights (no BatchNorm)',
        'include': lambda name: 'weight' in name and 'bn' not in name.lower() and 'num_batches' not in name
    },
    'batchnorm_only': {
        'description': 'Merge only BatchNorm parameters',
        'include': lambda name: any(x in name for x in ['model.3', 'model.6', 'model.9'])
    },
}


def merge_models_selective(base_model: nn.Module, collapsed_model: nn.Module,
                           w: float, strategy: str) -> nn.Module:
    """
    Apply NEON with selective layer merging.

    Only merges parameters that match the strategy's include function.
    """
    merged = copy.deepcopy(base_model)
    include_fn = MERGE_STRATEGIES[strategy]['include']

    with torch.no_grad():
        base_dict = dict(base_model.named_parameters())
        collapsed_dict = dict(collapsed_model.named_parameters())

        merged_count = 0
        skipped_count = 0

        for name, param in merged.named_parameters():
            if include_fn(name):
                # Apply NEON: (1+w)*base - w*collapsed
                param.copy_((1.0 + w) * base_dict[name] - w * collapsed_dict[name])
                merged_count += 1
            else:
                # Keep base model's parameters
                skipped_count += 1

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
    print("Selective Layer NEON for GAN-MNIST")
    print("="*70)

    # Print strategies
    print("\nMerge Strategies:")
    for name, info in MERGE_STRATEGIES.items():
        print(f"  {name}: {info['description']}")
    print()

    # Load LeNet
    lenet_path = os.path.join(GAN_DIR, 'lenet_mnist.pth')
    lenet = LeNet().to(device)
    if os.path.exists(lenet_path):
        lenet.load_state_dict(torch.load(lenet_path, map_location=device))
    else:
        raise FileNotFoundError(f"LeNet not found at {lenet_path}")
    lenet.eval()

    # Get real features
    print("Extracting real MNIST features...")
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

    # Load models
    print("\nLoading models...")
    gen_base = load_generator(0)

    # Compute base FID
    base_imgs = generate_samples(gen_base, N_eval)
    base_features = extract_features_batch(lenet, base_imgs)
    fid_base = compute_fid(real_features, base_features)
    print(f"Base model (gen 0) FID: {fid_base:.2f}")

    # Test with generation 5 (mid-collapse) and generation 7 (severe collapse)
    test_generations = [3, 5, 7]
    w_values = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    results = {
        'fid_base': fid_base,
        'strategies': {},
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

        for strategy_name in MERGE_STRATEGIES.keys():
            print(f"\n  Strategy: {strategy_name}")

            if strategy_name not in results['strategies']:
                results['strategies'][strategy_name] = {}

            if collapsed_gen not in results['strategies'][strategy_name]:
                results['strategies'][strategy_name][collapsed_gen] = {
                    'fid_collapsed': fid_collapsed,
                    'fid_at_w': {}
                }

            best_fid = float('inf')
            best_w = None

            for w in w_values:
                gen_neon = merge_models_selective(gen_base, gen_collapsed, w, strategy_name)
                neon_imgs = generate_samples(gen_neon, N_eval)

                if torch.isnan(neon_imgs).any() or torch.isinf(neon_imgs).any():
                    print(f"    w={w:.3f}: UNSTABLE")
                    results['strategies'][strategy_name][collapsed_gen]['fid_at_w'][w] = float('inf')
                    continue

                neon_features = extract_features_batch(lenet, neon_imgs)
                fid = compute_fid(real_features, neon_features)
                results['strategies'][strategy_name][collapsed_gen]['fid_at_w'][w] = fid

                if fid < best_fid:
                    best_fid = fid
                    best_w = w

                print(f"    w={w:.3f}: FID={fid:.2f}")

            results['strategies'][strategy_name][collapsed_gen]['best_w'] = best_w
            results['strategies'][strategy_name][collapsed_gen]['best_fid'] = best_fid

            improvement = fid_base - best_fid
            print(f"    Best: w={best_w}, FID={best_fid:.2f}, Improvement: {improvement:.2f}")

    # Save results
    with open(SAVE_DIR + 'selective_neon_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {SAVE_DIR}selective_neon_results.pkl")

    # Create summary plot
    plot_results(results, test_generations, w_values)
    print_summary(results, test_generations)


def plot_results(results, test_generations, w_values):
    """Create comparison plots"""
    strategies = list(MERGE_STRATEGIES.keys())
    n_strategies = len(strategies)

    fig, axes = plt.subplots(1, len(test_generations), figsize=(5*len(test_generations), 5))
    if len(test_generations) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_strategies))

    for ax_idx, gen in enumerate(test_generations):
        ax = axes[ax_idx]

        for s_idx, strategy in enumerate(strategies):
            fid_at_w = results['strategies'][strategy][gen]['fid_at_w']
            ws = sorted(fid_at_w.keys())
            fids = [fid_at_w[w] for w in ws]

            # Filter out inf
            valid = [(w, f) for w, f in zip(ws, fids) if f != float('inf')]
            if valid:
                w_plot, fid_plot = zip(*valid)
                ax.plot(w_plot, fid_plot, 'o-', color=colors[s_idx],
                       label=strategy, linewidth=1.5, markersize=4)

        ax.axhline(y=results['fid_base'], color='green', linestyle='--',
                   label=f'Base: {results["fid_base"]:.1f}')
        ax.set_xlabel('NEON Weight (w)')
        ax.set_ylabel('FID Score')
        ax.set_title(f'Generation {gen}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Selective Layer NEON: FID vs Weight for Different Strategies', y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + 'selective_neon_comparison.pdf', bbox_inches='tight')
    plt.savefig(SAVE_DIR + 'selective_neon_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {SAVE_DIR}selective_neon_comparison.pdf")


def print_summary(results, test_generations):
    """Print summary table"""
    print("\n" + "="*90)
    print("SUMMARY: Selective Layer NEON")
    print("="*90)
    print(f"Base model FID: {results['fid_base']:.2f}")
    print()

    for gen in test_generations:
        print(f"\nGeneration {gen}:")
        print("-"*70)
        print(f"{'Strategy':<20} {'Best w':<10} {'Best FID':<12} {'Improvement':<12}")
        print("-"*70)

        for strategy in MERGE_STRATEGIES.keys():
            data = results['strategies'][strategy][gen]
            improvement = results['fid_base'] - data['best_fid']
            print(f"{strategy:<20} {data['best_w']:<10.3f} {data['best_fid']:<12.2f} {improvement:<12.2f}")

    print("="*90)


if __name__ == "__main__":
    main()
