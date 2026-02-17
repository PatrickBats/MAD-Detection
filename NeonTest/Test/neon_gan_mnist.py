#!/usr/bin/env python3
"""
NEON (Negative Extrapolation) for GAN-MNIST

NEON applies negative extrapolation to model weights:
    θ_neon = (1 + w) * θ_base - w * θ_collapsed

Where:
- θ_base: Base model trained on real data (generator_0)
- θ_collapsed: Collapsed model from iterative synthetic training (generator_t)
- w: Extrapolation weight
    - w = 0: Use base model
    - w > 0: Extrapolate away from collapsed model
    - w = 1: Standard extrapolation (2*θ_base - θ_collapsed)

This script tests whether NEON can improve GAN quality after MADness degradation.
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
N_eval = 10000  # Samples for FID evaluation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
GAN_DIR = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/'
SAVE_DIR = '/home/patrick/MAD-Detection/NeonTest/Test/results/'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + "samples/", exist_ok=True)


class Generator(nn.Module):
    """Conditional Generator for MNIST (must match training architecture)"""
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
    """LeNet for feature extraction (FID computation)"""
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
        """Extract 84-dim features before final layer"""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


def load_generator(generation):
    """Load generator checkpoint for a specific generation"""
    checkpoint_path = os.path.join(GAN_DIR, f'generator_{generation}.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.eval()

    return generator


def merge_models_neon(base_model: nn.Module, collapsed_model: nn.Module, w: float) -> nn.Module:
    """
    Apply NEON negative extrapolation to model weights.

    Formula: θ_neon = (1 + w) * θ_base - w * θ_collapsed

    Only merges learnable parameters (weights, biases), NOT BatchNorm running stats.
    BatchNorm running_mean and running_var are copied from the base model.

    Args:
        base_model: Model trained on real data
        collapsed_model: Model that has degraded from synthetic training
        w: Extrapolation weight (w=0 gives base, w=1 gives 2*base - collapsed)

    Returns:
        New model with NEON-extrapolated weights
    """
    merged = copy.deepcopy(base_model)

    with torch.no_grad():
        # Merge only learnable parameters
        for pm, pb, pc in zip(merged.parameters(), base_model.parameters(), collapsed_model.parameters()):
            pm.copy_((1.0 + w) * pb - w * pc)

    return merged


def generate_samples(generator, N):
    """Generate N samples from the generator"""
    generator.eval()
    all_imgs = []

    batch_size = 500
    n_batches = N // batch_size

    with torch.no_grad():
        for _ in range(n_batches):
            z = torch.randn(batch_size, latent_dim).to(device)
            labels = torch.randint(0, n_classes, (batch_size,)).to(device)
            imgs = generator(z, labels)
            imgs = (imgs + 1) / 2  # Convert from [-1, 1] to [0, 1]
            all_imgs.append(imgs.cpu())

    return torch.cat(all_imgs, dim=0)


def extract_features_batch(model, images, batch_size=256):
    """Extract features from images using LeNet"""
    model.eval()
    features = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = model.extract_features(batch)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_fid(real_features, fake_features):
    """Compute FID between real and fake feature distributions"""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Add small regularization to handle singular matrices
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_fake += eps * np.eye(sigma_fake.shape[0])

    diff = mu_real - mu_fake

    try:
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    except Exception as e:
        print(f"    Warning: FID computation failed ({e}), returning inf")
        fid = float('inf')

    return fid


def main():
    print("="*70)
    print("NEON (Negative Extrapolation) for GAN-MNIST")
    print("="*70)
    print(f"\nFormula: θ_neon = (1 + w) * θ_base - w * θ_collapsed")
    print(f"  w = 0: Use base model")
    print(f"  w > 0: Extrapolate away from collapsed model")
    print(f"  w = 1: Standard extrapolation (2*θ_base - θ_collapsed)")
    print()

    # Check for existing checkpoints
    print("Checking for GAN checkpoints...")
    available_gens = []
    for gen in range(8):
        path = os.path.join(GAN_DIR, f'generator_{gen}.pth')
        if os.path.exists(path):
            available_gens.append(gen)
            print(f"  Found: generator_{gen}.pth")
        else:
            print(f"  Missing: generator_{gen}.pth")

    if 0 not in available_gens:
        raise FileNotFoundError("generator_0.pth (base model) is required!")

    collapsed_gens = [g for g in available_gens if g > 0]
    print(f"\nBase model: generation 0")
    print(f"Collapsed models to test: generations {collapsed_gens}")

    # Load or use existing LeNet for FID
    lenet_path = os.path.join(GAN_DIR, 'lenet_mnist.pth')
    lenet = LeNet().to(device)
    if os.path.exists(lenet_path):
        print(f"\nLoading LeNet from {lenet_path}")
        lenet.load_state_dict(torch.load(lenet_path, map_location=device))
    else:
        print("\nTraining LeNet for FID computation...")
        import torch.optim as optim
        optimizer = optim.Adam(lenet.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        tf = transforms.Compose([transforms.ToTensor()])
        train_data = MNIST("../../data", train=True, download=True, transform=tf)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

        for epoch in range(5):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = lenet(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"  Epoch {epoch+1}/5 done")

        torch.save(lenet.state_dict(), lenet_path)
        print(f"LeNet saved to {lenet_path}")

    lenet.eval()

    # Get real MNIST features
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

    # Load base model (generation 0)
    print("\nLoading base model (generation 0)...")
    gen_base = load_generator(0)

    # Compute base model FID
    print("Computing base model FID...")
    base_imgs = generate_samples(gen_base, N_eval)
    base_features = extract_features_batch(lenet, base_imgs)
    fid_base = compute_fid(real_features, base_features)
    print(f"Base model (gen 0) FID: {fid_base:.2f}")

    save_image(base_imgs[:100], SAVE_DIR + "samples/gen0_base.png", nrow=10)

    # W values to test - using much smaller values since GANs are more sensitive
    w_values = [-0.1, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]

    # Results storage
    results = {
        'fid_base': fid_base,
        'w_values': w_values,
        'generations': {}
    }

    # Test each collapsed generation
    for collapsed_gen in collapsed_gens:
        print(f"\n{'='*60}")
        print(f"Testing NEON with collapsed model from generation {collapsed_gen}")
        print(f"{'='*60}")

        # Load collapsed model
        gen_collapsed = load_generator(collapsed_gen)

        # Compute collapsed model FID
        collapsed_imgs = generate_samples(gen_collapsed, N_eval)
        collapsed_features = extract_features_batch(lenet, collapsed_imgs)
        fid_collapsed = compute_fid(real_features, collapsed_features)
        print(f"Collapsed model (gen {collapsed_gen}) FID: {fid_collapsed:.2f}")

        save_image(collapsed_imgs[:100],
                   SAVE_DIR + f"samples/gen{collapsed_gen}_collapsed.png", nrow=10)

        gen_results = {
            'fid_collapsed': fid_collapsed,
            'fid_at_w': {},
            'best_w': None,
            'best_fid': None
        }

        best_fid = float('inf')
        best_w = None

        for w in w_values:
            # Apply NEON extrapolation
            gen_neon = merge_models_neon(gen_base, gen_collapsed, w)

            # Generate and evaluate
            neon_imgs = generate_samples(gen_neon, N_eval)

            # Check for NaN/Inf (GAN instability)
            if torch.isnan(neon_imgs).any() or torch.isinf(neon_imgs).any():
                print(f"  w={w:+.2f}: UNSTABLE (NaN/Inf in output)")
                gen_results['fid_at_w'][w] = float('inf')
                continue

            neon_features = extract_features_batch(lenet, neon_imgs)
            fid = compute_fid(real_features, neon_features)

            gen_results['fid_at_w'][w] = fid
            print(f"  w={w:+.2f}: FID={fid:.2f}")

            if fid < best_fid:
                best_fid = fid
                best_w = w

            # Save samples at key w values
            if w in [0.0, 0.05, 0.1, 0.2]:
                save_image(neon_imgs[:100],
                          SAVE_DIR + f"samples/gen{collapsed_gen}_neon_w{w}.png", nrow=10)

        gen_results['best_w'] = best_w
        gen_results['best_fid'] = best_fid
        results['generations'][collapsed_gen] = gen_results

        print(f"\n  Best for gen {collapsed_gen}: w={best_w:.2f}, FID={best_fid:.2f}")
        print(f"  Improvement over base: {fid_base - best_fid:.2f}")
        print(f"  Improvement over collapsed: {fid_collapsed - best_fid:.2f}")

    # Save results
    with open(SAVE_DIR + 'neon_gan_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {SAVE_DIR}neon_gan_results.pkl")

    # Create plots
    plot_results(results, collapsed_gens)

    # Print summary
    print_summary(results, collapsed_gens, fid_base)


def plot_results(results, collapsed_gens):
    """Create visualization plots"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: FID vs w for each collapsed generation
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(collapsed_gens)))

    for idx, gen in enumerate(collapsed_gens):
        w_vals = list(results['generations'][gen]['fid_at_w'].keys())
        fid_vals = list(results['generations'][gen]['fid_at_w'].values())
        # Filter out inf values for plotting
        valid = [(w, f) for w, f in zip(w_vals, fid_vals) if f != float('inf')]
        if valid:
            w_plot, fid_plot = zip(*valid)
            ax1.plot(w_plot, fid_plot, 'o-', color=colors[idx],
                    label=f'Gen {gen}', linewidth=2, markersize=6)

    ax1.axhline(y=results['fid_base'], color='green', linestyle='--',
                label=f'Base (gen 0): {results["fid_base"]:.1f}')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('NEON Weight (w)')
    ax1.set_ylabel('FID Score')
    ax1.set_title('FID vs NEON Weight for Different Collapse Severities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best FID improvement vs generation (collapse severity)
    ax2 = axes[0, 1]

    gens = list(results['generations'].keys())
    best_fids = [results['generations'][g]['best_fid'] for g in gens]
    fid_collapsed = [results['generations'][g]['fid_collapsed'] for g in gens]
    improvements = [results['fid_base'] - bf for bf in best_fids]

    ax2.bar(range(len(gens)), improvements, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(gens)))
    ax2.set_xticklabels([f'Gen {g}' for g in gens])
    ax2.set_xlabel('Collapsed Generation Used')
    ax2.set_ylabel('FID Improvement over Base')
    ax2.set_title('NEON Improvement vs Collapse Severity')
    ax2.axhline(y=0, color='red', linestyle='--', label='No improvement')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Best w value vs generation
    ax3 = axes[1, 0]

    best_ws = [results['generations'][g]['best_w'] for g in gens]
    ax3.bar(range(len(gens)), best_ws, color='coral', alpha=0.7)
    ax3.set_xticks(range(len(gens)))
    ax3.set_xticklabels([f'Gen {g}' for g in gens])
    ax3.set_xlabel('Collapsed Generation Used')
    ax3.set_ylabel('Optimal NEON Weight (w)')
    ax3.set_title('Optimal w vs Collapse Severity')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='w=1 (standard extrapolation)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: FID comparison (base vs collapsed vs best NEON)
    ax4 = axes[1, 1]

    x = np.arange(len(gens))
    width = 0.25

    ax4.bar(x - width, [results['fid_base']] * len(gens), width,
            label='Base (gen 0)', color='green', alpha=0.7)
    ax4.bar(x, fid_collapsed, width,
            label='Collapsed', color='red', alpha=0.7)
    ax4.bar(x + width, best_fids, width,
            label='Best NEON', color='blue', alpha=0.7)

    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Gen {g}' for g in gens])
    ax4.set_xlabel('Collapsed Generation')
    ax4.set_ylabel('FID Score')
    ax4.set_title('FID Comparison: Base vs Collapsed vs NEON')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('NEON Analysis on MNIST GAN', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + 'neon_gan_analysis.pdf', bbox_inches='tight')
    plt.savefig(SAVE_DIR + 'neon_gan_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {SAVE_DIR}neon_gan_analysis.pdf")


def print_summary(results, collapsed_gens, fid_base):
    """Print summary table"""

    print("\n" + "="*80)
    print("SUMMARY: NEON for GAN-MNIST")
    print("="*80)
    print(f"\nBase model (gen 0) FID: {fid_base:.2f}")
    print("\n" + "-"*80)
    print(f"{'Gen':<6} {'FID Collapsed':<15} {'Best NEON FID':<15} {'Best w':<10} {'Improvement':<12}")
    print("-"*80)

    for gen in collapsed_gens:
        r = results['generations'][gen]
        improvement = fid_base - r['best_fid']
        print(f"{gen:<6} {r['fid_collapsed']:<15.2f} {r['best_fid']:<15.2f} {r['best_w']:<10.2f} {improvement:<12.2f}")

    print("="*80)

    # Find which generation works best
    best_gen = min(collapsed_gens,
                   key=lambda g: results['generations'][g]['best_fid'])
    best_overall = results['generations'][best_gen]['best_fid']
    best_overall_w = results['generations'][best_gen]['best_w']

    print(f"\nBest overall: Use gen {best_gen} with w={best_overall_w:.2f}")
    print(f"  -> FID: {best_overall:.2f} (improvement of {fid_base - best_overall:.2f} over base)")

    print("="*80)


if __name__ == "__main__":
    main()
