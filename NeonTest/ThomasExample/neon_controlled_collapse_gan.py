#!/usr/bin/env python3
"""
NEON with Controlled Spectral Collapse for MNIST GAN

Adapted from Thomas's VAE notebook to test:
"Do synthetic samples with lower effective ranks improve NEON effectiveness?"

Key Idea:
- Instead of generating normal synthetic samples, we CONTROL the collapse level
- alpha=0: Normal generation (full rank Jacobian)
- alpha=1: Full collapse (rank-1 Jacobian, all energy in first singular value)

Process:
1. Train/load base GAN on real MNIST
2. For each alpha (collapse level):
   a. Generate synthetic samples with controlled Jacobian collapse
   b. Fine-tune base model on collapsed samples → θ_s
   c. Apply NEON at various w: θ_NEON = (1+w)*θ_r - w*θ_s
   d. Measure FID
3. Plot: Does higher alpha (more collapse) → better NEON improvement?
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd.functional import jacobian
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pickle
import os
from scipy import linalg

# ============================================================================
# Configuration
# ============================================================================

# GAN Architecture
latent_dim = 100
img_size = 28
channels = 1
n_classes = 10

# Experiment parameters
N_SYNTHETIC = 2000          # Samples to generate with controlled collapse
N_FINETUNE_EPOCHS = 10      # Fine-tuning epochs
LR_FINETUNE = 1e-4          # Fine-tuning learning rate
batch_size = 128

# NEON sweep
W_SWEEPS = np.linspace(-0.5, 1.5, 21)  # w values to test
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Collapse levels

# FID evaluation
N_FID_SAMPLES = 10000

# Seeds for reproducibility
SEEDS = [0, 1, 2]

# Paths
BASE_MODEL_PATH = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/generator_0.pth'
BASE_DISC_PATH = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/discriminator_0.pth'
LENET_PATH = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/lenet_mnist.pth'
SAVE_DIR = '/home/patrick/MAD-Detection/NeonTest/ThomasExample/results_gan/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + 'samples/', exist_ok=True)

# ============================================================================
# Model Definitions
# ============================================================================

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


class Discriminator(nn.Module):
    """Conditional Discriminator for MNIST"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + img_size * img_size * channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        d_input = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        validity = self.model(d_input)
        return validity


class LeNet(nn.Module):
    """LeNet for FID feature extraction"""
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
        return self.fc3(x)

    def extract_features(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


# ============================================================================
# Controlled Collapse Functions
# ============================================================================

def forward_collapse_gan(generator, z, label, alpha):
    """
    Generate image with controlled Jacobian spectral collapse.

    Args:
        generator: GAN generator
        z: Latent vector [latent_dim]
        label: Class label (scalar tensor)
        alpha: Collapse level (0=normal, 1=full collapse to rank-1)

    Returns:
        x_collapsed: Image with collapsed Jacobian
        S_new: New singular values after collapse
    """
    assert 0.0 <= alpha <= 1.0

    if z.dim() == 1:
        z = z.unsqueeze(0)
    if label.dim() == 0:
        label = label.unsqueeze(0)

    with torch.enable_grad():
        z_input = z.detach().requires_grad_(True)

        def gen_func(z_):
            return generator(z_.unsqueeze(0), label).flatten()

        # Normal forward pass
        x = generator(z_input, label).flatten()

        # Compute Jacobian: [784, 100]
        J = jacobian(gen_func, z_input.squeeze(0))

    with torch.no_grad():
        # Affine approximation: x ≈ J @ z + b
        b = x - J @ z_input.squeeze(0)

        # SVD of Jacobian
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)

        # Total energy in spectrum
        energy = torch.sum(S ** 2)

        # Collapse spectrum based on alpha
        # - Shrink all singular values by sqrt(1-alpha)
        # - First singular value gets extra energy
        alpha_t = torch.tensor(alpha, device=S.device, dtype=S.dtype)
        S_new = torch.sqrt(1 - alpha_t) * S.clone()
        S_new[0] = torch.sqrt((1 - alpha_t) * S[0] ** 2 + alpha_t * energy)

        # Reconstruct with collapsed Jacobian
        J_collapsed = U @ torch.diag(S_new) @ Vh

        # Generate collapsed image
        x_collapsed = J_collapsed @ z_input.squeeze(0) + b

        # Reshape to image
        x_collapsed = x_collapsed.view(1, channels, img_size, img_size)

        # Clamp to valid range [-1, 1]
        x_collapsed = torch.clamp(x_collapsed, -1, 1)

    return x_collapsed.squeeze(0), S_new


def generate_collapsed_samples(generator, n_samples, alpha, seed=0):
    """
    Generate batch of samples with controlled Jacobian collapse.

    Args:
        generator: GAN generator
        n_samples: Number of samples to generate
        alpha: Collapse level
        seed: Random seed

    Returns:
        images: [n_samples, 1, 28, 28]
        labels: [n_samples]
        S_list: List of singular value spectra
    """
    torch.manual_seed(seed)
    generator.eval()

    images = []
    labels = []
    S_list = []

    print(f"Generating {n_samples} samples with alpha={alpha:.2f}...")

    for i in tqdm(range(n_samples)):
        z = torch.randn(latent_dim, device=device)
        label = torch.randint(0, n_classes, (1,), device=device)

        x_collapsed, S = forward_collapse_gan(generator, z, label, alpha)

        images.append(x_collapsed)
        labels.append(label)
        S_list.append(S)

    images = torch.stack(images)
    labels = torch.cat(labels)

    return images, labels, S_list


def compute_effective_rank(S_list, eps=1e-12):
    """Compute effective rank from list of singular value spectra."""
    eranks = []
    for S in S_list:
        p = S ** 2
        p = p / (p.sum() + eps)
        erank = torch.exp(-(p * torch.log(p + eps)).sum())
        eranks.append(erank.item())
    return np.mean(eranks), np.std(eranks)


# ============================================================================
# NEON Functions
# ============================================================================

def merge_models(base, other, w):
    """Apply NEON merge: θ_NEON = (1+w)*θ_base - w*θ_other"""
    merged = copy.deepcopy(base)
    with torch.no_grad():
        for pm, pb, po in zip(merged.parameters(),
                              base.parameters(),
                              other.parameters()):
            pm.copy_((1.0 + w) * pb - w * po)
    return merged


def finetune_gan(generator, discriminator, dataloader, n_epochs, lr):
    """Fine-tune GAN on synthetic data."""
    generator.train()
    discriminator.train()

    adversarial_loss = nn.BCELoss()
    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(n_epochs):
        g_losses, d_losses = [], []

        for imgs, labels in dataloader:
            bs = imgs.size(0)
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            imgs = imgs.to(device)
            labels = labels.to(device)

            # Train Generator
            opt_G.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            gen_labels = torch.randint(0, n_classes, (bs,), device=device)
            gen_imgs = generator(z, gen_labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            opt_G.step()

            # Train Discriminator
            opt_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: G_loss={np.mean(g_losses):.4f}, D_loss={np.mean(d_losses):.4f}")

    generator.eval()
    return generator, discriminator


# ============================================================================
# FID Computation
# ============================================================================

def compute_fid(real_features, fake_features):
    """Compute FID between feature distributions."""
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

    return float(fid)


def extract_features(lenet, images):
    """Extract features for FID."""
    lenet.eval()
    features = []

    for i in range(0, len(images), 256):
        batch = images[i:i+256].to(device)
        # Convert from [-1,1] to [0,1] for LeNet
        batch = (batch + 1) / 2
        with torch.no_grad():
            feat = lenet.extract_features(batch)
        features.append(feat.cpu().numpy())

    return np.concatenate(features)


def generate_samples_for_fid(generator, n_samples):
    """Generate normal samples for FID evaluation."""
    generator.eval()
    images = []

    with torch.no_grad():
        for _ in range(n_samples // 500):
            z = torch.randn(500, latent_dim, device=device)
            labels = torch.randint(0, n_classes, (500,), device=device)
            imgs = generator(z, labels)
            images.append(imgs.cpu())

    return torch.cat(images)


def fid_of_model(generator, lenet, real_features, n_samples=N_FID_SAMPLES):
    """Compute FID for a generator."""
    fake_imgs = generate_samples_for_fid(generator, n_samples)
    fake_features = extract_features(lenet, fake_imgs)
    return compute_fid(real_features, fake_features)


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("="*70)
    print("NEON with Controlled Spectral Collapse for MNIST GAN")
    print("="*70)
    print(f"Device: {device}")
    print(f"Alphas: {ALPHAS}")
    print(f"W sweeps: {len(W_SWEEPS)} values from {W_SWEEPS[0]:.2f} to {W_SWEEPS[-1]:.2f}")
    print(f"Seeds: {SEEDS}")

    # Load base model
    print("\nLoading base GAN model...")
    base_generator = Generator().to(device)
    base_generator.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    base_generator.eval()

    base_discriminator = Discriminator().to(device)
    if os.path.exists(BASE_DISC_PATH):
        base_discriminator.load_state_dict(torch.load(BASE_DISC_PATH, map_location=device))

    # Load LeNet for FID
    print("Loading LeNet for FID...")
    lenet = LeNet().to(device)
    if os.path.exists(LENET_PATH):
        lenet.load_state_dict(torch.load(LENET_PATH, map_location=device))
    else:
        print("Training LeNet...")
        # Train LeNet if not available
        opt = optim.Adam(lenet.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        tf = transforms.Compose([transforms.ToTensor()])
        train_data = MNIST("./data", train=True, download=True, transform=tf)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

        for epoch in range(5):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                loss = criterion(lenet(imgs), labels)
                loss.backward()
                opt.step()

        torch.save(lenet.state_dict(), LENET_PATH)
    lenet.eval()

    # Get real MNIST features
    print("Extracting real MNIST features...")
    tf = transforms.Compose([transforms.ToTensor()])
    mnist = MNIST("./data", train=True, download=True, transform=tf)
    real_loader = DataLoader(mnist, batch_size=256, shuffle=True)

    real_imgs = []
    for imgs, _ in real_loader:
        real_imgs.append(imgs * 2 - 1)  # Convert to [-1, 1]
        if len(real_imgs) * 256 >= N_FID_SAMPLES:
            break
    real_imgs = torch.cat(real_imgs)[:N_FID_SAMPLES]
    real_features = extract_features(lenet, real_imgs)

    # Base model FID
    base_fid = fid_of_model(base_generator, lenet, real_features)
    print(f"Base model FID: {base_fid:.2f}")

    # ========================================================================
    # Main experiment loop
    # ========================================================================
    all_results = []
    spectrum_results = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        for alpha in ALPHAS:
            print(f"\n--- Alpha = {alpha:.2f} ---")

            # Generate collapsed samples
            collapsed_imgs, collapsed_labels, S_list = generate_collapsed_samples(
                base_generator, N_SYNTHETIC, alpha, seed=seed
            )

            # Compute effective rank of collapsed samples
            erank_mean, erank_std = compute_effective_rank(S_list)
            print(f"  Effective rank: {erank_mean:.2f} ± {erank_std:.2f}")

            # Save sample images
            if seed == 0:
                save_image(
                    (collapsed_imgs[:100] + 1) / 2,
                    f"{SAVE_DIR}samples/collapsed_alpha{alpha:.2f}.png",
                    nrow=10
                )

            # Store spectrum results
            if alpha not in spectrum_results:
                spectrum_results[alpha] = {'S_list': [], 'erank': []}
            spectrum_results[alpha]['S_list'].extend([s.cpu().numpy() for s in S_list[:100]])
            spectrum_results[alpha]['erank'].append(erank_mean)

            # Fine-tune on collapsed samples
            print(f"  Fine-tuning on collapsed samples...")
            aux_generator = copy.deepcopy(base_generator)
            aux_discriminator = copy.deepcopy(base_discriminator)

            collapsed_dataset = TensorDataset(collapsed_imgs, collapsed_labels)
            collapsed_loader = DataLoader(collapsed_dataset, batch_size=batch_size,
                                         shuffle=True, drop_last=True)

            aux_generator, aux_discriminator = finetune_gan(
                aux_generator, aux_discriminator, collapsed_loader,
                N_FINETUNE_EPOCHS, LR_FINETUNE
            )

            # Sweep NEON w values
            print(f"  Sweeping NEON w values...")
            for w in tqdm(W_SWEEPS, desc=f"  NEON sweep"):
                merged = merge_models(base_generator, aux_generator, w)
                fid = fid_of_model(merged, lenet, real_features)

                all_results.append({
                    'seed': seed,
                    'alpha': alpha,
                    'w': w,
                    'fid': fid,
                    'erank': erank_mean
                })

    # ========================================================================
    # Save and plot results
    # ========================================================================

    # Save raw results
    df = pd.DataFrame(all_results)
    df.to_csv(f"{SAVE_DIR}neon_results.csv", index=False)

    with open(f"{SAVE_DIR}spectrum_results.pkl", 'wb') as f:
        pickle.dump(spectrum_results, f)

    # Aggregate statistics
    df_stats = (
        df.groupby(["alpha", "w"])
        .agg(
            fid_mean=("fid", "mean"),
            fid_std=("fid", "std"),
        )
        .reset_index()
    )

    # Plot FID vs w for each alpha
    plot_fid_results(df_stats, base_fid)

    # Plot effective rank vs alpha
    plot_erank_results(spectrum_results)

    # Plot singular value spectra
    plot_spectrum_results(spectrum_results)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to {SAVE_DIR}")


def plot_fid_results(df_stats, base_fid):
    """Plot FID vs NEON w for each alpha."""
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(8, 6))

    alphas = sorted(df_stats["alpha"].unique())
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(alphas)))

    for i, alpha in enumerate(alphas):
        df_alpha = df_stats[df_stats["alpha"] == alpha].sort_values("w")

        linestyle = "--" if alpha == 0.0 else "-"
        linewidth = 1.5 if alpha == 0.0 else 2.0
        label = r"$\alpha=0$ (normal)" if alpha == 0.0 else rf"$\alpha={alpha:.2f}$"
        color = "black" if alpha == 0.0 else colors[i]

        ax.plot(df_alpha["w"], df_alpha["fid_mean"],
                linestyle=linestyle, linewidth=linewidth,
                color=color, label=label, marker='o', markersize=3)

        ax.fill_between(
            df_alpha["w"],
            df_alpha["fid_mean"] - df_alpha["fid_std"],
            df_alpha["fid_mean"] + df_alpha["fid_std"],
            color=color, alpha=0.2
        )

    ax.axhline(y=base_fid, color='green', linestyle=':', label=f'Base model: {base_fid:.1f}')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel(r"NEON $w$", fontsize=14)
    ax.set_ylabel("FID", fontsize=14)
    ax.set_title("NEON with Controlled Spectral Collapse (MNIST GAN)", fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}neon_fid_vs_w.png", dpi=150)
    plt.savefig(f"{SAVE_DIR}neon_fid_vs_w.pdf")
    plt.close()

    print(f"Saved FID plot to {SAVE_DIR}neon_fid_vs_w.png")


def plot_erank_results(spectrum_results):
    """Plot effective rank vs alpha."""
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(6, 4))

    alphas = sorted(spectrum_results.keys())
    erank_means = [np.mean(spectrum_results[a]['erank']) for a in alphas]
    erank_stds = [np.std(spectrum_results[a]['erank']) for a in alphas]

    ax.errorbar(alphas, erank_means, yerr=erank_stds,
                marker='o', capsize=5, linewidth=2, markersize=8)

    ax.set_xlabel(r"Collapse Level $\alpha$", fontsize=14)
    ax.set_ylabel("Effective Rank", fontsize=14)
    ax.set_title("Effective Rank vs Collapse Level", fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}effective_rank_vs_alpha.png", dpi=150)
    plt.savefig(f"{SAVE_DIR}effective_rank_vs_alpha.pdf")
    plt.close()

    print(f"Saved effective rank plot to {SAVE_DIR}effective_rank_vs_alpha.png")


def plot_spectrum_results(spectrum_results):
    """Plot singular value spectra for each alpha."""
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(8, 6))

    alphas = sorted(spectrum_results.keys())
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(alphas)))

    for i, alpha in enumerate(alphas):
        S_arrays = spectrum_results[alpha]['S_list']
        S_mean = np.mean(S_arrays, axis=0)
        S_std = np.std(S_arrays, axis=0)

        idx = np.arange(len(S_mean))
        color = "black" if alpha == 0.0 else colors[i]
        label = r"$\alpha=0$" if alpha == 0.0 else rf"$\alpha={alpha:.2f}$"

        ax.semilogy(idx, S_mean, color=color, label=label, linewidth=2)
        ax.fill_between(idx, S_mean - S_std, S_mean + S_std, color=color, alpha=0.2)

    ax.set_xlabel("Singular Value Index", fontsize=14)
    ax.set_ylabel("Singular Value (log scale)", fontsize=14)
    ax.set_title("Jacobian Singular Value Spectrum", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}singular_values_vs_alpha.png", dpi=150)
    plt.savefig(f"{SAVE_DIR}singular_values_vs_alpha.pdf")
    plt.close()

    print(f"Saved spectrum plot to {SAVE_DIR}singular_values_vs_alpha.png")


if __name__ == "__main__":
    main()
