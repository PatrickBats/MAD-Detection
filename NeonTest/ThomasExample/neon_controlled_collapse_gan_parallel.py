#!/usr/bin/env python3
"""
NEON with Controlled Spectral Collapse for MNIST GAN - PARALLEL VERSION

Runs different (alpha, seed) combinations in parallel across 8 GPUs.
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
import multiprocessing as mp
from itertools import product

# ============================================================================
# Configuration
# ============================================================================

# GAN Architecture
latent_dim = 100
img_size = 28
channels = 1
n_classes = 10

# Experiment parameters
N_SYNTHETIC = 10000         # Samples to generate with controlled collapse (match data ratio from notebook)
N_FINETUNE_EPOCHS = 1000    # Fine-tuning epochs (match notebook)
LR_FINETUNE = 0.0002        # Fine-tuning learning rate (match original GAN training)
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
SAVE_DIR = '/home/patrick/MAD-Detection/NeonTest/ThomasExample/results_gan_parallel_reconstruction/'

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

def forward_collapse_gan(generator, z, label, alpha, device):
    """Generate image with controlled Jacobian spectral collapse."""
    assert 0.0 <= alpha <= 1.0

    if z.dim() == 1:
        z = z.unsqueeze(0)
    if label.dim() == 0:
        label = label.unsqueeze(0)

    with torch.enable_grad():
        z_input = z.detach().requires_grad_(True)

        def gen_func(z_):
            return generator(z_.unsqueeze(0), label).flatten()

        x = generator(z_input, label).flatten()
        J = jacobian(gen_func, z_input.squeeze(0))

    with torch.no_grad():
        b = x - J @ z_input.squeeze(0)
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        energy = torch.sum(S ** 2)

        alpha_t = torch.tensor(alpha, device=S.device, dtype=S.dtype)
        S_new = torch.sqrt(1 - alpha_t) * S.clone()
        S_new[0] = torch.sqrt((1 - alpha_t) * S[0] ** 2 + alpha_t * energy)

        J_collapsed = U @ torch.diag(S_new) @ Vh
        x_collapsed = J_collapsed @ z_input.squeeze(0) + b
        x_collapsed = x_collapsed.view(1, channels, img_size, img_size)
        x_collapsed = torch.clamp(x_collapsed, -1, 1)

    return x_collapsed.squeeze(0), S_new


def generate_collapsed_samples(generator, n_samples, alpha, seed, device):
    """Generate batch of samples with controlled Jacobian collapse.

    Returns images, labels, S_list, AND the latent vectors z used for generation.
    """
    torch.manual_seed(seed)
    generator.eval()

    images = []
    labels = []
    latents = []
    S_list = []

    for i in tqdm(range(n_samples), desc=f"Generating α={alpha:.2f}", leave=False):
        z = torch.randn(latent_dim, device=device)
        label = torch.randint(0, n_classes, (1,), device=device)
        x_collapsed, S = forward_collapse_gan(generator, z, label, alpha, device)
        images.append(x_collapsed)
        labels.append(label)
        latents.append(z)
        S_list.append(S)

    images = torch.stack(images)
    labels = torch.cat(labels)
    latents = torch.stack(latents)

    return images, labels, latents, S_list


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


def finetune_gan_reconstruction(generator, dataloader, n_epochs, lr, device):
    """Fine-tune generator using reconstruction loss (matches VAE notebook approach).

    Trains generator to reconstruct collapsed samples from their original latent vectors.
    This creates a clear "synthetic direction" for NEON extrapolation.
    """
    generator.train()

    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    recon_loss = nn.MSELoss()

    for epoch in range(n_epochs):
        epoch_losses = []
        for imgs, labels, zs in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            zs = zs.to(device)

            optimizer.zero_grad()

            # Generate from stored latents
            gen_imgs = generator(zs, labels)

            # Reconstruction loss: make generator reproduce collapsed samples
            loss = recon_loss(gen_imgs, imgs)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Print progress occasionally
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: Recon Loss={np.mean(epoch_losses):.6f}")

    generator.eval()
    return generator


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


def extract_features(lenet, images, device):
    """Extract features for FID."""
    lenet.eval()
    features = []

    for i in range(0, len(images), 256):
        batch = images[i:i+256].to(device)
        batch = (batch + 1) / 2  # [-1,1] to [0,1]
        with torch.no_grad():
            feat = lenet.extract_features(batch)
        features.append(feat.cpu().numpy())

    return np.concatenate(features)


def generate_samples_for_fid(generator, n_samples, device):
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


def fid_of_model(generator, lenet, real_features, device, n_samples=N_FID_SAMPLES):
    """Compute FID for a generator."""
    fake_imgs = generate_samples_for_fid(generator, n_samples, device)
    fake_features = extract_features(lenet, fake_imgs, device)
    return compute_fid(real_features, fake_features)


# ============================================================================
# Worker Function (runs on one GPU)
# ============================================================================

def run_experiment(args):
    """Run a single (alpha, seed) experiment on a specific GPU."""
    alpha, seed, gpu_id, real_features, base_fid = args

    device = f'cuda:{gpu_id}'
    print(f"[GPU {gpu_id}] Starting α={alpha:.2f}, seed={seed}")

    # Load generator on this GPU (no discriminator needed for reconstruction training)
    base_generator = Generator().to(device)
    base_generator.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device, weights_only=True))
    base_generator.eval()

    lenet = LeNet().to(device)
    lenet.load_state_dict(torch.load(LENET_PATH, map_location=device, weights_only=True))
    lenet.eval()

    results = []
    spectrum_data = {'S_list': [], 'erank': []}

    # Generate collapsed samples (now returns latents too!)
    collapsed_imgs, collapsed_labels, collapsed_latents, S_list = generate_collapsed_samples(
        base_generator, N_SYNTHETIC, alpha, seed=seed, device=device
    )

    # Compute effective rank
    erank_mean, erank_std = compute_effective_rank(S_list)
    print(f"[GPU {gpu_id}] α={alpha:.2f}, seed={seed} - Effective rank: {erank_mean:.2f} ± {erank_std:.2f}")

    # Save sample images (only for seed 0)
    if seed == 0:
        save_image(
            (collapsed_imgs[:100] + 1) / 2,
            f"{SAVE_DIR}samples/collapsed_alpha{alpha:.2f}.png",
            nrow=10
        )

    # Store spectrum results
    spectrum_data['S_list'] = [s.cpu().numpy() for s in S_list[:100]]
    spectrum_data['erank'] = erank_mean

    # Fine-tune on collapsed samples using reconstruction loss
    print(f"[GPU {gpu_id}] α={alpha:.2f}, seed={seed} - Fine-tuning with reconstruction loss for {N_FINETUNE_EPOCHS} epochs...")
    aux_generator = copy.deepcopy(base_generator)

    # Create dataset with (image, label, latent) tuples
    collapsed_dataset = TensorDataset(collapsed_imgs, collapsed_labels, collapsed_latents)
    collapsed_loader = DataLoader(collapsed_dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=True)

    aux_generator = finetune_gan_reconstruction(
        aux_generator, collapsed_loader,
        N_FINETUNE_EPOCHS, LR_FINETUNE, device
    )

    # Sweep NEON w values
    print(f"[GPU {gpu_id}] α={alpha:.2f}, seed={seed} - Sweeping NEON w values...")
    for w in W_SWEEPS:
        merged = merge_models(base_generator, aux_generator, w)
        fid = fid_of_model(merged, lenet, real_features, device)

        results.append({
            'seed': seed,
            'alpha': alpha,
            'w': w,
            'fid': fid,
            'erank': erank_mean
        })

    print(f"[GPU {gpu_id}] Completed α={alpha:.2f}, seed={seed}")

    return results, (alpha, spectrum_data)


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("NEON with Controlled Spectral Collapse - PARALLEL (8 GPUs)")
    print("="*70)
    print(f"Alphas: {ALPHAS}")
    print(f"Seeds: {SEEDS}")
    print(f"Fine-tuning epochs: {N_FINETUNE_EPOCHS}")
    print(f"W sweeps: {len(W_SWEEPS)} values")

    # Pre-compute real features (do this once on GPU 0)
    device = 'cuda:0'

    print("\nLoading LeNet and computing real MNIST features...")
    lenet = LeNet().to(device)
    lenet.load_state_dict(torch.load(LENET_PATH, map_location=device, weights_only=True))
    lenet.eval()

    tf = transforms.Compose([transforms.ToTensor()])
    mnist = MNIST("./data", train=True, download=True, transform=tf)
    real_loader = DataLoader(mnist, batch_size=256, shuffle=True)

    real_imgs = []
    for imgs, _ in real_loader:
        real_imgs.append(imgs * 2 - 1)  # [0,1] to [-1,1]
        if len(real_imgs) * 256 >= N_FID_SAMPLES:
            break
    real_imgs = torch.cat(real_imgs)[:N_FID_SAMPLES]
    real_features = extract_features(lenet, real_imgs, device)

    # Compute base FID
    base_generator = Generator().to(device)
    base_generator.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device, weights_only=True))
    base_generator.eval()
    base_fid = fid_of_model(base_generator, lenet, real_features, device)
    print(f"Base model FID: {base_fid:.2f}")

    # Clean up GPU 0
    del lenet, base_generator
    torch.cuda.empty_cache()

    # Create all (alpha, seed) combinations
    all_combinations = list(product(ALPHAS, SEEDS))
    n_jobs = len(all_combinations)
    n_gpus = 8

    print(f"\nTotal jobs: {n_jobs} ({len(ALPHAS)} alphas × {len(SEEDS)} seeds)")
    print(f"Running on {n_gpus} GPUs in parallel\n")

    # Assign GPU IDs round-robin
    job_args = []
    for i, (alpha, seed) in enumerate(all_combinations):
        gpu_id = i % n_gpus
        job_args.append((alpha, seed, gpu_id, real_features, base_fid))

    # Run in parallel using multiprocessing
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=n_gpus) as pool:
        results_list = pool.map(run_experiment, job_args)

    # Aggregate results
    all_results = []
    spectrum_results = {}

    for results, (alpha, spectrum_data) in results_list:
        all_results.extend(results)
        if alpha not in spectrum_results:
            spectrum_results[alpha] = {'S_list': [], 'erank': []}
        spectrum_results[alpha]['S_list'].extend(spectrum_data['S_list'])
        spectrum_results[alpha]['erank'].append(spectrum_data['erank'])

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(f"{SAVE_DIR}neon_results.csv", index=False)

    with open(f"{SAVE_DIR}spectrum_results.pkl", 'wb') as f:
        pickle.dump(spectrum_results, f)

    # Aggregate and plot
    df_stats = (
        df.groupby(["alpha", "w"])
        .agg(
            fid_mean=("fid", "mean"),
            fid_std=("fid", "std"),
        )
        .reset_index()
    )

    plot_fid_results(df_stats, base_fid)
    plot_erank_results(spectrum_results)
    plot_spectrum_results(spectrum_results)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to {SAVE_DIR}")


def plot_fid_results(df_stats, base_fid):
    """Plot FID vs NEON w for each alpha."""
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150, 'font.family': 'serif'})
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
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}neon_fid_vs_w.png", dpi=150)
    plt.savefig(f"{SAVE_DIR}neon_fid_vs_w.pdf")
    plt.close()
    print(f"Saved FID plot to {SAVE_DIR}neon_fid_vs_w.pdf")


def plot_erank_results(spectrum_results):
    """Plot effective rank vs alpha."""
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(6, 4))

    alphas = sorted(spectrum_results.keys())
    erank_means = [np.mean(spectrum_results[a]['erank']) for a in alphas]
    erank_stds = [np.std(spectrum_results[a]['erank']) for a in alphas]

    ax.errorbar(alphas, erank_means, yerr=erank_stds,
                marker='o', capsize=5, linewidth=2, markersize=8)

    ax.set_xlabel(r"Collapse Level $\alpha$", fontsize=14)
    ax.set_ylabel("Effective Rank", fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}effective_rank_vs_alpha.png", dpi=150)
    plt.savefig(f"{SAVE_DIR}effective_rank_vs_alpha.pdf")
    plt.close()
    print(f"Saved effective rank plot to {SAVE_DIR}effective_rank_vs_alpha.pdf")


def plot_spectrum_results(spectrum_results):
    """Plot singular value spectra for each alpha."""
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 6))

    alphas = sorted(spectrum_results.keys())
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(alphas)))

    for i, alpha in enumerate(alphas):
        S_arrays = spectrum_results[alpha]['S_list']
        if len(S_arrays) == 0:
            continue
        S_mean = np.mean(S_arrays, axis=0)
        S_std = np.std(S_arrays, axis=0)

        idx = np.arange(len(S_mean))
        color = "black" if alpha == 0.0 else colors[i]
        label = r"$\alpha=0$" if alpha == 0.0 else rf"$\alpha={alpha:.2f}$"

        ax.semilogy(idx, S_mean, color=color, label=label, linewidth=2)
        ax.fill_between(idx, S_mean - S_std, S_mean + S_std, color=color, alpha=0.2)

    ax.set_xlabel("Singular Value Index", fontsize=14)
    ax.set_ylabel("Singular Value (log scale)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}singular_values_vs_alpha.png", dpi=150)
    plt.savefig(f"{SAVE_DIR}singular_values_vs_alpha.pdf")
    plt.close()
    print(f"Saved spectrum plot to {SAVE_DIR}singular_values_vs_alpha.pdf")


if __name__ == "__main__":
    main()
