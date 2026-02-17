#!/usr/bin/env python3
"""
Protocol A: Faithful NEON Implementation for MNIST GAN

This implements the exact NEON methodology from the paper:

1. Load base model θ_r (trained on real data)
2. Generate synthetic dataset S from θ_r (6k samples)
3. Fine-tune θ_r briefly on S → θ_s (intentional degradation)
4. Measure Jacobian effective rank before/after fine-tuning
5. Apply NEON: θ_NEON = (1+w)*θ_r - w*θ_s
6. Evaluate FID at different w values

Key insight: Fine-tuning is NOT to improve the model.
It's to measure the degradation direction, which NEON then reverses.

The Jacobian analysis shows HOW the model's geometry changes during
the degradation step, connecting to your MADness research.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.functional as F_auto
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import os
import pickle
import copy
from scipy import linalg
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

# GAN Architecture (must match original training)
latent_dim = 100
img_size = 28
channels = 1
n_classes = 10

# Fine-tuning parameters (matching Neon paper)
N_SYNTHETIC = 6000          # |S| = 6k in Neon paper
N_FINETUNE_EPOCHS = 6       # Short fine-tuning (Neon uses ~1% of training budget)
FINETUNE_LR = 1e-4          # Lower LR for fine-tuning
batch_size = 128
b1, b2 = 0.5, 0.999

# Jacobian analysis
N_JACOBIAN_SAMPLES = 20     # Samples for Jacobian effective rank
JACOBIAN_CLASS = 7          # Fixed class for consistency

# NEON evaluation
W_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_FID_SAMPLES = 10000       # Samples for FID evaluation

# Paths
BASE_MODEL_PATH = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/generator_0.pth'
BASE_DISC_PATH = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/discriminator_0.pth'
SAVE_DIR = '/home/patrick/MAD-Detection/FineTuning/protocol_a_results/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + 'samples/', exist_ok=True)
os.makedirs(SAVE_DIR + 'checkpoints/', exist_ok=True)

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
# Utility Functions
# ============================================================================

def generate_synthetic_data(generator, n_samples):
    """Generate synthetic data from generator"""
    generator.eval()
    all_imgs, all_labels = [], []

    with torch.no_grad():
        for _ in range(n_samples // 500):
            z = torch.randn(500, latent_dim).to(device)
            labels = torch.randint(0, n_classes, (500,)).to(device)
            imgs = generator(z, labels)
            all_imgs.append(imgs.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_imgs), torch.cat(all_labels)


def generate_samples_for_fid(generator, n_samples):
    """Generate samples for FID (returns [0,1] range)"""
    generator.eval()
    all_imgs = []

    with torch.no_grad():
        for _ in range(n_samples // 500):
            z = torch.randn(500, latent_dim).to(device)
            labels = torch.randint(0, n_classes, (500,)).to(device)
            imgs = generator(z, labels)
            imgs = (imgs + 1) / 2  # [-1,1] -> [0,1]
            all_imgs.append(imgs.cpu())

    return torch.cat(all_imgs)


def compute_jacobian_metrics(generator, n_samples=20, class_label=7):
    """Compute Jacobian effective rank averaged over multiple samples"""
    generator.eval()

    effective_ranks = []
    max_eigenvalues = []

    for i in range(n_samples):
        torch.manual_seed(100 + i)
        z = torch.randn(1, latent_dim).to(device).requires_grad_(True)
        c = torch.tensor([class_label], device=device)

        z_flat = z.flatten()

        def gen_func(z_input):
            z_shaped = z_input.reshape(1, latent_dim)
            return generator(z_shaped, c).flatten()

        try:
            jacobian = F_auto.jacobian(gen_func, z_flat)
            U, S, V = torch.linalg.svd(jacobian)
            eigenvalues = S.pow(2).cpu().numpy()

            # Effective rank (entropy-based)
            eigs_norm = eigenvalues / eigenvalues.sum()
            eff_rank = np.exp(-np.sum(eigs_norm * np.log(eigs_norm + 1e-10)))

            effective_ranks.append(eff_rank)
            max_eigenvalues.append(eigenvalues[0])

        except Exception as e:
            print(f"  Warning: Jacobian computation failed for sample {i}: {e}")
            continue

    generator.train()

    return {
        'effective_rank_mean': np.mean(effective_ranks),
        'effective_rank_std': np.std(effective_ranks),
        'max_eigenvalue_mean': np.mean(max_eigenvalues),
        'max_eigenvalue_std': np.std(max_eigenvalues),
        'n_samples': len(effective_ranks)
    }


def compute_fid(real_features, fake_features):
    """Compute FID between feature distributions"""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Regularization for stability
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


def extract_features(model, images):
    """Extract features for FID"""
    model.eval()
    features = []

    for i in range(0, len(images), 256):
        batch = images[i:i+256].to(device)
        with torch.no_grad():
            feat = model.extract_features(batch)
        features.append(feat.cpu().numpy())

    return np.concatenate(features)


def apply_neon(base_state, finetuned_state, w):
    """Apply NEON: θ_NEON = (1+w)*θ_r - w*θ_s"""
    neon_state = {}
    for key in base_state:
        neon_state[key] = (1 + w) * base_state[key] - w * finetuned_state[key]
    return neon_state


def finetune_one_epoch(generator, discriminator, dataloader, opt_G, opt_D):
    """Fine-tune for one epoch"""
    loss_fn = nn.BCELoss()
    g_losses, d_losses = [], []

    for imgs, labels in dataloader:
        bs = imgs.size(0)
        valid = torch.ones(bs, 1).to(device)
        fake = torch.zeros(bs, 1).to(device)

        imgs = imgs.to(device)
        labels = labels.to(device)

        # Generator
        opt_G.zero_grad()
        z = torch.randn(bs, latent_dim).to(device)
        gen_labels = torch.randint(0, n_classes, (bs,)).to(device)
        gen_imgs = generator(z, gen_labels)
        g_loss = loss_fn(discriminator(gen_imgs, gen_labels), valid)
        g_loss.backward()
        opt_G.step()

        # Discriminator
        opt_D.zero_grad()
        real_loss = loss_fn(discriminator(imgs, labels), valid)
        fake_loss = loss_fn(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    return np.mean(g_losses), np.mean(d_losses)

# ============================================================================
# Main Protocol A Experiment
# ============================================================================

def main():
    print("="*70)
    print("PROTOCOL A: Faithful NEON Implementation")
    print("="*70)

    results = {
        'config': {
            'n_synthetic': N_SYNTHETIC,
            'n_finetune_epochs': N_FINETUNE_EPOCHS,
            'finetune_lr': FINETUNE_LR,
            'w_values': W_VALUES
        }
    }

    # ========================================================================
    # Step 1: Load base model θ_r
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Load base model θ_r (trained on real MNIST)")
    print("="*70)

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))

    discriminator = Discriminator().to(device)
    if os.path.exists(BASE_DISC_PATH):
        discriminator.load_state_dict(torch.load(BASE_DISC_PATH, map_location=device))

    # Save base state
    theta_r = copy.deepcopy(generator.state_dict())
    torch.save(theta_r, os.path.join(SAVE_DIR, 'checkpoints', 'theta_r.pth'))

    print(f"Loaded base model from {BASE_MODEL_PATH}")

    # ========================================================================
    # Step 2: Generate synthetic dataset S
    # ========================================================================
    print("\n" + "="*70)
    print(f"STEP 2: Generate synthetic dataset S ({N_SYNTHETIC} samples)")
    print("="*70)

    syn_imgs, syn_labels = generate_synthetic_data(generator, N_SYNTHETIC)
    print(f"Generated synthetic data: {syn_imgs.shape}")

    # Save sample visualization
    save_image((syn_imgs[:100] + 1) / 2,
               os.path.join(SAVE_DIR, 'samples', 'synthetic_training_data.png'), nrow=10)

    syn_dataset = TensorDataset(syn_imgs, syn_labels)
    syn_loader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ========================================================================
    # Step 3: Measure Jacobian effective rank of θ_r (before fine-tuning)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Compute Jacobian metrics for θ_r (before fine-tuning)")
    print("="*70)

    print(f"Computing Jacobian over {N_JACOBIAN_SAMPLES} samples (class {JACOBIAN_CLASS})...")
    jacobian_before = compute_jacobian_metrics(generator, N_JACOBIAN_SAMPLES, JACOBIAN_CLASS)

    print(f"  Effective Rank: {jacobian_before['effective_rank_mean']:.2f} ± {jacobian_before['effective_rank_std']:.2f}")
    print(f"  Max Eigenvalue: {jacobian_before['max_eigenvalue_mean']:.4f} ± {jacobian_before['max_eigenvalue_std']:.4f}")

    results['jacobian_before'] = jacobian_before

    # ========================================================================
    # Step 4: Fine-tune θ_r on S → θ_s (intentional degradation)
    # ========================================================================
    print("\n" + "="*70)
    print(f"STEP 4: Fine-tune θ_r on synthetic data ({N_FINETUNE_EPOCHS} epochs)")
    print("="*70)
    print("NOTE: This is intentional degradation to measure collapse direction")

    opt_G = optim.Adam(generator.parameters(), lr=FINETUNE_LR, betas=(b1, b2))
    opt_D = optim.Adam(discriminator.parameters(), lr=FINETUNE_LR, betas=(b1, b2))

    finetune_log = {'epochs': [], 'g_loss': [], 'd_loss': [], 'jacobian': []}

    for epoch in range(1, N_FINETUNE_EPOCHS + 1):
        g_loss, d_loss = finetune_one_epoch(
            generator, discriminator, syn_loader, opt_G, opt_D
        )

        finetune_log['epochs'].append(epoch)
        finetune_log['g_loss'].append(g_loss)
        finetune_log['d_loss'].append(d_loss)

        # Compute Jacobian at each epoch
        jac_metrics = compute_jacobian_metrics(generator, N_JACOBIAN_SAMPLES, JACOBIAN_CLASS)
        finetune_log['jacobian'].append(jac_metrics)

        print(f"Epoch {epoch}/{N_FINETUNE_EPOCHS}: G_loss={g_loss:.4f}, D_loss={d_loss:.4f}, "
              f"EffRank={jac_metrics['effective_rank_mean']:.2f}")

        # Save checkpoint
        torch.save(generator.state_dict(),
                   os.path.join(SAVE_DIR, 'checkpoints', f'theta_s_epoch{epoch}.pth'))

    # Final fine-tuned state
    theta_s = copy.deepcopy(generator.state_dict())
    torch.save(theta_s, os.path.join(SAVE_DIR, 'checkpoints', 'theta_s_final.pth'))

    results['finetune_log'] = finetune_log

    # ========================================================================
    # Step 5: Measure Jacobian effective rank of θ_s (after fine-tuning)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Compute Jacobian metrics for θ_s (after fine-tuning)")
    print("="*70)

    jacobian_after = compute_jacobian_metrics(generator, N_JACOBIAN_SAMPLES, JACOBIAN_CLASS)

    print(f"  Effective Rank: {jacobian_after['effective_rank_mean']:.2f} ± {jacobian_after['effective_rank_std']:.2f}")
    print(f"  Max Eigenvalue: {jacobian_after['max_eigenvalue_mean']:.4f} ± {jacobian_after['max_eigenvalue_std']:.4f}")

    rank_change = jacobian_after['effective_rank_mean'] - jacobian_before['effective_rank_mean']
    print(f"\n  Effective Rank Change: {rank_change:+.2f} "
          f"({'DECREASED' if rank_change < 0 else 'INCREASED'} - {'expected' if rank_change < 0 else 'unexpected'})")

    results['jacobian_after'] = jacobian_after

    # ========================================================================
    # Step 6: Load/Train LeNet for FID
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Prepare FID evaluation")
    print("="*70)

    lenet_path = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/lenet_mnist.pth'
    lenet = LeNet().to(device)

    if os.path.exists(lenet_path):
        lenet.load_state_dict(torch.load(lenet_path, map_location=device))
        print(f"Loaded LeNet from {lenet_path}")
    else:
        print("Training LeNet for FID...")
        opt = optim.Adam(lenet.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        tf = transforms.Compose([transforms.ToTensor()])
        train_data = MNIST("../data", train=True, download=True, transform=tf)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

        for epoch in range(5):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                loss = criterion(lenet(imgs), labels)
                loss.backward()
                opt.step()

        torch.save(lenet.state_dict(), lenet_path)
        print(f"Saved LeNet to {lenet_path}")

    lenet.eval()

    # Get real MNIST features
    print("Extracting real MNIST features...")
    tf = transforms.Compose([transforms.ToTensor()])
    mnist = MNIST("../data", train=True, download=True, transform=tf)
    real_loader = DataLoader(mnist, batch_size=256, shuffle=True)

    real_imgs = []
    for imgs, _ in real_loader:
        real_imgs.append(imgs)
        if len(real_imgs) * 256 >= N_FID_SAMPLES:
            break
    real_imgs = torch.cat(real_imgs)[:N_FID_SAMPLES]
    real_features = extract_features(lenet, real_imgs)

    # ========================================================================
    # Step 7: Compute FID for θ_r and θ_s
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: Compute baseline FIDs")
    print("="*70)

    # FID for base model θ_r
    generator.load_state_dict(theta_r)
    base_imgs = generate_samples_for_fid(generator, N_FID_SAMPLES)
    base_features = extract_features(lenet, base_imgs)
    fid_base = compute_fid(real_features, base_features)
    print(f"FID(θ_r): {fid_base:.2f}")

    save_image(base_imgs[:100], os.path.join(SAVE_DIR, 'samples', 'theta_r_samples.png'), nrow=10)

    # FID for fine-tuned model θ_s
    generator.load_state_dict(theta_s)
    ft_imgs = generate_samples_for_fid(generator, N_FID_SAMPLES)
    ft_features = extract_features(lenet, ft_imgs)
    fid_finetuned = compute_fid(real_features, ft_features)
    print(f"FID(θ_s): {fid_finetuned:.2f}")

    save_image(ft_imgs[:100], os.path.join(SAVE_DIR, 'samples', 'theta_s_samples.png'), nrow=10)

    fid_degradation = fid_finetuned - fid_base
    print(f"\nFID Degradation: {fid_degradation:+.2f} "
          f"({'WORSE' if fid_degradation > 0 else 'BETTER'} - {'expected' if fid_degradation > 0 else 'unexpected'})")

    results['fid_base'] = fid_base
    results['fid_finetuned'] = fid_finetuned

    # ========================================================================
    # Step 8: Apply NEON at different w values
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: Apply NEON extrapolation")
    print("="*70)
    print("θ_NEON = (1+w)*θ_r - w*θ_s")
    print()

    neon_results = {'w': [], 'fid': [], 'effective_rank': []}

    for w in W_VALUES:
        # Apply NEON
        theta_neon = apply_neon(theta_r, theta_s, w)
        generator.load_state_dict(theta_neon)

        # Compute FID
        neon_imgs = generate_samples_for_fid(generator, N_FID_SAMPLES)
        neon_features = extract_features(lenet, neon_imgs)
        fid = compute_fid(real_features, neon_features)

        # Compute Jacobian
        jac = compute_jacobian_metrics(generator, N_JACOBIAN_SAMPLES, JACOBIAN_CLASS)

        neon_results['w'].append(w)
        neon_results['fid'].append(fid)
        neon_results['effective_rank'].append(jac['effective_rank_mean'])

        print(f"w={w:.2f}: FID={fid:.2f}, EffRank={jac['effective_rank_mean']:.2f}")

        # Save samples for key w values
        if w in [0.0, 1.0, 2.0]:
            save_image(neon_imgs[:100],
                      os.path.join(SAVE_DIR, 'samples', f'theta_neon_w{w:.1f}.png'), nrow=10)

    results['neon'] = neon_results

    # ========================================================================
    # Step 9: Find optimal w and report results
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 9: Results Summary")
    print("="*70)

    best_idx = np.argmin(neon_results['fid'])
    best_w = neon_results['w'][best_idx]
    best_fid = neon_results['fid'][best_idx]

    print(f"\nBaseline FID (θ_r):        {fid_base:.2f}")
    print(f"Fine-tuned FID (θ_s):      {fid_finetuned:.2f}")
    print(f"Best NEON FID (w={best_w:.2f}):  {best_fid:.2f}")

    print(f"\nImprovement over θ_r:  {fid_base - best_fid:+.2f}")
    print(f"Improvement over θ_s:  {fid_finetuned - best_fid:+.2f}")

    print(f"\nJacobian Effective Rank:")
    print(f"  θ_r (before):  {jacobian_before['effective_rank_mean']:.2f}")
    print(f"  θ_s (after):   {jacobian_after['effective_rank_mean']:.2f}")
    print(f"  Change:        {rank_change:+.2f}")

    results['best_w'] = best_w
    results['best_fid'] = best_fid

    # ========================================================================
    # Step 10: Save results and create plots
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 10: Save results and plots")
    print("="*70)

    with open(os.path.join(SAVE_DIR, 'protocol_a_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {SAVE_DIR}protocol_a_results.pkl")

    # Create plots
    create_plots(results)

    print("\n" + "="*70)
    print("PROTOCOL A COMPLETE")
    print("="*70)


def create_plots(results):
    """Create visualization plots"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: FID vs w
    ax1 = axes[0, 0]
    ax1.plot(results['neon']['w'], results['neon']['fid'], 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=results['fid_base'], color='green', linestyle='--',
                label=f'θ_r (base): {results["fid_base"]:.1f}')
    ax1.axhline(y=results['fid_finetuned'], color='red', linestyle='--',
                label=f'θ_s (fine-tuned): {results["fid_finetuned"]:.1f}')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    best_idx = np.argmin(results['neon']['fid'])
    ax1.scatter([results['neon']['w'][best_idx]], [results['neon']['fid'][best_idx]],
                color='gold', s=200, zorder=5, marker='*', label=f'Best: w={results["best_w"]:.2f}')

    ax1.set_xlabel('NEON Weight (w)', fontsize=12)
    ax1.set_ylabel('FID (lower is better)', fontsize=12)
    ax1.set_title('NEON: FID vs Extrapolation Weight', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Effective Rank vs w
    ax2 = axes[0, 1]
    ax2.plot(results['neon']['w'], results['neon']['effective_rank'], 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=results['jacobian_before']['effective_rank_mean'], color='green', linestyle='--',
                label=f'θ_r: {results["jacobian_before"]["effective_rank_mean"]:.1f}')
    ax2.axhline(y=results['jacobian_after']['effective_rank_mean'], color='red', linestyle='--',
                label=f'θ_s: {results["jacobian_after"]["effective_rank_mean"]:.1f}')

    ax2.set_xlabel('NEON Weight (w)', fontsize=12)
    ax2.set_ylabel('Effective Rank', fontsize=12)
    ax2.set_title('NEON: Effective Rank vs Extrapolation Weight', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effective Rank during fine-tuning
    ax3 = axes[1, 0]
    epochs = [0] + results['finetune_log']['epochs']
    eff_ranks = [results['jacobian_before']['effective_rank_mean']]
    eff_ranks += [j['effective_rank_mean'] for j in results['finetune_log']['jacobian']]

    ax3.plot(epochs, eff_ranks, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Fine-tuning Epoch', fontsize=12)
    ax3.set_ylabel('Effective Rank', fontsize=12)
    ax3.set_title('Jacobian Effective Rank During Fine-tuning', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: FID vs Effective Rank (scatter)
    ax4 = axes[1, 1]
    ax4.scatter(results['neon']['effective_rank'], results['neon']['fid'],
                c=results['neon']['w'], cmap='viridis', s=100)

    for i, w in enumerate(results['neon']['w']):
        ax4.annotate(f'w={w:.1f}',
                    (results['neon']['effective_rank'][i], results['neon']['fid'][i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax4.set_xlabel('Effective Rank', fontsize=12)
    ax4.set_ylabel('FID', fontsize=12)
    ax4.set_title('FID vs Effective Rank at Different w', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Protocol A: Faithful NEON Implementation on MNIST GAN', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'protocol_a_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {SAVE_DIR}protocol_a_analysis.png")


if __name__ == "__main__":
    main()
