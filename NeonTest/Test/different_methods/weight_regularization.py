#!/usr/bin/env python3
"""
Weight Regularization to Prevent MADness in GAN-MNIST

Instead of post-hoc correction (NEON), this approach constrains the generator
during training to stay close to the original model trained on real data.

Loss = L_GAN + λ * ||θ - θ_base||²

This prevents parameter drift that causes mode collapse during iterative
synthetic training.

We simulate this by:
1. Training gen_0 on real data (already done)
2. Training gen_1+ with weight regularization toward gen_0
3. Comparing FID degradation with and without regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from scipy import linalg
import os
import copy
import pickle
import matplotlib.pyplot as plt
import time

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

# Training hyperparameters (match original experiment)
batch_size = 128
n_epochs = 50
lr = 0.0002
b1, b2 = 0.5, 0.999

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
GAN_DIR = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/'
SAVE_DIR = '/home/patrick/MAD-Detection/NeonTest/Test/different_methods/results_regularization/'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + "samples/", exist_ok=True)
os.makedirs(SAVE_DIR + "checkpoints/", exist_ok=True)


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
        d_in = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        validity = self.model(d_in)
        return validity


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


def compute_weight_reg_loss(model, base_model):
    """Compute L2 regularization loss toward base model weights"""
    reg_loss = 0.0
    for (name, param), (_, base_param) in zip(model.named_parameters(),
                                               base_model.named_parameters()):
        reg_loss += torch.sum((param - base_param.detach()) ** 2)
    return reg_loss


def generate_synthetic_dataset(generator, n_samples=60000):
    """Generate synthetic dataset from generator.

    Mimics the original experiment: save as uint8 [0,255], reload as float [-1,1].
    This quantization is important for matching the original FID progression.
    """
    generator.eval()
    all_imgs = []
    all_labels = []

    batch_sz = 500
    n_batches = n_samples // batch_sz

    with torch.no_grad():
        for _ in range(n_batches):
            z = torch.randn(batch_sz, latent_dim).to(device)
            labels = torch.randint(0, n_classes, (batch_sz,)).to(device)
            imgs = generator(z, labels)

            # Convert to [0, 1] then to uint8 [0, 255] then back to float [-1, 1]
            # This matches the original experiment's data pipeline
            imgs = (imgs + 1) / 2  # [-1, 1] -> [0, 1]
            imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)  # [0, 1] -> [0, 255] uint8
            imgs = imgs.float() / 255.0  # [0, 255] -> [0, 1] float
            imgs = imgs * 2 - 1  # [0, 1] -> [-1, 1]

            all_imgs.append(imgs.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_imgs), torch.cat(all_labels)


def train_gan_with_regularization(train_loader, base_generator, lambda_reg, n_epochs=50):
    """
    Train GAN with weight regularization toward base model.

    Args:
        train_loader: DataLoader with training images
        base_generator: The gen_0 model to regularize toward
        lambda_reg: Regularization strength (0 = no regularization)
        n_epochs: Number of training epochs

    Returns:
        Trained generator
    """
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Initialize from base if lambda > 0 (warm start)
    if lambda_reg > 0:
        generator.load_state_dict(base_generator.state_dict())

    # Loss and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            batch_size_curr = imgs.size(0)

            # Ground truths
            valid = torch.ones(batch_size_curr, 1).to(device)
            fake = torch.zeros(batch_size_curr, 1).to(device)

            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            z = torch.randn(batch_size_curr, latent_dim).to(device)
            gen_labels = torch.randint(0, n_classes, (batch_size_curr,)).to(device)
            gen_imgs = generator(z, gen_labels)

            validity = discriminator(gen_imgs, gen_labels)
            g_loss_adv = adversarial_loss(validity, valid)

            # Weight regularization
            if lambda_reg > 0:
                reg_loss = compute_weight_reg_loss(generator, base_generator)
                g_loss = g_loss_adv + lambda_reg * reg_loss
            else:
                g_loss = g_loss_adv
                reg_loss = torch.tensor(0.0)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} - D: {d_loss.item():.4f}, G_adv: {g_loss_adv.item():.4f}, Reg: {reg_loss.item():.6f}")

    generator.eval()
    return generator


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
    print("Weight Regularization for MADness Prevention in GAN-MNIST")
    print("="*70)
    print(f"\nLoss = L_GAN + λ * ||θ - θ_base||²")
    print(f"This constrains the generator to stay close to the base model.\n")

    # Load LeNet for FID
    lenet_path = os.path.join(GAN_DIR, 'lenet_mnist.pth')
    lenet = LeNet().to(device)
    if os.path.exists(lenet_path):
        lenet.load_state_dict(torch.load(lenet_path, map_location=device))
    else:
        raise FileNotFoundError(f"LeNet not found at {lenet_path}")
    lenet.eval()

    # Get real MNIST features
    print("Loading real MNIST data...")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    mnist_data = MNIST("../../data", train=True, download=True, transform=tf)

    # For FID computation (unnormalized)
    tf_eval = transforms.Compose([transforms.ToTensor()])
    mnist_eval = MNIST("../../data", train=True, download=True, transform=tf_eval)
    eval_loader = DataLoader(mnist_eval, batch_size=256, shuffle=True)

    real_images = []
    for imgs, _ in eval_loader:
        real_images.append(imgs)
        if len(real_images) * 256 >= N_eval:
            break
    real_images = torch.cat(real_images)[:N_eval]
    real_features = extract_features_batch(lenet, real_images)

    # Load base generator (gen_0)
    print("\nLoading base generator (gen_0)...")
    gen_base = load_generator(0)

    # Compute base FID
    base_imgs = generate_samples(gen_base, N_eval)
    base_features = extract_features_batch(lenet, base_imgs)
    fid_base = compute_fid(real_features, base_features)
    print(f"Base model (gen 0) FID: {fid_base:.2f}")

    # Lambda values to test (focused on promising range based on initial results)
    lambda_values = [0.0, 1e-6, 1e-5, 1e-4]

    # Number of synthetic generations to simulate
    n_generations = 4

    results = {
        'fid_base': fid_base,
        'lambda_values': lambda_values,
        'generations': {lam: [] for lam in lambda_values}
    }

    for lambda_reg in lambda_values:
        print(f"\n{'='*60}")
        print(f"Testing λ = {lambda_reg}")
        print(f"{'='*60}")

        # Start with base generator
        current_gen = copy.deepcopy(gen_base)

        fid_history = [fid_base]  # Gen 0 FID

        for gen_idx in range(1, n_generations + 1):
            print(f"\n  Generation {gen_idx}:")

            # Generate synthetic data from current generator
            print(f"    Generating synthetic training data...")
            syn_imgs, syn_labels = generate_synthetic_dataset(current_gen, n_samples=60000)

            # Create dataloader
            # Normalize to [-1, 1] for training (already in this range from generator)
            syn_dataset = TensorDataset(syn_imgs, syn_labels)
            syn_loader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=True)

            # Train new generator on synthetic data with regularization
            print(f"    Training with λ={lambda_reg}...")
            start_time = time.time()
            new_gen = train_gan_with_regularization(
                syn_loader, gen_base, lambda_reg, n_epochs=n_epochs
            )
            train_time = time.time() - start_time

            # Compute FID
            gen_imgs = generate_samples(new_gen, N_eval)
            gen_features = extract_features_batch(lenet, gen_imgs)
            fid = compute_fid(real_features, gen_features)
            fid_history.append(fid)

            print(f"    FID: {fid:.2f} (training time: {train_time:.1f}s)")

            # Save sample images
            save_image(gen_imgs[:100],
                      SAVE_DIR + f"samples/lambda{lambda_reg}_gen{gen_idx}.png", nrow=10)

            # Update current generator for next iteration
            current_gen = new_gen

            # Save checkpoint
            torch.save(new_gen.state_dict(),
                      SAVE_DIR + f"checkpoints/lambda{lambda_reg}_gen{gen_idx}.pth")

        results['generations'][lambda_reg] = fid_history
        print(f"\n  FID progression: {' -> '.join([f'{f:.2f}' for f in fid_history])}")

    # Save results
    with open(SAVE_DIR + 'regularization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {SAVE_DIR}regularization_results.pkl")

    # Plot results
    plot_results(results, lambda_values, n_generations)
    print_summary(results, lambda_values)


def plot_results(results, lambda_values, n_generations):
    """Create visualization plots"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: FID vs Generation for each lambda
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_values)))

    for idx, lam in enumerate(lambda_values):
        fid_history = results['generations'][lam]
        gens = list(range(len(fid_history)))
        label = f'λ={lam}' if lam > 0 else 'No reg (λ=0)'
        ax1.plot(gens, fid_history, 'o-', color=colors[idx],
                label=label, linewidth=2, markersize=6)

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('FID Score')
    ax1.set_title('FID vs Generation for Different Regularization Strengths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(n_generations + 1))

    # Plot 2: Final FID vs Lambda
    ax2 = axes[1]

    final_fids = [results['generations'][lam][-1] for lam in lambda_values]
    x_labels = [f'{lam}' for lam in lambda_values]

    bars = ax2.bar(range(len(lambda_values)), final_fids, color='steelblue', alpha=0.7)
    ax2.axhline(y=results['fid_base'], color='green', linestyle='--',
                label=f'Base (gen 0): {results["fid_base"]:.1f}')

    ax2.set_xticks(range(len(lambda_values)))
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.set_xlabel('Regularization λ')
    ax2.set_ylabel('Final FID (after 5 generations)')
    ax2.set_title('Final FID vs Regularization Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Weight Regularization to Prevent MADness', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + 'regularization_analysis.pdf', bbox_inches='tight')
    plt.savefig(SAVE_DIR + 'regularization_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {SAVE_DIR}regularization_analysis.pdf")


def print_summary(results, lambda_values):
    """Print summary table"""
    print("\n" + "="*80)
    print("SUMMARY: Weight Regularization Results")
    print("="*80)
    print(f"Base model FID: {results['fid_base']:.2f}")
    print()

    # Header
    header = "Gen   " + "   ".join([f"λ={lam}" for lam in lambda_values])
    print(header)
    print("-"*80)

    n_gens = len(results['generations'][lambda_values[0]])
    for gen in range(n_gens):
        row = f"{gen:<6}"
        for lam in lambda_values:
            fid = results['generations'][lam][gen]
            row += f"{fid:>8.2f}"
        print(row)

    print("-"*80)

    # Degradation summary
    print("\nFID Degradation (Gen 5 - Gen 0):")
    for lam in lambda_values:
        fid_0 = results['generations'][lam][0]
        fid_final = results['generations'][lam][-1]
        degradation = fid_final - fid_0
        print(f"  λ={lam}: {degradation:+.2f}")

    print("="*80)


if __name__ == "__main__":
    main()
