"""
GAN-MNIST Constant Proportion Synthetic Loop Experiment

Implements a synthetic augmentation loop where:
- Total training samples stays CONSTANT (60k)
- Each generation uses p% synthetic + (1-p)% real
- Synthetic data comes from the PREVIOUS generation only

This allows testing different synthetic proportions:
- p=0.0: 100% real (baseline, no synthetic)
- p=0.25: 25% synthetic, 75% real
- p=0.5: 50% synthetic, 50% real (paper-style)
- p=0.75: 75% synthetic, 25% real
- p=0.9: 90% synthetic, 10% real
- p=1.0: 100% synthetic (full synthetic loop)

Key differences from other approaches:
- Training set size is CONSTANT (unlike accumulating which grows)
- Real:synthetic ratio is FIXED across generations
- Only uses synthetic from previous gen (not accumulated)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import os
import argparse

# Hyperparameters
latent_dim = 100
img_size = 28
channels = 1
n_epochs = 50
batch_size = 128
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_classes = 10
N_TOTAL = 60000  # Total training samples (constant)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_gpus = torch.cuda.device_count()
print(f"Using device: {device}, GPUs available: {n_gpus}")


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


def load_real_mnist(data_path="../data"):
    """Load real MNIST dataset"""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    dataset = MNIST(data_path, train=True, download=True, transform=tf)
    return dataset


def load_constant_proportion_dataset(generation, save_dir, real_dataset, synthetic_proportion):
    """
    Load dataset with constant proportion of synthetic data.

    Args:
        generation: Current generation number (0 = train on real only)
        save_dir: Directory containing synthetic data
        real_dataset: Real MNIST dataset
        synthetic_proportion: Proportion of synthetic data (0.0 to 1.0)

    Returns:
        DataLoader with mixed real/synthetic data
    """
    n_synthetic = int(N_TOTAL * synthetic_proportion)
    n_real = N_TOTAL - n_synthetic

    # Get real data tensors
    real_X = real_dataset.data.float()
    real_X = real_X.unsqueeze(1) / 255.0  # Add channel dim, normalize to [0,1]
    real_X = real_X * 2 - 1  # Convert to [-1, 1]
    real_C = real_dataset.targets

    if generation == 0 or synthetic_proportion == 0:
        # First generation or no synthetic: train on real data only
        # Subsample if needed
        indices = torch.randperm(len(real_X))[:N_TOTAL]
        X = real_X[indices]
        C = real_C[indices]
        print(f"Generation {generation}: Training on {N_TOTAL} REAL samples (0% synthetic)")
    else:
        # Load synthetic data from previous generation
        prev_gen = generation - 1
        syn_path = save_dir + f"gen_data_{prev_gen}.pt"
        syn_labels_path = save_dir + f"gen_labels_{prev_gen}.pt"

        if not os.path.exists(syn_path):
            raise FileNotFoundError(f"Synthetic data not found: {syn_path}")

        syn_X = torch.load(syn_path).float()
        if syn_X.max() > 1:
            syn_X = syn_X / 255.0
        syn_X = syn_X * 2 - 1  # Convert to [-1, 1]
        syn_C = torch.load(syn_labels_path).long()

        # Sample n_synthetic from synthetic data
        syn_indices = torch.randperm(len(syn_X))[:n_synthetic]
        syn_X = syn_X[syn_indices]
        syn_C = syn_C[syn_indices]

        # Sample n_real from real data
        real_indices = torch.randperm(len(real_X))[:n_real]
        sampled_real_X = real_X[real_indices]
        sampled_real_C = real_C[real_indices]

        # Combine
        X = torch.cat([sampled_real_X, syn_X], dim=0)
        C = torch.cat([sampled_real_C, syn_C], dim=0)

        print(f"Generation {generation}: Training on {n_real} real + {n_synthetic} synthetic = {len(X)} total")
        print(f"  Synthetic proportion: {100*synthetic_proportion:.0f}%")

    # Shuffle
    shuffle_idx = torch.randperm(len(X))
    X = X[shuffle_idx]
    C = C[shuffle_idx]

    dataset = TensorDataset(X, C)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader


def generate_samples(generator, N, device):
    """Generate N samples from the generator"""
    generator.eval()
    all_imgs = []
    all_labels = []

    gen_batch_size = 500
    n_batches = N // gen_batch_size

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Generating samples"):
            z = torch.randn(gen_batch_size, latent_dim).to(device)
            labels = torch.randint(0, n_classes, (gen_batch_size,)).to(device)

            imgs = generator(z, labels)
            imgs = (imgs + 1) / 2  # Convert to [0, 1]

            all_imgs.append(imgs.cpu())
            all_labels.append(labels.cpu())

    generator.train()
    return torch.cat(all_imgs, dim=0), torch.cat(all_labels, dim=0)


def train_gan(dataloader, generation, device, save_dir):
    """Train GAN for one generation"""

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    if n_gpus > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        g_losses = []
        d_losses = []

        pbar = tqdm(dataloader, desc=f"Gen {generation} Epoch {epoch+1}/{n_epochs}")
        for imgs, labels in pbar:
            batch_size_curr = imgs.size(0)

            valid = torch.ones(batch_size_curr, 1).to(device)
            fake = torch.zeros(batch_size_curr, 1).to(device)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size_curr, latent_dim).to(device)
            gen_labels = torch.randint(0, n_classes, (batch_size_curr,)).to(device)
            gen_imgs = generator(z, gen_labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            pbar.set_postfix({
                'D_loss': f'{np.mean(d_losses):.4f}',
                'G_loss': f'{np.mean(g_losses):.4f}'
            })

        # Save sample images
        if (epoch + 1) % 10 == 0:
            z = torch.randn(100, latent_dim).to(device)
            sample_labels = torch.arange(10).repeat(10).to(device)
            gen_imgs = generator(z, sample_labels)
            gen_imgs = (gen_imgs + 1) / 2
            save_image(gen_imgs.data, save_dir + f"samples/gen{generation}_epoch{epoch+1}.png",
                      nrow=10, normalize=False)

    # Save model
    gen_state = generator.module.state_dict() if n_gpus > 1 else generator.state_dict()
    disc_state = discriminator.module.state_dict() if n_gpus > 1 else discriminator.state_dict()
    torch.save(gen_state, save_dir + f"generator_{generation}.pth")
    torch.save(disc_state, save_dir + f"discriminator_{generation}.pth")

    if n_gpus > 1:
        return generator.module
    return generator


def main(args):
    save_dir = f'./data/gan_outputs_p{int(args.proportion*100)}/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "samples/", exist_ok=True)

    print("="*60)
    print("GAN-MNIST Constant Proportion Synthetic Loop")
    print("="*60)
    print(f"Synthetic proportion: {args.proportion*100:.0f}%")
    print(f"Real proportion: {(1-args.proportion)*100:.0f}%")
    print(f"Total samples per generation: {N_TOTAL}")
    print(f"Number of generations: {args.generations}")
    print(f"Save directory: {save_dir}")
    print()
    print("Data structure:")
    print(f"  Gen 0: 100% REAL ({N_TOTAL} samples)")
    n_syn = int(N_TOTAL * args.proportion)
    n_real = N_TOTAL - n_syn
    print(f"  Gen 1+: {n_real} real + {n_syn} synthetic = {N_TOTAL} total")
    print("="*60)
    print()

    # Load real MNIST
    real_dataset = load_real_mnist(args.data_path)
    print(f"Loaded real MNIST: {len(real_dataset)} samples")

    for generation in range(args.generations):
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")

        # Load dataset with constant proportion
        dataloader = load_constant_proportion_dataset(
            generation, save_dir, real_dataset, args.proportion
        )

        # Train GAN
        generator = train_gan(dataloader, generation, device, save_dir)

        # Generate samples for next generation
        print(f"\nGenerating {N_TOTAL} samples for next generation...")
        X_gen, C_gen = generate_samples(generator, N_TOTAL, device)

        # Save as uint8 [0, 255]
        torch.save((255 * X_gen).byte(), save_dir + f"gen_data_{generation}.pt")
        torch.save(C_gen, save_dir + f"gen_labels_{generation}.pt")

        # Save sample grid
        save_image(X_gen[:100], save_dir + f"samples/generation_{generation}_final.png",
                  nrow=10, normalize=False)

        print(f"Generation {generation} complete!")

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAN-MNIST Constant Proportion Synthetic Loop')
    parser.add_argument('--proportion', type=float, default=0.5,
                        help='Proportion of synthetic data (0.0 to 1.0, default: 0.5)')
    parser.add_argument('--generations', type=int, default=9,
                        help='Number of generations to train (default: 9, giving Gen 0-8)')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='Path to MNIST data')

    args = parser.parse_args()

    if not 0.0 <= args.proportion <= 1.0:
        raise ValueError("Proportion must be between 0.0 and 1.0")

    main(args)
