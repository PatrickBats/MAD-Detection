#!/usr/bin/env python3
"""
Progressive Decay Synthetic Loop Experiment

Tests MADness with progressively decreasing real data proportion:
- Gen 0: 100% real (60k real, 0 synthetic)
- Gen 1: 90% real, 10% synthetic (54k real, 6k synthetic)
- Gen 2: 80% real, 20% synthetic (48k real, 12k synthetic)
- Gen 3: 70% real, 30% synthetic (42k real, 18k synthetic)
- ...
- Gen 10: 0% real, 100% synthetic (0 real, 60k synthetic)

Total training size stays CONSTANT at 60k samples.
Synthetic data comes from the previous generation only.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import os
import argparse

# Constants
N_TOTAL = 60000  # Total training samples (constant)
BATCH_SIZE = 64
N_EPOCHS = 50
LATENT_DIM = 100
N_CLASSES = 10
IMG_SIZE = 28
CHANNELS = 1
LR = 0.0002
BETAS = (0.5, 0.999)


class Generator(nn.Module):
    """Conditional Generator for MNIST"""
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(N_CLASSES, N_CLASSES)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(LATENT_DIM + N_CLASSES, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, IMG_SIZE * IMG_SIZE * CHANNELS),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), CHANNELS, IMG_SIZE, IMG_SIZE)
        return img


class Discriminator(nn.Module):
    """Conditional Discriminator for MNIST"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(N_CLASSES, N_CLASSES)

        self.model = nn.Sequential(
            nn.Linear(N_CLASSES + IMG_SIZE * IMG_SIZE * CHANNELS, 512),
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
        return self.model(d_in)


def get_real_proportion(generation, total_generations=10):
    """
    Calculate real data proportion for a given generation.
    Gen 0: 100% real
    Gen N: (1 - N/total_generations) real
    """
    if generation == 0:
        return 1.0
    return max(0.0, 1.0 - generation / total_generations)


def load_progressive_dataset(generation, save_dir, real_X, real_C, total_generations=10):
    """
    Load dataset with progressively decaying real data proportion.

    Args:
        generation: Current generation number
        save_dir: Directory containing synthetic data
        real_X: Real MNIST images tensor
        real_C: Real MNIST labels tensor
        total_generations: Number of generations until 100% synthetic

    Returns:
        X, C tensors for training
    """
    real_proportion = get_real_proportion(generation, total_generations)
    synthetic_proportion = 1.0 - real_proportion

    n_real = int(N_TOTAL * real_proportion)
    n_synthetic = N_TOTAL - n_real

    print(f"  Gen {generation}: {real_proportion*100:.0f}% real ({n_real}) + {synthetic_proportion*100:.0f}% synthetic ({n_synthetic})")

    if generation == 0 or n_synthetic == 0:
        # First generation or 100% real: train on real data only
        indices = torch.randperm(len(real_X))[:N_TOTAL]
        X = real_X[indices]
        C = real_C[indices]
    else:
        # Load synthetic data from previous generation
        prev_gen = generation - 1
        syn_path = os.path.join(save_dir, f"gen_data_{prev_gen}.pt")
        syn_labels_path = os.path.join(save_dir, f"gen_labels_{prev_gen}.pt")

        if not os.path.exists(syn_path):
            raise FileNotFoundError(f"Synthetic data not found: {syn_path}")

        syn_X = torch.load(syn_path).float()
        if syn_X.max() > 1:
            syn_X = syn_X / 255.0
        syn_C = torch.load(syn_labels_path)

        # Sample from real data
        real_indices = torch.randperm(len(real_X))[:n_real]
        sampled_real_X = real_X[real_indices]
        sampled_real_C = real_C[real_indices]

        # Sample from synthetic data
        syn_indices = torch.randperm(len(syn_X))[:n_synthetic]
        sampled_syn_X = syn_X[syn_indices]
        sampled_syn_C = syn_C[syn_indices]

        # Combine
        X = torch.cat([sampled_real_X, sampled_syn_X], dim=0)
        C = torch.cat([sampled_real_C, sampled_syn_C], dim=0)

        # Shuffle
        shuffle_idx = torch.randperm(len(X))
        X = X[shuffle_idx]
        C = C[shuffle_idx]

    return X, C


def train_gan(generator, discriminator, dataloader, device, n_epochs):
    """Train GAN for one generation."""
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_labels = torch.randint(0, N_CLASSES, (batch_size,), device=device)
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

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} - D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

    return generator, discriminator


def generate_samples(generator, device, n_samples=60000):
    """Generate synthetic samples."""
    generator.eval()
    all_imgs = []
    all_labels = []

    samples_per_class = n_samples // N_CLASSES

    with torch.no_grad():
        for class_label in range(N_CLASSES):
            for _ in range(0, samples_per_class, BATCH_SIZE):
                batch_size = min(BATCH_SIZE, samples_per_class - len([l for l in all_labels if l == class_label]))
                if batch_size <= 0:
                    break
                z = torch.randn(batch_size, LATENT_DIM, device=device)
                labels = torch.full((batch_size,), class_label, dtype=torch.long, device=device)
                imgs = generator(z, labels)
                all_imgs.append(imgs.cpu())
                all_labels.extend([class_label] * batch_size)

    generator.train()
    return torch.cat(all_imgs, dim=0), torch.tensor(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Progressive decay synthetic loop experiment')
    parser.add_argument('--generations', type=int, default=11,
                        help='Number of generations (0 to N, where N is 100%% synthetic)')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Training epochs per generation (default: {N_EPOCHS})')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='Path to MNIST data')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Setup save directory
    save_dir = './data/gan_outputs_progressive/'
    os.makedirs(save_dir, exist_ok=True)

    # Load real MNIST
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    real_dataset = MNIST(args.data_path, train=True, download=True, transform=tf)
    real_X = real_dataset.data.float().unsqueeze(1) / 255.0
    real_X = real_X * 2 - 1  # Normalize to [-1, 1]
    real_C = real_dataset.targets

    print(f"Loaded {len(real_X)} real MNIST images")
    print(f"Total training samples per generation: {N_TOTAL}")
    print(f"Generations: {args.generations} (Gen 0 = 100% real, Gen {args.generations-1} = 100% synthetic)")
    print()

    # Progressive decay schedule
    print("Progressive Decay Schedule:")
    print("-" * 50)
    for gen in range(args.generations):
        real_prop = get_real_proportion(gen, args.generations - 1)
        n_real = int(N_TOTAL * real_prop)
        n_syn = N_TOTAL - n_real
        print(f"  Gen {gen}: {real_prop*100:5.1f}% real ({n_real:5d}) + {(1-real_prop)*100:5.1f}% syn ({n_syn:5d})")
    print("-" * 50)
    print()

    # Training loop
    for generation in range(args.generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {generation}")
        print(f"{'='*60}")

        # Load dataset with progressive decay
        X, C = load_progressive_dataset(
            generation, save_dir, real_X, real_C,
            total_generations=args.generations - 1
        )

        dataset = TensorDataset(X, C)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        # Initialize fresh models each generation
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)

        # Train
        print(f"  Training for {args.epochs} epochs...")
        generator, discriminator = train_gan(
            generator, discriminator, dataloader, device, args.epochs
        )

        # Save models
        torch.save(generator.state_dict(), os.path.join(save_dir, f'generator_{generation}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, f'discriminator_{generation}.pth'))

        # Generate synthetic samples for next generation
        print(f"  Generating {N_TOTAL} synthetic samples...")
        syn_imgs, syn_labels = generate_samples(generator, device, N_TOTAL)

        # Save synthetic data (convert back to [0, 1] range)
        syn_imgs_save = (syn_imgs + 1) / 2  # [-1, 1] -> [0, 1]
        torch.save(syn_imgs_save.squeeze(1), os.path.join(save_dir, f'gen_data_{generation}.pt'))
        torch.save(syn_labels, os.path.join(save_dir, f'gen_labels_{generation}.pt'))

        print(f"  Saved to {save_dir}")

    print("\n" + "="*60)
    print("PROGRESSIVE DECAY EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {save_dir}")
    print("\nNext steps:")
    print("  python compute_fid_progressive.py")
    print("  python jacobian_progressive.py")


if __name__ == "__main__":
    main()
