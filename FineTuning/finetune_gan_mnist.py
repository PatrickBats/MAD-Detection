#!/usr/bin/env python3
"""
Fine-tune GAN on Synthetic Data (Neon-style)

This script replicates the Neon paper's fine-tuning methodology:
1. Load pretrained GAN (Gen 0, trained on real MNIST)
2. Generate synthetic data from it (6k samples, matching Neon)
3. Fine-tune the SAME model on synthetic data only
4. Save checkpoints at regular intervals for Jacobian analysis

Key difference from Full Synthetic Loop:
- Full Synthetic: Train fresh model from scratch on synthetic data
- Fine-tuning: Continue training existing model on synthetic data

This allows us to study how Jacobian effective rank changes during
the fine-tuning process that causes model collapse.
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
import copy

# Hyperparameters (must match original GAN training)
latent_dim = 100
img_size = 28
channels = 1
n_classes = 10
batch_size = 128
lr = 1e-4  # Lower LR for fine-tuning (Neon uses 1e-4)
b1 = 0.5
b2 = 0.999

# Fine-tuning parameters (matching Neon paper)
N_SYNTHETIC = 6000  # |S| = 6k in Neon paper
N_FINETUNE_EPOCHS = 20  # More epochs to see collapse progression
CHECKPOINT_EVERY = 2  # Save checkpoint every N epochs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
BASE_MODEL_PATH = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/generator_0.pth'
BASE_DISC_PATH = '/home/patrick/MAD-Detection/FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/discriminator_0.pth'
SAVE_DIR = '/home/patrick/MAD-Detection/FineTuning/outputs/'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + 'checkpoints/', exist_ok=True)
os.makedirs(SAVE_DIR + 'samples/', exist_ok=True)


class Generator(nn.Module):
    """Conditional Generator for MNIST (must match original architecture)"""
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
    """Conditional Discriminator for MNIST (must match original architecture)"""
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


def generate_synthetic_data(generator, n_samples):
    """Generate synthetic data from the generator"""
    generator.eval()
    all_imgs = []
    all_labels = []

    batch_size_gen = 500
    n_batches = n_samples // batch_size_gen

    print(f"Generating {n_samples} synthetic samples...")
    with torch.no_grad():
        for _ in tqdm(range(n_batches)):
            z = torch.randn(batch_size_gen, latent_dim).to(device)
            labels = torch.randint(0, n_classes, (batch_size_gen,)).to(device)
            imgs = generator(z, labels)
            # Keep in [-1, 1] range for training
            all_imgs.append(imgs.cpu())
            all_labels.append(labels.cpu())

    generator.train()
    return torch.cat(all_imgs, dim=0), torch.cat(all_labels, dim=0)


def finetune_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, epoch):
    """Fine-tune for one epoch"""
    adversarial_loss = nn.BCELoss()

    g_losses = []
    d_losses = []

    pbar = tqdm(dataloader, desc=f"Fine-tune Epoch {epoch}")
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

    return np.mean(g_losses), np.mean(d_losses)


def save_checkpoint(generator, discriminator, epoch, save_dir):
    """Save model checkpoint"""
    torch.save(generator.state_dict(),
               os.path.join(save_dir, 'checkpoints', f'generator_ft_epoch{epoch}.pth'))
    torch.save(discriminator.state_dict(),
               os.path.join(save_dir, 'checkpoints', f'discriminator_ft_epoch{epoch}.pth'))
    print(f"Saved checkpoint at epoch {epoch}")


def save_samples(generator, epoch, save_dir):
    """Save sample images"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(100, latent_dim).to(device)
        labels = torch.arange(10).repeat(10).to(device)
        gen_imgs = generator(z, labels)
        gen_imgs = (gen_imgs + 1) / 2  # Convert to [0, 1]
        save_image(gen_imgs,
                   os.path.join(save_dir, 'samples', f'samples_epoch{epoch}.png'),
                   nrow=10, normalize=False)
    generator.train()


def main():
    print("="*70)
    print("GAN Fine-tuning on Synthetic Data (Neon-style)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Synthetic samples: {N_SYNTHETIC}")
    print(f"  Fine-tune epochs: {N_FINETUNE_EPOCHS}")
    print(f"  Learning rate: {lr}")
    print(f"  Checkpoint every: {CHECKPOINT_EVERY} epochs")
    print(f"  Save directory: {SAVE_DIR}")

    # Load base model (Gen 0 trained on real MNIST)
    print(f"\nLoading base model from {BASE_MODEL_PATH}")

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))

    discriminator = Discriminator().to(device)
    if os.path.exists(BASE_DISC_PATH):
        discriminator.load_state_dict(torch.load(BASE_DISC_PATH, map_location=device))
        print("Loaded discriminator from checkpoint")
    else:
        print("Warning: No discriminator checkpoint found, using fresh discriminator")

    # Save base model as epoch 0
    save_checkpoint(generator, discriminator, 0, SAVE_DIR)
    save_samples(generator, 0, SAVE_DIR)

    # Generate synthetic data from base model
    print("\n" + "="*70)
    print("Step 1: Generate synthetic data from base model")
    print("="*70)

    syn_imgs, syn_labels = generate_synthetic_data(generator, N_SYNTHETIC)

    # Save some synthetic samples for inspection
    save_image((syn_imgs[:100] + 1) / 2,
               os.path.join(SAVE_DIR, 'synthetic_training_data.png'),
               nrow=10, normalize=False)
    print(f"Synthetic data shape: {syn_imgs.shape}")

    # Create dataloader for synthetic data
    syn_dataset = TensorDataset(syn_imgs, syn_labels)
    syn_loader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Setup optimizers (lower LR for fine-tuning)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Fine-tuning loop
    print("\n" + "="*70)
    print("Step 2: Fine-tune on synthetic data")
    print("="*70)

    training_log = {
        'epochs': [],
        'g_losses': [],
        'd_losses': []
    }

    for epoch in range(1, N_FINETUNE_EPOCHS + 1):
        g_loss, d_loss = finetune_epoch(
            generator, discriminator, syn_loader,
            optimizer_G, optimizer_D, epoch
        )

        training_log['epochs'].append(epoch)
        training_log['g_losses'].append(g_loss)
        training_log['d_losses'].append(d_loss)

        # Save checkpoint at intervals
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(generator, discriminator, epoch, SAVE_DIR)
            save_samples(generator, epoch, SAVE_DIR)

    # Save final checkpoint
    save_checkpoint(generator, discriminator, N_FINETUNE_EPOCHS, SAVE_DIR)
    save_samples(generator, N_FINETUNE_EPOCHS, SAVE_DIR)

    # Save training log
    import pickle
    with open(os.path.join(SAVE_DIR, 'training_log.pkl'), 'wb') as f:
        pickle.dump(training_log, f)

    # Plot training curves
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(training_log['epochs'], training_log['g_losses'], label='Generator Loss')
    ax.plot(training_log['epochs'], training_log['d_losses'], label='Discriminator Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('GAN Fine-tuning on Synthetic Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
    plt.close()

    print("\n" + "="*70)
    print("Fine-tuning Complete!")
    print("="*70)
    print(f"\nCheckpoints saved in: {SAVE_DIR}checkpoints/")
    print(f"  - generator_ft_epoch0.pth (base model)")
    for epoch in range(CHECKPOINT_EVERY, N_FINETUNE_EPOCHS + 1, CHECKPOINT_EVERY):
        print(f"  - generator_ft_epoch{epoch}.pth")
    print(f"\nNext step: Run Jacobian analysis on these checkpoints")
    print(f"to measure effective rank collapse during fine-tuning.")


if __name__ == "__main__":
    main()
