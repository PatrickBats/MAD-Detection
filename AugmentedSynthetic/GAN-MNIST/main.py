3"""
GAN-MNIST Synthetic Augmentation Loop Experiment

Implements the synthetic augmentation loop from the paper:
- Gen 1: Train on REAL (60k)
- Gen 2: Train on REAL (60k) + SYNTHETIC from Gen 1 (60k) = 120k
- Gen 3: Train on REAL (60k) + SYNTHETIC from Gen 1-2 (120k) = 180k
- Gen t: Train on REAL (60k) + ALL SYNTHETIC from Gen 1 to t-1

The real dataset is FIXED and always present.
Synthetic data ACCUMULATES from all previous generations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import os

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
generation_number = 8
N_synthetic = 60000  # Synthetic samples to generate per generation
save_dir = './data/gan_outputs/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_gpus = 2
print(f"Using device: {device}, using {n_gpus} GPUs")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir + "samples/", exist_ok=True)


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


def load_real_mnist():
    """Load real MNIST dataset (always the same)"""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for tanh
    ])
    dataset = MNIST("../data", train=True, download=True, transform=tf)
    return dataset


def load_augmented_dataset(generation, save_dir, batch_size, real_dataset):
    """
    Load dataset for given generation using synthetic augmentation loop.

    Gen 1: Real only (60k)
    Gen 2: Real (60k) + Synthetic from Gen 1 (60k) = 120k
    Gen 3: Real (60k) + Synthetic from Gen 1-2 (120k) = 180k
    ...
    Gen t: Real (60k) + ALL Synthetic from Gen 1 to t-1
    """
    if generation == 1:
        # First generation: train on real data only
        print(f"Generation {generation}: Training on REAL data only (60k samples)")
        dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return dataloader

    # For generation >= 2: combine real + all accumulated synthetic data
    # Get real data tensors
    real_X = real_dataset.data.float()
    real_X = real_X.unsqueeze(1) / 255.0  # Add channel dim and normalize to [0,1]
    real_X = real_X * 2 - 1  # Convert to [-1, 1]
    real_C = real_dataset.targets

    # Accumulate all synthetic data from previous generations
    synthetic_X_list = []
    synthetic_C_list = []

    for prev_gen in range(1, generation):
        data_path = save_dir + f"gen_data_{prev_gen}.pt"
        labels_path = save_dir + f"gen_labels_{prev_gen}.pt"

        if os.path.exists(data_path):
            X = torch.load(data_path).float()
            # Convert from [0, 255] to [-1, 1]
            X = (X / 255.0) * 2 - 1
            C = torch.load(labels_path).long()

            synthetic_X_list.append(X)
            synthetic_C_list.append(C)
            print(f"  Loaded synthetic data from Gen {prev_gen}: {X.shape[0]} samples")

    # Combine real + all synthetic
    all_X = [real_X] + synthetic_X_list
    all_C = [real_C] + synthetic_C_list

    combined_X = torch.cat(all_X, dim=0)
    combined_C = torch.cat(all_C, dim=0)

    total_real = real_X.shape[0]
    total_synthetic = sum(x.shape[0] for x in synthetic_X_list)
    total = combined_X.shape[0]

    print(f"Generation {generation}: Training on {total_real} real + {total_synthetic} synthetic = {total} total samples")

    dataset = TensorDataset(combined_X, combined_C)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader


def generate_samples(generator, N, device):
    """Generate N samples from the generator"""
    generator.eval()
    all_imgs = []
    all_labels = []

    batch_size = 500
    n_batches = N // batch_size

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Generating samples"):
            z = torch.randn(batch_size, latent_dim).to(device)
            labels = torch.randint(0, n_classes, (batch_size,)).to(device)

            imgs = generator(z, labels)
            # Convert from [-1, 1] to [0, 1]
            imgs = (imgs + 1) / 2

            all_imgs.append(imgs.cpu())
            all_labels.append(labels.cpu())

    generator.train()
    return torch.cat(all_imgs, dim=0), torch.cat(all_labels, dim=0)


def train_gan(dataloader, generation, device, save_dir):
    """Train GAN for one generation"""

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Wrap with DataParallel for multi-GPU
    if n_gpus > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        print(f"Using {n_gpus} GPUs")

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Training
    for epoch in range(n_epochs):
        g_losses = []
        d_losses = []

        pbar = tqdm(dataloader, desc=f"Gen {generation} Epoch {epoch+1}/{n_epochs}")
        for i, (imgs, labels) in enumerate(pbar):
            batch_size_curr = imgs.size(0)

            # Ground truths
            valid = torch.ones(batch_size_curr, 1).to(device)
            fake = torch.zeros(batch_size_curr, 1).to(device)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            z = torch.randn(batch_size_curr, latent_dim).to(device)
            gen_labels = torch.randint(0, n_classes, (batch_size_curr,)).to(device)

            gen_imgs = generator(z, gen_labels)

            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
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

        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            z = torch.randn(100, latent_dim).to(device)
            labels = torch.arange(10).repeat(10).to(device)
            gen_imgs = generator(z, labels)
            gen_imgs = (gen_imgs + 1) / 2  # Convert to [0, 1]
            save_image(gen_imgs.data, save_dir + f"samples/gen{generation}_epoch{epoch+1}.png", nrow=10, normalize=False)

    # Save model (unwrap DataParallel if needed)
    gen_state = generator.module.state_dict() if n_gpus > 1 else generator.state_dict()
    disc_state = discriminator.module.state_dict() if n_gpus > 1 else discriminator.state_dict()
    torch.save(gen_state, save_dir + f"generator_{generation}.pth")
    torch.save(disc_state, save_dir + f"discriminator_{generation}.pth")

    # Return unwrapped generator for sample generation
    if n_gpus > 1:
        return generator.module
    return generator


if __name__ == "__main__":
    print(f"Starting Synthetic Augmentation Loop experiment with {generation_number} generations")
    print(f"Saving outputs to: {save_dir}")
    print()
    print("Data structure:")
    print("  Gen 1: REAL (60k)")
    print("  Gen 2: REAL (60k) + SYNTHETIC_Gen1 (60k) = 120k")
    print("  Gen 3: REAL (60k) + SYNTHETIC_Gen1-2 (120k) = 180k")
    print("  ...")
    print()

    # Load real MNIST once (it stays fixed)
    real_dataset = load_real_mnist()
    print(f"Loaded real MNIST: {len(real_dataset)} samples")

    for generation in range(1, generation_number + 1):
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")

        # Load augmented dataset (real + accumulated synthetic)
        dataloader = load_augmented_dataset(generation, save_dir, batch_size, real_dataset)

        # Train GAN
        generator = train_gan(dataloader, generation, device, save_dir)

        # Generate samples for future generations
        print(f"\nGenerating {N_synthetic} samples for future generations...")
        X_gen, C_gen = generate_samples(generator, N_synthetic, device)

        # Save as uint8 [0, 255]
        torch.save((255 * X_gen).byte(), save_dir + f"gen_data_{generation}.pt")
        torch.save(C_gen, save_dir + f"gen_labels_{generation}.pt")

        # Save a grid of samples for visual inspection
        save_image(X_gen[:100], save_dir + f"samples/generation_{generation}_final.png", nrow=10, normalize=False)

        print(f"Generation {generation} complete!")

    print(f"\n{'='*60}")
    print("Synthetic Augmentation Loop experiment complete!")
    print(f"{'='*60}")
