"""
GAN-MNIST MADness Experiment
Train GANs iteratively on their own synthetic outputs for 8 generations.
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
N_next = 60000  # Samples to generate for next generation
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


def load_dataset(generation, save_dir, batch_size):
    """Load dataset for given generation"""
    if generation == 0:
        # Load real MNIST
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for tanh
        ])
        dataset = MNIST("../data", train=True, download=True, transform=tf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        # Load synthetic data from previous generation
        data_path = save_dir + f"gen_data_{generation-1}.pt"
        labels_path = save_dir + f"gen_labels_{generation-1}.pt"

        X = torch.load(data_path).float()
        # Convert from [0, 255] to [-1, 1]
        X = (X / 255.0) * 2 - 1
        C = torch.load(labels_path).long()

        dataset = TensorDataset(X, C)
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
    print(f"Starting MADness experiment with {generation_number} generations")
    print(f"Saving outputs to: {save_dir}")

    for generation in range(generation_number):
        print(f"\n{'='*50}")
        print(f"Generation {generation}")
        print(f"{'='*50}")

        # Load dataset
        dataloader = load_dataset(generation, save_dir, batch_size)

        # Train GAN
        generator = train_gan(dataloader, generation, device, save_dir)

        # Generate samples for next generation
        print(f"\nGenerating {N_next} samples for next generation...")
        X_gen, C_gen = generate_samples(generator, N_next, device)

        # Save as uint8 [0, 255] to match DDPM format
        torch.save((255 * X_gen).byte(), save_dir + f"gen_data_{generation}.pt")
        torch.save(C_gen, save_dir + f"gen_labels_{generation}.pt")

        # Save a grid of samples for visual inspection
        save_image(X_gen[:100], save_dir + f"samples/generation_{generation}_final.png", nrow=10, normalize=False)

        print(f"Generation {generation} complete!")

    print(f"\n{'='*50}")
    print("MADness experiment complete!")
    print(f"{'='*50}")
