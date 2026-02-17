#!/usr/bin/env python3
"""
Compute FID scores for Augmented Synthetic GAN-MNIST across generations.
Uses trained LeNet as feature extractor (84-dim features from penultimate layer).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from scipy import linalg
from tqdm import tqdm
import os
import pickle


class LeNet(nn.Module):
    """LeNet architecture for feature extraction"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc3(out)
        return out

    def extract_features(self, x):
        """Extract 84-dim features from penultimate layer"""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def compute_fid(real_features, fake_features):
    """
    Compute Frechet Inception Distance between two feature distributions.
    """
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(fake_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)


def extract_features(model, dataloader, device, n_samples=10000):
    """Extract features from dataset using LeNet"""
    model.eval()
    all_features = []
    count = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            if count >= n_samples:
                break
            images = images.to(device)
            features = model.extract_features(images)
            all_features.append(features.cpu().numpy())
            count += images.size(0)

    features = np.concatenate(all_features, axis=0)[:n_samples]
    return features


def train_lenet(device, save_path):
    """Train LeNet on MNIST if model doesn't exist"""
    print("Training LeNet on MNIST...")

    tf = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST("../data", train=True, download=True, transform=tf)
    test_dataset = MNIST("../data", train=False, download=True, transform=tf)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={acc:.2f}%")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'lenet_mnist.pth')
    data_dir = os.path.join(script_dir, 'data', 'gan_outputs')

    # Load or train LeNet
    model = LeNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        model = train_lenet(device, model_path)

    model.eval()

    # Load real MNIST data
    tf = transforms.Compose([transforms.ToTensor()])
    real_dataset = MNIST("../data", train=True, download=True, transform=tf)
    real_loader = DataLoader(real_dataset, batch_size=256, shuffle=False, num_workers=4)

    n_samples = 10000
    print(f"\nExtracting real data features (n={n_samples})...")
    real_features = extract_features(model, real_loader, device, n_samples=n_samples)
    print(f"Real features shape: {real_features.shape}")

    # Find available generations
    gen_files = sorted([f for f in os.listdir(data_dir) if f.startswith('gen_data_') and f.endswith('.pt')])
    generations = [int(f.split('_')[2].split('.')[0]) for f in gen_files]
    print(f"Found generations: {generations}")

    # Compute FID for each generation
    fid_scores = {}

    for gen in generations:
        print(f"\nProcessing generation {gen}...")

        syn_path = os.path.join(data_dir, f"gen_data_{gen}.pt")
        syn_data = torch.load(syn_path).float()

        # Convert from [0, 255] to [0, 1] if needed
        if syn_data.max() > 1:
            syn_data = syn_data / 255.0

        # Create dataloader
        syn_labels = torch.zeros(syn_data.size(0), dtype=torch.long)
        syn_dataset = TensorDataset(syn_data, syn_labels)
        syn_loader = DataLoader(syn_dataset, batch_size=256, shuffle=False, num_workers=4)

        # Extract features
        syn_features = extract_features(model, syn_loader, device, n_samples=n_samples)

        # Compute FID
        fid = compute_fid(real_features, syn_features)
        fid_scores[gen] = fid
        print(f"Generation {gen}: FID = {fid:.2f}")

    # Summary
    print("\n" + "="*50)
    print("FID Scores Summary (Augmented Synthetic Loop)")
    print("="*50)
    print(f"{'Generation':<12} {'FID':>10}")
    print("-"*22)
    for gen in sorted(fid_scores.keys()):
        print(f"{gen:<12} {fid_scores[gen]:>10.2f}")
    print("="*50)

    # Save results
    results = {
        'fid_scores': fid_scores,
        'n_samples': n_samples
    }
    save_path = os.path.join(data_dir, 'fid_scores.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {save_path}")

    # Plot
    import matplotlib.pyplot as plt

    gens = sorted(fid_scores.keys())
    fids = [fid_scores[g] for g in gens]

    plt.figure(figsize=(8, 5))
    plt.plot(gens, fids, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('FID Score', fontsize=12)
    plt.title('Augmented Synthetic GAN-MNIST - FID Across Generations', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(data_dir, 'fid_evolution.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
