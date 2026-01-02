"""
Compute FID scores for GAN experiments across generations.
Uses trained LeNet as feature extractor (84-dim features from penultimate layer).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import KMNIST, FashionMNIST
from scipy import linalg
from tqdm import tqdm
import argparse
import os


class LeNet(nn.Module):
    """LeNet architecture matching DDPM-MNIST metrics.py"""
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

    Args:
        real_features: numpy array [N, feature_dim]
        fake_features: numpy array [N, feature_dim]

    Returns:
        FID score (float)
    """
    # Compute mean and covariance
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(fake_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    # Compute FID
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical error might give slight imaginary component
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


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load appropriate LeNet model
    model = LeNet().to(device)

    if args.dataset == 'kmnist':
        model_path = './models/lenet_kmnist.pth'
        RealDataset = KMNIST
        data_dir = './GAN-MNIST-Kuzushiji/data/gan_outputs/'
    elif args.dataset == 'fashionmnist':
        model_path = './models/lenet_fashionmnist.pth'
        RealDataset = FashionMNIST
        data_dir = './GAN-MNIST-Fashion/data/gan_outputs/'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run: python train_lenet.py --dataset", args.dataset)
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load real data
    tf = transforms.Compose([transforms.ToTensor()])
    real_dataset = RealDataset("../data", train=True, download=True, transform=tf)
    real_loader = DataLoader(real_dataset, batch_size=256, shuffle=False, num_workers=4)

    print("Extracting real data features...")
    real_features = extract_features(model, real_loader, device, n_samples=args.n_samples)
    print(f"Real features shape: {real_features.shape}")

    # Find available generations
    gen_files = sorted([f for f in os.listdir(data_dir) if f.startswith('gen_data_') and f.endswith('.pt')])
    generations = [int(f.split('_')[2].split('.')[0]) for f in gen_files]
    print(f"Found generations: {generations}")

    # Compute FID for each generation
    fid_scores = {}

    for gen in generations:
        print(f"\nProcessing generation {gen}...")

        # Load synthetic data
        syn_path = data_dir + f"gen_data_{gen}.pt"
        syn_data = torch.load(syn_path).float()

        # Convert from [0, 255] to [0, 1] if needed
        if syn_data.max() > 1:
            syn_data = syn_data / 255.0

        # Create dataloader
        syn_labels = torch.zeros(syn_data.size(0), dtype=torch.long)  # Dummy labels
        syn_dataset = TensorDataset(syn_data, syn_labels)
        syn_loader = DataLoader(syn_dataset, batch_size=256, shuffle=False, num_workers=4)

        # Extract features
        syn_features = extract_features(model, syn_loader, device, n_samples=args.n_samples)

        # Compute FID
        fid = compute_fid(real_features, syn_features)
        fid_scores[gen] = fid
        print(f"Generation {gen}: FID = {fid:.2f}")

    # Summary
    print("\n" + "="*50)
    print("FID Scores Summary")
    print("="*50)
    print(f"{'Generation':<12} {'FID':>10}")
    print("-"*22)
    for gen in sorted(fid_scores.keys()):
        print(f"{gen:<12} {fid_scores[gen]:>10.2f}")

    # Save results
    results = {
        'fid_scores': fid_scores,
        'dataset': args.dataset,
        'n_samples': args.n_samples
    }
    save_path = data_dir + 'fid_scores.pkl'
    torch.save(results, save_path)
    print(f"\nResults saved to {save_path}")

    # Plot
    import matplotlib.pyplot as plt

    gens = sorted(fid_scores.keys())
    fids = [fid_scores[g] for g in gens]

    plt.figure(figsize=(8, 5))
    plt.plot(gens, fids, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('FID Score', fontsize=12)
    plt.title(f'{args.dataset.upper()} GAN - FID Across Generations', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = data_dir + 'fid_evolution.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['kmnist', 'fashionmnist'],
                        help='Dataset to compute FID for')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples to use for FID computation')
    args = parser.parse_args()

    main(args)
