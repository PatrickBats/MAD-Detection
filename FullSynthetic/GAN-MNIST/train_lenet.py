"""
Train LeNet classifiers on KMNIST and FashionMNIST for FID feature extraction.
Saves models to be used as feature extractors for FID computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import KMNIST, FashionMNIST
from tqdm import tqdm
import os
import argparse


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


def train_lenet(dataset_name, save_path, epochs=20, batch_size=128, lr=0.001):
    """Train LeNet on specified dataset"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training LeNet on {dataset_name} using {device}")

    # Load dataset
    tf = transforms.Compose([transforms.ToTensor()])

    if dataset_name == 'kmnist':
        train_dataset = KMNIST("../data", train=True, download=True, transform=tf)
        test_dataset = KMNIST("../data", train=False, download=True, transform=tf)
    elif dataset_name == 'fashionmnist':
        train_dataset = FashionMNIST("../data", train=True, download=True, transform=tf)
        test_dataset = FashionMNIST("../data", train=False, download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
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

            pbar.set_postfix({
                'loss': f'{train_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        scheduler.step()

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with {test_acc:.2f}% accuracy")

    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['kmnist', 'fashionmnist', 'both'],
                        help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    os.makedirs('./models', exist_ok=True)

    if args.dataset in ['kmnist', 'both']:
        print("\n" + "="*50)
        print("Training LeNet on KMNIST")
        print("="*50)
        train_lenet('kmnist', './models/lenet_kmnist.pth', epochs=args.epochs)

    if args.dataset in ['fashionmnist', 'both']:
        print("\n" + "="*50)
        print("Training LeNet on FashionMNIST")
        print("="*50)
        train_lenet('fashionmnist', './models/lenet_fashionmnist.pth', epochs=args.epochs)

    print("\nAll done!")
