"""
Compute FID scores across all generations using the SAME method as the original experiments.
This uses the pretrained LeNet model from prmodel.pth for feature extraction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

# Import from metrics (use the existing functions)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import extract_mnist_features, calculate_frechet_distance


def compute_fid_for_generation(generation, script_dir, real_mu, real_cov, device='cuda'):
    """Compute FID for a specific generation using the original method."""

    # Load generated data
    if generation == 0:
        # Use initial real MNIST data
        data_file = 'gen_data_with_w_initial_w0'
    else:
        # Use generated data from previous generation
        data_file = f'gen_data_without_w{generation-1}_w0'

    data_path = os.path.join(script_dir, 'data', 'diffusion_outputs10', data_file)

    # Try zipped version if needed
    if not os.path.exists(data_path):
        data_path = data_path + '.zip'

    if not os.path.exists(data_path):
        print(f"Warning: Cannot find {data_file}")
        return None

    # Load data
    x = torch.load(data_path, map_location=device)
    x = x.to(device)
    x = x.float() / 255.0

    # Extract features using pretrained LeNet
    gen_features = extract_mnist_features(x, device)

    # Compute statistics
    mu_gen = np.mean(np.transpose(gen_features), axis=1)
    cov_gen = np.cov(np.transpose(gen_features))

    # Compute FID
    fid = calculate_frechet_distance(mu_gen, cov_gen, real_mu, real_cov)

    return fid


def main():
    """Compute FID scores for all generations using the original method."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print("\nThis script uses the PRETRAINED LeNet model for feature extraction")
    print("(same as the original experiments)\n")

    # Load real MNIST data and compute statistics (as reference)
    print("="*80)
    print("Loading real MNIST reference data...")
    print("="*80)

    # The original code uses all real data for reference
    from torchvision.datasets import MNIST
    from torchvision import transforms

    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(os.path.join(script_dir, "data"), train=True, download=True, transform=tf)

    # Convert to tensor
    real_data = dataset.data.float().unsqueeze(1) / 255.0
    real_data = real_data.to(device)

    # Extract features
    print("Extracting features from real MNIST...")
    real_features = extract_mnist_features(real_data, device)

    # Compute statistics
    mu_real = np.mean(np.transpose(real_features), axis=1)
    cov_real = np.cov(np.transpose(real_features))

    print(f"Real MNIST: {len(real_features)} samples, feature dim={real_features.shape[1]}")

    # Find available generations
    generations = []
    for gen in range(20):
        model_path = os.path.join(script_dir, 'data', 'diffusion_outputs10', f'model_{gen}_w0.pth')
        if os.path.exists(model_path):
            generations.append(gen)

    print(f"\nFound {len(generations)} generations: {generations}\n")

    # Compute FID for each generation
    fid_scores = {}

    for gen in generations:
        print("="*80)
        print(f"Processing Generation {gen}")
        print("="*80)

        try:
            fid = compute_fid_for_generation(gen, script_dir, mu_real, cov_real, device)

            if fid is not None:
                fid_scores[gen] = fid
                print(f"Generation {gen}: FID = {fid:.4f}")
            else:
                fid_scores[gen] = None
                print(f"Generation {gen}: No data available")

        except Exception as e:
            print(f"Error processing generation {gen}: {e}")
            fid_scores[gen] = None

    # Save results
    results_path = os.path.join(script_dir, 'fid_scores_correct.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(fid_scores, f)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "="*80)
    print("FID SCORE SUMMARY (Using Pretrained LeNet Features)")
    print("="*80)

    valid_scores = []
    for gen in generations:
        if fid_scores[gen] is not None:
            print(f"Generation {gen:2d}: FID = {fid_scores[gen]:8.4f}")
            valid_scores.append(fid_scores[gen])

    if len(valid_scores) > 1:
        print(f"\nFID range: {min(valid_scores):.4f} to {max(valid_scores):.4f}")
        print(f"FID increase from Gen 0: {valid_scores[-1] - valid_scores[0]:.4f}")

    # Plot results
    plot_fid_scores(fid_scores, generations, script_dir)


def plot_fid_scores(fid_scores, generations, script_dir):
    """Create visualization of FID scores across generations."""

    # Filter valid scores
    valid_gens = [g for g in generations if fid_scores[g] is not None]
    valid_scores = [fid_scores[g] for g in valid_gens]

    if not valid_scores:
        print("No valid FID scores to plot")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(valid_gens, valid_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('FID Score', fontsize=14)
    ax.set_title('FID Score Evolution (Pretrained LeNet Features)\n(Lower is better)', fontsize=16)
    ax.grid(True, alpha=0.3)

    # Add annotation for Gen 0
    if 0 in valid_gens:
        idx = valid_gens.index(0)
        ax.annotate('Real MNIST\n(reference)',
                   xy=(0, valid_scores[idx]),
                   xytext=(0.5, valid_scores[idx] + max(valid_scores)*0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, color='red')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(script_dir, 'fid_evolution_correct.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
