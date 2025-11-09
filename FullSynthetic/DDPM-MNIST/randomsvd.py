"""
Improved local Jacobian spectrum computation for MADness detection.
Uses randomized SVD for numerically stable and accurate spectrum estimation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import torch.multiprocessing as mp
from torchvision.datasets import MNIST
from torchvision import transforms

# Import model architecture
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import DDPM, ContextUnet


def load_real_mnist_anchors(n_anchors=50, device='cuda'):
    """Load real MNIST data to use as anchor points."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load MNIST dataset
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(os.path.join(script_dir, "data"), train=True, download=True, transform=tf)

    # Sample random anchors
    indices = torch.randperm(len(dataset))[:n_anchors]
    anchors = []

    for idx in indices:
        img, label = dataset[idx]
        anchors.append(img.unsqueeze(0))  # Add batch dimension

    anchors = torch.cat(anchors, dim=0).to(device)  # [n_anchors, 1, 28, 28]

    # Normalize to [0, 1] range (MNIST is already in this range)
    return anchors


def load_model(generation, device='cuda'):
    """Load a DDPM model from a specific generation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Model configuration
    n_T = 500
    n_feat = 128
    n_classes = 10

    # Create model
    nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes)
    ddpm = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

    # Load checkpoint
    checkpoint_path = os.path.join(script_dir, 'data', 'diffusion_outputs10', f'model_{generation}_w0.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(checkpoint)
    ddpm.eval()

    return ddpm


def compute_jacobian_spectrum_randomized(model, x, t, n_probes=100, batch_size=10):
    """
    Compute spectrum of J^T J using randomized linear algebra.

    Args:
        model: DDPM model
        x: anchor point (1, 1, 28, 28)
        t: timestep (absolute value, e.g., 1, 100, 250, 500)
        n_probes: number of random directions to probe
        batch_size: number of Jacobian-vector products to compute at once

    Returns:
        eigenvalues: eigenvalues of J^T J (sorted descending)
    """
    # Prepare model inputs
    x = x.detach()
    device = x.device

    # Flatten x to vector
    x_shape = x.shape
    x_flat = x.flatten()
    d = x_flat.numel()

    # Generate random orthonormal directions using QR decomposition
    V = torch.randn(n_probes, d, device=device)
    V, _ = torch.linalg.qr(V.T)
    V = V.T  # Now V is [n_probes, d] with orthonormal rows

    # Prepare fixed inputs for the model
    c = torch.zeros(1, dtype=torch.long, device=device)  # Class label
    t_norm = torch.tensor([t / model.n_T], device=device).reshape(1, 1, 1, 1)  # Normalized timestep
    context_mask = torch.zeros(1, device=device)

    # Compute J·v for each direction
    JV = []

    for i in tqdm(range(0, n_probes, batch_size), desc=f"Computing J·v (t={t})", leave=False):
        batch_end = min(i + batch_size, n_probes)
        batch_v = V[i:batch_end]  # [batch_size, d]

        batch_jv = []
        for v_flat in batch_v:
            v = v_flat.reshape(x_shape)

            # Compute J·v via autograd
            x_input = x.detach().requires_grad_(True)

            with torch.enable_grad():
                eps_pred = model.nn_model(x_input, c, t_norm, context_mask)

                # J·v is the directional derivative
                jv = torch.autograd.grad(
                    outputs=eps_pred,
                    inputs=x_input,
                    grad_outputs=v,
                    create_graph=False,
                    retain_graph=False
                )[0]

                batch_jv.append(jv.flatten())

        JV.extend(batch_jv)

    # Stack into matrix [n_probes, d]
    JV = torch.stack(JV)  # [n_probes, d]

    # Compute SVD of JV^T (which is [d, n_probes])
    # The singular values of JV^T are the same as JV
    try:
        U, S, Vh = torch.linalg.svd(JV.T, full_matrices=False)
        # S contains singular values of the Jacobian (via random sketch)

        # Eigenvalues of J^T J are squared singular values
        eigenvalues = (S ** 2).cpu().numpy()

    except Exception as e:
        print(f"SVD failed: {e}")
        eigenvalues = np.array([])

    return eigenvalues


def compute_spectrum_metrics(eigenvalues):
    """Compute summary metrics from eigenvalue spectrum."""
    if len(eigenvalues) == 0:
        return {}

    metrics = {
        'max': np.max(eigenvalues),
        'min': np.min(eigenvalues[eigenvalues > 1e-10]),  # Avoid zeros
        'mean': np.mean(eigenvalues),
        'median': np.median(eigenvalues),
        'condition_number': np.max(eigenvalues) / (np.min(eigenvalues[eigenvalues > 1e-10]) + 1e-10),
        'effective_rank': np.sum(eigenvalues) / (np.max(eigenvalues) + 1e-10),
        'spectral_gap': eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0
    }

    return metrics


def process_generation(gen, gpu_id, script_dir, timesteps=[50, 100, 250, 400], n_anchors=20, n_probes=100):
    """Process a single generation on a specific GPU."""
    device = f'cuda:{gpu_id}'

    try:
        # Load model
        model = load_model(gen, device)

        # Load real MNIST anchors (same for all generations)
        anchors = load_real_mnist_anchors(n_anchors, device)

        # Store results
        results = {
            'generation': gen,
            'timesteps': {},
        }

        # Compute spectrum at multiple timesteps
        for t in timesteps:
            print(f"\n[Gen {gen}, GPU {gpu_id}] Processing timestep t={t}")

            timestep_eigenvalues = []
            timestep_metrics = []

            for anchor_idx, anchor in enumerate(anchors):
                anchor_input = anchor.unsqueeze(0)  # [1, 1, 28, 28]

                try:
                    eigenvalues = compute_jacobian_spectrum_randomized(
                        model, anchor_input, t, n_probes=n_probes
                    )

                    if len(eigenvalues) > 0:
                        timestep_eigenvalues.append(eigenvalues)
                        metrics = compute_spectrum_metrics(eigenvalues)
                        timestep_metrics.append(metrics)

                except Exception as e:
                    print(f"Error at anchor {anchor_idx}: {e}")
                    continue

            # Store averaged metrics and all eigenvalues
            results['timesteps'][t] = {
                'all_eigenvalues': timestep_eigenvalues,
                'metrics': timestep_metrics,
                'avg_metrics': {
                    key: np.mean([m[key] for m in timestep_metrics])
                    for key in timestep_metrics[0].keys()
                } if timestep_metrics else {}
            }

            print(f"[Gen {gen}, t={t}] Collected {len(timestep_eigenvalues)} spectra")

        return results

    except Exception as e:
        print(f"Failed to process generation {gen} on GPU {gpu_id}: {e}")
        return {'generation': gen, 'error': str(e)}


def main():
    """Compute and visualize eigenvalue evolution across generations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration
    generations = list(range(20))  # All 20 generations (0-19)
    timesteps = [50, 100, 250, 400]  # Different stages of denoising
    n_anchors = 20  # Number of real MNIST samples to use
    n_probes = 100  # Number of random directions
    num_gpus = 3

    # Find available generations
    available_gens = []
    for gen in generations:
        if os.path.exists(os.path.join(script_dir, 'data', 'diffusion_outputs10', f'model_{gen}_w0.pth')):
            available_gens.append(gen)

    print(f"Found {len(available_gens)} generations: {available_gens}")

    # Compute spectra using multiple GPUs
    all_results = {}

    # Process generations in batches
    for i in range(0, len(available_gens), num_gpus):
        batch_gens = available_gens[i:i+num_gpus]
        print(f"\n{'='*60}")
        print(f"Processing batch: {batch_gens}")
        print(f"{'='*60}")

        # Create a pool of workers
        with mp.Pool(processes=len(batch_gens)) as pool:
            results = pool.starmap(
                process_generation,
                [(gen, idx % num_gpus, script_dir, timesteps, n_anchors, n_probes)
                 for idx, gen in enumerate(batch_gens)]
            )

        # Collect results
        for result in results:
            all_results[result['generation']] = result

    # Save results
    output_path = os.path.join(script_dir, 'spectrum_results_randomsvd.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {output_path}")

    # Visualize results
    visualize_results(all_results, available_gens, timesteps, script_dir)


def visualize_results(all_results, generations, timesteps, script_dir):
    """Create comprehensive visualizations of the spectral analysis."""

    # 1. Plot condition number evolution across generations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for t_idx, t in enumerate(timesteps):
        ax = axes[t_idx // 2, t_idx % 2]

        condition_numbers = []
        gens_plotted = []

        for gen in generations:
            if gen in all_results and 'timesteps' in all_results[gen]:
                if t in all_results[gen]['timesteps']:
                    avg_metrics = all_results[gen]['timesteps'][t]['avg_metrics']
                    if 'condition_number' in avg_metrics:
                        condition_numbers.append(avg_metrics['condition_number'])
                        gens_plotted.append(gen)

        if condition_numbers:
            ax.plot(gens_plotted, condition_numbers, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Generation', fontsize=12)
            ax.set_ylabel('Condition Number', fontsize=12)
            ax.set_title(f'Timestep t={t}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

    plt.suptitle('Jacobian Condition Number Evolution (MADness Detection)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'condition_number_evolution.png'), dpi=150)
    plt.close()

    # 2. Plot eigenvalue distributions for each generation
    n_gens = len(generations)
    fig, axes = plt.subplots(n_gens, len(timesteps), figsize=(16, 3*n_gens))

    if n_gens == 1:
        axes = axes.reshape(1, -1)

    for gen_idx, gen in enumerate(generations):
        for t_idx, t in enumerate(timesteps):
            ax = axes[gen_idx, t_idx]

            if gen in all_results and 'timesteps' in all_results[gen]:
                if t in all_results[gen]['timesteps']:
                    all_eigs = all_results[gen]['timesteps'][t]['all_eigenvalues']

                    if all_eigs:
                        # Flatten all eigenvalues
                        flat_eigs = np.concatenate(all_eigs)
                        log_eigs = np.log10(flat_eigs + 1e-10)

                        ax.hist(log_eigs, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                        ax.set_xlabel('log10(eigenvalue)')
                        ax.set_ylabel('Count')
                        ax.set_title(f'Gen {gen}, t={t}')
                        ax.grid(True, alpha=0.3)

    plt.suptitle('Eigenvalue Distributions Across Generations', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'eigenvalue_distributions.png'), dpi=150)
    plt.close()

    # 3. Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for gen in generations:
        if gen in all_results and 'timesteps' in all_results[gen]:
            print(f"\nGeneration {gen}:")
            for t in timesteps:
                if t in all_results[gen]['timesteps']:
                    metrics = all_results[gen]['timesteps'][t]['avg_metrics']
                    if metrics:
                        print(f"  t={t:3d}: cond={metrics['condition_number']:8.1f}, "
                              f"max={metrics['max']:8.4f}, mean={metrics['mean']:8.4f}, "
                              f"eff_rank={metrics['effective_rank']:6.1f}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
