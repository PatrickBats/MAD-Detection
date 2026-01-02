#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from metrics import DDPM, ContextUnet


def load_model(generation=0, device='cuda'):
    """Load DDPM model for a specific generation.

    Args:
        generation: Generation number, or 'initial' for model_initial.pth
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    n_T = 500
    nn_model = ContextUnet(in_channels=1, n_feat=128, n_classes=10)
    ddpm = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)

    if generation == 'initial':
        checkpoint_path = os.path.join(parent_dir, 'data', 'diffusion_outputs10', 'model_initial.pth')
    else:
        checkpoint_path = os.path.join(parent_dir, 'data', 'diffusion_outputs10', f'model_{generation}_w0.pth')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    nn_state = {k[9:]: v for k, v in checkpoint.items() if k.startswith('nn_model.')}
    ddpm.nn_model.load_state_dict(nn_state)

    ddpm.nn_model.eval()
    ddpm.eval()

    return ddpm


def generate_ddim_samples(model, n_samples=10, device='cuda'):
    """Generate samples using DDIM (deterministic)."""

    samples = []

    with torch.no_grad():
        for i in range(n_samples):
            # Generate initial noise
            torch.manual_seed(42 + i)
            z = torch.randn(1, 1, 28, 28).to(device)

            # DDIM sampling
            x = z.clone()
            c = torch.tensor([i % 10], device=device, dtype=torch.long)  # Cycle through digits
            context_mask = torch.zeros(1, device=device, dtype=torch.float32)

            for t in range(500, 0, -1):
                t_norm = torch.tensor([t / model.n_T], device=device, dtype=torch.float32).view(1, 1, 1, 1)
                eps = model.nn_model(x, c, t_norm, context_mask)

                if t > 1:
                    x_0_pred = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]
                    x = model.sqrtab[t-1] * x_0_pred + model.sqrtmab[t-1] * eps
                else:
                    x = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]

            samples.append(x.cpu().numpy())

    samples = np.array(samples).squeeze()
    labels = np.arange(n_samples) % 10

    return samples, labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    generations = ['initial', 0, 1, 2, 3, 4, 5]
    n_samples = 10

    # Create figure with 6 rows (one per generation), 10 columns (one per sample)
    fig, axes = plt.subplots(len(generations), n_samples, figsize=(n_samples*2, len(generations)*2))

    for gen_idx, gen in enumerate(generations):
        model = load_model(generation=gen, device=device)

        # Generate samples with DDIM
        ddim_samples, ddim_labels = generate_ddim_samples(model, n_samples, device)

        # Plot samples for this generation
        for i in range(n_samples):
            ddim_img = ddim_samples[i].squeeze()
            axes[gen_idx, i].imshow(ddim_img, cmap='gray')
            axes[gen_idx, i].axis('off')

            # Add column titles for first row
            if gen_idx == 0:
                axes[gen_idx, i].set_title(f'Digit {ddim_labels[i]}', fontsize=10)

        # Add row label
        row_label = 'Initial' if gen == 'initial' else f'Gen {gen}'
        axes[gen_idx, 0].text(-0.1, 0.5, row_label,
                             transform=axes[gen_idx, 0].transAxes,
                             fontsize=12, fontweight='bold',
                             va='center', ha='right')


    plt.suptitle('DDIM Sampling Across Generations (Initial + Gen 0-5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ddim_samples_gen0_to_5.png', dpi=150, bbox_inches='tight')



if __name__ == "__main__":
    main()
