"""
Generate comparison images across generations for MADness visualization.
Generates 30 images per digit (0-9) for generations 1, 5, 10, 15, and 19.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from tqdm import tqdm

# Import model architecture
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import DDPM, ContextUnet


def load_model(generation, device='cuda'):
    """Load a DDPM model from a specific generation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Model configuration (same as training)
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


def generate_samples_for_digit(ddpm, digit, n_samples=30, device='cuda', guide_w=0.0):
    """Generate n_samples images for a specific digit."""
    ddpm.eval()

    with torch.no_grad():
        # Start from noise
        x_i = torch.randn(n_samples, 1, 28, 28).to(device)
        c_i = torch.tensor([digit] * n_samples).to(device)

        # Context mask (0 = use context)
        context_mask = torch.zeros_like(c_i).to(device)

        # Double the batch for classifier-free guidance
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_samples:] = 1.  # makes second half of batch context free

        # Denoise step by step
        for i in range(ddpm.n_T, 0, -1):
            t_is = torch.tensor([i / ddpm.n_T]).to(device)
            t_is = t_is.repeat(n_samples, 1, 1, 1)

            # Double batch
            x_i_double = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_samples, 1, 28, 28).to(device) if i > 1 else 0

            # Predict noise
            eps = ddpm.nn_model(x_i_double, c_i, t_is, context_mask)
            eps1 = eps[:n_samples]
            eps2 = eps[n_samples:]

            # Classifier-free guidance
            eps = ((1 + guide_w) * eps1 - guide_w * eps2)

            x_i = x_i[:n_samples]
            x_i = (
                ddpm.oneover_sqrta[i] * (x_i - eps * ddpm.mab_over_sqrtmab[i])
                + ddpm.sqrt_beta_t[i] * z
            )

    # Clip to [0, 1]
    x_i = torch.clip(x_i, 0, 1)

    return x_i


def generate_and_save_all(generations=[1, 5, 10, 15, 19], n_samples=30, device='cuda'):
    """Generate images for all digits across specified generations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'generation_comparison')
    os.makedirs(output_dir, exist_ok=True)

    for generation in generations:
        print(f"\n=== Processing Generation {generation} ===")

        # Load model
        try:
            ddpm = load_model(generation, device)
        except Exception as e:
            print(f"Error loading model for generation {generation}: {e}")
            continue

        # Generate 3 samples for each digit (30 total images)
        all_images = []
        for digit in tqdm(range(10), desc=f"Gen {generation}"):
            # Generate 3 samples for this digit
            images = generate_samples_for_digit(ddpm, digit, n_samples=3, device=device)
            all_images.append(images)

        # Stack all images (10 digits × 3 samples = 30 images)
        all_images = torch.cat(all_images, dim=0)  # Shape: [30, 1, 28, 28]

        # Create a grid with 10 rows (one per digit) and 3 columns (samples per digit)
        grid = make_grid(all_images, nrow=3, padding=2, normalize=False)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 24))
        ax.imshow(grid_np[:, :, 0], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Generation {generation} - 3 samples per digit', fontsize=20)

        # Add digit labels on the left
        for i in range(10):
            y_pos = (i * 3 + 1.5) * 30  # Adjusted for 3 images per row
            ax.text(-10, y_pos, f'{i}', fontsize=18, ha='right', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'generation_{generation}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved images for generation {generation}")

    print(f"\n=== All images saved to {output_dir} ===")


if __name__ == "__main__":
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate images
    generate_and_save_all(
        generations=[1, 5, 10, 15, 19],
        n_samples=30,
        device=device
    )
