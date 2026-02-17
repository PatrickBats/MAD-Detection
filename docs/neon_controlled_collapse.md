# NEON with Controlled Collapse (Figures 1c, 7)

## Overview

Tests whether synthetic data with controlled Jacobian collapse improves NEON (Negative Extrapolation) effectiveness. Instead of generating normal synthetic samples, the Jacobian singular value spectrum is explicitly collapsed by a parameter `alpha`.

**Directory**: `NeonTest/ThomasExample/`

## Key Idea

- `alpha = 0`: Normal generation (full rank Jacobian)
- `alpha = 1`: Full collapse (rank-1 Jacobian, all energy in first singular value)

The question: does more collapse in synthetic data lead to better NEON improvement?

## Protocol

1. Load base GAN `theta_r` (trained on real MNIST, Gen 0 from FullSynthetic)
2. For each `alpha` (collapse level):
   a. Generate synthetic samples with controlled Jacobian collapse
   b. Fine-tune base model on collapsed samples -> `theta_s`
   c. Apply NEON at various `w`: `theta_NEON = (1+w)*theta_r - w*theta_s`
   d. Measure FID

## Controlled Collapse Formula

From Appendix B of the paper. Given Jacobian `J` with SVD `J = U S V^T`:

```
S_new[i] = sqrt(1 - alpha) * S[i]      for i > 0
S_new[0] = sqrt((1 - alpha) * S[0]^2 + alpha * sum(S^2))
```

This redistributes spectral energy toward the first singular value while preserving total energy `sum(S^2)`.

The collapsed image is generated via affine approximation:
```
x_collapsed = J_collapsed @ z + b
where b = x_normal - J @ z
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Alpha values | 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 |
| NEON w sweep | -0.5 to 1.5 (21 values) |
| Synthetic samples | 2,000 |
| Fine-tune epochs | 10 |
| Fine-tune LR | 1e-4 |
| FID evaluation samples | 10,000 |
| Seeds | 0, 1, 2 |

## Key Results

- `alpha ~ 0.1` gives the best NEON improvement (moderate mode-seeking synthetic data)
- Too much collapse (`alpha > 0.5`) degrades NEON effectiveness
- NEON formula effectively reverses the collapse direction

## Scripts

| Script | Purpose |
|--------|---------|
| `neon_controlled_collapse_gan.py` | Main experiment (serial) |
| `neon_controlled_collapse_gan_parallel.py` | Parallelized version |

## Running

```bash
PYTHON=/home/patrick/miniconda3/bin/python

# Run controlled collapse experiment
$PYTHON NeonTest/ThomasExample/neon_controlled_collapse_gan.py
```

## Dependencies

Requires the base GAN checkpoint from the full synthetic experiment:
- `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/generator_0.pth`
- `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/discriminator_0.pth`
- `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/lenet_mnist.pth`

## Outputs

Saved to `NeonTest/ThomasExample/results_gan/`:
- `neon_results.csv` -- FID vs w for each alpha and seed
- `spectrum_results.pkl` -- Singular value spectra
- `neon_fid_vs_w.png/pdf` -- FID vs NEON w plot
- `effective_rank_vs_alpha.png/pdf` -- ER vs collapse level
- `singular_values_vs_alpha.png/pdf` -- Spectrum comparison
- `samples/collapsed_alpha*.png` -- Sample visualizations
