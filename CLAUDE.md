# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for the paper **"The Geometry of Training on Synthetic Data"** (under review, DeLTa Workshop at ICLR 2026). It investigates **Model Autophagy Disorder (MADness)** — the degradation that occurs when deep generative networks (DGNs) are iteratively trained on their own synthetic outputs.

The core finding: MADness manifests as **Jacobian effective rank collapse**. When a generator's Jacobian `J = ∂G(z)/∂z` is decomposed via SVD, the singular value spectrum concentrates on fewer directions over successive synthetic-training generations, causing both mode collapse and quality degradation.

## Environment

- **Python**: Use `/home/patrick/miniconda3/bin/python` (Python 3.x with all packages). The system `python` is Python 2.7 and will not work.
- **GPUs**: 8 GPUs available. Most training scripts use `nn.DataParallel`. Some scripts hardcode `n_gpus=2`; override as needed.
- **Dependencies**: PyTorch 2.0+, torchvision, numpy, matplotlib, tqdm, scikit-learn, scipy, Pillow, seaborn, pandas.
- **Data**: MNIST/FashionMNIST stored at `FullSynthetic/data/`. Scripts auto-download if missing.
- **Documentation**: See `docs/` folder for detailed experiment descriptions and reproduction guide.

## Key Concepts

### The MADness Loop
```
Gen 0: Train on real data → Generate 60k synthetic samples
Gen 1: Train on Gen 0's synthetic output → Generate 60k synthetic samples
Gen 2: Train on Gen 1's synthetic output → ...
```
Each generation trains a fresh model from scratch on the previous generation's output.

### Effective Rank (Core Metric)
```python
# Singular values of Jacobian J = ∂G(z)/∂z
s = torch.linalg.svdvals(J)
p = s / s.sum()                              # normalize to distribution
entropy = -torch.sum(p * torch.log(p + eps)) # Shannon entropy
eff_rank = torch.exp(entropy)                # effective rank
```
- ER ≈ 4.5 at Gen 0 (healthy), drops to ≈ 1.0 by Gen 7 (collapsed)
- Maximum possible ER = min(output_dim, latent_dim) = 100 for MNIST GAN

### Shared GAN Architecture (all experiments)
- **Generator**: `110 → 128 → 256 → 512 → 1024 → 784` (Linear + BatchNorm + LeakyReLU, Tanh output)
- **Discriminator**: `794 → 512 → 512 → 512 → 1` (Linear + Dropout + LeakyReLU, Sigmoid output)
- **Training**: BCE loss, Adam (lr=2e-4, β1=0.5, β2=0.999), 50 epochs, batch_size=128
- Class-conditional via `nn.Embedding(10, 10)` concatenated with input
- Images normalized to [-1, 1] (Tanh), saved as uint8 [0, 255]

## Architecture & Directory Structure

### FullSynthetic/ — Core MADness experiments (pure synthetic loop)
Each generation trains exclusively on previous generation's synthetic output.
- `GAN-MNIST/GAN-MNIST-Default/main.py` — **Primary GAN baseline** (8 generations, Figure 2)
- `GAN-MNIST/GAN-MNIST-Fashion/` — FashionMNIST variant (Figure 5)
- `GAN-MNIST/models/` — Pretrained LeNet for FID
- `GAN-MNIST/train_lenet.py` — Train LeNet feature extractor
- `GAN-MNIST/compute_fid.py` — Cross-dataset FID computation
- `GAN-MNIST/tsne_gan.py` — t-SNE visualization
- `create_publication_figures.py` — Generates publication plots from `.pkl` result files
- `figures/` — Publication PDF figures

### AugmentedSynthetic/ — Real + synthetic mixing experiments (Figure 6)
Tests whether including real data prevents collapse.
- `GAN-MNIST/main.py` — Accumulating augmentation: Gen t trains on REAL + all synthetic from Gen 1..t-1
- `GAN-MNIST/main_constant_proportion.py` — Fixed ratio of real:synthetic each generation
- `GAN-MNIST/main_progressive_decay.py` — Synthetic influence decays with age
- `GAN-MNIST-PaperStyle/main.py` — Paper-style 50/50 augmentation

### FineTuning/ — NEON extrapolation method
- `GAN_NEON/protocol_a_neon.py` — NEON for GAN: `θ_NEON = (1+w)*θ_r - w*θ_s`
- `DDIM_NEON/protocol_a_ddim.py` — NEON for DDPM/DDIM

### NeonTest/ — Advanced NEON and controlled collapse experiments (Figures 1c, 7)
- `ThomasExample/neon_controlled_collapse_gan.py` — Generate synthetic samples with controlled Jacobian collapse (alpha parameter), then apply NEON
- `ThomasExample/neon_controlled_collapse_gan_parallel.py` — Parallelized version
- `Test/neon_gan_mnist.py` — Basic NEON test
- `Test/different_methods/` — Alternative regularization approaches (BatchNorm, weight reg, selective layer NEON)
- `23288_Neon_Negative_Extrapolat_Supplementary Material/` — External model repos (VAR, xAR, IMM)

### Jacobian Analysis Scripts (in each experiment directory)
Common pattern: load saved `generator_N.pth`, compute Jacobian via `torch.autograd.functional.jacobian()`, extract SVD, save results to `.pkl`.
- `jacobian_gan.py` — Single-sample analysis
- `jacobian_gan_multi.py` — Multi-sample with different seeds
- `jacobian_gan_multi_avg.py` — Averaged metrics across samples (most reliable)

## Common Commands

```bash
# Run GAN baseline MADness experiment (8 generations)
/home/patrick/miniconda3/bin/python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/main.py

# Run augmented synthetic experiment
/home/patrick/miniconda3/bin/python AugmentedSynthetic/GAN-MNIST/main.py

# Run NEON protocol
/home/patrick/miniconda3/bin/python FineTuning/GAN_NEON/protocol_a_neon.py

# Run controlled collapse NEON experiment
/home/patrick/miniconda3/bin/python NeonTest/ThomasExample/neon_controlled_collapse_gan.py

# Compute Jacobian analysis on trained generators
/home/patrick/miniconda3/bin/python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/jacobian_gan_multi_avg.py

# Generate publication figures
/home/patrick/miniconda3/bin/python FullSynthetic/create_publication_figures.py

# Compute FID scores
/home/patrick/miniconda3/bin/python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/compute_fid.py
```

## Key Data Flow

### Saved artifacts per generation (in each experiment's results/data directory):
- `generator_N.pth` — Generator state dict
- `discriminator_N.pth` — Discriminator state dict
- `gen_data_N.pt` — Synthetic images (uint8, [0,255], shape [60000, 1, 28, 28])
- `gen_labels_N.pt` — Labels (long, shape [60000])
- `samples/genN_epochM.png` — Sample grids during training
- `samples/generation_N_final.png` — Final generation sample grid

### Analysis outputs:
- `jacobian_gan_avg_results.pkl` — Dict with `results[gen]` → list of `{'eigenvalues': ..., 'effective_rank': ...}`
- Publication figures saved as `.pdf` in `FullSynthetic/figures/`

## Important Technical Notes

- **BatchNorm and Jacobian**: When computing Jacobians, always use `model.eval()` mode for stable BatchNorm statistics. This is critical for reproducible ER measurements.
- **DataParallel and Jacobian**: Extract `.module` before Jacobian computation. Jacobian must run on a single device.
- **Image normalization**: Generator outputs [-1, 1] (Tanh). Convert to [0, 1] with `(x+1)/2` before saving. Saved as uint8 [0, 255]. When loading for next generation, convert back: `x/255 * 2 - 1`.
- **FID computation**: Uses features from a LeNet classifier trained on MNIST (84-dimensional penultimate layer), not Inception. See `docs/metrics.md`.
