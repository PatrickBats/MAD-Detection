# A Geometric Perspective on Recursive Synthetic Training

**Paper**: "A Geometric Perspective on Recursive Synthetic Training"
**Venue**: 2nd DeLTa Workshop, ICLR 2026

## Summary

This codebase investigates **Model Autophagy Disorder (MADness)** -- the degradation that occurs when deep generative networks (DGNs) are iteratively trained on their own synthetic outputs.

**Core finding**: MADness manifests as **Jacobian effective rank collapse**. When a generator's Jacobian `J = dG(z)/dz` is decomposed via SVD, the singular value spectrum concentrates on fewer directions over successive synthetic-training generations. This causes both mode collapse and quality degradation.

The **MADness loop**:

```
Gen 0: Train on real data       -> Generate 60k synthetic samples
Gen 1: Train on Gen 0's output  -> Generate 60k synthetic samples
Gen 2: Train on Gen 1's output  -> ...
```

Each generation trains a fresh model from scratch on the previous generation's output.

## Repository Structure

```
MAD-Detection/
├── CLAUDE.md                          # Claude Code project instructions
├── docs/                              # Documentation (this folder)
│   ├── README.md                      # This file
│   ├── architecture.md                # GAN architecture details
│   ├── metrics.md                     # Effective Rank + FID computation
│   ├── full_synthetic_mnist.md        # Figure 2 experiment
│   ├── full_synthetic_fashion.md      # Figure 6 experiment
│   ├── augmented_synthetic.md         # Figure 7 experiment
│   ├── neon_controlled_collapse.md    # Figures 4, 8, 9 experiment
│   ├── neon.md                        # NEON protocol details
│   └── reproduction.md               # Step-by-step reproduction guide
├── FullSynthetic/                     # Core MADness experiments
│   ├── create_publication_figures.py  # Generates figures from .pkl results
│   ├── figures/                       # Publication PDF figures
│   ├── data/                          # MNIST + FashionMNIST datasets
│   └── GAN-MNIST/
│       ├── GAN-MNIST-Default/         # Figure 2: MNIST (8 gens, ~1G)
│       ├── GAN-MNIST-Fashion/         # Figure 6: FashionMNIST (8 gens, ~1G)
│       ├── models/                    # Pretrained LeNet for FID
│       ├── train_lenet.py             # Train LeNet feature extractor
│       ├── compute_fid.py            # Cross-dataset FID computation
│       └── tsne_gan.py               # t-SNE visualization
├── AugmentedSynthetic/                # Real + synthetic mixing experiments
│   ├── GAN-MNIST/                     # Figure 7: augmentation loop (~632M)
│   └── GAN-MNIST-PaperStyle/         # Paper-style 50/50 augmentation (~82M)
├── FineTuning/                        # NEON implementations (~249M)
│   ├── GAN_NEON/                      # NEON for GAN
│   └── DDIM_NEON/                     # NEON for DDPM/DDIM
└── NeonTest/                          # NEON experiments (~292M)
    ├── ThomasExample/                 # Figures 4, 8, 9: controlled collapse
    ├── Test/                          # NEON tests + alternative methods
    └── 23288_.../Neon_SM/             # External supplementary models
```

## Quick Start

```bash
# Environment
export PYTHON=/home/patrick/miniconda3/bin/python

# Run the core MADness experiment (Figure 2)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/main.py

# Compute Jacobian analysis
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/jacobian_gan_multi_avg.py

# Compute FID scores
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/compute_fid.py

# Generate publication figures
$PYTHON FullSynthetic/create_publication_figures.py
```

See [reproduction.md](reproduction.md) for detailed reproduction instructions.

## Key Figures

| Figure | Experiment | Directory |
|--------|------------|-----------|
| Figure 2 | Full synthetic loop (MNIST) | `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/` |
| Figure 6 | Full synthetic loop (FashionMNIST) | `FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/` |
| Figure 7 | Augmented synthetic (real+synthetic mix) | `AugmentedSynthetic/GAN-MNIST/` |
| Figures 4, 8, 9 | Controlled collapse + NEON | `NeonTest/ThomasExample/` |

## Further Reading

- [architecture.md](architecture.md) -- GAN model architecture and training details
- [metrics.md](metrics.md) -- Effective Rank and FID computation
- [neon.md](neon.md) -- NEON (Negative Extrapolation) protocol
