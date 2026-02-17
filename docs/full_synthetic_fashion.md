# Full Synthetic Loop -- FashionMNIST (Figure 5)

## Overview

Same MADness protocol as the MNIST experiment, applied to FashionMNIST to demonstrate generality of the ER collapse phenomenon.

**Directory**: `FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/`

## Protocol

Identical to the MNIST experiment:
- 8 generations of fully synthetic training
- 60,000 samples generated per generation
- Fresh model trained from scratch each generation
- Same GAN architecture and hyperparameters

The only difference is the dataset: FashionMNIST instead of MNIST.

## Key Results

- Similar ER decline pattern as MNIST, demonstrating the MADness phenomenon is not dataset-specific
- FashionMNIST is more challenging (more complex textures), so degradation may be visible earlier

## Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Training loop for 8 generations |
| `jacobian_gan_multi_avg.py` | Averaged Jacobian ER analysis |
| `jacobian_gan_multi.py` | Multi-sample Jacobian analysis |
| `jacobian_gan_single_class.py` | Per-class Jacobian analysis |

## Running

```bash
PYTHON=/home/patrick/miniconda3/bin/python

# Step 1: Train the MADness loop
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/main.py

# Step 2: Compute Jacobian analysis
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/jacobian_gan_multi_avg.py
```

## Saved Artifacts

Same structure as MNIST, saved under `FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/data/gan_outputs/`:
- `generator_N.pth`, `discriminator_N.pth` -- Model checkpoints
- `gen_data_N.pt`, `gen_labels_N.pt` -- Synthetic data
- `samples/` -- Sample visualizations

## Configuration

Same as MNIST experiment (see [full_synthetic_mnist.md](full_synthetic_mnist.md)) except:
- Dataset: FashionMNIST (loaded via `torchvision.datasets.FashionMNIST`)
- Data path: `../data` (relative to script, uses `FullSynthetic/data/`)
