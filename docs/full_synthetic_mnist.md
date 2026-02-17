# Full Synthetic Loop -- MNIST (Figure 2)

## Overview

The core MADness experiment: train a GAN on real MNIST, generate synthetic data, train a new GAN on the synthetic data, and repeat for 8 generations.

**Directory**: `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/`

## Protocol

```
Gen 0: Train GAN on real MNIST (60k images) -> Generate 60k synthetic samples
Gen 1: Train GAN on Gen 0's synthetic output -> Generate 60k synthetic samples
Gen 2: Train GAN on Gen 1's synthetic output -> ...
...
Gen 7: Train GAN on Gen 6's synthetic output -> Generate 60k synthetic samples
```

Each generation trains a **fresh model from scratch** (randomly initialized) on the previous generation's output.

## Key Results

- **Effective Rank**: Drops from ~4.5 (Gen 0) to ~1.0 (Gen 7)
- **FID**: Increases from ~3 (Gen 0) to ~30 (Gen 7)
- Visual quality degrades progressively; by Gen 7, most digits are unrecognizable

## Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Training loop for 8 generations |
| `jacobian_gan_multi_avg.py` | Jacobian ER analysis (20 samples, 8 GPUs) |
| `jacobian_gan_multi.py` | Multi-sample Jacobian with different seeds |
| `jacobian_gan.py` | Single-sample Jacobian analysis |
| `compute_fid.py` | FID scores across generations |
| `visualize_low_high_ER_samples.py` | Visualize samples by ER |

## Running

```bash
PYTHON=/home/patrick/miniconda3/bin/python

# Step 1: Train the MADness loop (8 generations, ~4 hours on 2 GPUs)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/main.py

# Step 2: Compute Jacobian analysis (~2 hours on 8 GPUs)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/jacobian_gan_multi_avg.py

# Step 3: Compute FID scores (~10 minutes)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/compute_fid.py

# Step 4: Generate publication figures
$PYTHON FullSynthetic/create_publication_figures.py
```

## Saved Artifacts

All artifacts saved under `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/`:

| File | Description |
|------|-------------|
| `generator_N.pth` | Generator state dict for generation N |
| `discriminator_N.pth` | Discriminator state dict for generation N |
| `gen_data_N.pt` | Synthetic images (uint8, [0,255], [60000, 1, 28, 28]) |
| `gen_labels_N.pt` | Labels (long, [60000]) |
| `lenet_mnist.pth` | LeNet weights for FID computation |
| `samples/genN_epochM.png` | Sample grids during training |
| `samples/generation_N_final.png` | Final sample grid |

## Configuration

- **Generations**: 8 (Gen 0 through Gen 7)
- **Samples per generation**: 60,000
- **Training**: 50 epochs per generation, batch_size=128
- **Architecture**: Standard GAN (see [architecture.md](architecture.md))
- **Dataset**: MNIST, loaded from `../data` (relative to script)
