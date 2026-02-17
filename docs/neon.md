# NEON Protocol

## Overview

NEON (Negative Extrapolation for Overcoming Noise) is a method that improves a generative model by extrapolating in the opposite direction of synthetic data degradation:

```
theta_NEON = (1 + w) * theta_r - w * theta_s
```

Where:
- `theta_r`: Base model trained on real data
- `theta_s`: Model fine-tuned on synthetic data (intentionally degraded)
- `w`: Extrapolation weight (typically 0 < w < 2)

The intuition: fine-tuning on synthetic data moves model parameters in the "collapse direction." NEON reverses this direction to improve the model.

## Implementations

### GAN NEON (`FineTuning/GAN_NEON/protocol_a_neon.py`)

Protocol A for MNIST GAN:

1. Load base GAN `theta_r` (Gen 0, trained on real MNIST)
2. Generate 6,000 synthetic samples from `theta_r`
3. Fine-tune `theta_r` on synthetic data for 6 epochs -> `theta_s`
4. Measure Jacobian ER before and after fine-tuning
5. Apply NEON at w = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}
6. Evaluate FID at each w

**Configuration**:
- Synthetic samples: 6,000 (matching NEON paper |S|=6k)
- Fine-tune epochs: 6
- Fine-tune LR: 1e-4
- Jacobian samples: 20 (class 7)
- FID samples: 10,000

### DDIM NEON (`FineTuning/DDIM_NEON/protocol_a_ddim.py`)

Same Protocol A applied to a DDPM/DDIM model:
- Uses `ContextUnet` with 128 features, 500 timesteps
- DDIM deterministic sampling for generation
- Otherwise identical protocol to GAN NEON

**Note**: The DDIM NEON script imports from the (now deleted) `FullSynthetic/DDPM-MNIST/metrics.py`. If you need to run this, you'll need to restore that dependency or copy the `ContextUnet`, `DDPM`, and `LeNet` classes locally.

## Alternative Methods (`NeonTest/Test/different_methods/`)

Three alternative regularization approaches tested alongside NEON:

| Script | Method |
|--------|--------|
| `batchnorm_neon_fine.py` | NEON applied only to BatchNorm parameters |
| `weight_regularization.py` | Direct weight regularization during fine-tuning |
| `selective_layer_neon.py` | NEON applied to selected layers only |

## NEON Test Scripts (`NeonTest/Test/`)

- `neon_gan_mnist.py` -- Basic NEON test on MNIST GAN

## External Supplementary Models (`NeonTest/23288_.../Neon_SM/`)

External model repositories from the NEON paper's supplementary material:
- VAR (Visual Autoregressive) -- ImageNet 256 and 512
- xAR (Extended Autoregressive)
- IMM (Image-to-Image Matching)

These are large external codebases, not part of this project's experiments.

## Running

```bash
PYTHON=/home/patrick/miniconda3/bin/python

# GAN NEON Protocol A
$PYTHON FineTuning/GAN_NEON/protocol_a_neon.py

# Controlled collapse NEON (see neon_controlled_collapse.md)
$PYTHON NeonTest/ThomasExample/neon_controlled_collapse_gan.py
```

## Dependencies

All NEON scripts require the Gen 0 checkpoints from the full synthetic experiment:
- `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/generator_0.pth`
- `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/discriminator_0.pth`
- `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/lenet_mnist.pth`

## Outputs

GAN NEON results saved to `FineTuning/protocol_a_results/`:
- `protocol_a_results.pkl` -- Full results dictionary
- `protocol_a_analysis.png` -- 4-panel analysis plot
- `checkpoints/theta_r.pth` -- Base model state
- `checkpoints/theta_s_final.pth` -- Fine-tuned model state
- `checkpoints/theta_s_epoch*.pth` -- Per-epoch fine-tuning checkpoints
- `samples/` -- Sample images at different w values
