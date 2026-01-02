# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAD-Detection is a research project investigating **Model Autophagy Disorder (MADness)** in generative models. The core concept: when generative models are iteratively trained on their own synthetic outputs, they exhibit mode collapse and quality degradation over successive generations.

### Model Architecture Details

**ContextUnet** (n_feat=128 default):
- Input: 28x28 grayscale MNIST images
- Conditioning: Class labels (0-9) + timestep embedding
- Structure: 2-layer downsampling, bottleneck, 2-layer upsampling
- Uses residual connections and GroupNorm
- Context dropout: 10% during training for classifier-free guidance

**DDPM Sampling**:
- Beta schedule: Linear from 1e-4 to 0.02
- Classifier-free guidance: Doubles batch, mixes conditional/unconditional predictions
- Default guidance weight w=0 (no guidance) for MADness experiments

## Key Files

- **metrics.py**: All utility functions, model architectures, DDPM class, PRDC computation, feature extraction
- **main.py**: Main training loop for iterative generations
- **spectrumcompute.py**: Jacobian eigenvalue analysis for MAD detection
- **compute_fid.py**: FID score computation across generations
- **image_comparison.py**: Visual comparison of samples across generations

## Development Notes

**CUDA/GPU Requirements**:
- Experiments designed for multi-GPU setup (uses `nn.DataParallel`)
- spectrumcompute.py uses 4 GPUs by default

**Data Persistence**:
- Generated datasets are saved as PyTorch tensors (`.pth` files or zipped)
- Images stored as uint8 (0-255) but normalized to [0,1] for training
- Each generation's data is ~2.3GB (60K images)

**Experiment Variants**:
- **mainall.py** (AugmentedSynthetic): Tests multiple guidance weights w=[0, 0.25, 0.5, 1]
- **initialmain.py**: Initial generation setup scripts
- **newinitial.py** (Fresh): Control experiments with fresh models

**Visualization**:
- t-SNE: 10,000 real + 10,000 synthetic samples
- Plots auto-display during training (must close window to continue)
- Use `view_samples.py` for post-hoc visualization

## MADness Detection Strategy

The core hypothesis tested: Iterative training on synthetic data causes **model autophagy disorder**.

**Primary detection signals**:
1. FID score increases over generations
2. PRDC metrics (especially Coverage, Recall) decline
3. t-SNE shows increasing separation between real/synthetic
4. **Jacobian eigenvalue spectrum shifts**
