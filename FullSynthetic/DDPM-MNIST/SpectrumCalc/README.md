# DDIM Jacobian Spectrum Analysis

This directory contains scripts for analyzing Model Autophagy Disorder (MADness) through Jacobian eigenvalue spectrum analysis using deterministic DDIM sampling.

## Overview

The core approach: Compute the Jacobian (output)/(input) of the full 500-step DDIM generation process to measure model sensitivity across generations. Eigenvalue collapse indicates loss of generative diversity and rank degradation.

## Key Scripts

### 1. `jacobian_ddim.py` - Compute Jacobian Eigenvalues

**Purpose**: Computes the full 500-step DDIM Jacobian for multiple model generations in parallel.

**What it does**:
- Loads model checkpoints (`model_initial.pth`, `model_0_w0.pth`, ..., `model_5_w0.pth`)
- Uses a **fixed initial noise** (seed 42) for all generations
- Computes Jacobian via automatic differentiation through 500 deterministic DDIM steps
- Performs SVD to extract eigenvalues: `U, S, V = torch.linalg.svd(jacobian)`
- Saves results to `jacobian_ddim_results.pkl`

**Usage**:
```bash
python jacobian_ddim.py
```

**Requirements**:
- Multi-GPU setup (processes generations in parallel)
- Model checkpoints in `../data/diffusion_outputs10/`
- Takes ~15-20 minutes per generation

**Outputs**:
- `jacobian_ddim_results.pkl` - Eigenvalues, generated images, timing data
- `jacobian_ddim_spectrum.png` - Eigenvalue spectrum plots
- `jacobian_ddim_images.png` - Sample images from each generation

**Key insight**: DDIM is deterministic (no random noise), so the Jacobian measures pure model behavior without stochastic confounding.

---

### 2. `plot_ddim_results.py` - Visualize Precomputed Results

**Purpose**: Create plots from existing `jacobian_ddim_results.pkl` without recomputing Jacobians.

**What it does**:
- Loads the `.pkl` file
- Generates the same plots as `jacobian_ddim.py`
- Prints eigenvalue statistics (effective rank, min/max, etc.)

**Usage**:
```bash
python plot_ddim_results.py
```

**Requirements**:
- `jacobian_ddim_results.pkl` (from running `jacobian_ddim.py`)

**Outputs**:
- `jacobian_ddim_spectrum.png` - Left: eigenvalue decay curves, Right: log-eigenvalue distribution
- `jacobian_ddim_images.png` - Generated digit samples

**Use case**: Fast iteration on visualizations without 15+ minute recomputation. Perfect for sharing results.

---

### 3. `plot_explained_variance.py` - Variance Analysis

**Purpose**: Compute cumulative explained variance and effective dimensionality from eigenvalues.

**What it does**:
- Loads eigenvalues from `jacobian_ddim_results.pkl`
- Computes cumulative variance: how many components needed for 90%, 95%, 99% variance
- Plots cumulative variance curves and bar charts

**Usage**:
```bash
python plot_explained_variance.py
```

**Outputs**:
- `explained_variance_ddim.png` - Two subplots:
  - Left: Cumulative variance curves (shows how quickly variance accumulates)
  - Right: Bar chart of components needed for variance thresholds
- Terminal table showing exact component counts

**Key metrics**:
- **90% variance**: Top principal components capturing bulk of sensitivity
- **99% variance**: Total effective dimensionality (how many dimensions have meaningful sensitivity)

**Interpretation**:
- **Healthy model**: Long tail - variance distributed across many components
- **Collapsed model**: Short tail - variance concentrated in few components, rest are dead

---

### 4. `generate_images_gen0.py` - Generate Sample Images

**Purpose**: Generate DDIM samples from each model checkpoint for visual comparison.

**What it does**:
- Loads models for Initial, Gen 0-5
- Generates 10 samples (digits 0-9) using deterministic DDIM
- Creates grid showing samples across generations

**Usage**:
```bash
python generate_images_gen0.py
```

**Outputs**:
- `ddim_samples_gen0_to_5.png` - 7×10 grid (7 generations × 10 digits)

**Use case**: Visual inspection of image quality degradation across MADness generations.

---

## Typical Workflow

### First-time setup:
1. Ensure model checkpoints exist in `../data/diffusion_outputs10/`
2. Run `jacobian_ddim.py` to compute eigenvalues (~2 hours for 7 generations)
3. Run `plot_explained_variance.py` for variance analysis
4. Run `generate_images_gen0.py` for visual samples

### Working with existing results:
1. Use `plot_ddim_results.py` to regenerate plots from `.pkl` file
2. Modify plotting code and re-run (no recomputation needed)

---

## Key Results

**Expected eigenvalue behavior**:

- **Initial model** (trained on real MNIST):
  - Wide eigenvalue distribution (10^-10 to 10^2)
  - Long tail extending across 700+ dimensions
  - 99% variance explained by ~10 components (very efficient)
  - BUT maintains sensitivity across nearly all 784 dimensions

- **Gen 0** (first synthetic training):
  - Moderate collapse
  - 99% variance explained by ~480 components
  - Beginning of tail degradation

- **Gen 1-5** (MADness regime):
  - Severe collapse - eigenvalues nearly identical across generations
  - 99% variance explained by only ~25-34 components
  - Tail completely dead by component 100-200
  - Lost 95%+ of initial model's effective dimensionality

**MADness signature**: Dramatic reduction in effective rank (number of meaningful dimensions) as models are iteratively trained on synthetic data.

---

## File Dependencies

**Model checkpoints** (in `../data/diffusion_outputs10/`):
- `model_initial.pth` - Trained on real MNIST
- `model_0_w0.pth` through `model_N_w0.pth` - Successive synthetic generations

**Generated data**:
- `jacobian_ddim_results.pkl` - Precomputed eigenvalues and images
- Various `.png` plot files

**Code dependencies**:
- `../metrics.py` - DDPM class, ContextUnet, sampling utilities
- PyTorch, NumPy, Matplotlib, Pickle

---

## Technical Details

### DDIM Sampling
Uses deterministic DDIM formula:
```python
x_0_pred = (x - sqrt(1-±_t) * eps) / sqrt(±_t)
x_{t-1} = sqrt(±_{t-1}) * x_0_pred + sqrt(1-±_{t-1}) * eps
```

No random noise - completely deterministic trajectory from initial noise to final image.

### Jacobian Computation
- Input: 784-dimensional initial noise (28×28 flattened)
- Output: 784-dimensional final image (28×28 flattened)
- Jacobian: 784×784 matrix computed via autograd
- SVD: `U, S, V = torch.linalg.svd(jacobian)`
  - S: Singular values (square root of eigenvalues)
  - V: Right singular vectors (directions in noise space)
  - U: Left singular vectors (directions in image space)

### Effective Rank
Entropy-based measure:
```python
p_i = eigenvalue_i / sum(eigenvalues)
effective_rank = exp(-sum(p_i * log(p_i)))
```

Captures "number of meaningful dimensions" - more robust than hard thresholding.

---

## Notes

- **Fixed noise**: All generations use the same initial noise (seed 42) for fair comparison
- **Class label**: Analysis uses class 7 by default
- **GPU memory**: Each Jacobian computation requires ~8GB VRAM
- **Parallelization**: Multi-GPU support for faster computation
