# DDIM Jacobian Spectrum Analysis

This directory contains scripts for analyzing Model Autophagy Disorder (MADness) through Jacobian eigenvalue spectrum analysis using DDIM sampling.

## Overview

The core approach: Compute the Jacobian of the full 500-step DDIM generation process to measure model sensitivity across generations. 

## Key Scripts

### 1. `jacobian_ddim.py` - Compute Jacobian Eigenvalues

**Purpose**: Computes the full 500-step DDIM Jacobian for multiple model generations in parallel.

**What it does**:
- Loads model checkpoints (`model_initial.pth`, `model_0_w0.pth`, `model_5_w0.pth`)
- Uses a **fixed initial noise** (seed 42) for all generations
- Computes Jacobian via automatic differentiation through 500 deterministic DDIM steps
- Performs SVD to extract eigenvalues: `U, S, V = torch.linalg.svd(jacobian)`
- Saves results to `jacobian_ddim_results.pkl`

**Usage**:
```bash
python jacobian_ddim.py
```


**Outputs**:
- `jacobian_ddim_results.pkl` - Eigenvalues, generated images, timing data
- `jacobian_ddim_spectrum.png` - Eigenvalue spectrum plots
- `jacobian_ddim_images.png` - Sample images from each generation


---

### 2. `plot_ddim_results.py` - Visualize Precomputed Results

**Purpose**: Create plots from existing `jacobian_ddim_results.pkl` without recomputing Jacobians.


**Usage**:
```bash
python plot_ddim_results.py
```

**Requirements**:
- `jacobian_ddim_results.pkl` (from running `jacobian_ddim.py`)

**Outputs**:
- `jacobian_ddim_spectrum.png` - Left: eigenvalue decay curves, Right: log-eigenvalue distribution
- `jacobian_ddim_images.png` - Generated digit samples


---

### 3. `imagegen.py` - Generate Sample Images

**Purpose**: Generate DDIM samples from each model checkpoint for visual comparison.

**What it does**:
- Loads models for Initial, Gen 0-5
- Generates 10 samples (digits 0-9) using deterministic DDIM
- Creates grid showing samples across generations

**Usage**:
```bash
python imagegen.py
```

**Outputs**:
- `ddim_samples_gen0_to_5.png` - 7x10 grid (7 generations x 10 digits)

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
x_0_pred = (x - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
x_{t-1} = sqrt(alpha_{t-1}) * x_0_pred + sqrt(1-alpha_{t-1}) * eps
```

### Jacobian Computation
- Input: 784-dimensional initial noise (28x28 flattened)
- Output: 784-dimensional final image (28x28 flattened)
- Jacobian: 784x784 matrix computed via autograd
- SVD: `U, S, V = torch.linalg.svd(jacobian)`






