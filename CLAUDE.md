# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **MADness (Model Autophagy Disorder) Detection** research project that studies how synthetic data quality degrades when generative models are iteratively trained on their own outputs. The primary goal is to detect MADness onset by monitoring the **spectrum of Jacobian outer products** across training generations.

## Environment Setup

### Activation
```bash
# Windows Command Prompt
activate.bat

# PowerShell
.\activate.ps1

# Manual
venv\Scripts\activate
```

### Verify GPU and Dependencies
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi
```

## Core Architecture

### Three Training Scenarios

The codebase contains three experimental scenarios for studying synthetic data degradation:

**1. FullSynthetic** (Primary focus for Jacobian analysis)
- Location: `FullSynthetic/DDPM-MNIST/`
- Pure synthetic-to-synthetic training loop
- Each generation (after Gen 0) trains only on previous generation's synthetic output
- Fast: 1 epoch per generation, 20 generations total (~2-3 hours)
- Maximum MADness expected

**2. Fresh**
- Location: `Fresh/MNIST-DDPM/`
- Mixes 5k synthetic + 1k NEW real data per generation
- Slow: 1000 epochs per generation, 20 generations (~8 days)
- Tests if fresh data injection prevents MADness

**3. AugmentedSynthetic**
- Location: `AugmentedSynthetic/DDPM-MNIST/`
- Heavy augmentation: 60k real + 60k synthetic per generation
- Moderate: 60 epochs per generation, 10 generations (~5 days)
- Tests maximum real data augmentation

### Two-Stage Training Process

**Stage 1: Initial Data Generation**
```bash
cd FullSynthetic/DDPM-MNIST
python initialmain.py
```
- Trains DDPM on real MNIST (40 epochs)
- Generates initial synthetic datasets for multiple guidance weights (w ∈ [0, 0.25, ..., 2])
- Saves to `./data/t500/gen_data_with_w_initial_w{w}` and `model_initial.pth`
- Takes ~1-2 hours

**Stage 2: Generational Training Loop**
```bash
python main.py
```
- Runs 20 generations of training
- Each generation:
  1. Loads data from previous generation (or initial data for Gen 0)
  2. Trains DDPM for 1 epoch
  3. Generates synthetic data for next generation
  4. Computes FID, PRDC metrics
  5. Creates t-SNE visualization
  6. Saves checkpoint: `model_{generation}_w0.pth`
- Takes ~2-3 hours total

### Key Files and Their Roles

**Training Scripts:**
- `initialmain.py` - Generate initial synthetic data (run first)
- `main.py` - Single-condition FullSynthetic training (fast: 1 epoch/gen, ~2-3 hours)
- `mainall.py` - Multi-condition comparative training (slow: 40 epochs/gen, ~30-40 hours)
- `metrics.py` - Shared utilities for DDPM model, data loading, metrics (FID, PRDC)

**Script Comparison:**
- **main.py**: Runs ONE 20-generation experiment with w=0 (single condition, fast)
- **mainall.py**: Runs MULTIPLE 20-generation experiments (one per w value in Wall) to compare guidance effects (slow)
- For Jacobian analysis, use **main.py** only

**Configuration in initialmain.py:**
```python
N_with_w = 60000        # Number of synthetic samples to generate
max_c = 500             # Batch size for generation
Wall = [0]              # Guidance weights to generate datasets for
                        # [0] = only w=0 (60k images, for main.py)
                        # [0,0.25,...,2] = all w values (540k images, for mainall.py)
n_epoch = 40            # Training epochs for initial model
```

**Configuration in main.py:**
```python
N_eval = 1000           # Samples for evaluation metrics
N_next = 10             # Samples for next generation training
generation_number = 20  # Total generations
n_epoch = 1             # Epochs per generation (fast)
n_T = 500               # DDPM diffusion timesteps
w = 0                   # Single fixed guidance weight
save_dir = './data/diffusion_outputs10/'  # Output directory
```

**Configuration in mainall.py:**
```python
Wall = [1.5, 1.75, 2]   # Multiple guidance weights to compare
generation_number = 20  # Generations PER w value (60 total)
n_epoch = 40            # Epochs per generation (slow, for quality)
save_dir = './data/t500/'  # Different output directory
```

## Understanding the Guidance Weight (w)

The `w` parameter controls **how strongly the DDPM follows class labels during generation**:
- `w=0`: No guidance (unconditional generation)
- `w>0`: Increasing strength of class-conditional guidance
- `w=2`: Strong guidance (very class-specific samples)

**Key Points:**
- `w` affects GENERATION, not training (training always uses guide_w=0)
- Different `w` values produce different "styles" of synthetic data
- `main.py` only uses w=0 for the entire experiment
- `mainall.py` compares multiple w values to study their effect on MADness
- If `Wall = [0, 0.25, 0.5]` in initialmain.py, it generates 60k images × 3 = 180k total images

## Critical Windows-Specific Fix

**IMPORTANT:** On Windows, PyTorch DataLoader multiprocessing requires `num_workers=0` to avoid spawn errors.

All DataLoader instances must use:
```python
DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
```

If you see `RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly`, search for `num_workers=5` and change to `num_workers=0`.

## Model Architecture

The core model is a **DDPM (Denoising Diffusion Probabilistic Model)** with:
- `ContextUnet` backbone (defined in metrics.py)
- Class-conditional generation (10 MNIST classes)
- 500 diffusion timesteps
- Trained with MSE loss between predicted and actual noise

## Data Flow

```
Real MNIST (60k images)
    ↓
initialmain.py trains DDPM
    ↓
Generate synthetic images (w=0, 0.25, ..., 2)
    ↓
Save: gen_data_with_w_initial_w{w}
    ↓
main.py Gen 0: Load initial synthetic data
    ↓
Train 1 epoch → Generate new synthetic
    ↓
main.py Gen 1: Load Gen 0 synthetic
    ↓
Train 1 epoch → Generate new synthetic
    ↓
... (repeat 20 times)
    ↓
Save checkpoints: model_0_w0.pth ... model_19_w0.pth
```

## Checkpoints and Outputs

**Model Checkpoints (for Jacobian analysis):**
- `./data/diffusion_outputs10/model_{0-19}_w0.pth` - One per generation
- These contain the trained DDPM weights (state_dict)
- Used later to compute Jacobians via autograd

**Generated Data:**
- `gen_data_with_w{generation}_w{w}` - Synthetic images (tensors)
- `gen_index_with_w{generation}_w{w}` - Class labels

**Metrics:**
- FID (Fréchet Inception Distance)
- PRDC (Precision, Recall, Density, Coverage)
- t-SNE visualizations saved as PNG

## Typical Workflow for Jacobian Analysis

1. **Generate initial data:**
   ```bash
   cd FullSynthetic/DDPM-MNIST
   python initialmain.py  # ~1-2 hours
   ```

2. **Run generational training:**
   ```bash
   python main.py  # ~2-3 hours, creates 20 checkpoints
   ```

3. **Compute Jacobians (future work):**
   - Load each checkpoint: `torch.load('model_{g}_w0.pth')`
   - For fixed samples, compute `J = ∂ε_θ(x_t, t)/∂x_t`
   - Analyze eigenvalue spectrum of `J^T J` across generations
   - Compare Gen 0 (baseline) vs Gen 19 (maximum MADness)

## Metrics Computation

**FID (Fréchet Inception Distance):**
```python
# Extract features using LeNet (trained on MNIST)
real_features = extract_mnist_features(real_data, device)
gen_features = extract_mnist_features(synthetic_data, device)

# Compute FID
mu_real, cov_real = mean(real_features), cov(real_features)
mu_gen, cov_gen = mean(gen_features), cov(gen_features)
FID = ||mu_real - mu_gen||² + Tr(cov_real + cov_gen - 2√(cov_real·cov_gen))
```

**PRDC (Precision, Recall, Density, Coverage):**
- Implemented in `compute_prdc_slice()` in metrics.py
- Uses k-NN distances to measure manifold overlap

## Common Issues and Solutions

**"RuntimeError: DataLoader worker exited unexpectedly"**
- Set `num_workers=0` in all DataLoader calls
- Required on Windows due to multiprocessing spawn behavior

**"FileNotFoundError: gen_data_with_w-1_w0"**
- Must run `initialmain.py` before `main.py`
- Gen 0 loads initial synthetic data from initialmain.py

**"FileNotFoundError: prmodel.pth"**
- Run scripts from correct directory: `cd FullSynthetic/DDPM-MNIST`
- Scripts use relative paths like `./data/` which only work from script directory
- DO NOT run from project root (MadCode/)

**Scripts running for 12+ hours**
- Check `Wall` in initialmain.py - each w value generates 60k images
- For main.py, only need `Wall = [0]` (generates 60k images, ~2-3 hours)
- Multiple w values multiply generation time: `[0,0.25,...,2]` = 540k images

**"Model training when I expected generation"**
- initialmain.py checks for existing model_initial.pth
- If model exists, loads it and skips to generation
- If model doesn't exist, trains from scratch (40 epochs)

**GPU not detected**
- Verify: `nvidia-smi` shows GPU
- Verify: `python -c "import torch; print('CUDA:', torch.cuda.is_available())"`
- Ensure CUDA 12.6 drivers installed

## Research Context

This codebase implements experiments from MADness research studying:
- How synthetic data degrades across generations (FullSynthetic)
- Whether fresh data injection prevents degradation (Fresh)
- Whether heavy augmentation helps (AugmentedSynthetic)

The primary innovation for this project is using **Jacobian spectrum analysis** to detect MADness earlier than global metrics like FID.
