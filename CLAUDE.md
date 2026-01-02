# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAD-Detection is a research project investigating **Model Autophagy Disorder (MADness)** in generative models. The core concept: when generative models are iteratively trained on their own synthetic outputs, they exhibit mode collapse and quality degradation over successive generations.

The repository contains three experimental paradigms:
- **FullSynthetic**: Models trained exclusively on synthetic data from previous generations
- **AugmentedSynthetic**: Models trained on a mix of real and synthetic data
- **Fresh**: Control experiments with fresh models

Primary model architectures tested:
- DDPM (Denoising Diffusion Probabilistic Models) on MNIST
- Normalizing Flows (Julia implementation)
- StyleGAN2 and WGAN variants

## Repository Structure

```
MAD-Detection/
├── FullSynthetic/        # Pure synthetic training experiments
│   ├── DDPM-MNIST/       # Main DDPM experiments (primary focus)
│   ├── NF/               # Normalizing Flow experiments (Julia)
│   ├── stylegan2/
│   ├── WGAN/
│   └── GMM/
├── AugmentedSynthetic/   # Mixed real+synthetic training
│   └── DDPM-MNIST/
├── Fresh/                # Control experiments
│   └── MNIST-DDPM/
└── data/                 # Datasets and outputs
```

## Common Commands

### DDPM-MNIST (Python)

**Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# For CUDA 12.x specifically:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Run main experiment (20 generations):**
```bash
cd FullSynthetic/DDPM-MNIST
python main.py
```

**Compute Jacobian spectrum for MAD detection:**
```bash
cd FullSynthetic/DDPM-MNIST
python spectrumcompute.py
```

**Compute FID scores:**
```bash
cd FullSynthetic/DDPM-MNIST
python compute_fid.py
```

**View/compare generated samples:**
```bash
cd FullSynthetic/DDPM-MNIST
python view_samples.py
python image_comparison.py
```

### Normalizing Flows (Julia)

**Setup:**
```bash
# Ensure Python matplotlib/seaborn installed first
pip install matplotlib seaborn

# Setup Julia environment
export PYTHON=$(which python3)
julia -e 'using Pkg; Pkg.add("DrWatson"); Pkg.add("PyCall"); Pkg.build("PyCall")'
julia --project -e 'using Pkg; Pkg.instantiate()'
```

**Run experiments:**
```bash
cd FullSynthetic/NF

# Train on 2D toy dataset for 10 madcow iterations
julia --project scripts/train_rosenbrock_madcow.jl --madcow 10

# Plot results
julia --project scripts/test_rosenbrock_madcow.jl --madcow 10
```

## Architecture Details

### DDPM Training Pipeline (main.py)

Each generation performs a complete cycle:

1. **Data Loading** (`metrics.load_datasets`):
   - Generation 0: Uses pre-generated MNIST data from `gen_data_with_w_initial_w0`
   - Generation N>0: Loads synthetic data from generation N-1

2. **Model Training**:
   - Architecture: `ContextUnet` (U-Net with context embedding)
   - Training: 40 epochs, batch size 128
   - Optimizer: Adam with linear learning rate decay (1e-4 → 0.2e-4)
   - Loss: MSE between predicted and actual noise
   - Timesteps: n_T = 500 denoising steps

3. **Sample Generation**:
   - 60,000 samples for next generation training (`N_next`)
   - 1,000 samples for evaluation (`N_eval`)
   - Guidance weight w=0 (classifier-free guidance disabled)

4. **Evaluation Metrics**:
   - **FID**: Fréchet Inception Distance (uses pre-trained LeNet features)
   - **Precision/Recall**: Distribution matching quality
   - **Density/Coverage**: Mode coverage analysis
   - **t-SNE visualization**: 2D projection comparing real vs synthetic

5. **Outputs** (saved to `./data/diffusion_outputs10/`):
   - `model_{generation}_w0.pth`: Trained model checkpoint
   - `gen_data_with_w{generation}_w0`: 60K synthetic images for next gen
   - `gen_data_without_w{generation}_w0`: 1K evaluation samples
   - `All-genration={generation}w=0.png`: t-SNE visualization

### Jacobian Spectrum Analysis (spectrumcompute.py)

Novel MADness detection approach that isolates model behavior from data distribution:

**Key Innovation**: All models evaluated on **fixed real MNIST anchor points** (not their own generated data). This measures how model sensitivity changes across generations, independent of data drift.

**Method**:
1. Load model checkpoint for each generation
2. Use 20 fixed anchor points from real MNIST (generation 0)
3. Compute top-5 eigenvalues of J^T J where J is Jacobian of denoiser
4. Uses power iteration with deflation for eigenvalue computation

**Interpretation**:
- Decreasing eigenvalues → model less sensitive to input diversity → mode collapse
- Increasing condition number → numerical instability

**Multi-GPU execution**: Processes generations in parallel across 4 GPUs using multiprocessing.

### Feature Extraction for Metrics

Uses a pre-trained LeNet model (`prmodel.pth`) to extract 84-dimensional features from images. This is used for:
- FID computation (comparing feature distribution statistics)
- PRDC metrics (manifold distance calculations)

**Important**: The LeNet model is expected at `./data/diffusion_outputs10/prmodel.pth`. This must exist before running experiments.

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
- Minimum 8GB VRAM recommended per GPU

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
4. **Jacobian eigenvalue spectrum shifts** (novel contribution)

The Jacobian analysis (spectrumcompute.py) is unique because it measures intrinsic model behavior, not data quality.
