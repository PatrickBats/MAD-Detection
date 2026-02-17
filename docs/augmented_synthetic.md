# Augmented Synthetic Experiments (Figure 6)

## Overview

Tests whether including real data alongside synthetic data prevents MADness collapse. The key question: can access to real data mitigate ER degradation?

**Directory**: `AugmentedSynthetic/GAN-MNIST/` and `AugmentedSynthetic/GAN-MNIST-PaperStyle/`

## Experiment Variants

### 1. Accumulating Augmentation (`main.py`)

Each generation trains on all real data plus all accumulated synthetic data:

```
Gen 1: REAL (60k)
Gen 2: REAL (60k) + SYNTHETIC_Gen1 (60k) = 120k
Gen 3: REAL (60k) + SYNTHETIC_Gen1-2 (120k) = 180k
...
Gen t: REAL (60k) + ALL SYNTHETIC from Gen 1..t-1
```

The training set grows each generation. Real data is always present.

### 2. Constant Proportion (`main_constant_proportion.py`)

Total training samples stays fixed at 60k. Each generation uses `p%` synthetic + `(1-p)%` real:

| Proportion | Real | Synthetic |
|-----------|------|-----------|
| p=0.0 | 60k | 0k (baseline) |
| p=0.25 | 45k | 15k |
| p=0.50 | 30k | 30k |
| p=0.75 | 15k | 45k |
| p=0.90 | 6k | 54k |
| p=1.0 | 0k | 60k (full synthetic) |

Synthetic data comes from the **previous generation only**.

### 3. Progressive Decay (`main_progressive_decay.py`)

Synthetic influence decays with age: older synthetic data contributes less to the training set.

### 4. Paper-Style 50/50 (`GAN-MNIST-PaperStyle/main.py`)

Fixed 50% real / 50% synthetic split matching the paper's protocol.

## Key Results

- Real data **slows** but does not **prevent** ER collapse
- At 75% synthetic / 25% real: ER drops from ~3.64 to ~1.99 over 9 generations
- Full synthetic (100%): ER collapses much faster
- Even 50/50 mixing shows gradual degradation

## Scripts

### `AugmentedSynthetic/GAN-MNIST/`

| Script | Purpose |
|--------|---------|
| `main.py` | Accumulating augmentation loop |
| `main_constant_proportion.py` | Fixed ratio real:synthetic |
| `main_progressive_decay.py` | Decaying synthetic influence |
| `run_constant_proportion_experiments.py` | Run sweep over proportions |
| `compare_proportions.py` | Compare results across proportions |
| `jacobian_gan_avg.py` | Jacobian ER analysis (accumulating) |
| `jacobian_constant_proportion.py` | Jacobian ER for constant proportion |
| `jacobian_gan_single_class.py` | Per-class Jacobian analysis |
| `jacobian_p75_single_class.py` | Jacobian for p=0.75 config |
| `compute_fid.py` | FID for accumulating augmentation |
| `compute_fid_constant_proportion.py` | FID for constant proportion |
| `plot_jacobian_proportions.py` | Plot ER across proportions |
| `plot_er_scatter_75pct.py` | ER scatter for 75% synthetic |
| `plot_p75_jacobian.py` | Detailed p=0.75 Jacobian plots |
| `visualize_low_high_ER_samples.py` | Visualize by ER |

### `AugmentedSynthetic/GAN-MNIST-PaperStyle/`

| Script | Purpose |
|--------|---------|
| `main.py` | Paper-style 50/50 augmentation |
| `compute_fid.py` | FID computation |
| `jacobian_gan_single_class.py` | Jacobian analysis |
| `visualize_low_high_ER_samples.py` | Visualize by ER |

## Running

```bash
PYTHON=/home/patrick/miniconda3/bin/python

# Accumulating augmentation (8 generations)
$PYTHON AugmentedSynthetic/GAN-MNIST/main.py

# Constant proportion sweep
$PYTHON AugmentedSynthetic/GAN-MNIST/run_constant_proportion_experiments.py

# Or a single proportion (e.g., 75% synthetic)
$PYTHON AugmentedSynthetic/GAN-MNIST/main_constant_proportion.py --proportion 0.75 --generations 10

# Paper-style 50/50
$PYTHON AugmentedSynthetic/GAN-MNIST-PaperStyle/main.py

# Analysis
$PYTHON AugmentedSynthetic/GAN-MNIST/jacobian_gan_avg.py
$PYTHON AugmentedSynthetic/GAN-MNIST/compute_fid.py
```

## Saved Artifacts

Same format as full synthetic experiments:
- `data/gan_outputs/generator_N.pth` -- Model checkpoints
- `data/gan_outputs/gen_data_N.pt` -- Synthetic images
- Constant proportion results saved in per-proportion subdirectories
