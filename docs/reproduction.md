# Reproduction Guide

Step-by-step instructions to reproduce all experiments and figures from the paper.

## Environment Setup

### Python

```bash
# Use the miniconda Python (system Python is 2.7)
export PYTHON=/home/patrick/miniconda3/bin/python
```

### Dependencies

```
PyTorch 2.0+
torchvision
numpy
matplotlib
tqdm
scikit-learn
scipy
Pillow
seaborn
pandas (for NEON experiments)
```

### GPU

8 GPUs available (indices 0-7). Most training scripts use `nn.DataParallel` with 2 GPUs by default. Jacobian analysis distributes across all available GPUs.

### Data

MNIST and FashionMNIST are auto-downloaded to `FullSynthetic/data/` on first run.

---

## Figure 2: Full Synthetic Loop (MNIST)

**Result**: ER drops from ~4.5 to ~1.0 over 8 generations; FID increases from ~3 to ~30.

```bash
# Step 1: Train 8 generations (~4 hours)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/main.py

# Step 2: Jacobian analysis (~2 hours on 8 GPUs)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/jacobian_gan_multi_avg.py

# Step 3: FID scores (~10 minutes)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Default/compute_fid.py

# Step 4: Publication figures
$PYTHON FullSynthetic/create_publication_figures.py
```

**Outputs**:
- Checkpoints: `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/`
- Figures: `FullSynthetic/figures/`

---

## Figure 6: Full Synthetic Loop (FashionMNIST)

**Result**: Similar ER decline pattern, demonstrating generality.

```bash
# Step 1: Train 8 generations (~4 hours)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/main.py

# Step 2: Jacobian analysis (~2 hours)
$PYTHON FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/jacobian_gan_multi_avg.py
```

**Outputs**: `FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/data/gan_outputs/`

---

## Figure 7: Augmented Synthetic (Real + Synthetic Mix)

**Result**: Real data slows but doesn't prevent ER collapse. At 75% synthetic: ER drops from ~3.64 to ~1.99 over 9 generations.

```bash
# Accumulating augmentation (8 generations, ~5 hours)
$PYTHON AugmentedSynthetic/GAN-MNIST/main.py

# Constant proportion sweep (multiple runs)
$PYTHON AugmentedSynthetic/GAN-MNIST/run_constant_proportion_experiments.py

# Or single proportion
$PYTHON AugmentedSynthetic/GAN-MNIST/main_constant_proportion.py --proportion 0.75 --generations 10

# Paper-style 50/50
$PYTHON AugmentedSynthetic/GAN-MNIST-PaperStyle/main.py

# Analysis
$PYTHON AugmentedSynthetic/GAN-MNIST/jacobian_gan_avg.py
$PYTHON AugmentedSynthetic/GAN-MNIST/compute_fid.py
$PYTHON AugmentedSynthetic/GAN-MNIST/plot_jacobian_proportions.py
```

**Outputs**: `AugmentedSynthetic/GAN-MNIST/data/gan_outputs/`

---

## Figures 4, 8, 9: Controlled Collapse + NEON

**Result**: alpha ~ 0.1 gives best NEON improvement (moderate collapse).

**Prerequisite**: Figure 2 must be run first (needs Gen 0 checkpoints).

```bash
# Controlled collapse experiment (~6 hours)
$PYTHON NeonTest/ThomasExample/neon_controlled_collapse_gan.py
```

**Outputs**: `NeonTest/ThomasExample/results_gan/`

---

## NEON Protocol A

**Prerequisite**: Figure 2 must be run first (needs Gen 0 checkpoints).

```bash
# GAN NEON (~2 hours)
$PYTHON FineTuning/GAN_NEON/protocol_a_neon.py
```

**Outputs**: `FineTuning/protocol_a_results/`

---

## Publication Figures

After running all experiments:

```bash
$PYTHON FullSynthetic/create_publication_figures.py
```

Generates PDF figures in `FullSynthetic/figures/`.

---

## Execution Order

If reproducing everything from scratch:

1. **Figure 2** (MNIST full synthetic) -- required first, provides Gen 0 checkpoints
2. **Figure 6** (FashionMNIST) -- independent, can run in parallel with #1
3. **Figure 7** (augmented synthetic) -- independent
4. **NEON Protocol A** -- depends on #1 (Gen 0 checkpoints)
5. **Figures 4, 8, 9** (controlled collapse) -- depends on #1 (Gen 0 checkpoints)
6. **Publication figures** -- depends on all above

Steps 2 and 3 can run in parallel with each other. Steps 4 and 5 can also run in parallel, but both depend on step 1.

## Notes

- All scripts use relative paths for data directories. Run from the repo root or from the script's directory.
- `n_gpus=2` is hardcoded in some scripts. Override in the source if needed.
- Jacobian analysis is the most compute-intensive step. It parallelizes across GPUs automatically.
- FID computation requires a trained LeNet model. If `lenet_mnist.pth` is not found, the FID scripts will warn but still run (with unreliable scores).
