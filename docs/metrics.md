# Metrics

## Effective Rank (ER)

The core metric of the paper. Measures how many directions of the generator's Jacobian carry significant energy.

### Definition

Given a generator `G(z)` and a latent vector `z`:

1. Compute the Jacobian: `J = dG(z)/dz` (shape: [784, 100] for MNIST)
2. Compute singular values: `s = svd(J).S`
3. Normalize to a distribution: `p_i = s_i / sum(s)`
4. Compute Shannon entropy: `H = -sum(p_i * log(p_i))`
5. Effective rank: `ER = exp(H)`

### Code

```python
import torch

def effective_rank(J, eps=1e-10):
    """Compute effective rank of Jacobian matrix."""
    s = torch.linalg.svdvals(J)
    p = s / s.sum()
    entropy = -torch.sum(p * torch.log(p + eps))
    return torch.exp(entropy)
```

### Interpretation

- **ER = 1**: Jacobian is rank-1; all variation collapses to a single direction
- **ER = 100**: All singular values are equal; maximum diversity (for 100-dim latent space)
- **Typical Gen 0**: ER ~ 4-9 (healthy generator)
- **Gen 7**: ER ~ 1-2.5 (collapsed generator)

### Averaging Protocol

ER varies across latent vectors, so the analysis scripts sample multiple points:
- `jacobian_gan_multi_avg.py`: Samples 20 latent vectors with seeds 100-119
- Reports mean +/- std of ER across samples
- Uses `model.eval()` for stable BatchNorm statistics

### Note on Eigenvalues vs Singular Values

The codebase sometimes refers to "eigenvalues" when computing `S.pow(2)` -- these are the squared singular values of the Jacobian (equivalent to eigenvalues of `J^T J`). The ER formula uses singular values directly (not squared).

## FID (Frechet Inception Distance)

Measures the distance between real and generated image distributions in feature space.

### MNIST-specific Implementation

Unlike standard FID which uses Inception-v3 (2048-dim features), this codebase uses a **LeNet classifier trained on MNIST** with 84-dimensional features from the penultimate layer.

### LeNet Architecture

```
Conv2d(1, 6, 5, padding=2) -> ReLU -> MaxPool2d(2)
Conv2d(6, 16, 5) -> ReLU -> MaxPool2d(2)
Linear(16*5*5, 120) -> ReLU
Linear(120, 84) -> ReLU          # Features extracted here (84-dim)
Linear(84, 10)                    # Classification head (not used for FID)
```

### FID Formula

```
FID = ||mu_real - mu_gen||^2 + Tr(Sigma_real + Sigma_gen - 2*(Sigma_real @ Sigma_gen)^(1/2))
```

Where `mu` and `Sigma` are the mean and covariance of the feature distributions.

### Source Files

- `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/compute_fid.py` -- FID for full synthetic experiment
- `AugmentedSynthetic/GAN-MNIST/compute_fid.py` -- FID for augmented experiments
- LeNet weights saved at `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/data/gan_outputs/lenet_mnist.pth`

## Jacobian Computation

### Method

Uses `torch.autograd.functional.jacobian()`:

```python
import torch.autograd.functional as F

def gen_func(z_input):
    z_shaped = z_input.reshape(1, latent_dim)
    return generator(z_shaped, class_label).flatten()

J = F.jacobian(gen_func, z_flat)  # [784, 100]
```

### Multi-GPU Considerations

- Jacobian computation must run on a single device
- Extract `.module` from `DataParallel` wrapper before computing
- Use `model.eval()` for stable BatchNorm running statistics

### Analysis Scripts

Each experiment directory contains Jacobian analysis scripts:

| Script | Description |
|--------|-------------|
| `jacobian_gan.py` | Single latent vector analysis |
| `jacobian_gan_multi.py` | Multiple vectors with different seeds |
| `jacobian_gan_multi_avg.py` | Averaged metrics (most reliable) |
| `jacobian_gan_single_class.py` | Per-class analysis |
