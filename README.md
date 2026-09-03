# A Geometric Perspective on Recursive Synthetic Training

Patrick Batsell, Thomas Walker, Richard Baraniuk (Rice University)

Published at the 2nd DeLTa Workshop, ICLR 2026. [[paper]](paper/A_Geometric_Perspective_on_Recursive_Synthetic_Training.pdf)

Training a deep generative network on its own samples (Model Autophagy Disorder, "MADness") shows up in the local geometry of the generator: the input-output Jacobians lose effective rank and their singular vectors become noisy. This repo contains the code for the small-scale experiments in the paper, plus the NEON experiments that use spectrum-collapsed synthetic data as a negative-guidance direction.

## Layout

| Paper | Experiment | Code |
|---|---|---|
| Fig. 2, App. A.2 | MNIST conditional GAN, fully synthetic loop (8 generations) | `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/` |
| Fig. 6, App. A.4 | Same loop on FashionMNIST | `FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/` |
| Fig. 7, App. A.5 | Augmentation loop (25% real / 75% synthetic) | `AugmentedSynthetic/` |
| Fig. 4a, 8, 9, App. A.6, B | 5-D Gaussian VAE, NEON with spectrum collapse (alpha sweep) | `NeonTest/ThomasExample/madness_neon.ipynb` |
| | GAN-MNIST port of the controlled-collapse NEON experiment | `NeonTest/ThomasExample/neon_controlled_collapse_gan.py` |
| | NEON (Protocol A) on the MNIST GAN and on a DDIM | `FineTuning/` |

Effective rank and FID (LeNet features) are computed by the `jacobian_*.py` and `compute_fid.py` scripts next to each experiment. `FullSynthetic/create_publication_figures.py` turns the saved `.pkl` results into the paper plots.

Not included here: the 2-D circle VAE (Fig. 1, 5), the BigGAN CIFAR-10 loop (Fig. 3, base model from [ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)), and the spectrum-flattening fine-tuning sweep (Fig. 4b).

## Setup

```bash
pip install -r requirements.txt
```

MNIST and FashionMNIST download automatically on first run.

## Running

```bash
# MNIST synthetic loop (Fig. 2): train 8 generations, then Jacobian + FID analysis
python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/main.py
python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/jacobian_gan_multi_avg.py
python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/compute_fid.py

# Augmentation loop (Fig. 7)
python AugmentedSynthetic/GAN-MNIST/main_constant_proportion.py --proportion 0.75 --generations 10

# Controlled-collapse NEON on the MNIST GAN (needs the generation-0 checkpoint from above)
python NeonTest/ThomasExample/neon_controlled_collapse_gan.py
```

The effective rank of a Jacobian `J` is `exp(H(s / s.sum()))` where `s = svdvals(J)` and `H` is the Shannon entropy. The spectrum collapse used to build the NEON fine-tuning set is, for `alpha` in [0, 1],

```
s_1 <- sqrt((1 - alpha) s_1^2 + alpha * sum_i s_i^2)
s_i <- sqrt(1 - alpha) s_i          (i >= 2)
```

More detail on each experiment is in `docs/`.

## Citation

```bibtex
@inproceedings{batsell2026geometric,
  title     = {A Geometric Perspective on Recursive Synthetic Training},
  author    = {Batsell, Patrick and Walker, Thomas and Baraniuk, Richard},
  booktitle = {2nd Workshop on Deep Learning Theory and Applications (DeLTa), ICLR},
  year      = {2026}
}
```
