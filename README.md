# A Geometric Perspective on Recursive Synthetic Training

Patrick Batsell, Thomas Walker, Richard Baraniuk (Rice University)

2nd DeLTa Workshop, ICLR 2026. [[paper]](paper/A_Geometric_Perspective_on_Recursive_Synthetic_Training.pdf)

Code for the experiments in the paper: generative models retrained on their own samples, the effective rank of their Jacobians across generations, and NEON with spectrum-collapsed synthetic data.

## Contents

- `FullSynthetic/` MNIST and FashionMNIST GAN synthetic loops
- `AugmentedSynthetic/` real + synthetic mixing loops
- `NeonTest/` NEON with spectrum-collapsed synthetic data (VAE notebook and MNIST GAN script)
- `FineTuning/` NEON fine-tuning on the MNIST GAN and a DDIM
- `docs/` experiment details

The 2-D VAE and BigGAN CIFAR-10 experiments are not included.

## Setup

```bash
pip install -r requirements.txt
```

MNIST and FashionMNIST download automatically.

## Running

```bash
python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/main.py
python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/jacobian_gan_multi_avg.py
python FullSynthetic/GAN-MNIST/GAN-MNIST-Default/compute_fid.py
python NeonTest/ThomasExample/neon_controlled_collapse_gan.py
```

## Citation

```bibtex
@inproceedings{batsell2026geometric,
  title     = {A Geometric Perspective on Recursive Synthetic Training},
  author    = {Batsell, Patrick and Walker, Thomas and Baraniuk, Richard},
  booktitle = {2nd Workshop on Deep Learning Theory and Applications (DeLTa), ICLR},
  year      = {2026}
}
```
