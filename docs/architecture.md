# GAN Architecture

All experiments use the same class-conditional GAN architecture for MNIST/FashionMNIST.

## Generator

**Input**: Latent vector `z` (100-dim) + class label (10-dim embedding) = 110-dim

```
Linear(110, 128) -> LeakyReLU(0.2)
Linear(128, 256) -> BatchNorm1d(256) -> LeakyReLU(0.2)
Linear(256, 512) -> BatchNorm1d(512) -> LeakyReLU(0.2)
Linear(512, 1024) -> BatchNorm1d(1024) -> LeakyReLU(0.2)
Linear(1024, 784) -> Tanh()
```

- Output reshaped to `[batch, 1, 28, 28]`
- No BatchNorm on first layer
- BatchNorm momentum = 0.8

## Discriminator

**Input**: Flattened image (784-dim) + class label (10-dim embedding) = 794-dim

```
Linear(794, 512) -> LeakyReLU(0.2)
Linear(512, 512) -> Dropout(0.4) -> LeakyReLU(0.2)
Linear(512, 512) -> Dropout(0.4) -> LeakyReLU(0.2)
Linear(512, 1) -> Sigmoid()
```

## Class Conditioning

Both generator and discriminator use `nn.Embedding(10, 10)` for class conditioning. The embedding is concatenated with the input vector:
- Generator: `[label_emb(10) | noise(100)]` = 110-dim input
- Discriminator: `[image(784) | label_emb(10)]` = 794-dim input

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | BCELoss |
| Optimizer | Adam |
| Learning rate | 2e-4 |
| Beta1 | 0.5 |
| Beta2 | 0.999 |
| Epochs per generation | 50 |
| Batch size | 128 |
| Latent dimension | 100 |
| Multi-GPU | DataParallel (2 GPUs default) |

## Image Normalization

- **Generator output**: [-1, 1] (Tanh activation)
- **Saved format**: uint8 [0, 255], conversion: `(img + 1) / 2 * 255`
- **Loading for training**: `img / 255 * 2 - 1` back to [-1, 1]
- **Real MNIST**: `transforms.Normalize([0.5], [0.5])` maps [0, 1] to [-1, 1]

## Jacobian Considerations

- Always use `model.eval()` mode when computing Jacobians (BatchNorm statistics must be stable)
- Extract `.module` from DataParallel wrapper before Jacobian computation
- Jacobian must run on a single GPU device

## Source Files

The architecture is defined identically in all experiment files:
- `FullSynthetic/GAN-MNIST/GAN-MNIST-Default/main.py`
- `FullSynthetic/GAN-MNIST/GAN-MNIST-Fashion/main.py`
- `AugmentedSynthetic/GAN-MNIST/main.py`
- `NeonTest/ThomasExample/neon_controlled_collapse_gan.py`
- `FineTuning/GAN_NEON/protocol_a_neon.py`
