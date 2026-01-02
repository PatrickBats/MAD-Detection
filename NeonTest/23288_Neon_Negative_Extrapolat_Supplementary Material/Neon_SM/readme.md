```markdown
# Reproducing Results for ICLR Submission

This README provides instructions for running the provided code to reproduce the experimental results from our ICLR submission. The `toy_appendix.ipynb` notebook includes the toy example referenced in Appendix B.9, offering a lightweight implementation to test our algorithm and experiment with it interactively.

**FID Calculation**: For all experiments, FID scores are computed using the torch-fidelity library. Install it via:
```
pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity
```
This ensures high-fidelity metrics matching reference implementations.

## Quick Test with Toy Example

To get started quickly:

1. Open `toy_appendix.ipynb` in Jupyter Notebook or Google Colab.
2. Run the cells sequentially to see the toy example from Appendix B.9 in action.
3. Modify parameters in the notebook to play around with the algorithm.

## Experiments

The provided zip includes folders for different experiments. Each has its own setup and running instructions. Follow the relevant subsection below.

### xAR (xAR-B and xAR-L on ImageNet-256)

This experiment reproduces results using the xAR models for synthetic data generation, fine-tuning, and evaluation on ImageNet-256.

#### Setup for xAR

1. Unzip the provided file to a local directory and navigate to the `xAR` folder.

2. Set up the environment by following the instructions in the [xAR repository](https://github.com/OliverRensu/xAR). This includes:
   - Adopting the tokenizer from [MAR](https://github.com/LTH14/mar).
   - Downloading the ImageNet dataset and caching VAE latents as described in MAR.
   - Installing dependencies (e.g., via `pip install torch torchvision`).

3. Download the pretrained models `xAR_B.pth` and `xAR_L.pth`, along with the xAR tokenizer, from the [xAR Hugging Face repository](https://huggingface.co/OliverRen/xAR). Place them in the `pretrained/` folder within the `xAR` directory.

4. Download `adm_in256_stats.npz` from the `fid_stats` folder in the [xAR repository](https://github.com/OliverRensu/xAR) and place it in `xAR/fid_stats/`.

#### Running xAR

To generate synthetic data, perform fine-tuning, and run evaluation:

- For xAR-B:
  ```
  cd xAR
  bash run_xarb.sh
  ```

- For xAR-L:
  ```
  cd xAR
  bash run_xarl.sh
  ```

These scripts handle the full pipeline. Expected runtime depends on hardware (multi-GPU recommended for efficiency).

### EDM (EDM-VP on CIFAR-10 and FFHQ)

This experiment reproduces results using EDM-VP for synthetic data generation, fine-tuning, and evaluation on CIFAR-10 and FFHQ.

#### Setup for EDM

1. Set up the environment by following the instructions in the [EDM repository](https://github.com/NVlabs/edm). This includes:
   - Using Conda: `conda env create -f environment.yml -n edm` followed by `conda activate edm`.
   - Requires Python 3.8+ and PyTorch 1.12.0+ (install via official PyTorch site if needed).
   - Supported on Linux (recommended) and Windows; Docker option available for containerized runs.

#### Running EDM

To generate synthetic data, perform fine-tuning, and run evaluation:

- For CIFAR-10:
  ```
  cd edm
  bash run_cifar10.sh
  ```

- For FFHQ:
  ```
  cd edm
  bash run_ffhq.sh
  ```

These scripts handle the full pipeline. Expected runtime depends on hardware (multi-GPU recommended for efficiency).

### IMM (on ImageNet-256)

This experiment reproduces results using IMM (Inductive Moment Matching) for synthetic data generation, fine-tuning, and evaluation on ImageNet-256.

#### Setup for IMM

1. Unzip the provided file to a local directory and navigate to the `imm` folder.

2. Set up the environment by following the instructions in the [IMM repository](https://github.com/lumalabs/imm). This includes:
   - Create and activate the Conda environment: `conda env create -f env.yml` followed by `conda activate imm`.
3. Download the pretrained model `imagenet256_ts_a2.pkl` from the [IMM Hugging Face repository](https://huggingface.co/lumaai/imm) and place it in the `imm/` folder.

4. Download `adm_in256_stats.npz` (e.g., from the relevant stats source) and place it in `imm/fid_stats/`.

#### Running IMM

To generate synthetic data, perform fine-tuning, and run evaluation:

```
cd imm
bash run_imm_imagenet_256.sh
```

This script handles the full pipeline. Expected runtime depends on hardware (multi-GPU recommended for efficiency).

### VAR (on ImageNet-256 and ImageNet-512)

This experiment reproduces results using VAR (Visual Autoregressive Modeling) for synthetic data generation, fine-tuning, and evaluation on ImageNet at 256x256 and 512x512 resolutions.

#### Setup for VAR


1. Set up the environment by following the instructions in the [VAR repository](https://github.com/FoundationVision/VAR). This includes:
   - Install PyTorch >=2.0.0: `pip install torch torchvision torchaudio`.
   - Install other dependencies: `pip install -r requirements.txt`.
   - (Optional) Install flash-attn and xformers for faster training: Follow the repo's compilation instructions.

2. Download the required VAE model `vae_ch160v4096z32.pth` from [https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth) and place it in the `VAR/` folder.

3. For the `VAR_imagenet_256` subfolder:
   - Download `var_d16.pth` from [https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth) and place it in `VAR/VAR_imagenet_256/`.
   - Download `adm_in256_stats.npz` (e.g., from the relevant stats source) and place it in `VAR/VAR_imagenet_256/fid_stats/`.

4. For the `VAR_imagenet_512` subfolder:
   - Download `var_d36.pth` from [https://huggingface.co/FoundationVision/var/resolve/main/var_d36.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d36.pth) and place it in `VAR/VAR_imagenet_512/`.
   - Download `adm_in512_stats.npz` from the `evaluations` folder in the [guided-diffusion repository](https://github.com/openai/guided-diffusion/tree/main/evaluations) and place it in `VAR/VAR_imagenet_512/fid_stats/`.


#### Running VAR

To generate synthetic data, perform fine-tuning, and run evaluation:

- For ImageNet-256:
  ```
  cd VAR/VAR_imagenet_256
  bash run_var_imagenet_256.sh
  ```

- For ImageNet-512:
  ```
  cd VAR/VAR_imagenet_512
  bash run_var_imagenet_512.sh
  ```

These scripts handle the full pipeline. Expected runtime depends on hardware (multi-GPU recommended for efficiency).

**Hardware Note**: Experiments require multi-GPU setups (e.g., 8 GPUs per node)

If issues arise, enable verbose logging in the scripts or consult the relevant repo issues.
```
