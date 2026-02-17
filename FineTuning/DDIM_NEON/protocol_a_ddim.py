#!/usr/bin/env python3
"""
Protocol A: Faithful NEON Implementation for MNIST DDIM

Uses DDIM (Denoising Diffusion Implicit Models) for deterministic sampling.
DDIM is exactly the type of model NEON was designed for!

This should work better than GAN because:
1. DDIM has iterative inference with controllable guidance (w parameter)
2. Mode-seeking behavior from classifier-free guidance
3. Same architecture family as EDM (tested in NEON paper)

Protocol A:
1. Load base model θ_r (Gen 0, trained on real MNIST)
2. Generate synthetic dataset S from θ_r (6k samples) - for FINE-TUNING
3. Fine-tune θ_r briefly on S → θ_s (intentional degradation)
4. Measure Jacobian effective rank before/after fine-tuning
   (Jacobian computed over full DDIM generation: noise → image)
5. Apply NEON: θ_NEON = (1+w)*θ_r - w*θ_s
6. Evaluate FID at different w values

Note on sample counts:
- N_SYNTHETIC = 6000: Samples for fine-tuning (matching NEON paper |S|=6k)
- N_JACOBIAN_SAMPLES: Samples for measuring Jacobian (diagnostic only, expensive)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.functional as F_auto
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import os
import sys
import pickle
import copy
from scipy import linalg
import matplotlib.pyplot as plt

# Add path to import from DDPM-MNIST
sys.path.insert(0, '/home/patrick/MAD-Detection/FullSynthetic/DDPM-MNIST')
from metrics import ContextUnet, DDPM, LeNet

# ============================================================================
# Configuration
# ============================================================================

# DDPM Architecture parameters
n_feat = 128
n_classes = 10
n_T = 500
beta1 = 1e-4
beta2 = 0.02

# Fine-tuning parameters (matching Neon paper)
N_SYNTHETIC = 6000          # |S| = 6k in Neon paper
N_FINETUNE_EPOCHS = 6       # Short fine-tuning
FINETUNE_LR = 1e-5          # Lower LR for fine-tuning diffusion models
batch_size = 128
GUIDE_W = 0.0               # Guidance weight for sampling

# Jacobian analysis (DDIM is expensive - ~10-20 min per sample)
N_JACOBIAN_SAMPLES = 3      # Few samples due to computational cost
JACOBIAN_CLASS = 7          # Fixed class

# NEON evaluation
W_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_FID_SAMPLES = 10000       # Samples for FID evaluation

# Paths
DDPM_DIR = '/home/patrick/MAD-Detection/FullSynthetic/DDPM-MNIST/data/diffusion_outputs10/'
BASE_MODEL_PATH = DDPM_DIR + 'model_0_w0.pth'
SAVE_DIR = '/home/patrick/MAD-Detection/FineTuning/DDIM_NEON/protocol_a_results/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + 'samples/', exist_ok=True)
os.makedirs(SAVE_DIR + 'checkpoints/', exist_ok=True)


# ============================================================================
# Utility Functions
# ============================================================================

def load_ddpm_model(checkpoint_path):
    """Load DDPM model from checkpoint"""
    nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes)
    model = DDPM(nn_model=nn_model, betas=(beta1, beta2), n_T=n_T, device=device, drop_prob=0.1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model


def generate_synthetic_data(model, n_samples, guide_w=0.0):
    """Generate synthetic data from DDPM"""
    model.eval()
    all_imgs = []
    all_labels = []

    batch_size_gen = 100  # Smaller batch for diffusion
    n_batches = n_samples // batch_size_gen

    print(f"Generating {n_samples} synthetic samples (guide_w={guide_w})...")
    with torch.no_grad():
        for _ in tqdm(range(n_batches)):
            x_gen, c_gen = model.sample(batch_size_gen, (1, 28, 28), device, guide_w=guide_w)
            # x_gen is in [0, 1] range after sampling
            all_imgs.append(x_gen.cpu())
            all_labels.append(c_gen[:batch_size_gen].cpu())

    return torch.cat(all_imgs), torch.cat(all_labels)


def generate_samples_for_fid(model, n_samples, guide_w=0.0):
    """Generate samples for FID evaluation"""
    model.eval()
    all_imgs = []

    batch_size_gen = 100
    n_batches = n_samples // batch_size_gen

    print(f"Generating {n_samples} samples for FID...")
    with torch.no_grad():
        for _ in tqdm(range(n_batches)):
            x_gen, _ = model.sample(batch_size_gen, (1, 28, 28), device, guide_w=guide_w)
            all_imgs.append(x_gen.cpu())

    return torch.cat(all_imgs)


def finetune_ddpm_epoch(model, dataloader, optimizer):
    """Fine-tune DDPM for one epoch on synthetic data"""
    model.train()
    losses = []

    for imgs, labels in tqdm(dataloader, desc="Fine-tuning"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # flag=1 means training mode (returns loss)
        loss = model(imgs, labels, flag=1, n_sample=0, guide_w=0)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def generation_full_ddim(z, model, class_label=7):
    """
    Full 500-step DDIM generation (deterministic).
    Maps noise z -> image x_0 through entire denoising process.

    Args:
        z: Initial noise [1, 1, 28, 28]
        model: DDPM model (using DDIM sampling formula)
        class_label: Class to generate
    """
    device = z.device

    # Setup class and context
    c = torch.tensor([class_label], device=device, dtype=torch.long)
    context_mask = torch.zeros(1, device=device, dtype=torch.float32)

    # Start with noise
    x = z.clone().float()

    # Full DDIM denoising loop (500 steps)
    for t in range(n_T, 0, -1):
        t_norm = torch.tensor([t / model.n_T], device=device, dtype=torch.float32).view(1, 1, 1, 1)

        # Predict noise
        eps = model.nn_model(x, c, t_norm, context_mask)

        # DDIM denoising formula
        if t > 1:
            # Predict x_0 from current x_t
            x_0_pred = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]
            # Compute x_{t-1}
            x = model.sqrtab[t-1] * x_0_pred + model.sqrtmab[t-1] * eps
        else:
            # Final step: predict x_0
            x = (x - model.sqrtmab[t] * eps) / model.sqrtab[t]

    return x


def compute_jacobian_metrics_ddim(model, n_samples=5, class_label=7):
    """
    Compute Jacobian effective rank using full DDIM generation.

    This computes the Jacobian of the ENTIRE noise->image mapping,
    not just a single denoiser step. This is the correct way to
    analyze the full generative process.

    WARNING: This is expensive (~10-20 min per sample due to 500 steps).
    """
    model.eval()

    effective_ranks = []
    max_eigenvalues = []

    for i in range(n_samples):
        torch.manual_seed(100 + i)
        z = torch.randn(1, 1, 28, 28).to(device)
        z_flat = z.flatten().detach().float().requires_grad_(True)

        def gen_func(z_input):
            z_shaped = z_input.reshape(1, 1, 28, 28)
            output = generation_full_ddim(z_shaped, model, class_label)
            return output.flatten()

        try:
            print(f"  Sample {i+1}/{n_samples}: Computing full DDIM Jacobian (500 steps)...")
            import time
            start = time.time()

            jacobian = F_auto.jacobian(gen_func, z_flat)

            elapsed = time.time() - start
            print(f"    Jacobian computed in {elapsed:.1f}s")

            U, S, V = torch.linalg.svd(jacobian)
            eigenvalues = S.pow(2).cpu().numpy()

            # Effective rank
            eigs_norm = eigenvalues / eigenvalues.sum()
            eff_rank = np.exp(-np.sum(eigs_norm * np.log(eigs_norm + 1e-10)))

            effective_ranks.append(eff_rank)
            max_eigenvalues.append(eigenvalues[0])

            print(f"    EffRank={eff_rank:.2f}, MaxEig={eigenvalues[0]:.4f}")

            # Clear GPU memory
            del jacobian, U, S, V
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Warning: Jacobian failed for sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return {
        'effective_rank_mean': np.mean(effective_ranks) if effective_ranks else 0,
        'effective_rank_std': np.std(effective_ranks) if effective_ranks else 0,
        'max_eigenvalue_mean': np.mean(max_eigenvalues) if max_eigenvalues else 0,
        'max_eigenvalue_std': np.std(max_eigenvalues) if max_eigenvalues else 0,
        'n_samples': len(effective_ranks)
    }


def compute_fid(real_features, fake_features):
    """Compute FID between feature distributions"""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_fake += eps * np.eye(sigma_fake.shape[0])

    diff = mu_real - mu_fake

    try:
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    except:
        fid = float('inf')

    return fid


def extract_features(lenet, images):
    """Extract features for FID using LeNet"""
    lenet.eval()
    features = []

    for i in range(0, len(images), 256):
        batch = images[i:i+256].to(device)
        with torch.no_grad():
            before_fc = lenet.features(batch)
            before_fc = before_fc.view(-1, 256)
            feat = lenet.fc(before_fc)
        features.append(feat.cpu().numpy())

    return np.concatenate(features)


def apply_neon(base_state, finetuned_state, w):
    """Apply NEON: θ_NEON = (1+w)*θ_r - w*θ_s"""
    neon_state = {}
    for key in base_state:
        neon_state[key] = (1 + w) * base_state[key] - w * finetuned_state[key]
    return neon_state


# ============================================================================
# Main Protocol A Experiment
# ============================================================================

def main():
    print("="*70)
    print("PROTOCOL A: NEON for MNIST DDPM")
    print("="*70)
    print("\nDDPM is a diffusion model - this should work with NEON!")

    results = {
        'config': {
            'n_synthetic': N_SYNTHETIC,
            'n_finetune_epochs': N_FINETUNE_EPOCHS,
            'finetune_lr': FINETUNE_LR,
            'guide_w': GUIDE_W,
            'w_values': W_VALUES
        }
    }

    # ========================================================================
    # Step 1: Load base model θ_r
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Load base model θ_r (Gen 0, trained on real MNIST)")
    print("="*70)

    model = load_ddpm_model(BASE_MODEL_PATH)
    theta_r = copy.deepcopy(model.state_dict())
    torch.save(theta_r, os.path.join(SAVE_DIR, 'checkpoints', 'theta_r.pth'))
    print(f"Loaded base model from {BASE_MODEL_PATH}")

    # ========================================================================
    # Step 2: Generate synthetic dataset S
    # ========================================================================
    print("\n" + "="*70)
    print(f"STEP 2: Generate synthetic dataset S ({N_SYNTHETIC} samples)")
    print("="*70)

    syn_imgs, syn_labels = generate_synthetic_data(model, N_SYNTHETIC, guide_w=GUIDE_W)
    print(f"Generated synthetic data: {syn_imgs.shape}")

    # Save sample visualization
    save_image(syn_imgs[:100], os.path.join(SAVE_DIR, 'samples', 'synthetic_training_data.png'), nrow=10)

    # Create dataloader (images should already be in [0, 1] from sampling)
    syn_dataset = TensorDataset(syn_imgs, syn_labels)
    syn_loader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ========================================================================
    # Step 3: Measure Jacobian of θ_r (before fine-tuning)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Compute Jacobian metrics for θ_r (before fine-tuning)")
    print("="*70)

    print(f"Computing full DDIM Jacobian over {N_JACOBIAN_SAMPLES} samples...")
    print("(This is slow - ~10-20 min per sample)")
    jacobian_before = compute_jacobian_metrics_ddim(model, N_JACOBIAN_SAMPLES, JACOBIAN_CLASS)
    print(f"\n  Effective Rank: {jacobian_before['effective_rank_mean']:.2f} ± {jacobian_before['effective_rank_std']:.2f}")

    results['jacobian_before'] = jacobian_before

    # ========================================================================
    # Step 4: Fine-tune θ_r on S → θ_s
    # ========================================================================
    print("\n" + "="*70)
    print(f"STEP 4: Fine-tune θ_r on synthetic data ({N_FINETUNE_EPOCHS} epochs)")
    print("="*70)

    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)

    finetune_log = {'epochs': [], 'loss': [], 'jacobian': []}

    for epoch in range(1, N_FINETUNE_EPOCHS + 1):
        loss = finetune_ddpm_epoch(model, syn_loader, optimizer)
        finetune_log['epochs'].append(epoch)
        finetune_log['loss'].append(loss)

        # Compute Jacobian at each epoch (expensive!)
        jac = compute_jacobian_metrics_ddim(model, N_JACOBIAN_SAMPLES, JACOBIAN_CLASS)
        finetune_log['jacobian'].append(jac)

        print(f"Epoch {epoch}/{N_FINETUNE_EPOCHS}: Loss={loss:.4f}, EffRank={jac['effective_rank_mean']:.2f}")

        # Save checkpoint
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, 'checkpoints', f'theta_s_epoch{epoch}.pth'))

    theta_s = copy.deepcopy(model.state_dict())
    torch.save(theta_s, os.path.join(SAVE_DIR, 'checkpoints', 'theta_s_final.pth'))

    results['finetune_log'] = finetune_log

    # ========================================================================
    # Step 5: Measure Jacobian of θ_s (after fine-tuning)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Compute Jacobian metrics for θ_s (after fine-tuning)")
    print("="*70)

    jacobian_after = compute_jacobian_metrics_ddim(model, N_JACOBIAN_SAMPLES, JACOBIAN_CLASS)
    print(f"  Effective Rank: {jacobian_after['effective_rank_mean']:.2f} ± {jacobian_after['effective_rank_std']:.2f}")

    rank_change = jacobian_after['effective_rank_mean'] - jacobian_before['effective_rank_mean']
    print(f"\n  Effective Rank Change: {rank_change:+.2f}")

    results['jacobian_after'] = jacobian_after

    # ========================================================================
    # Step 6: Load LeNet for FID
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Prepare FID evaluation")
    print("="*70)

    lenet_path = DDPM_DIR + 'prmodel.pth'
    lenet = LeNet()
    lenet.load_state_dict(torch.load(lenet_path, map_location=device))
    lenet.to(device)
    lenet.eval()
    print(f"Loaded LeNet from {lenet_path}")

    # Get real MNIST features
    print("Extracting real MNIST features...")
    tf = transforms.Compose([transforms.ToTensor()])
    mnist = MNIST("/home/patrick/MAD-Detection/FullSynthetic/DDPM-MNIST/data", train=True, download=True, transform=tf)
    real_loader = DataLoader(mnist, batch_size=256, shuffle=True)

    real_imgs = []
    for imgs, _ in real_loader:
        real_imgs.append(imgs)
        if len(real_imgs) * 256 >= N_FID_SAMPLES:
            break
    real_imgs = torch.cat(real_imgs)[:N_FID_SAMPLES]
    real_features = extract_features(lenet, real_imgs)

    # ========================================================================
    # Step 7: Compute FID for θ_r and θ_s
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: Compute baseline FIDs")
    print("="*70)

    # FID for base model θ_r
    model.load_state_dict(theta_r)
    base_imgs = generate_samples_for_fid(model, N_FID_SAMPLES, guide_w=GUIDE_W)
    base_features = extract_features(lenet, base_imgs)
    fid_base = compute_fid(real_features, base_features)
    print(f"FID(θ_r): {fid_base:.2f}")

    save_image(base_imgs[:100], os.path.join(SAVE_DIR, 'samples', 'theta_r_samples.png'), nrow=10)

    # FID for fine-tuned model θ_s
    model.load_state_dict(theta_s)
    ft_imgs = generate_samples_for_fid(model, N_FID_SAMPLES, guide_w=GUIDE_W)
    ft_features = extract_features(lenet, ft_imgs)
    fid_finetuned = compute_fid(real_features, ft_features)
    print(f"FID(θ_s): {fid_finetuned:.2f}")

    save_image(ft_imgs[:100], os.path.join(SAVE_DIR, 'samples', 'theta_s_samples.png'), nrow=10)

    print(f"\nFID Degradation: {fid_finetuned - fid_base:+.2f}")

    results['fid_base'] = fid_base
    results['fid_finetuned'] = fid_finetuned

    # ========================================================================
    # Step 8: Apply NEON at different w values
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 8: Apply NEON extrapolation")
    print("="*70)

    neon_results = {'w': [], 'fid': [], 'effective_rank': []}

    for w in W_VALUES:
        print(f"\nTesting w={w:.2f}...")

        # Apply NEON
        theta_neon = apply_neon(theta_r, theta_s, w)
        model.load_state_dict(theta_neon)

        # Compute FID
        neon_imgs = generate_samples_for_fid(model, N_FID_SAMPLES, guide_w=GUIDE_W)
        neon_features = extract_features(lenet, neon_imgs)
        fid = compute_fid(real_features, neon_features)

        # Compute Jacobian (expensive!)
        jac = compute_jacobian_metrics_ddim(model, N_JACOBIAN_SAMPLES, JACOBIAN_CLASS)

        neon_results['w'].append(w)
        neon_results['fid'].append(fid)
        neon_results['effective_rank'].append(jac['effective_rank_mean'])

        print(f"  w={w:.2f}: FID={fid:.2f}, EffRank={jac['effective_rank_mean']:.2f}")

        if w in [0.0, 1.0, 2.0]:
            save_image(neon_imgs[:100],
                      os.path.join(SAVE_DIR, 'samples', f'theta_neon_w{w:.1f}.png'), nrow=10)

    results['neon'] = neon_results

    # ========================================================================
    # Step 9: Results Summary
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 9: Results Summary")
    print("="*70)

    best_idx = np.argmin(neon_results['fid'])
    best_w = neon_results['w'][best_idx]
    best_fid = neon_results['fid'][best_idx]

    print(f"\nBaseline FID (θ_r):        {fid_base:.2f}")
    print(f"Fine-tuned FID (θ_s):      {fid_finetuned:.2f}")
    print(f"Best NEON FID (w={best_w:.2f}):  {best_fid:.2f}")

    print(f"\nImprovement over θ_r:  {fid_base - best_fid:+.2f}")
    print(f"Improvement over θ_s:  {fid_finetuned - best_fid:+.2f}")

    print(f"\nJacobian Effective Rank:")
    print(f"  θ_r (before):  {jacobian_before['effective_rank_mean']:.2f}")
    print(f"  θ_s (after):   {jacobian_after['effective_rank_mean']:.2f}")

    results['best_w'] = best_w
    results['best_fid'] = best_fid

    # ========================================================================
    # Step 10: Save and plot
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 10: Save results and plots")
    print("="*70)

    with open(os.path.join(SAVE_DIR, 'protocol_a_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {SAVE_DIR}protocol_a_results.pkl")

    create_plots(results)

    print("\n" + "="*70)
    print("PROTOCOL A COMPLETE")
    print("="*70)


def create_plots(results):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: FID vs w
    ax1 = axes[0, 0]
    ax1.plot(results['neon']['w'], results['neon']['fid'], 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=results['fid_base'], color='green', linestyle='--',
                label=f'θ_r (base): {results["fid_base"]:.1f}')
    ax1.axhline(y=results['fid_finetuned'], color='red', linestyle='--',
                label=f'θ_s (fine-tuned): {results["fid_finetuned"]:.1f}')

    best_idx = np.argmin(results['neon']['fid'])
    ax1.scatter([results['neon']['w'][best_idx]], [results['neon']['fid'][best_idx]],
                color='gold', s=200, zorder=5, marker='*', label=f'Best: w={results["best_w"]:.2f}')

    ax1.set_xlabel('NEON Weight (w)', fontsize=12)
    ax1.set_ylabel('FID (lower is better)', fontsize=12)
    ax1.set_title('NEON: FID vs Extrapolation Weight (DDPM)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Effective Rank vs w
    ax2 = axes[0, 1]
    ax2.plot(results['neon']['w'], results['neon']['effective_rank'], 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=results['jacobian_before']['effective_rank_mean'], color='green', linestyle='--',
                label=f'θ_r: {results["jacobian_before"]["effective_rank_mean"]:.1f}')
    ax2.axhline(y=results['jacobian_after']['effective_rank_mean'], color='red', linestyle='--',
                label=f'θ_s: {results["jacobian_after"]["effective_rank_mean"]:.1f}')

    ax2.set_xlabel('NEON Weight (w)', fontsize=12)
    ax2.set_ylabel('Effective Rank', fontsize=12)
    ax2.set_title('NEON: Effective Rank vs Extrapolation Weight', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effective Rank during fine-tuning
    ax3 = axes[1, 0]
    epochs = [0] + results['finetune_log']['epochs']
    eff_ranks = [results['jacobian_before']['effective_rank_mean']]
    eff_ranks += [j['effective_rank_mean'] for j in results['finetune_log']['jacobian']]

    ax3.plot(epochs, eff_ranks, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Fine-tuning Epoch', fontsize=12)
    ax3.set_ylabel('Effective Rank', fontsize=12)
    ax3.set_title('Jacobian Effective Rank During Fine-tuning', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Training loss during fine-tuning
    ax4 = axes[1, 1]
    ax4.plot(results['finetune_log']['epochs'], results['finetune_log']['loss'],
             'b-', linewidth=2, markersize=8)
    ax4.set_xlabel('Fine-tuning Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Fine-tuning Loss', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Protocol A: NEON on MNIST DDPM', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'protocol_a_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {SAVE_DIR}protocol_a_analysis.png")


if __name__ == "__main__":
    main()
