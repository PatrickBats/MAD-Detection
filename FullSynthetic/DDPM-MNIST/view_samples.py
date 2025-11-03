"""
Quick script to visualize generated samples
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the generated images
X = torch.load('./data/t500/gen_data_with_w_initial_w0')
X = X.float() / 255  # Normalize

# Load labels
C = torch.load('./data/t500/gen_index_with_w_initial_w0')

print(f"Loaded {X.shape[0]} images")
print(f"Image shape: {X.shape}")

# Show first 25 images in a 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle('Generated MNIST Samples (w=0)', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(X):
        ax.imshow(X[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {C[i].item()}', fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('./data/t500/generated_samples_preview.png', dpi=150, bbox_inches='tight')
print("\nSaved preview to: ./data/t500/generated_samples_preview.png")
plt.show()