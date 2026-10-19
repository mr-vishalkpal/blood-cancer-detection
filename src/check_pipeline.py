from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import get_dataloaders, MEAN, STD

loaders, classes, weights = get_dataloaders()
print("Classes:", classes)
print("Class weights:", weights)

images, labels = next(iter(loaders["train"]))
print("Batch shape:", images.shape)   # should be [32, 3, 224, 224]

mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, img, lab in zip(axes.flat, images, labels):
    ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0))
    ax.set_title(classes[lab])
    ax.axis("off")

Path("results").mkdir(exist_ok=True)
plt.savefig("results/sample_batch.png")
plt.show()