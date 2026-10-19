import sys
sys.path.append("..")            # so we can import from src/
import os
os.chdir("..")                   # run from project root so paths in the CSV work

from src.dataset import get_dataloaders, MEAN, STD
import matplotlib.pyplot as plt
import torch

loaders, classes, weights = get_dataloaders()
print("Classes:", classes)
print("Class weights:", weights)

images, labels = next(iter(loaders["train"]))
print("Batch shape:", images.shape)   # should be [32, 3, 224, 224]

# undo normalization so we can display the images
mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, img, lab in zip(axes.flat, images, labels):
    ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0))
    ax.set_title(classes[lab])
    ax.axis("off")
plt.show()