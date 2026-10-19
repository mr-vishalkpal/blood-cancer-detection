from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random

data_dir = Path("../data/raw")
classes = [d for d in data_dir.iterdir() if d.is_dir()]

# count images per class
for c in classes:
    n = len(list(c.glob("*.*")))
    print(f"{c.name}: {n} images")

# show 4 random images per class
fig, axes = plt.subplots(len(classes), 4, figsize=(12, 3 * len(classes)))
for i, c in enumerate(classes):
    imgs = random.sample(list(c.glob("*.*")), 4)
    for j, p in enumerate(imgs):
        axes[i, j].imshow(Image.open(p))
        axes[i, j].set_title(c.name)
        axes[i, j].axis("off")
plt.tight_layout()
plt.show()