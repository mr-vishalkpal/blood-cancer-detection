import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from dataset import get_dataloaders
from evaluate import predict
from model import build_model

device = torch.device("cpu")
loaders, classes, _ = get_dataloaders()
rng = np.random.default_rng(42)

for name in ["simple_cnn", "efficientnet_b0", "resnet50"]:
    ckpt = torch.load(f"models/{name}_best.pth", map_location=device, weights_only=False)
    model = build_model(name, len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    probs, y = predict(model, loaders["test"], device)
    pred = probs.argmax(1)

    accs, f1s = [], []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))      # resample test set with replacement
        accs.append(accuracy_score(y[idx], pred[idx]))
        f1s.append(f1_score(y[idx], pred[idx], average="macro"))

    lo, hi = np.percentile(accs, [2.5, 97.5])
    flo, fhi = np.percentile(f1s, [2.5, 97.5])
    print(f"{name:16s} acc {accuracy_score(y, pred):.4f} (95% CI {lo:.4f}-{hi:.4f}) | "
          f"macro-F1 {f1_score(y, pred, average='macro'):.4f} (95% CI {flo:.4f}-{fhi:.4f})")