import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import label_binarize

from dataset import get_dataloaders
from model import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        out = model(x.to(device))
        probs.append(torch.softmax(out, 1).cpu().numpy())
        targets += y.tolist()
    return np.concatenate(probs), np.array(targets)


def plot_confusion(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc(y_true, probs, classes, title, path):
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    plt.figure(figsize=(5.5, 4.5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(y_bin[:, i], probs[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["simple_cnn", "resnet50", "efficientnet_b0"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, classes, _ = get_dataloaders()
    test_df = pd.read_csv("data/processed/splits.csv").query("split == 'test'").reset_index(drop=True)
    Path("results").mkdir(exist_ok=True)

    benign_idx = classes.index("Benign")
    summary = []

    for name in args.models:
        ckpt = torch.load(f"models/{name}_best.pth", map_location=device, weights_only=False)
        model = build_model(name, len(classes), pretrained=False).to(device)
        model.load_state_dict(ckpt["state_dict"])

        probs, y_true = predict(model, loaders["test"], device)
        y_pred = probs.argmax(1)
        cm = confusion_matrix(y_true, y_pred)

        # Medical view: Benign = negative, any malignant class = positive
        true_cancer = y_true != benign_idx
        pred_cancer = y_pred != benign_idx
        sensitivity = (pred_cancer & true_cancer).sum() / true_cancer.sum()
        specificity = (~pred_cancer & ~true_cancer).sum() / (~true_cancer).sum()

        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "macro_auc": auc,
            "sensitivity_cancer": sensitivity,
            "specificity_benign": specificity,
            "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        }
        summary.append(metrics)

        print(f"\n===== {name} =====")
        print(classification_report(y_true, y_pred, target_names=classes, digits=4))
        print(cm)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

        plot_confusion(cm, classes, f"{name} - test confusion matrix", f"results/{name}_confusion.png")
        plot_roc(y_true, probs, classes, f"{name} - ROC (one-vs-rest)", f"results/{name}_roc.png")

        wrong = test_df.assign(pred=[classes[i] for i in y_pred])[y_pred != y_true]
        wrong.to_csv(f"results/{name}_misclassified.csv", index=False)

    pd.DataFrame(summary).round(4).to_csv("results/model_comparison.csv", index=False)
    print("\n", pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()