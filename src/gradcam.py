import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import eval_tf, IMG_SIZE
from model import build_model


def get_target_layer(name, model):
    """The last convolutional stage of each network."""
    if name == "resnet50":
        return [model.layer4[-1]]
    if name == "efficientnet_b0":
        return [model.features[-1]]
    return [model.features[-1][0]]          # simple_cnn: last conv layer


def explain(cam, model, path):
    img = Image.open(path).convert("RGB")
    tensor = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1)[0]
    pred = int(probs.argmax())
    heat = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred)])[0]
    rgb = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    return rgb, overlay, pred, float(probs[pred])


def make_figure(rows, cam, model, classes, title, out_path, cols=3):
    """rows: list of (path, true_label). Each sample uses 2 panels: original + heatmap."""
    n = len(rows)
    n_rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(n_rows, cols * 2, figsize=(cols * 4.2, n_rows * 2.3))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, (path, true) in enumerate(rows):
        r, c = divmod(k, cols)
        rgb, overlay, pred, conf = explain(cam, model, path)
        axes[r, c * 2].imshow(rgb)
        axes[r, c * 2].set_title(f"True: {true}", fontsize=8)
        axes[r, c * 2 + 1].imshow(overlay)
        colour = "green" if classes[pred] == true else "red"
        axes[r, c * 2 + 1].set_title(f"Pred: {classes[pred]} ({conf:.2f})",
                                     fontsize=8, color=colour)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--per_class", type=int, default=3)
    args = ap.parse_args()
    random.seed(42)

    ckpt = torch.load(f"models/{args.model}_best.pth", map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = build_model(args.model, len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cam = GradCAM(model=model, target_layers=get_target_layer(args.model, model))
    test_df = pd.read_csv("data/processed/splits.csv").query("split == 'test'")
    Path("results").mkdir(exist_ok=True)

    # Figure 1: random test images from each class
    rows = []
    for c in classes:
        sub = test_df[test_df.label == c]
        for p in sub.sample(args.per_class, random_state=42)["path"]:
            rows.append((p, c))
    make_figure(rows, cam, model, classes,
                f"{args.model} - Grad-CAM on test images",
                f"results/{args.model}_gradcam.png")

    # Figure 2: the images the model got wrong (if any)
    wrong_csv = Path(f"results/{args.model}_misclassified.csv")
    if wrong_csv.exists():
        wrong = pd.read_csv(wrong_csv)
        if len(wrong) > 0:
            rows = list(zip(wrong["path"], wrong["label"]))[:9]
            make_figure(rows, cam, model, classes,
                        f"{args.model} - Grad-CAM on misclassified images",
                        f"results/{args.model}_gradcam_errors.png")
        else:
            print("No misclassified images, nothing to plot.")


if __name__ == "__main__":
    main()