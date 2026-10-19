import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds += out.argmax(1).cpu().tolist()
            targets += y.cpu().tolist()
    n = len(targets)
    acc = sum(p == t for p, t in zip(preds, targets)) / n
    f1 = f1_score(targets, preds, average="macro")
    return total_loss / n, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    loaders, classes, weights = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers)
    model = build_model(args.model, len(classes)).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    best_f1, bad_epochs, history = 0.0, 0, []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(va_f1)

        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1))
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} F1 {va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1, bad_epochs = va_f1, 0
            torch.save({"model": args.model, "classes": classes,
                        "state_dict": model.state_dict()},
                       f"models/{args.model}_best.pth")
            print("  saved new best model")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    pd.DataFrame(history).to_csv(f"results/{args.model}_history.csv", index=False)
    print("Best val macro-F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()