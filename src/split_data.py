import hashlib
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT_CSV = Path("data/processed/splits.csv")
SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}

rows = []
for class_dir in sorted(RAW_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    for p in class_dir.iterdir():
        if p.suffix.lower() in EXTS:
            rows.append({
                "path": p.as_posix(),
                "label": class_dir.name,
                "hash": hashlib.md5(p.read_bytes()).hexdigest(),
            })

df = pd.DataFrame(rows)
print("Total images found:", len(df))

# remove exact duplicate files
n_dup = df.duplicated("hash").sum()
df = df.drop_duplicates("hash").reset_index(drop=True)
print("Duplicates removed:", n_dup)

# 70% train, 15% val, 15% test, stratified so each split keeps class proportions
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

train_df = train_df.assign(split="train")
val_df = val_df.assign(split="val")
test_df = test_df.assign(split="test")
out = pd.concat([train_df, val_df, test_df]).drop(columns="hash")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print("\nImages per split and class:")
print(pd.crosstab(out["label"], out["split"]))
print("\nSaved to", OUT_CSV)