import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats, needed for transfer learning later
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([       # NO augmentation for val/test
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class BloodDataset(Dataset):
    def __init__(self, df, class_to_idx, transform):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), self.class_to_idx[row["label"]]


def get_dataloaders(csv_path="data/processed/splits.csv", batch_size=32, num_workers=0):
    df = pd.read_csv(csv_path)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    test_df = df[df.split == "test"]

    loaders = {
        "train": DataLoader(BloodDataset(train_df, class_to_idx, train_tf),
                            batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(BloodDataset(val_df, class_to_idx, eval_tf),
                          batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(BloodDataset(test_df, class_to_idx, eval_tf),
                           batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    # class weights: rarer classes get a larger weight (handles imbalance)
    counts = train_df["label"].value_counts()
    weights = torch.tensor([len(train_df) / (len(classes) * counts[c]) for c in classes],
                           dtype=torch.float32)
    return loaders, classes, weights