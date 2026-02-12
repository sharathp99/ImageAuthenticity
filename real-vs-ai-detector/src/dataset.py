from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import CLASS_TO_IDX

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Sample:
    path: Path
    label: int


class ImageBinaryDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: pd.DataFrame, transform: transforms.Compose):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), int(row["label"])


def build_samples(real_dir: Path, ai_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for label_name, folder in (("real", real_dir), ("ai", ai_dir)):
        for image_path in folder.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                samples.append(Sample(path=image_path, label=CLASS_TO_IDX[label_name]))
    if not samples:
        raise ValueError("No images found in discovered class directories.")
    return samples


def create_split_df(samples: list[Sample], seed: int) -> pd.DataFrame:
    df = pd.DataFrame({"path": [str(s.path) for s in samples], "label": [s.label for s in samples]})
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out


def get_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            norm,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            norm,
        ]
    )
    return train_tf, eval_tf
