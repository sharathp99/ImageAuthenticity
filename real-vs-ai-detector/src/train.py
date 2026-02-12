from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainConfig
from src.data_discovery import download_and_discover
from src.dataset import ImageBinaryDataset, build_samples, create_split_df, get_transforms
from src.utils import resolve_device, set_seed


def build_model(config: TrainConfig, device: str) -> nn.Module:
    model_name = config.model_gpu if device == "cuda" else config.model_cpu
    return timm.create_model(model_name, pretrained=True, num_classes=2)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: AdamW | None, device: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    probs, labels = [], []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    probs_np = np.concatenate(probs)
    labels_np = np.concatenate(labels)
    auc = roc_auc_score(labels_np, probs_np)
    return float(np.mean(losses)), auc, probs_np, labels_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    args = parser.parse_args()

    config = TrainConfig(epochs=args.epochs)
    set_seed(config.seed)
    device = resolve_device()

    _, real_dir, ai_dir = download_and_discover(config.dataset_ref)
    samples = build_samples(real_dir, ai_dir)
    split_df = create_split_df(samples, config.seed)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(config.split_csv, index=False)

    train_tf, eval_tf = get_transforms(config.image_size)
    train_ds = ImageBinaryDataset(split_df[split_df.split == "train"], train_tf)
    val_ds = ImageBinaryDataset(split_df[split_df.split == "val"], eval_tf)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_model(config, device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_auc = -1.0
    bad_epochs = 0
    config.model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        train_loss, train_auc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, _, _ = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        print(f"epoch={epoch+1} train_loss={train_loss:.4f} train_auc={train_auc:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": config.model_gpu if device == "cuda" else config.model_cpu,
                    "image_size": config.image_size,
                },
                config.model_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
