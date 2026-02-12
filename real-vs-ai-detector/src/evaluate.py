from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.dataset import ImageBinaryDataset, get_transforms
from src.utils import resolve_device, set_seed


def load_model(model_path: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(model_path, map_location=device)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = resolve_device()

    if not cfg.split_csv.exists() or not cfg.model_path.exists():
        raise FileNotFoundError("Missing reports/split.csv or models/best_model.pt. Run training first.")

    split_df = pd.read_csv(cfg.split_csv)
    test_df = split_df[split_df["split"] == "test"]
    _, eval_tf = get_transforms(cfg.image_size)
    test_ds = ImageBinaryDataset(test_df, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = load_model(cfg.model_path, device)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1], labels=["REAL", "AI"])
    ax.set_yticks([0, 1], labels=["REAL", "AI"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(cfg.confusion_matrix_png)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
