from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    dataset_ref: str = "cashbowman/ai-generated-images-vs-real-images"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 2
    seed: int = 42
    uncertain_threshold: float = 0.60
    model_cpu: str = "efficientnet_b0"
    model_gpu: str = "convnext_tiny"
    project_root: Path = Path(__file__).resolve().parents[1]
    model_path: Path = project_root / "models" / "best_model.pt"
    reports_dir: Path = project_root / "reports"
    split_csv: Path = reports_dir / "split.csv"
    metrics_json: Path = reports_dir / "metrics.json"
    confusion_matrix_png: Path = reports_dir / "confusion_matrix.png"


CLASS_TO_IDX = {"real": 0, "ai": 1}
IDX_TO_LABEL = {0: "REAL", 1: "AI"}
