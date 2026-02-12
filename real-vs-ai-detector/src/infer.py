from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import timm
import torch
from PIL import Image

from src.config import IDX_TO_LABEL, TrainConfig
from src.dataset import get_transforms
from src.utils import resolve_device


def _load_image(input_data: bytes | str | Path) -> Image.Image:
    if isinstance(input_data, (str, Path)):
        return Image.open(input_data).convert("RGB")
    if isinstance(input_data, bytes):
        return Image.open(io.BytesIO(input_data)).convert("RGB")
    raise TypeError("input must be bytes or file path")


def load_model(model_path: str | Path | None = None) -> tuple[torch.nn.Module, str, int]:
    cfg = TrainConfig()
    device = resolve_device()
    path = Path(model_path) if model_path else cfg.model_path
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, device, int(ckpt.get("image_size", cfg.image_size))


def predict_image(input_data: bytes | str | Path, model_path: str | Path | None = None, uncertain_threshold: float = 0.60) -> dict[str, Any]:
    model, device, image_size = load_model(model_path)
    _, eval_tf = get_transforms(image_size)
    image = _load_image(input_data)

    tensor = eval_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    label = IDX_TO_LABEL[pred_idx]
    if confidence < uncertain_threshold:
        label = "UNCERTAIN"

    return {"label": label, "confidence": confidence, "probs": [float(probs[0]), float(probs[1])]}
