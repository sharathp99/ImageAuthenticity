from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from src.config import TrainConfig
from src.infer import predict_image


def test_predict_image_smoke() -> None:
    cfg = TrainConfig()
    if not cfg.model_path.exists():
        pytest.skip(f"Model file missing at {cfg.model_path}; skipping smoke inference test.")

    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    result = predict_image(buffer.getvalue())
    assert result["label"] in {"REAL", "AI", "UNCERTAIN"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probs"]) == 2
