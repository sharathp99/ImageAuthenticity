from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import TrainConfig
from src.infer import predict_image

cfg = TrainConfig()

st.set_page_config(page_title="Real vs AI Image Classifier", layout="centered")
st.title("🖼️ Real vs AI-Generated Image Classifier")
st.write("Upload an image to estimate whether it is real photography or AI-generated.")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=str(cfg.model_path))
    threshold = st.slider("Uncertain threshold", min_value=0.50, max_value=0.90, value=0.60, step=0.01)

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)
    try:
        result = predict_image(uploaded.getvalue(), model_path=model_path, uncertain_threshold=threshold)
        st.subheader(f"Prediction: {result['label']}")
        st.write(f"Confidence: {result['confidence'] * 100:.2f}%")
        st.write(f"Probabilities → REAL: {result['probs'][0]:.3f}, AI: {result['probs'][1]:.3f}")
    except Exception as exc:
        st.error(f"Inference error: {exc}")

metrics_path = Path(cfg.metrics_json)
if metrics_path.exists():
    st.markdown("### Latest Evaluation Metrics")
    st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
else:
    st.info("No metrics found yet. Run evaluation to generate reports/metrics.json.")

st.warning(
    "Disclaimer: This is a best-effort classifier and can fail on unseen generators, heavy compression, "
    "or edited/re-encoded images. Do not use as sole evidence for authenticity decisions."
)
