# Real vs AI-Generated Image Classifier

Portfolio-grade binary image classifier for distinguishing **real photographs** from **AI-generated images** using the Kaggle dataset `cashbowman/ai-generated-images-vs-real-images` via `kagglehub`.

## Problem Scope
- Build a reproducible ML pipeline with training, evaluation, and inference.
- Deliver a Streamlit app for CPU-friendly local inference.
- Provide confidence-aware output with an **UNCERTAIN** class.

## Explicit Limitations
- Not a forensic proof system.
- Performance can degrade on unseen generators, heavy compression, screenshots, crops, or edited/re-encoded images.
- Any prediction should be treated as probabilistic guidance, not legal/authenticity evidence.

## Project Structure

```text
real-vs-ai-detector/
  app/
    streamlit_app.py
  src/
    config.py
    data_discovery.py
    dataset.py
    train.py
    evaluate.py
    infer.py
    utils.py
  models/
    best_model.pt (generated)
  reports/
    metrics.json (generated)
    confusion_matrix.png (generated)
  tests/
    test_infer_smoke.py
  .github/workflows/
    ci.yml
  .gitignore
  requirements.txt
  README.md
  LICENSE
```

## Setup

```bash
cd real-vs-ai-detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset (kagglehub)
Training automatically downloads data with:

```python
import kagglehub
path = kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images")
print(path)
```

Ensure Kaggle credentials are configured for your environment when needed.

## Train

```bash
PYTHONPATH=. python src/train.py --epochs 10
```

Training behavior:
- Auto-discovers class folders by case-insensitive patterns.
- Builds deterministic stratified split (80/10/10).
- Saves split to `reports/split.csv`.
- Uses AdamW + cosine scheduler + early stopping (patience=2) on val ROC-AUC.
- Saves best checkpoint to `models/best_model.pt`.

## Evaluate

```bash
PYTHONPATH=. python src/evaluate.py
```

Outputs:
- `reports/metrics.json`
- `reports/confusion_matrix.png`

## Run Streamlit

```bash
PYTHONPATH=. streamlit run app/streamlit_app.py
```

App outputs:
- Label: `REAL`, `AI`, or `UNCERTAIN`
- Confidence (0–100%)
- Probabilities `[p_real, p_ai]`
- Metrics display if `reports/metrics.json` exists
- Limitations disclaimer

  <img width="1865" height="916" alt="image" src="https://github.com/user-attachments/assets/78db31cd-dc5d-464f-bfa9-cc67d8170638" />


## Streamlit Community Cloud Deployment
1. Push repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set main file to `real-vs-ai-detector/app/streamlit_app.py`.
4. Ensure `requirements.txt` is in project root used by Streamlit.
5. Provide or mount `models/best_model.pt` (or update sidebar model path to your hosted checkpoint).

## Sample Metrics Template
If `reports/metrics.json` is missing, use this shape:

```json
{
  "accuracy": 0.90,
  "precision": 0.90,
  "recall": 0.90,
  "f1": 0.90,
  "roc_auc": 0.95
}
```
