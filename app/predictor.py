"""
app/predictor.py - Model loading and inference.

Loads the Production-aliased model at startup.
Resolves the model source (models:/m-<uuid>) to a local artifact path
so the container can load it directly from the mounted mlruns volume,
without triggering MLflow artifact download (which fails with local-path artifacts).
"""
import os
import mlflow.xgboost
from mlflow import MlflowClient
import pandas as pd
from datetime import datetime

from app.config import MODEL_NAME, MODEL_ALIAS

# Feature column order expected by the model
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# mlruns base: inside the container it is /app/ml/mlruns (mounted from host)
MLRUNS_BASE = os.getenv("MLRUNS_BASE", "/app/ml/mlruns")

client = MlflowClient()

def _load_production_model():
    """Resolve Production alias -> model source -> local artifact path -> load."""
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    version = str(mv.version)

    source = mv.source  # e.g. "models:/m-fa4d9b5a13f24aa18591086f41ed74dd"
    model_id = source.replace("models:/", "").strip("/")

    # Local path pattern used by MLflow 3 for registry-backed models:
    # <mlruns>/1/models/<model-id>/artifacts/
    local_path = os.path.join(MLRUNS_BASE, "1", "models", model_id, "artifacts")

    if not os.path.isdir(local_path):
        raise RuntimeError(
            f"Model artifact directory not found: {local_path}\n"
            f"  source={source}\n"
            f"  Ensure the mlruns volume is mounted correctly."
        )

    loaded = mlflow.xgboost.load_model(local_path)
    return loaded, version


model, MODEL_VERSION = _load_production_model()
print(f"[predictor] Loaded model v{MODEL_VERSION} from mlruns", flush=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(transaction: dict) -> dict:
    df = pd.DataFrame([transaction])

    existing = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(existing) == len(FEATURE_COLUMNS):
        df = df[FEATURE_COLUMNS]

    prediction = int(model.predict(df)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        probability = float(probs[1])
    else:
        probability = 0.0

    return {
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(probability, 6),
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
