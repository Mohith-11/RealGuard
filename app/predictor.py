import mlflow.pyfunc
from mlflow import MlflowClient
import pandas as pd
from datetime import datetime

from app.config import MODEL_NAME, MODEL_ALIAS

# Load model using the 'Production' alias
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model = mlflow.pyfunc.load_model(MODEL_URI)

# Fetch the version details dynamically from the registry
client = MlflowClient()
try:
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    MODEL_VERSION = mv.version
except Exception:
    MODEL_VERSION = "unknown"


def predict(transaction):

    df = pd.DataFrame([transaction])

    # Get binary prediction (0 or 1)
    prediction = int(model.predict(df)[0])

    # Extract class probability if supported
    raw_model = model.get_raw_model()
    if hasattr(raw_model, "predict_proba"):
        probs = raw_model.predict_proba(df)[0]
        # Index 1 is the probability of class 1 (Fraud)
        probability = float(probs[1])
    else:
        probability = 0.0

    return {
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": probability,
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
