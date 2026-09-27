import time
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.schemas import Transaction
from app.predictor import predict, MODEL_VERSION, MODEL_NAME as PREDICTOR_MODEL_NAME
from app.config import MODEL_NAME, MODEL_ALIAS
from app.metrics import (
    prediction_requests,
    fraud_predictions,
    legitimate_predictions,
    prediction_latency,
)

app = FastAPI(
    title="RealGuard Fraud Detection",
    version="1.0"
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
def root():
    return {
        "message": "RealGuard API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    """Return the model version currently loaded in this process.
    Used by the promotion controller to verify the correct model is serving.
    """
    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": MODEL_VERSION,
        "status": "loaded"
    }


@app.post("/predict")
def fraud_predict(transaction: Transaction):
    prediction_requests.inc()

    start = time.perf_counter()

    result = predict(transaction.model_dump())

    prediction_latency.observe(time.perf_counter() - start)

    if result.get("prediction") == 1:
        fraud_predictions.inc()
    else:
        legitimate_predictions.inc()

    return result
