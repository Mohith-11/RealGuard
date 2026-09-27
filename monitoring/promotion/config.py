"""
monitoring/promotion/config.py - Promotion controller configuration.
All values are overridable via environment variables.
"""
import os

MODEL_NAME          = os.getenv("MODEL_NAME",          "RealGuard-FraudDetector")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PRODUCTION_ALIAS    = os.getenv("PRODUCTION_ALIAS",    "Production")

API_BASE_URL        = os.getenv("API_BASE_URL",  "http://localhost:8000").rstrip("/")
HEALTH_PATH         = os.getenv("HEALTH_PATH",   "/health")
MODEL_INFO_PATH     = os.getenv("MODEL_INFO_PATH", "/model-info")
PREDICT_PATH        = os.getenv("PREDICT_PATH",  "/predict")

# Docker Compose service name for the FastAPI container
API_SERVICE         = os.getenv("API_SERVICE",   "realguard-api")

# Health-poll settings
HEALTH_TIMEOUT      = int(os.getenv("HEALTH_TIMEOUT",  "120"))
HEALTH_INTERVAL     = int(os.getenv("HEALTH_INTERVAL", "3"))

import pathlib
_HERE       = pathlib.Path(__file__).parent
REPORT_DIR  = str(_HERE / "reports")
LOG_DIR     = str(_HERE / "logs")
