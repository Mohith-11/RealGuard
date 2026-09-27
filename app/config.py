import os
import mlflow

MODEL_NAME  = "RealGuard-FraudDetector"
MODEL_ALIAS = "Production"

# Read tracking URI from environment (set in docker-compose.yml for container,
# or from local env for direct runs). Env var takes absolute precedence.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)

POSTGRES_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
POSTGRES_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB       = os.getenv("POSTGRES_DB",       "frauddb")
POSTGRES_USER     = os.getenv("POSTGRES_USER",     "realguard")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "realguard")
