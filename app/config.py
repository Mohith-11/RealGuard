from pathlib import Path
import mlflow

MODEL_NAME = "RealGuard-FraudDetector"
MODEL_ALIAS = "Production"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DB = PROJECT_ROOT / "ml" / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")

