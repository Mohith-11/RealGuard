"""
config.py - Retraining pipeline configuration for RealGuard.
"""

import os

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME",           "RealGuard-FraudDetector")
PRODUCTION_ALIAS    = os.getenv("PRODUCTION_ALIAS",     "Production")
EXPERIMENT_NAME     = os.getenv("EXPERIMENT_NAME",      "RealGuard-Retraining")

# Feature schema - matches FEATURE_COLUMNS in app/predictor.py exactly
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Minimum verified production labels needed before they supplement training
MIN_PROD_SAMPLES       = int(os.getenv("RETRAIN_MIN_PROD_SAMPLES", "100"))
MIN_PROD_FRAUD_SAMPLES = int(os.getenv("RETRAIN_MIN_PROD_FRAUD",   "5"))

# Candidate validation gates - ALL must pass before registration
MIN_RECALL         = float(os.getenv("RETRAIN_MIN_RECALL",         "0.75"))
MAX_PRECISION_DROP = float(os.getenv("RETRAIN_MAX_PRECISION_DROP", "0.02"))
MAX_RECALL_DROP    = float(os.getenv("RETRAIN_MAX_RECALL_DROP",    "0.02"))
MIN_PR_AUC         = float(os.getenv("RETRAIN_MIN_PR_AUC",         "0.70"))

# XGBoost hyperparameters - mirrors ml/train.py production config
XGB_PARAMS = {
    "n_estimators":     int(os.getenv("XGB_N_ESTIMATORS", "300")),
    "max_depth":        int(os.getenv("XGB_MAX_DEPTH",     "6")),
    "learning_rate":    float(os.getenv("XGB_LR",          "0.1")),
    "subsample":        float(os.getenv("XGB_SUBSAMPLE",   "0.8")),
    "colsample_bytree": float(os.getenv("XGB_COLSAMPLE",   "0.8")),
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}

# PostgreSQL connection
DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST",     "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname":   os.getenv("POSTGRES_DB",       "frauddb"),
    "user":     os.getenv("POSTGRES_USER",     "realguard"),
    "password": os.getenv("POSTGRES_PASSWORD", "realguard"),
}

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE    = 0.20

