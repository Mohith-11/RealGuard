"""
retrainer.py - Automated model retraining pipeline for RealGuard.

Strategy:
  - Baseline training data: data/models/train.pkl  (creditcard.csv ground truth)
  - Supplemental data:      verified labels from PostgreSQL (if sufficient)
  - Evaluation set:         data/models/valid.pkl  (fixed holdout, never used in training)
  - Candidate model:        XGBoost, matching the production model type (ml/train.py)
  - Production comparison:  loaded from MLflow via Production alias
  - Registration:           only if candidate passes ALL validation gates

Usage:
    python -m monitoring.retraining.retrainer              # train + register if pass
    python -m monitoring.retraining.retrainer --no-register  # dry run
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
import mlflow
import mlflow.xgboost
import mlflow.pyfunc
from mlflow import MlflowClient
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

#  Paths 
BASE_DIR   = Path(__file__).resolve().parents[2]
MODEL_DIR  = BASE_DIR / "data"  / "models"
REPORT_DIR = BASE_DIR / "monitoring" / "retraining" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

#  Config import 
sys.path.insert(0, str(BASE_DIR))
from monitoring.retraining.config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PRODUCTION_ALIAS,
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    MIN_PROD_SAMPLES,
    MIN_PROD_FRAUD_SAMPLES,
    MIN_RECALL,
    MAX_PRECISION_DROP,
    MAX_RECALL_DROP,
    MIN_PR_AUC,
    XGB_PARAMS,
    DB_CONFIG,
    RANDOM_STATE,
)


#  Data loading 

def load_baseline_data():
    """Load the original creditcard training set from data/models/train.pkl."""
    train_path = MODEL_DIR / "train.pkl"
    valid_path = MODEL_DIR / "valid.pkl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation data not found: {valid_path}")

    X_train, y_train = joblib.load(train_path)
    X_valid, y_valid = joblib.load(valid_path)

    # Ensure pandas DataFrames with correct column names
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    if not isinstance(X_valid, pd.DataFrame):
        X_valid = pd.DataFrame(X_valid, columns=FEATURE_COLUMNS)

    X_train = X_train[FEATURE_COLUMNS]
    X_valid = X_valid[FEATURE_COLUMNS]

    print(f"  Baseline train : {len(X_train):,} rows  "
          f"({int((y_train==1).sum()):,} fraud)")
    print(f"  Validation set : {len(X_valid):,} rows  "
          f"({int((y_valid==1).sum()):,} fraud)")

    return X_train, y_train, X_valid, y_valid


def fetch_production_labels():
    """
    Fetch verified production labels from PostgreSQL.

    Only rows where actual_label was set from the creditcard.csv Class column
    (via the consumer pipeline) are considered ground truth.
    Returns None if insufficient data; returns (X, y) DataFrame pair if sufficient.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT transaction_features, actual_label
                FROM   predictions
                WHERE  actual_label IN (0, 1)
                  AND  transaction_features IS NOT NULL
                ORDER  BY labeled_at ASC, id ASC
            """)
            rows_data = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows_data, columns=col_names)
    finally:
        conn.close()

    if df.empty:
        print("  No production labels in PostgreSQL.")
        return None, None

    fraud_count = int((df["actual_label"] == 1).sum())
    print(f"  Production labels: {len(df):,} rows  ({fraud_count:,} fraud)")

    if len(df) < MIN_PROD_SAMPLES or fraud_count < MIN_PROD_FRAUD_SAMPLES:
        print(f"  [SKIP] Insufficient production labels "
              f"(need {MIN_PROD_SAMPLES} total / {MIN_PROD_FRAUD_SAMPLES} fraud). "
              f"Using baseline data only.")
        return None, None

    rows = []
    for _, row in df.iterrows():
        feat = row["transaction_features"]
        if isinstance(feat, str):
            feat = json.loads(feat)
        rows.append({col: feat.get(col, np.nan) for col in FEATURE_COLUMNS})

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = df["actual_label"].astype(int).reset_index(drop=True)

    # Remove rows with missing/non-finite values
    mask = X.apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    mask &= np.isfinite(X.apply(pd.to_numeric, errors="coerce").to_numpy()).all(axis=1)
    X, y = X[mask], y[mask]

    return X, y


#  Metrics 

def compute_metrics(model, X, y, source="pyfunc"):
    """Compute classification metrics. Handles both pyfunc and raw XGBoost."""
    if source == "pyfunc":
        y_pred  = model.predict(pd.DataFrame(X, columns=FEATURE_COLUMNS))
        raw     = model.get_raw_model() if hasattr(model, "get_raw_model") else None
        if raw and hasattr(raw, "predict_proba"):
            y_prob = raw.predict_proba(X)[:, 1]
        else:
            y_prob = y_pred.astype(float)
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return {
        "accuracy":         round(float(accuracy_score(y, y_pred)), 4),
        "precision":        round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1":               round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "pr_auc":           round(float(average_precision_score(y, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
    }


#  Validation 

def validate_candidate(candidate_m, production_m):
    """
    Gate: candidate must pass ALL checks to be eligible for registration.
    Returns dict with 'passed' (bool) and per-check results.
    """
    checks = {
        "recall_minimum": {
            "passed":    candidate_m["recall"] >= MIN_RECALL,
            "candidate": candidate_m["recall"],
            "threshold": MIN_RECALL,
        },
        "recall_non_degradation": {
            "passed":    candidate_m["recall"] >= production_m["recall"] - MAX_RECALL_DROP,
            "candidate": candidate_m["recall"],
            "production": production_m["recall"],
            "max_drop":  MAX_RECALL_DROP,
        },
        "precision_non_degradation": {
            "passed":    candidate_m["precision"] >= production_m["precision"] - MAX_PRECISION_DROP,
            "candidate": candidate_m["precision"],
            "production": production_m["precision"],
            "max_drop":  MAX_PRECISION_DROP,
        },
        "pr_auc_minimum": {
            "passed":    candidate_m["pr_auc"] >= MIN_PR_AUC,
            "candidate": candidate_m["pr_auc"],
            "threshold": MIN_PR_AUC,
        },
    }
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


#  Report 

def save_report(report: dict):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"retraining_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {path}")
    return path


def print_comparison(candidate_m, production_m, validation):
    print("\n" + "=" * 70)
    print("  CANDIDATE vs PRODUCTION")
    print("=" * 70)
    print(f"  {'Metric':<20} {'Candidate':>12} {'Production':>12}")
    print(f"  {'-'*44}")
    for key in ["accuracy", "precision", "recall", "f1", "pr_auc"]:
        c = candidate_m[key]
        p = production_m[key]
        arrow = "[OK] " if c >= p else "[!!] "
        print(f"  {key:<20} {c:>12.4f} {p:>12.4f}  {arrow}")
    print("=" * 70)
    print(f"\n  Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
    for name, detail in validation["checks"].items():
        icon = "[OK] " if detail["passed"] else "[FAIL]"
        print(f"    {icon} {name}")
    print()


#  Main pipeline 

def run_retraining(register: bool = True) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    report = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "status":       "STARTED",
        "registered":   False,
    }

    #  Step 1: Load baseline training data 
    print("\n[1/7] Loading baseline training data (data/models/train.pkl)...")
    X_train, y_train, X_valid, y_valid = load_baseline_data()

    #  Step 2: Supplement with verified production labels 
    print("\n[2/7] Checking PostgreSQL for verified production labels...")
    X_prod, y_prod = fetch_production_labels()

    if X_prod is not None:
        X_train = pd.concat([X_train, X_prod], ignore_index=True)
        y_train = pd.concat(
            [pd.Series(y_train), pd.Series(y_prod)], ignore_index=True
        )
        print(f"  Merged dataset: {len(X_train):,} rows  "
              f"({int((y_train==1).sum()):,} fraud)")

    report["training_samples"]  = int(len(X_train))
    report["fraud_samples"]     = int((y_train == 1).sum())
    report["legit_samples"]     = int((y_train == 0).sum())
    report["validation_samples"] = int(len(X_valid))
    report["features"]          = FEATURE_COLUMNS

    #  Step 3: Compute scale_pos_weight (matches ml/train.py) 
    scale_pos_weight = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    print(f"\n[3/7] scale_pos_weight = {scale_pos_weight:.2f}")

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    report["hyperparameters"] = params

    #  Step 4: Train candidate XGBoost 
    print("\n[4/7] Training XGBoost candidate...")
    candidate = XGBClassifier(**params)
    candidate.fit(X_train, y_train, verbose=False)
    print("  Training complete.")

    #  Step 5: Evaluate candidate on fixed validation set 
    print("\n[5/7] Evaluating candidate on validation set...")
    candidate_metrics = compute_metrics(candidate, X_valid, y_valid, source="raw")
    print(f"  Precision={candidate_metrics['precision']:.4f}  "
          f"Recall={candidate_metrics['recall']:.4f}  "
          f"F1={candidate_metrics['f1']:.4f}  "
          f"PR-AUC={candidate_metrics['pr_auc']:.4f}")

    #  Step 6: Load & evaluate production model on the same set 
    print("\n[6/7] Loading production model and evaluating on validation set...")
    try:
        mv           = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        prod_uri     = f"models:/{MODEL_NAME}/{mv.version}"
        prod_pyfunc  = mlflow.pyfunc.load_model(prod_uri)
        prod_version = str(mv.version)
        print(f"  Loaded {MODEL_NAME} v{prod_version}")
    except Exception as exc:
        raise RuntimeError(
            "Cannot load Production model  refusing to register without comparison."
        ) from exc

    production_metrics = compute_metrics(prod_pyfunc, X_valid, y_valid, source="pyfunc")
    print(f"  Precision={production_metrics['precision']:.4f}  "
          f"Recall={production_metrics['recall']:.4f}  "
          f"F1={production_metrics['f1']:.4f}  "
          f"PR-AUC={production_metrics['pr_auc']:.4f}")

    report["production_version"]  = prod_version
    report["candidate_metrics"]   = candidate_metrics
    report["production_metrics"]  = production_metrics

    #  Validate 
    validation = validate_candidate(candidate_metrics, production_metrics)
    report["validation"] = validation
    print_comparison(candidate_metrics, production_metrics, validation)

    #  Step 7: Register or reject 
    if not validation["passed"]:
        report["status"] = "REJECTED"
        save_report(report)
        print("[REJECTED] [FAIL]  Candidate did not pass validation. Production unchanged.")
        return report

    if not register:
        report["status"] = "VALIDATED_DRY_RUN"
        save_report(report)
        print("[DRY RUN] [OK]   Candidate passed validation. --no-register set; skipping MLflow registration.")
        return report

    print("\n[7/7] Registering candidate in MLflow...")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="automated-retraining") as run:
        # Log parameters
        mlflow.log_params({
            "model_type":        "XGBClassifier",
            "training_samples":  report["training_samples"],
            "validation_samples": report["validation_samples"],
            "fraud_samples":     report["fraud_samples"],
            **{f"xgb_{k}": v for k, v in params.items()},
        })

        # Log metrics (both candidate and production for comparison)
        for prefix, m in [("candidate", candidate_metrics), ("production", production_metrics)]:
            for name in ["accuracy", "precision", "recall", "f1", "pr_auc"]:
                mlflow.log_metric(f"{prefix}_{name}", m[name])
        mlflow.log_metric("validation_passed", 1)

        # Register model
        model_info = mlflow.xgboost.log_model(
            xgb_model=candidate,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )
        report["run_id"]     = run.info.run_id
        report["model_uri"]  = model_info.model_uri

    report["status"]     = "REGISTERED"
    report["registered"] = True
    save_report(report)

    # Fetch the newly registered version number
    new_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    new_version  = max(int(v.version) for v in new_versions)
    report["candidate_version"] = new_version

    print(f"\n[SUCCESS] [OK]   Candidate registered as {MODEL_NAME} v{new_version}")
    print(f"  Production alias still points to v{prod_version}")
    print(f"  To promote: mlflow models set-alias --name {MODEL_NAME} "
          f"--alias Production --version {new_version}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealGuard Automated Retraining Pipeline")
    parser.add_argument("--no-register", action="store_true",
                        help="Validate without registering to MLflow")
    args = parser.parse_args()

    try:
        run_retraining(register=not args.no_register)
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}")
        traceback.print_exc()
        sys.exit(1)
