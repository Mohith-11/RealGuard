import os
import warnings
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow import MlflowClient
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

warnings.filterwarnings("ignore")

# ==========================================================
# MLflow
# ==========================================================

# Configure MLflow SQLite tracking URI relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DB = PROJECT_ROOT / "ml" / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")

mlflow.set_experiment("RealGuard Fraud Detection")
client = MlflowClient()

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "metrics"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# Display Options
# ==========================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("Loading Processed Dataset")
print("=" * 70)

X_train, y_train = joblib.load(
    os.path.join(MODEL_DIR, "train.pkl")
)

X_valid, y_valid = joblib.load(
    os.path.join(MODEL_DIR, "valid.pkl")
)

print(f"Training Shape   : {X_train.shape}")
print(f"Validation Shape : {X_valid.shape}")

# ==========================================================
# Calculate Class Weight
# ==========================================================

scale_pos_weight = (
    len(y_train[y_train == 0])
    /
    len(y_train[y_train == 1])
)

print(f"\nScale Positive Weight : {scale_pos_weight:.2f}")

# ==========================================================
# Models
# ==========================================================

models = {

    "Logistic Regression":

        LogisticRegression(

            max_iter=3000,
            solver="saga",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1

        ),

    "Random Forest":

        RandomForestClassifier(

            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1

        ),

    "XGBoost":

        XGBClassifier(

            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1

        )

}

# ==========================================================
# Train Models
# ==========================================================

results = []

best_model = None
best_name = None
best_f1 = -1

print("\n")
print("=" * 70)
print("Training Models")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    with mlflow.start_run(run_name=name):

        # ===========================================
        # Train
        # ===========================================

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_valid
        )

        probabilities = model.predict_proba(
            X_valid
        )[:, 1]

        # ===========================================
        # Metrics
        # ===========================================

        precision = precision_score(
            y_valid,
            predictions
        )

        recall = recall_score(
            y_valid,
            predictions
        )

        f1 = f1_score(
            y_valid,
            predictions
        )

        roc_auc = roc_auc_score(
            y_valid,
            probabilities
        )

        pr_auc = average_precision_score(
            y_valid,
            probabilities
        )

        # ===========================================
        # Parameters
        # ===========================================

        mlflow.log_param(
            "model_name",
            name
        )

        if name == "Logistic Regression":

            mlflow.log_param(
                "max_iter",
                3000
            )

        elif name == "Random Forest":

            mlflow.log_param(
                "n_estimators",
                300
            )

            mlflow.log_param(
                "class_weight",
                "balanced"
            )

        elif name == "XGBoost":

            mlflow.log_param(
                "n_estimators",
                300
            )

            mlflow.log_param(
                "max_depth",
                6
            )

            mlflow.log_param(
                "learning_rate",
                0.1
            )

            mlflow.log_param(
                "subsample",
                0.8
            )

            mlflow.log_param(
                "colsample_bytree",
                0.8
            )

        # ===========================================
        # Log Metrics
        # ===========================================

        mlflow.log_metric(
            "Precision",
            precision
        )

        mlflow.log_metric(
            "Recall",
            recall
        )

        mlflow.log_metric(
            "F1 Score",
            f1
        )

        mlflow.log_metric(
            "ROC AUC",
            roc_auc
        )

        mlflow.log_metric(
            "PR AUC",
            pr_auc
        )

        # ===========================================
        # Save Model to MLflow
        # ===========================================

        if name == "XGBoost":
            mlflow.xgboost.log_model(
                xgb_model=model,
                name="model"
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )

        # ===========================================
        # Save Local Model
        # ===========================================

        filename = (
            name.replace(" ", "_")
            + ".pkl"
        )

        joblib.dump(
            model,
            os.path.join(
                MODEL_DIR,
                filename
            )
        )

        print(f"Saved {filename}")

        # ===========================================
        # Store Results
        # ===========================================

        results.append({

            "Model": name,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "ROC-AUC": roc_auc,

            "PR-AUC": pr_auc

        })

        if f1 > best_f1:

            best_f1 = f1
            best_model = model
            best_name = name

# ==========================================================
# Results Table
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n")
print("=" * 70)
print("Model Comparison")
print("=" * 70)

print(results_df)

# ==========================================================
# Save Metrics
# ==========================================================

results_df.to_csv(

    os.path.join(
        REPORT_DIR,
        "model_results.csv"
    ),

    index=False

)

# ==========================================================
# Save Best Model
# ==========================================================

joblib.dump(

    best_model,

    os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )

)

# ==========================================================
# Log Best Model Information
# ==========================================================

with mlflow.start_run(run_name="Best Model Summary") as run:

    mlflow.log_param(
        "Best Model",
        best_name
    )

    mlflow.log_metric(
        "Best F1",
        best_f1
    )

    if best_name == "XGBoost":

        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            name="best_model"
        )

    else:

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="best_model"
        )

    model_uri = model_info.model_uri

    mv = mlflow.register_model(
        model_uri=model_uri,
        name="RealGuard-FraudDetector"
    )

    client.set_registered_model_alias(
        name="RealGuard-FraudDetector",
        alias="Production",
        version=mv.version
    )

print("\n")
print("=" * 70)
print(f"Best Model : {best_name}")
print(f"Best F1    : {best_f1:.4f}")
print("=" * 70)

print("\nAll models saved successfully.")
print("Metrics saved successfully.")
print("MLflow logs created successfully.")
print("Training Completed Successfully!")