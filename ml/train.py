import os
import joblib
import warnings
import pandas as pd

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

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# Display Settings
# =====================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

# =====================================================
# Load Dataset
# =====================================================

print("=" * 60)
print("Loading Processed Dataset")
print("=" * 60)

X_train, y_train = joblib.load(
    os.path.join(MODEL_DIR, "train.pkl")
)

X_valid, y_valid = joblib.load(
    os.path.join(MODEL_DIR, "valid.pkl")
)

print(f"Training Samples   : {X_train.shape}")
print(f"Validation Samples : {X_valid.shape}")

# =====================================================
# Class Weight for XGBoost
# =====================================================

fraud_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

print(f"\nScale Pos Weight : {fraud_weight:.2f}")

# =====================================================
# Models
# =====================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        solver="saga",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fraud_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# =====================================================
# Train Models
# =====================================================

results = []

best_model = None
best_name = None
best_f1 = -1

print("\n")
print("=" * 60)
print("Training Models")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    probabilities = model.predict_proba(X_valid)[:, 1]

    precision = precision_score(y_valid, predictions)

    recall = recall_score(y_valid, predictions)

    f1 = f1_score(y_valid, predictions)

    roc_auc = roc_auc_score(y_valid, probabilities)

    pr_auc = average_precision_score(y_valid, probabilities)

    results.append({

        "Model": name,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc,

        "PR-AUC": pr_auc

    })

    # Save every model
    filename = name.replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, filename)
    )

    print(f"Saved {filename}")

    if f1 > best_f1:

        best_f1 = f1
        best_model = model
        best_name = name

# =====================================================
# Results Table
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n")
print("=" * 60)
print("Model Comparison")
print("=" * 60)

print(results_df)

# =====================================================
# Save Metrics
# =====================================================

results_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "model_results.csv"
    ),
    index=False
)

print("\nModel results saved.")

# =====================================================
# Save Best Model
# =====================================================

joblib.dump(

    best_model,

    os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )

)

print("\n")
print("=" * 60)
print(f"Best Model : {best_name}")
print(f"Best F1    : {best_f1:.4f}")
print("=" * 60)

print("\nBest model saved successfully!")

print("\nTraining Completed Successfully!")