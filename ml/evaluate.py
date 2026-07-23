import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score
)

# =====================================
# Paths
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "data", "models")

REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")
METRIC_DIR = os.path.join(REPORT_DIR, "metrics")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

# =====================================
# Load Model
# =====================================

print("Loading best model...")

model = joblib.load(
    os.path.join(MODEL_DIR, "best_model.pkl")
)

X_test, y_test = joblib.load(
    os.path.join(MODEL_DIR, "test.pkl")
)

# =====================================
# Prediction
# =====================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# =====================================
# Classification Report
# =====================================

report = classification_report(
    y_test,
    predictions
)

print(report)

with open(
    os.path.join(METRIC_DIR, "classification_report.txt"),
    "w"
) as f:
    f.write(report)

# =====================================
# Confusion Matrix
# =====================================

cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================
# ROC Curve
# =====================================

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("ROC Curve")

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================
# Precision Recall Curve
# =====================================

PrecisionRecallDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("Precision Recall Curve")

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "pr_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================
# Feature Importance
# =====================================

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": X_test.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        os.path.join(
            METRIC_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    plt.figure(figsize=(10,8))

    sns.barplot(
        data=importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Important Features")

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================
# Metrics
# =====================================

roc = roc_auc_score(
    y_test,
    probabilities
)

pr = average_precision_score(
    y_test,
    probabilities
)

summary = pd.DataFrame({

    "ROC-AUC":[roc],

    "PR-AUC":[pr]

})

summary.to_csv(

    os.path.join(
        METRIC_DIR,
        "metrics.csv"
    ),

    index=False

)

print("\nEvaluation Completed Successfully!")

print(f"ROC-AUC : {roc:.4f}")

print(f"PR-AUC  : {pr:.4f}")