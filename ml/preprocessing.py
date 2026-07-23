import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clean_creditcard.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(df.shape)

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("Class", axis=1)

y = df["Class"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nTrain:", X_train.shape)
print("Validation:", X_valid.shape)
print("Test:", X_test.shape)

# -----------------------------
# Scale Amount
# -----------------------------
scaler = StandardScaler()

X_train["Amount"] = scaler.fit_transform(
    X_train[["Amount"]]
)

X_valid["Amount"] = scaler.transform(
    X_valid[["Amount"]]
)

X_test["Amount"] = scaler.transform(
    X_test[["Amount"]]
)

# -----------------------------
# Save Scaler
# -----------------------------
joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "scaler.pkl")
)

print("\nScaler saved!")

# -----------------------------
# Compute Class Weights
# -----------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print("\nClass Weights")
print(class_weights)

# -----------------------------
# Save Processed Data
# -----------------------------
joblib.dump(
    (X_train,y_train),
    os.path.join(MODEL_DIR,"train.pkl")
)

joblib.dump(
    (X_valid,y_valid),
    os.path.join(MODEL_DIR,"valid.pkl")
)

joblib.dump(
    (X_test,y_test),
    os.path.join(MODEL_DIR,"test.pkl")
)

print("\nPreprocessing completed successfully.")