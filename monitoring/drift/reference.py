"""
reference.py — Load the training reference dataset for drift detection.

Loads the processed training split saved by preprocessing.py and returns it
as a pandas DataFrame containing all V1-V28 features, Time, and Amount.
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")


def load_reference_data(sample_size: int = 2000) -> pd.DataFrame:
    """
    Load the training split from preprocessing.py output as reference data.

    Args:
        sample_size: Number of rows to use as reference (default 2000).
                     Stratified sample to preserve class ratio.

    Returns:
        pd.DataFrame with all 30 features used by the model.
    """
    train_path = os.path.join(MODEL_DIR, "train.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Reference data not found at {train_path}. "
            "Run ml/preprocessing.py first."
        )

    X_train, y_train = joblib.load(train_path)

    # Convert to DataFrame (handles both DataFrame and ndarray)
    if isinstance(X_train, pd.DataFrame):
        reference_df = X_train.copy()
    else:
        reference_df = pd.DataFrame(X_train)

    reference_df["Class"] = y_train.values if hasattr(y_train, "values") else y_train

    # Stratified sample to keep the dataset size manageable
    if len(reference_df) > sample_size:
        fraud = reference_df[reference_df["Class"] == 1]
        legit = reference_df[reference_df["Class"] == 0]

        fraud_n = min(len(fraud), max(1, int(sample_size * 0.002)))
        legit_n = sample_size - fraud_n

        reference_df = pd.concat([
            legit.sample(n=legit_n, random_state=42),
            fraud.sample(n=fraud_n, random_state=42),
        ]).reset_index(drop=True)

    print(f"Reference data loaded: {len(reference_df)} rows, {reference_df.shape[1]} columns")
    return reference_df
