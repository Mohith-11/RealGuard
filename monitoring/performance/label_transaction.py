"""
label_transaction.py — Record confirmed fraud outcomes for model evaluation.

Usage:
    python -m monitoring.performance.label_transaction <prediction_id> <actual_label>

    actual_label: 0 = legitimate, 1 = fraud
"""

import os
import sys
import psycopg2


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "frauddb"),
        user=os.getenv("POSTGRES_USER", "realguard"),
        password=os.getenv("POSTGRES_PASSWORD", "realguard"),
    )


def label_transaction(prediction_id: int, actual_label: int):
    """Label a prediction with its confirmed ground-truth outcome."""
    if actual_label not in (0, 1):
        raise ValueError("actual_label must be 0 (legitimate) or 1 (fraud)")

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE predictions
                    SET actual_label = %s,
                        labeled_at   = NOW()
                    WHERE id = %s
                      AND actual_label IS NULL
                    RETURNING id
                    """,
                    (actual_label, prediction_id),
                )
                updated = cur.fetchone()

                if updated is None:
                    cur.execute(
                        "SELECT actual_label FROM predictions WHERE id = %s",
                        (prediction_id,),
                    )
                    existing = cur.fetchone()
                    if existing is None:
                        raise ValueError(f"Prediction {prediction_id} does not exist")
                    raise ValueError(
                        f"Prediction {prediction_id} is already labeled as {existing[0]}"
                    )

                label_str = "FRAUD" if actual_label == 1 else "LEGITIMATE"
                print(f"Prediction {updated[0]} labeled as {actual_label} ({label_str})")
    finally:
        conn.close()


def bulk_label_from_class_column(limit: int = 500):
    """
    Convenience helper: auto-label predictions using the Class column that was
    stored in transaction_features JSONB during the simulation run.
    This is ONLY valid for the simulator / dev environment where we have ground truth.
    In production, labels must come from an authoritative external source.
    """
    conn = get_connection()
    labeled = 0
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE predictions
                    SET actual_label = (transaction_features->>'Class')::SMALLINT,
                        labeled_at   = NOW()
                    WHERE actual_label IS NULL
                      AND transaction_features->>'Class' IS NOT NULL
                    RETURNING id
                    """,
                )
                rows = cur.fetchall()
                labeled = len(rows)
        print(f"Bulk-labeled {labeled} predictions from transaction_features.Class")
    finally:
        conn.close()
    return labeled


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--bulk-from-simulator":
        bulk_label_from_class_column()
        sys.exit(0)

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python -m monitoring.performance.label_transaction <id> <0|1>\n"
            "  python -m monitoring.performance.label_transaction --bulk-from-simulator"
        )
        sys.exit(1)

    label_transaction(int(sys.argv[1]), int(sys.argv[2]))
