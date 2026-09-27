import os
import time
import pandas as pd

try:
    from producer import TransactionProducer
except ImportError:
    from kafka.producer import TransactionProducer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")


def simulate(delay: float = 0.5):
    print(f"Reading dataset from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        csv_alt = os.path.join(BASE_DIR, "data", "processed", "clean_creditcard.csv")
        if os.path.exists(csv_alt):
            print(f"Raw CSV not found. Falling back to {csv_alt}")
            df = pd.read_csv(csv_alt)
        else:
            print("Error: No transaction CSV file found!")
            return
    else:
        df = pd.read_csv(CSV_PATH)

    # ── Ground-truth label handling ────────────────────────────────────────────
    # creditcard.csv contains a verified 'Class' column (0 = legitimate, 1 = fraud).
    # We keep it in the Kafka message so the consumer can store it as actual_label.
    # The model never sees 'Class' — it is removed from the feature dict inside
    # predictor.py before inference.
    has_class = "Class" in df.columns
    if not has_class:
        print("[WARNING] 'Class' column not found — ground-truth labels will not be recorded.")

    print("Initializing Kafka Producer...")
    producer = TransactionProducer(bootstrap_servers=["localhost:9092"])

    fraud_count = 0
    sent_count  = 0
    print("Starting streaming simulation (Ctrl+C to stop)...")
    print(f"Dataset size: {len(df)} rows"
          + (f" | Fraud rows: {int(df['Class'].sum())}" if has_class else ""))

    for _, row in df.iterrows():
        transaction = row.to_dict()

        # Convert Class to int so JSON serialisation is clean
        if has_class:
            transaction["Class"] = int(transaction["Class"])
            if transaction["Class"] == 1:
                fraud_count += 1

        producer.send_transaction("transactions", transaction)
        sent_count += 1
        status = f"[FRAUD]" if (has_class and transaction.get("Class") == 1) else ""
        print(
            f"[{sent_count}] Sent Time={transaction['Time']:.0f} "
            f"Amount={transaction['Amount']:.2f} {status}"
        )
        time.sleep(delay)


if __name__ == "__main__":
    simulate()
