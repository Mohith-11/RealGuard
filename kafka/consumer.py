import os
import sys
import json

# Prevent local project directory 'kafka' from shadowing installed kafka-python package
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAFKA_DIR = os.path.dirname(os.path.abspath(__file__))
orig_path = list(sys.path)

sys.path = [
    p for p in sys.path
    if p not in ("", ".")
    and os.path.abspath(p) not in (BASE_DIR, KAFKA_DIR)
]

for mod_name in list(sys.modules.keys()):
    if mod_name == "kafka" or mod_name.startswith("kafka."):
        mod_file = getattr(sys.modules[mod_name], "__file__", "")
        if not mod_file or not mod_file.startswith(sys.prefix):
            sys.modules.pop(mod_name, None)

from kafka import KafkaConsumer, KafkaProducer

# Restore sys.path and ensure project root is present for app module imports
sys.path = orig_path
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app.database import init_db, migrate_db, insert_prediction
from app.predictor import predict, FEATURE_COLUMNS

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")


class FraudDetectorConsumer:
    def __init__(self, bootstrap_servers=BOOTSTRAP_SERVERS):
        self.consumer = KafkaConsumer(
            "transactions",
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def start(self):
        print(f"Connecting to Kafka Brokers: {BOOTSTRAP_SERVERS}")

        try:
            init_db()
            migrate_db()
        except Exception as e:
            print(f"Database initialization warning: {e}")

        print("Kafka Fraud Detector Consumer listening on 'transactions' topic...")

        for message in self.consumer:
            transaction = message.value

            # ── Extract ground-truth label (from simulator) ────────────────────
            # 'Class' is the verified label from creditcard.csv.
            # It is extracted before inference and never passed to the model.
            actual_label = None
            if "Class" in transaction:
                raw_class = transaction["Class"]
                try:
                    actual_label = int(raw_class)
                    if actual_label not in (0, 1):
                        actual_label = None
                except (ValueError, TypeError):
                    actual_label = None

            # ── Build clean feature dict (no Class column for the model) ───────
            features_for_model = {k: v for k, v in transaction.items() if k != "Class"}
            # Store original transaction (with Class) for drift analysis
            features_for_storage = transaction

            transaction_time = float(features_for_model.get("Time", 0.0))
            amount           = float(features_for_model.get("Amount", 0.0))
            label_tag        = " [FRAUD]" if actual_label == 1 else ""
            print(
                f"Received transaction (Time={transaction_time}, "
                f"Amount={amount:.2f}){label_tag}"
            )

            try:
                result = predict(features_for_model)

                prediction  = int(result.get("prediction", 0))
                label       = str(result.get("label", "Legitimate"))
                prob        = float(result.get("fraud_probability", 0.0))
                raw_version = result.get("version", 1)
                try:
                    version = int(raw_version)
                except (ValueError, TypeError):
                    version = 1

                print(
                    f"  -> Predicted: {label} | Prob={prob:.4f} | "
                    f"GT: {'FRAUD' if actual_label==1 else 'LEGIT' if actual_label==0 else 'UNKNOWN'}"
                )

                try:
                    db_id = insert_prediction(
                        transaction_time=transaction_time,
                        amount=amount,
                        prediction=prediction,
                        label=label,
                        fraud_probability=prob,
                        model_version=version,
                        transaction_features=features_for_storage,
                        actual_label=actual_label,
                    )
                    print(f"  -> Saved to PostgreSQL (ID={db_id})")
                except Exception as db_err:
                    print(f"  -> DB write failed: {db_err}")

                self.producer.send("predictions", result)
                self.producer.flush()

            except Exception as e:
                print(f"Error processing transaction: {e}")


if __name__ == "__main__":
    consumer = FraudDetectorConsumer()
    consumer.start()
