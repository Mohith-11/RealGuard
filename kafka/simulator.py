import os
import time
import pandas as pd

try:
    from producer import TransactionProducer
except ImportError:
    from kafka.producer import TransactionProducer

# Set path to the CSV file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

def simulate():
    print(f"Reading dataset from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        # Fallback to processed data
        CSV_PATH_ALT = os.path.join(BASE_DIR, "data", "processed", "clean_creditcard.csv")
        if os.path.exists(CSV_PATH_ALT):
            print(f"Raw CSV not found. Reading processed dataset from {CSV_PATH_ALT}...")
            df = pd.read_csv(CSV_PATH_ALT)
        else:
            print("Error: No transaction CSV file found!")
            return
    else:
        df = pd.read_csv(CSV_PATH)
    
    # Drop Class if it exists
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
        
    print("Initializing Kafka Producer...")
    producer = TransactionProducer(bootstrap_servers=['localhost:9092'])
    
    print("Starting streaming simulation (Press Ctrl+C to stop)...")
    for _, row in df.iterrows():
        transaction = row.to_dict()
        producer.send_transaction("transactions", transaction)
        print(f"Sent transaction: Time={transaction['Time']}, Amount={transaction['Amount']}")
        time.sleep(1.0) # Send one transaction every second

if __name__ == "__main__":
    simulate()
