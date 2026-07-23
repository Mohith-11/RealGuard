import os
import json
import requests
from kafka import KafkaConsumer, KafkaProducer

# Load configuration from environment variables
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")

class FraudDetectorConsumer:
    def __init__(self, bootstrap_servers=BOOTSTRAP_SERVERS, api_url=API_URL):
        self.consumer = KafkaConsumer(
            "transactions",
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.api_url = api_url

    def start(self):
        print(f"Connecting to Kafka Brokers: {BOOTSTRAP_SERVERS}")
        print(f"Prediction API URL: {self.api_url}")
        print("Kafka Fraud Detector Consumer listening on 'transactions' topic...")
        
        for message in self.consumer:
            transaction = message.value
            print(f"Received transaction (Time={transaction.get('Time')}, Amount={transaction.get('Amount')})")
            
            try:
                # Call FastAPI predict endpoint
                response = requests.post(self.api_url, json=transaction)
                if response.status_code == 200:
                    result = response.json()
                    print(f"Prediction result: Label={result['label']}, Prob={result['fraud_probability']:.6f}")
                    
                    # Publish result to 'predictions' topic
                    self.producer.send("predictions", result)
                    self.producer.flush()
                else:
                    print(f"API Error: Status {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Error invoking prediction API: {e}")

if __name__ == "__main__":
    consumer = FraudDetectorConsumer()
    consumer.start()
