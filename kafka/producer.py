import json
from kafka import KafkaProducer

class TransactionProducer:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_transaction(self, topic, transaction):
        self.producer.send(topic, transaction)
        self.producer.flush()
