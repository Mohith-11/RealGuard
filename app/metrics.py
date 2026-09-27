from prometheus_client import Counter, Histogram

prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

fraud_predictions = Counter(
    "fraud_predictions_total",
    "Fraud predictions"
)

legitimate_predictions = Counter(
    "legitimate_predictions_total",
    "Legitimate predictions"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)
