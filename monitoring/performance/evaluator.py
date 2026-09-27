"""
evaluator.py - Model performance monitoring for RealGuard.

Improvements over v1:
  1. Metrics are CLEARED (set to -1 sentinel) when no fraud samples exist,
     preventing stale Prometheus gauge values from misleading dashboards.
  2. A minimum fraud-sample threshold (MIN_FRAUD_SAMPLES) gates whether
     precision/recall/F1/PR-AUC are considered reliable.
  3. model_metrics_reliable gauge distinguishes "no data" from "degraded".
  4. Evaluation window metadata (batch size, fraud %, window start) is
     exported so Grafana can surface it alongside metrics.

Usage:
    python -m monitoring.performance.evaluator          # run once
    python -m monitoring.performance.evaluator --serve  # loop + Prometheus
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

import psycopg2
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
)
from prometheus_client import Gauge, Counter, Info, start_http_server

# ── Configuration ─────────────────────────────────────────────────────────────
BATCH_SIZE        = int(os.getenv("PERF_BATCH_SIZE",      "1000"))
INTERVAL          = int(os.getenv("PERF_INTERVAL",        "300"))
METRICS_PORT      = int(os.getenv("PERF_METRICS_PORT",    "8002"))
# Minimum confirmed fraud samples required before recall/precision are reliable
MIN_FRAUD_SAMPLES = int(os.getenv("PERF_MIN_FRAUD_SAMPLES", "5"))

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_DIR = os.path.join(BASE_DIR, "monitoring", "performance", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Sentinel value written to gauges when the metric cannot be reliably computed.
# Grafana panels can filter on `!= -1` to distinguish unavailable from degraded.
UNAVAILABLE = -1.0

# ── Prometheus metrics ─────────────────────────────────────────────────────────
# Core classification metrics
precision_gauge    = Gauge("model_precision",    "Precision (reliable only when model_metrics_reliable==1)")
recall_gauge       = Gauge("model_recall",       "Recall    (reliable only when model_metrics_reliable==1)")
f1_gauge           = Gauge("model_f1_score",     "F1-score  (reliable only when model_metrics_reliable==1)")
pr_auc_gauge       = Gauge("model_pr_auc",       "PR-AUC   (reliable only when model_metrics_reliable==1)")
accuracy_gauge     = Gauge("model_accuracy",     "Accuracy (always available when labeled data exists)")

# Availability / reliability gates
performance_available  = Gauge(
    "model_performance_available",
    "1=labeled data exists, 0=no labeled data"
)
metrics_reliable = Gauge(
    "model_metrics_reliable",
    "1=both fraud+legit classes present AND fraud >= MIN_FRAUD_SAMPLES; 0=unreliable"
)

# Window / sample metadata
labeled_samples_gauge  = Gauge("model_labeled_samples",      "Total labeled samples in current window")
fraud_samples_gauge    = Gauge("model_actual_fraud_samples", "Confirmed fraud samples in current window")
legit_samples_gauge    = Gauge("model_legitimate_samples",   "Confirmed legitimate samples in current window")
fraud_rate_gauge       = Gauge("model_fraud_rate",           "Fraud rate in current evaluation window (0.0-1.0)")
min_fraud_threshold    = Gauge("model_min_fraud_threshold",  "Minimum fraud samples required for reliable metrics")

# Confusion matrix components
tn_gauge = Gauge("model_true_negatives",  "True negatives in latest batch")
fp_gauge = Gauge("model_false_positives", "False positives in latest batch")
fn_gauge = Gauge("model_false_negatives", "False negatives in latest batch")
tp_gauge = Gauge("model_true_positives",  "True positives in latest batch")

# Counters
performance_reports_ctr = Counter(
    "model_performance_reports_total",
    "Total evaluation runs completed"
)
unreliable_reports_ctr = Counter(
    "model_unreliable_reports_total",
    "Evaluation runs where fraud sample count was below threshold"
)


def _clear_fraud_metrics():
    """Set fraud-dependent metrics to the UNAVAILABLE sentinel (-1)."""
    precision_gauge.set(UNAVAILABLE)
    recall_gauge.set(UNAVAILABLE)
    f1_gauge.set(UNAVAILABLE)
    pr_auc_gauge.set(UNAVAILABLE)
    tp_gauge.set(UNAVAILABLE)
    fn_gauge.set(UNAVAILABLE)


# ── Database ──────────────────────────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST",     "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB",     "frauddb"),
        user=os.getenv("POSTGRES_USER",     "realguard"),
        password=os.getenv("POSTGRES_PASSWORD", "realguard"),
    )


def fetch_labeled_data():
    """Return (rows, window_start) where rows are (prediction, prob, actual)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT prediction, fraud_probability, actual_label, labeled_at
                FROM   predictions
                WHERE  actual_label IS NOT NULL
                ORDER  BY labeled_at DESC, id DESC
                LIMIT  %s
                """,
                (BATCH_SIZE,),
            )
            rows = cur.fetchall()
        if rows:
            window_start = min(r[3] for r in rows if r[3])
        else:
            window_start = None
        return rows, window_start
    finally:
        conn.close()


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate() -> dict | None:
    rows, window_start = fetch_labeled_data()

    # ── No labeled data at all ───────────────────────────────────────────────
    if not rows:
        performance_available.set(0)
        metrics_reliable.set(0)
        _clear_fraud_metrics()
        accuracy_gauge.set(UNAVAILABLE)
        labeled_samples_gauge.set(0)
        fraud_samples_gauge.set(0)
        legit_samples_gauge.set(0)
        fraud_rate_gauge.set(UNAVAILABLE)
        min_fraud_threshold.set(MIN_FRAUD_SAMPLES)
        print("[INFO] No labeled transactions available yet.")
        return None

    y_pred = [int(r[0])   for r in rows]
    y_prob = [float(r[1]) if r[1] is not None else None for r in rows]
    y_true = [int(r[2])   for r in rows]

    fraud_count = sum(y_true)
    legit_count = len(y_true) - fraud_count
    fraud_rate  = fraud_count / len(y_true)

    # ── Always-available metrics ─────────────────────────────────────────────
    acc = round(accuracy_score(y_true, y_pred), 4)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    # ── Reliability gate ─────────────────────────────────────────────────────
    reliable = (fraud_count >= MIN_FRAUD_SAMPLES) and (legit_count > 0)

    result = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "window_start":       window_start.isoformat() if window_start else None,
        "batch_size_limit":   BATCH_SIZE,
        "min_fraud_threshold": MIN_FRAUD_SAMPLES,
        "samples":            len(y_true),
        "fraud_samples":      fraud_count,
        "legitimate_samples": legit_count,
        "fraud_rate":         round(fraud_rate, 6),
        "metrics_reliable":   reliable,
        "accuracy":           acc,
        "confusion_matrix":   cm,
        "precision":          None,
        "recall":             None,
        "f1_score":           None,
        "pr_auc":             None,
    }

    if reliable:
        result["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
        result["recall"]    = round(recall_score(y_true,    y_pred, zero_division=0), 4)
        result["f1_score"]  = round(f1_score(y_true,        y_pred, zero_division=0), 4)
        valid_probs = all(p is not None for p in y_prob)
        if valid_probs:
            result["pr_auc"] = round(average_precision_score(y_true, y_prob), 4)

        precision_gauge.set(result["precision"])
        recall_gauge.set(result["recall"])
        f1_gauge.set(result["f1_score"])
        if result["pr_auc"] is not None:
            pr_auc_gauge.set(result["pr_auc"])
        else:
            pr_auc_gauge.set(UNAVAILABLE)

        tp_gauge.set(tp)
        fn_gauge.set(fn)
        metrics_reliable.set(1)
    else:
        # Clear stale values — do NOT leave old scores in gauges
        _clear_fraud_metrics()
        metrics_reliable.set(0)
        unreliable_reports_ctr.inc()

    # ── Always push these ────────────────────────────────────────────────────
    accuracy_gauge.set(acc)
    tn_gauge.set(tn)
    fp_gauge.set(fp)
    labeled_samples_gauge.set(len(y_true))
    fraud_samples_gauge.set(fraud_count)
    legit_samples_gauge.set(legit_count)
    fraud_rate_gauge.set(round(fraud_rate, 6))
    min_fraud_threshold.set(MIN_FRAUD_SAMPLES)
    performance_available.set(1)
    performance_reports_ctr.inc()

    return result


# ── Report & display ──────────────────────────────────────────────────────────
def save_report(result: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"performance_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Report saved: {path}")


def print_summary(result: dict):
    reliable = result.get("metrics_reliable", False)
    fraud    = result.get("fraud_samples", 0)

    print("\n" + "=" * 65)
    print("  MODEL PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  Timestamp      : {result['timestamp']}")
    print(f"  Window start   : {result.get('window_start', 'N/A')}")
    print(f"  Samples        : {result['samples']}  "
          f"({fraud} fraud / {result['legitimate_samples']} legit)")
    print(f"  Fraud rate     : {result['fraud_rate']:.4%}")
    print(f"  Min threshold  : {result['min_fraud_threshold']} fraud samples required")
    print(f"  Metrics reliable: {'YES' if reliable else 'NO — insufficient fraud samples'}")
    print(f"  Accuracy       : {result['accuracy']:.4f}")

    if reliable:
        pr = result.get("pr_auc")
        cm = result["confusion_matrix"]
        print(f"  Precision      : {result['precision']:.4f}")
        print(f"  Recall         : {result['recall']:.4f}")
        print(f"  F1-Score       : {result['f1_score']:.4f}")
        print(f"  PR-AUC         : {pr:.4f}" if pr else "  PR-AUC         : N/A")
        print(f"  Confusion      : TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    else:
        print(f"  [Precision/Recall/F1/PR-AUC cleared — need >= {result['min_fraud_threshold']} fraud samples]")
    print("=" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(serve: bool = False):
    # Initialise all gauges to UNAVAILABLE at startup so Grafana never
    # displays stale values from a previous process.
    _clear_fraud_metrics()
    accuracy_gauge.set(UNAVAILABLE)
    performance_available.set(0)
    metrics_reliable.set(0)
    min_fraud_threshold.set(MIN_FRAUD_SAMPLES)

    if serve:
        start_http_server(METRICS_PORT)
        print(f"Prometheus metrics at http://localhost:{METRICS_PORT}/metrics")
        print(f"Minimum fraud samples for reliable metrics: {MIN_FRAUD_SAMPLES}")

    while True:
        try:
            result = evaluate()
            if result:
                save_report(result)
                print_summary(result)
        except Exception as exc:
            performance_available.set(0)
            metrics_reliable.set(0)
            import traceback
            print(f"[ERROR] Evaluation failed: {exc}")
            traceback.print_exc()

        if not serve:
            break

        print(f"Next evaluation in {INTERVAL}s... (Ctrl+C to stop)")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealGuard Performance Evaluator")
    parser.add_argument("--serve", action="store_true",
                        help="Run in loop mode with Prometheus /metrics endpoint")
    args = parser.parse_args()
    main(serve=args.serve)
