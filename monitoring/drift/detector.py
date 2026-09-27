"""
detector.py — Drift detection engine for RealGuard using Evidently AI 0.7+.

Compares reference (training) data with production (current) data.
Generates HTML drift reports and exposes Prometheus metrics.

Usage:
    python -m monitoring.drift.detector            # one-shot report
    python -m monitoring.drift.detector --serve    # periodic loop (every 5 min)
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Feature columns expected by the model ─────────────────────────────────────
FEATURE_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

# ── Prometheus metrics ────────────────────────────────────────────────────────
try:
    from prometheus_client import Gauge, Counter, start_http_server

    drift_detected_gauge = Gauge(
        "drift_detected",
        "1 if feature drift was detected in the latest report, 0 otherwise",
    )
    drift_share_gauge = Gauge(
        "drift_share",
        "Fraction of features that drifted in the latest report (0.0 to 1.0)",
    )
    drift_reports_total = Counter(
        "drift_reports_total",
        "Total number of drift reports generated",
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("[WARNING] prometheus_client not installed; metrics will not be exported.")


def load_production_data(limit: int = 1000) -> pd.DataFrame:
    """Fetch recent production transactions from PostgreSQL."""
    from app.database import fetch_production_features

    rows = fetch_production_features(limit=limit)
    if not rows:
        raise ValueError(
            "No production transactions with features found. "
            "Run kafka/consumer.py and kafka/simulator.py first."
        )
    df = pd.DataFrame(rows)
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].astype(float)


def run_drift_report(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> dict:
    """
    Compare reference vs production data using Evidently AI 0.7+ API.
    Saves an HTML report and returns a summary dict.
    """
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset

    # Align columns
    common_cols = [c for c in reference_df.columns if c in production_df.columns]
    for drop_col in ["Class", "target", "label"]:
        if drop_col in common_cols:
            common_cols.remove(drop_col)

    ref_df = reference_df[common_cols].copy().astype(float)
    cur_df = production_df[common_cols].copy().astype(float)

    print(f"Running drift: reference={len(ref_df)} rows, production={len(cur_df)} rows, features={len(common_cols)}")

    # Wrap DataFrames in Evidently Dataset objects
    data_def = DataDefinition()
    ref_dataset = Dataset.from_pandas(ref_df, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(cur_df, data_definition=data_def)

    # Build and run the report
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=cur_dataset, reference_data=ref_dataset)

    # ── Save HTML report ──────────────────────────────────────────────────────
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_filename = f"drift_report_{timestamp}.html"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    snapshot.save_html(report_path)
    print(f"Drift report saved: {report_path}")

    # ── Parse summary from result dict ────────────────────────────────────────
    drift_summary = {
        "timestamp": timestamp,
        "report_file": report_filename,
        "dataset_drift_detected": False,
        "number_of_features": len(common_cols),
        "number_of_drifted_features": 0,
        "drift_share": 0.0,
    }

    try:
        result_dict = snapshot.dict() if hasattr(snapshot, "dict") else {}
        # Walk through metrics to find drift results
        metrics = result_dict.get("metrics", [])
        drifted = 0
        for metric in metrics:
            result = metric.get("result", {})
            if "drift_by_columns" in result:
                for col_info in result["drift_by_columns"].values():
                    if col_info.get("drift_detected", False):
                        drifted += 1
                dataset_drift = result.get("dataset_drift", False)
                drift_summary["dataset_drift_detected"] = dataset_drift
                drift_summary["number_of_drifted_features"] = drifted
                drift_summary["drift_share"] = round(drifted / max(len(common_cols), 1), 4)
                break
    except Exception as e:
        print(f"[WARNING] Could not parse drift result details: {e}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(REPORTS_DIR, f"drift_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(drift_summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    return drift_summary


def update_prometheus(summary: dict):
    """Push drift metrics to Prometheus gauges."""
    if not PROMETHEUS_AVAILABLE:
        return
    drift_detected_gauge.set(1 if summary.get("dataset_drift_detected") else 0)
    drift_share_gauge.set(summary.get("drift_share", 0.0))
    drift_reports_total.inc()


def print_summary(summary: dict):
    """Pretty-print the drift summary to console."""
    detected = summary.get("dataset_drift_detected", False)
    print("\n" + "=" * 60)
    print("  DRIFT DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Timestamp          : {summary.get('timestamp')}")
    print(f"  Report file        : {summary.get('report_file')}")
    print(f"  Dataset drift      : {'WARNING  YES - DRIFT DETECTED' if detected else 'OK  No significant drift'}")
    print(f"  Features drifted   : {summary.get('number_of_drifted_features', '?')} / {summary.get('number_of_features', '?')}")
    print(f"  Drift share        : {summary.get('drift_share', 0.0):.1%}")
    if detected:
        print("\n  Consider scheduling a model retraining run.")
    print("=" * 60 + "\n")


def main(serve: bool = False, interval: int = 300, port: int = 8001):
    from monitoring.drift.reference import load_reference_data

    reference_df = load_reference_data(sample_size=2000)

    if serve and PROMETHEUS_AVAILABLE:
        start_http_server(port)
        print(f"Drift metrics exposed at http://localhost:{port}/metrics")

    while True:
        try:
            production_df = load_production_data(limit=1000)
            summary = run_drift_report(reference_df, production_df)
            update_prometheus(summary)
            print_summary(summary)
        except ValueError as ve:
            print(f"[INFO] {ve}")
        except Exception as e:
            import traceback
            print(f"[ERROR] Drift detection failed: {e}")
            traceback.print_exc()

        if not serve:
            break

        print(f"Next drift check in {interval}s... (Ctrl+C to stop)")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealGuard Drift Detector")
    parser.add_argument("--serve", action="store_true",
                        help="Run in periodic loop mode")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between drift checks in serve mode (default: 300)")
    parser.add_argument("--port", type=int, default=8001,
                        help="Prometheus /metrics port (default: 8001)")
    args = parser.parse_args()
    main(serve=args.serve, interval=args.interval, port=args.port)
