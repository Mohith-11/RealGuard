# RealGuard — Complete MLOps Project Documentation

> **Credit Card Fraud Detection Platform**  
> End-to-end MLOps pipeline: data → training → serving → monitoring → retraining → promotion

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Phase-by-Phase Implementation](#5-phase-by-phase-implementation)
   - [Phase 1–3: Data & Baseline Models](#phase-13-data--baseline-models)
   - [Phase 4–6: MLflow Tracking & Registry](#phase-46-mlflow-tracking--registry)
   - [Phase 7–9: FastAPI Serving & Docker](#phase-79-fastapi-serving--docker)
   - [Phase 10–12: Kafka Streaming Pipeline](#phase-1012-kafka-streaming-pipeline)
   - [Phase 13: Data Drift Monitoring](#phase-13-data-drift-monitoring)
   - [Phase 14: Model Performance Monitoring](#phase-14-model-performance-monitoring)
   - [Phase 15: Automated Model Retraining](#phase-15-automated-model-retraining)
   - [Phase 16: Controlled Promotion & Rollback](#phase-16-controlled-promotion--rollback)
6. [Running the System](#6-running-the-system)
7. [Service Endpoints](#7-service-endpoints)
8. [Model Performance](#8-model-performance)
9. [Operational Runbook](#9-operational-runbook)

---

## 1. Project Overview

RealGuard is a production-grade MLOps platform for real-time credit card fraud detection. It demonstrates the complete machine learning lifecycle:

| Concern | Implementation |
|---|---|
| Data | Kaggle `creditcard.csv` (284,807 transactions, 492 fraud) |
| Training | XGBoost with `scale_pos_weight` for class imbalance |
| Experiment tracking | MLflow (SQLite backend, local artifact store) |
| Model registry | MLflow Model Registry with versioned aliases |
| Serving | FastAPI + uvicorn inside Docker |
| Streaming | Apache Kafka (transaction simulation + consumption) |
| Drift detection | PSI + KS-test on feature distributions |
| Performance monitoring | Precision / Recall / F1 / PR-AUC with Prometheus + Grafana |
| Retraining | Automated weekly XGBoost retraining with validation gates |
| Promotion | Controlled alias switch + health check + smoke test + auto-rollback |

---

## 2. Architecture

```
creditcard.csv
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│                        TRAINING LAYER                              │
│  ml/train.py  ──────────────────────────────────────────────────  │
│    • XGBoost (n=300, depth=6, lr=0.1, scale_pos_weight=602)       │
│    • Logs params / metrics / model to MLflow                       │
│    • Registers as RealGuard-FraudDetector in MLflow Registry       │
│    • Tags Production alias on best version                         │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                       STREAMING LAYER                              │
│  kafka/simulator.py  →  Kafka topic "transactions"                │
│  kafka/consumer.py   ←  consumes, calls /predict, saves to PG     │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                         SERVING LAYER                              │
│  Docker: realguard-api (FastAPI + uvicorn, port 8000)             │
│    GET  /health       → {status: healthy}                          │
│    GET  /model-info   → {model_version, model_alias, status}       │
│    POST /predict      → {prediction, label, fraud_probability}     │
│    GET  /metrics      → Prometheus text format                     │
└────────────────────────────────────────────────────────────────────┘
                              │
                  ┌───────────┴────────────┐
                  ▼                        ▼
┌───────────────────────┐    ┌──────────────────────────────────────┐
│     PostgreSQL        │    │         MONITORING LAYER              │
│  Table: predictions   │    │                                       │
│    - transaction data │    │  monitoring/drift/detector.py         │
│    - prediction       │    │    PSI + KS-test → Prometheus         │
│    - fraud_prob       │    │                                       │
│    - actual_label     │    │  monitoring/performance/evaluator.py  │
│    - labeled_at       │    │    P/R/F1/PR-AUC → Prometheus :8002  │
└───────────────────────┘    │                                       │
                              │  Prometheus :9090 → Grafana :3000    │
                              └──────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                      RETRAINING LAYER                              │
│  monitoring/retraining/retrainer.py  (scheduled: Sunday 2AM)      │
│    • Loads baseline train.pkl (198k rows)                          │
│    • Supplements with verified PostgreSQL labels                   │
│    • Trains XGBoost candidate                                      │
│    • Validates: Recall≥0.75, P/R non-degradation, PR-AUC≥0.70    │
│    • Registers in MLflow if PASS; rejects if FAIL                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      PROMOTION LAYER                               │
│  monitoring/promotion/promoter.py  (manual --approve required)    │
│    1. Require --approve flag                                       │
│    2. Record current Production version                            │
│    3. Set MLflow Production alias to candidate                     │
│    4. Restart API container                                        │
│    5. Poll /health until up (120s timeout)                         │
│    6. Verify /model-info shows correct version                     │
│    7. Run 4-point smoke test                                       │
│    8. On any failure → auto-rollback to previous version           │
│    9. Save JSON deployment report                                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.10 (host), 3.11 (container) |
| ML framework | XGBoost | 3.2.0 |
| Experiment tracking | MLflow | 3.14.0 |
| API framework | FastAPI + uvicorn | 0.139.2 / 0.51.0 |
| Database | PostgreSQL | 16 |
| Message broker | Apache Kafka | KRaft mode (no ZooKeeper) |
| Containerisation | Docker + Compose | — |
| Metrics | Prometheus | — |
| Dashboards | Grafana | — |
| Drift detection | SciPy (KS-test), custom PSI | — |
| Data processing | pandas, numpy, scikit-learn | 2.3.3 / 2.2.6 / 1.7.2 |
| Task scheduler | Windows Task Scheduler | — |

---

## 4. Project Structure

```
RealGuard/
├── app/                          # FastAPI inference service
│   ├── config.py                 # Env-driven config (MLflow URI, DB creds)
│   ├── database.py               # PostgreSQL connection + predictions table
│   ├── main.py                   # FastAPI app: /health /model-info /predict /metrics
│   ├── metrics.py                # Prometheus counters and histograms
│   ├── predictor.py              # Model loading from mlruns + inference
│   └── schemas.py                # Pydantic Transaction schema (30 fields)
│
├── kafka/
│   ├── producer.py               # Base Kafka producer helper
│   ├── simulator.py              # Reads creditcard.csv, publishes to Kafka
│   └── consumer.py               # Consumes Kafka, calls /predict, stores in PG
│
├── ml/
│   ├── train.py                  # XGBoost training + MLflow logging + registration
│   ├── evaluate.py               # Holdout evaluation, classification report
│   ├── predict.py                # Offline batch prediction helper
│   └── preprocessing.py          # StandardScaler fit/transform
│
├── monitoring/
│   ├── prometheus.yml            # Prometheus scrape config (API + evaluator)
│   ├── drift/
│   │   ├── detector.py           # PSI + KS-test drift detector
│   │   └── reference.py          # Reference distribution builder
│   ├── performance/
│   │   ├── evaluator.py          # P/R/F1/PR-AUC evaluator + Prometheus server
│   │   └── label_transaction.py  # CLI: assign actual_label to a prediction
│   ├── retraining/
│   │   ├── config.py             # XGBoost hyperparams + validation thresholds
│   │   └── retrainer.py          # Full retraining pipeline (7 steps)
│   └── promotion/
│       ├── config.py             # Promotion settings (API URL, service name)
│       ├── verify.py             # Health check, version check, smoke test
│       └── promoter.py           # Promotion controller with auto-rollback
│
├── data/models/
│   ├── best_model.pkl            # Saved XGBoost production model
│   ├── train.pkl                 # Training split (198,608 rows)
│   ├── valid.pkl                 # Validation split (42,559 rows)
│   └── test.pkl                  # Test split
│
├── ml/mlruns/                    # MLflow artifact store
├── ml/mlflow.db                  # MLflow SQLite tracking DB
│
├── Dockerfile                    # Lean API image (requirements.api.txt)
├── docker-compose.yml            # All services: api, mlflow, kafka, pg, prometheus, grafana
├── requirements.txt              # Full project deps (training + monitoring)
└── requirements.api.txt          # Minimal API runtime deps (9 packages)
```

---

## 5. Phase-by-Phase Implementation

---

### Phase 1–3: Data & Baseline Models

**Goal**: Establish the dataset, preprocessing, and baseline models.

**Dataset**: `creditcard.csv` — 284,807 transactions, 492 fraud (0.17% positive rate).  
Features: `Time`, `V1`–`V28` (PCA-anonymised), `Amount`, `Class`.

**Preprocessing** (`ml/preprocessing.py`):
- `StandardScaler` fit on `Time` and `Amount`
- Stratified 70/15/15 train/valid/test split
- Saved to `data/models/{train,valid,test}.pkl`

**Models trained**:
| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.86 | 0.70 | 0.77 | — |
| Random Forest | 0.93 | 0.77 | 0.84 | — |
| **XGBoost** | **0.95** | **0.79** | **0.84** | **0.844** |

XGBoost selected as production model.

---

### Phase 4–6: MLflow Tracking & Registry

**Goal**: Track experiments, register the production model, enable versioned aliases.

**Key decisions**:
- Backend store: `sqlite:///ml/mlflow.db` (single-file, no external DB needed)
- Artifact store: `ml/mlruns/` (local filesystem)
- Model name: `RealGuard-FraudDetector`
- Production alias: `Production` (not the deprecated `Staging/Production` stages)

**MLflow tracking** (`ml/train.py`):
```python
mlflow.log_params({n_estimators, max_depth, learning_rate, subsample, ...})
mlflow.log_metrics({accuracy, precision, recall, f1, pr_auc})
mlflow.xgboost.log_model(model, name="RealGuard-FraudDetector")
mlflow.register_model(model_uri, "RealGuard-FraudDetector")
client.set_registered_model_alias("RealGuard-FraudDetector", "Production", version)
```

**MLflow UI**: `http://localhost:5000`

**Model versions at project end**:
| Version | Alias | Source |
|---|---|---|
| v1 | — | Original training run |
| v2 | **Production** | Re-trained with tuned hyperparams |
| v3 | — | Phase 15 retraining (broken artifact path) |
| v4 | — | Tested via Phase 16 promotion (passes all checks) |

---

### Phase 7–9: FastAPI Serving & Docker

**Goal**: Serve the Production model via HTTP, containerised in Docker.

**API endpoints** (`app/main.py`):

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health ping |
| GET | `/health` | `{"status": "healthy"}` |
| GET | `/model-info` | `{model_name, model_alias, model_version, status}` |
| POST | `/predict` | `{prediction, label, fraud_probability, model, version, timestamp}` |
| GET | `/metrics` | Prometheus metrics (proxied from `prometheus_client`) |

**Request schema** (`app/schemas.py`):
```json
{
  "Time": 406.0,
  "V1": -1.36, ..., "V28": -0.02,
  "Amount": 149.62
}
```

**Response**:
```json
{
  "prediction": 0,
  "label": "Legitimate",
  "fraud_probability": 0.000012,
  "model": "RealGuard-FraudDetector",
  "version": "2",
  "timestamp": "2026-09-27T10:45:00Z"
}
```

**Model loading** (`app/predictor.py`):
- Resolves `Production` alias → model source UUID → local `mlruns/1/models/<uuid>/artifacts/`
- Loads via `mlflow.xgboost.load_model(local_path)` (avoids broken artifact download)
- Exposes `MODEL_VERSION` string for `/model-info`

**Docker setup**:
- `Dockerfile`: `python:3.11-slim` + `requirements.api.txt` (9 packages, ~500MB image vs 3GB)
- `docker-compose.yml`: 7 services — `realguard-api`, `mlflow-ui`, `kafka`, `kafka-ui`, `postgres`, `prometheus`, `grafana`

**Key fix**: API container uses `MLFLOW_TRACKING_URI: http://172.18.0.3:5000` (container IP, not hostname) to bypass MLflow's DNS rebinding protection.

---

### Phase 10–12: Kafka Streaming Pipeline

**Goal**: Simulate real-time transaction stream, consume predictions, persist to PostgreSQL.

**Components**:

**Simulator** (`kafka/simulator.py`):
- Reads `creditcard.csv` row by row
- Publishes each transaction as JSON to Kafka topic `transactions`
- Configurable rate (default: ~1 transaction/second)
- Loops through the dataset continuously

**Consumer** (`kafka/consumer.py`):
- Subscribes to `transactions` topic
- For each message:
  1. Calls `POST http://localhost:8000/predict`
  2. Stores result in PostgreSQL `predictions` table
  3. Stores original `Class` column as `actual_label` (ground truth)

**PostgreSQL schema** (`predictions` table):
```sql
CREATE TABLE predictions (
    id                  SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(64),
    transaction_features JSONB,
    prediction          INTEGER,          -- 0 or 1
    fraud_probability   FLOAT,
    model_version       VARCHAR(32),
    predicted_at        TIMESTAMPTZ DEFAULT NOW(),
    actual_label        INTEGER,          -- ground truth from dataset
    labeled_at          TIMESTAMPTZ
);
```

**Prometheus metrics** exposed by API:
| Metric | Type | Description |
|---|---|---|
| `prediction_requests_total` | Counter | Total predictions served |
| `fraud_predictions_total` | Counter | Fraud predictions |
| `legitimate_predictions_total` | Counter | Legitimate predictions |
| `prediction_latency_seconds` | Histogram | Inference latency |

---

### Phase 13: Data Drift Monitoring

**Goal**: Detect when incoming transaction distributions shift from the training baseline.

**Methods**:
- **PSI** (Population Stability Index): `PSI > 0.2` = significant drift
- **KS-test** (Kolmogorov-Smirnov): `p < 0.05` = statistically significant shift

**Reference distribution** (`monitoring/drift/reference.py`):
- Built from `data/models/train.pkl`
- Saved as `monitoring/drift/reference_stats.json`
- Contains per-feature mean, std, min, max, percentiles

**Detector** (`monitoring/drift/detector.py`):
- Compares recent predictions window (last N rows from PG) against reference
- Reports per-feature PSI and KS-test p-values
- Publishes drift flag to Prometheus

**Grafana panels**: Feature drift heatmap, PSI gauge per feature, drift alert threshold line.

---

### Phase 14: Model Performance Monitoring

**Goal**: Continuously measure actual model quality using ground-truth labels from the dataset.

**Label collection**:
- Consumer stores `Class` from `creditcard.csv` directly as `actual_label` on each prediction
- `labeled_at` timestamp recorded for windowed evaluation

**Evaluator** (`monitoring/performance/evaluator.py`):
- Runs on a configurable schedule (default: every 60 seconds via `--serve`)
- Queries PostgreSQL for records where `actual_label IS NOT NULL`
- Computes: Accuracy, Precision, Recall, F1, PR-AUC, Confusion Matrix
- **Reliability gating**: requires minimum fraud samples before reporting recall as reliable
- Saves JSON report to `monitoring/performance/reports/`
- Exposes Prometheus metrics at `:8002`

**Prometheus metrics**:
| Metric | Description |
|---|---|
| `model_accuracy` | Accuracy on labeled window |
| `model_precision` | Precision |
| `model_recall` | Recall |
| `model_f1_score` | F1 score |
| `model_pr_auc` | PR-AUC |
| `model_labeled_samples` | Total labeled samples seen |
| `model_metrics_reliable` | 1 if enough fraud samples exist, 0 otherwise |
| `model_performance_reports_total` | Evaluation runs count |

**Sample report output**:
```
======================================================
  MODEL PERFORMANCE SUMMARY
======================================================
  Timestamp    : 2026-09-27T08:51:57Z
  Samples      : 226 (1 fraud)
  Accuracy     : 0.9996
  Precision    : 0.9649
  Recall       : 0.7746
  F1-Score     : 0.8594
  PR-AUC       : 0.8456
  Confusion    : TN=225 FP=0 FN=0 TP=1
======================================================
```

**Running**:
```powershell
$env:PERF_INTERVAL="60"
python -m monitoring.performance.evaluator --serve
```

**Grafana dashboard**: `RealGuard Model Performance` at `http://localhost:3000`

---

### Phase 15: Automated Model Retraining

**Goal**: Automatically retrain XGBoost using accumulated verified labels, validate the candidate, and register in MLflow if it passes all gates.

**Configuration** (`monitoring/retraining/config.py`):
```python
MIN_RECALL      = 0.75     # Candidate must achieve at least 0.75 recall
MAX_DEGRADATION = 0.02     # Candidate may degrade at most 2% vs production
MIN_PR_AUC      = 0.70     # Minimum PR-AUC to register
```

**XGBoost hyperparameters** (matching original training):
```python
n_estimators=300, max_depth=6, learning_rate=0.1,
subsample=0.8, colsample_bytree=0.8,
scale_pos_weight=computed_from_data, random_state=42
```

**Retraining pipeline** (`monitoring/retraining/retrainer.py`):

| Step | Action |
|---|---|
| 1 | Load `data/models/train.pkl` as baseline (198,608 rows, 331 fraud) |
| 2 | Query PostgreSQL for verified labels; merge if sufficient |
| 3 | Compute `scale_pos_weight` from merged dataset |
| 4 | Train XGBoost candidate |
| 5 | Evaluate candidate on fixed `data/models/valid.pkl` holdout |
| 6 | Load Production model; evaluate on same holdout |
| 7 | Compare metrics; validate against gates; register if `PASS` |

**Validation gates** (all must pass):
```
[PASS] recall_minimum           candidate_recall >= 0.75
[PASS] recall_non_degradation   candidate_recall >= production_recall - 0.02
[PASS] precision_non_degradation candidate_precision >= production_precision - 0.02
[PASS] pr_auc_minimum           candidate_pr_auc >= 0.70
```

**Sample dry-run output**:
```
[5/7] Candidate:   Precision=0.9649  Recall=0.7746  F1=0.8594  PR-AUC=0.8606
[6/7] Production:  Precision=0.9483  Recall=0.7746  F1=0.8527  PR-AUC=0.8440
  Validation: PASSED
```

**Scheduling** (Windows Task Scheduler):
```
Task Name : RealGuard-Retraining
Trigger   : Every Sunday at 2:00 AM
Command   : python -m monitoring.retraining.retrainer
```

**Running manually**:
```powershell
# Dry run (no MLflow registration)
python -m monitoring.retraining.retrainer --no-register

# Full run with registration
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:PYTHONIOENCODING="utf-8"
python -m monitoring.retraining.retrainer
```

---

### Phase 16: Controlled Promotion & Rollback

**Goal**: Safely promote a validated candidate to Production with automatic rollback on failure.

**Configuration** (`monitoring/promotion/config.py`):
```python
MODEL_NAME        = "RealGuard-FraudDetector"
MLFLOW_TRACKING_URI = "http://localhost:5000"
PRODUCTION_ALIAS  = "Production"
API_BASE_URL      = "http://localhost:8000"
API_SERVICE       = "realguard-api"
HEALTH_TIMEOUT    = 120   # seconds
```

**Verification module** (`monitoring/promotion/verify.py`):
- `check_health()` — polls `/health`, requires `{"status": "healthy"}`
- `check_model_version(v)` — polls `/model-info`, requires `model_version == v`
- `run_smoke_test(v)` — 4-point test suite:
  1. Health endpoint → HTTP 200
  2. Model version → correct version loaded
  3. Valid prediction → HTTP 200, `prediction` field present
  4. Invalid request (missing fields) → HTTP 422

**Promotion controller** (`monitoring/promotion/promoter.py`):

```
Requires: --approve flag (safety gate)

Step 1: Confirm candidate is READY in MLflow
Step 2: Record previous Production version (for rollback)
Step 3: Set Production alias to candidate
Step 4: docker compose restart <api-service>
Step 5: Poll /health every 3s for up to 120s
Step 6: Call check_model_version(candidate)
Step 7: Run 4-point smoke test
Step 8: On any failure → restore old alias, restart, re-verify
Step 9: Save JSON report to monitoring/promotion/reports/
```

**Usage**:
```powershell
# Check without changing anything
python -m monitoring.promotion.promoter --candidate 4
# Output: "Promotion not started. Review the candidate, then re-run with --approve."

# Full promotion
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:PYTHONIOENCODING="utf-8"
python -m monitoring.promotion.promoter --candidate 4 --approve
```

**Promotion output (successful)**:
```
============================================================
  PROMOTION: v2 -> v4
============================================================
[1/5] Setting Production alias to v4...
[2/5] Restarting API container (realguard-api)...
[3/5] Waiting for API health...
  [health] OK: {'status': 'healthy'}
[4/5] Verifying loaded model version...
  [version] OK: {'model_version': '4', 'status': 'loaded'}
[5/5] Running smoke test...
  [PASS] health
  [PASS] model_version
  [PASS] prediction: {prediction: 0, fraud_probability: 1.2e-05}
  [PASS] invalid_request: HTTP 422
[SUCCESS] v4 is live and verified.
Report saved: monitoring/promotion/reports/promotion_20260927_104516.json
```

**Manual rollback**:
```powershell
# Set alias back
python -c "
from mlflow import MlflowClient
c = MlflowClient()
c.set_registered_model_alias('RealGuard-FraudDetector', 'Production', '2')
"
# Restart API
docker compose restart realguard-api
# Verify
Invoke-RestMethod http://localhost:8000/model-info
```

**Phase 16 acceptance results**:

| Check | Result |
|---|---|
| `--approve` required | ✅ `exit 2` without it |
| Candidate READY in MLflow | ✅ |
| Previous version recorded in report | ✅ |
| Container restarts after alias change | ✅ |
| `/health` passes after restart | ✅ |
| `/model-info` shows correct version | ✅ |
| Smoke test passes (4/4) | ✅ |
| Auto-rollback demonstrated | ✅ (v4 artifact missing → restored v2) |
| Manual rollback v4→v2 verified | ✅ |
| JSON report saved | ✅ |

---

## 6. Running the System

### Prerequisites
- Docker Desktop running
- PowerShell (Windows)
- Python 3.10 venv at `D:\mlops\RealGuard\venv\`

### Start all infrastructure
```powershell
cd D:\mlops\RealGuard
.\venv\Scripts\Activate.ps1
docker compose up -d
```

### Start the streaming pipeline
```powershell
# Terminal 1 - Simulator
python kafka/simulator.py

# Terminal 2 - Consumer
python kafka/consumer.py
```

### Start the performance evaluator
```powershell
$env:PERF_INTERVAL="60"
python -m monitoring.performance.evaluator --serve
```

### Run retraining (manual)
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:PYTHONIOENCODING="utf-8"
python -m monitoring.retraining.retrainer --no-register   # dry run
python -m monitoring.retraining.retrainer                  # register if passes
```

### Promote a candidate
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:PYTHONIOENCODING="utf-8"
$env:API_SERVICE="realguard-api"
python -m monitoring.promotion.promoter --candidate <version> --approve
```

---

## 7. Service Endpoints

| Service | URL | Notes |
|---|---|---|
| **FastAPI** | `http://localhost:8000` | Fraud prediction API |
| **FastAPI docs** | `http://localhost:8000/docs` | Swagger UI |
| **MLflow UI** | `http://localhost:5000` | Experiment tracking + model registry |
| **Kafka UI** | `http://localhost:8080` | Topic browser, consumer groups |
| **Prometheus** | `http://localhost:9090` | Metrics storage + query |
| **Grafana** | `http://localhost:3000` | Dashboards (admin/admin123) |
| **PostgreSQL** | `localhost:5432` | DB: `frauddb`, user: `realguard` |
| **Perf Evaluator** | `http://localhost:8002/metrics` | Model performance Prometheus endpoint |

---

## 8. Model Performance

### Production model (v2 — XGBoost)

Evaluated on `valid.pkl` holdout (42,559 rows, 71 fraud):

| Metric | Value |
|---|---|
| Accuracy | 99.96% |
| Precision | 94.83% |
| Recall | 77.46% |
| F1-Score | 85.27% |
| PR-AUC | 84.40% |
| True Negatives | 42,485 |
| False Positives | 3 |
| False Negatives | 16 |
| True Positives | 55 |

### Retraining validation thresholds

| Gate | Threshold | Rationale |
|---|---|---|
| Minimum Recall | ≥ 0.75 | Must catch 75% of fraud |
| Recall non-degradation | ≤ 2% drop vs Production | Protect recall on promotion |
| Precision non-degradation | ≤ 2% drop vs Production | Protect precision on promotion |
| Minimum PR-AUC | ≥ 0.70 | Overall ranking quality |

---

## 9. Operational Runbook

### Check system health
```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
```

### Check current Production version
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python -c "
from mlflow import MlflowClient
c = MlflowClient()
v = c.get_model_version_by_alias('RealGuard-FraudDetector','Production')
print('Production: v' + str(v.version))
"
```

### Check prediction throughput
```powershell
# Recent prediction count from PostgreSQL
python -c "
import psycopg2
conn = psycopg2.connect(host='localhost',dbname='frauddb',user='realguard',password='realguard')
cur = conn.cursor()
cur.execute('SELECT COUNT(*), SUM(CASE WHEN prediction=1 THEN 1 ELSE 0 END) FROM predictions')
total, fraud = cur.fetchone()
print(f'Total: {total}, Fraud: {fraud} ({100*fraud/total:.3f}%)')
conn.close()
"
```

### View latest retraining report
```powershell
$r = Get-ChildItem monitoring\retraining\reports\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content $r.FullName | ConvertFrom-Json | ConvertTo-Json -Depth 4
```

### View latest promotion report
```powershell
$r = Get-ChildItem monitoring\promotion\reports\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content $r.FullName | ConvertFrom-Json | ConvertTo-Json -Depth 4
```

### Trigger manual performance evaluation
```powershell
python -m monitoring.performance.evaluator
```

### Rollback to previous version
```powershell
# Replace <old_version> with version number to restore (e.g. 2)
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python -c "
from mlflow import MlflowClient
c = MlflowClient()
c.set_registered_model_alias('RealGuard-FraudDetector', 'Production', '<old_version>')
print('Alias restored')
"
docker compose restart realguard-api
Invoke-RestMethod http://localhost:8000/model-info
```

### Windows Task Scheduler — view retraining task
```powershell
Get-ScheduledTask -TaskName "RealGuard-Retraining" | Select-Object TaskName, State
(Get-ScheduledTask -TaskName "RealGuard-Retraining").Triggers
```

---

*Documentation generated: 2026-09-27 | RealGuard MLOps Project | Phases 1–16*
