"""
monitoring/promotion/promoter.py - Controlled MLflow model promotion with rollback.

Flow:
  1. Require --approve flag (safety gate)
  2. Record current Production version
  3. Confirm candidate is READY in MLflow
  4. Set Production alias to candidate
  5. Restart API container
  6. Wait for health
  7. Verify correct model version is serving
  8. Run smoke test
  9. On any failure: restore previous alias, restart, re-verify, save report
 10. Save JSON deployment report either way
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient

from monitoring.promotion import config
from monitoring.promotion.verify import (
    restart_api,
    wait_for_health,
    check_model_version,
    run_smoke_test,
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _save_report(report):
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    filename = "promotion_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".json"
    path = os.path.join(config.REPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Report saved: " + path)
    return path


def _get_production_version(client):
    try:
        mv = client.get_model_version_by_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS)
        return str(mv.version)
    except Exception:
        return None


def _set_alias(client, version):
    client.set_registered_model_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS, str(version))


def promote(candidate):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)

    report = {
        "model_name":        config.MODEL_NAME,
        "candidate_version": str(candidate),
        "started_at":        _utc_now(),
        "status":            "started",
        "previous_version":  None,
        "rollback_version":  None,
        "smoke_test":        None,
        "error":             None,
    }

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    old_version = _get_production_version(client)
    if old_version is None:
        raise RuntimeError(
            "No Production alias found. Cannot promote safely without a rollback target."
        )
    if str(candidate) == old_version:
        raise RuntimeError("Candidate v" + str(candidate) + " is already Production.")

    mv = client.get_model_version(config.MODEL_NAME, str(candidate))
    if mv.status != "READY":
        raise RuntimeError("Candidate is not READY (status=" + mv.status + ")")

    report["previous_version"] = old_version
    print("=" * 60)
    print("  PROMOTION: v" + str(old_version) + " -> v" + str(candidate))
    print("=" * 60)

    # ── Promotion sequence ────────────────────────────────────────────────────
    try:
        print("\n[1/5] Setting Production alias to v" + str(candidate) + "...")
        _set_alias(client, candidate)

        print("[2/5] Restarting API container (" + config.API_SERVICE + ")...")
        restart_api()

        print("[3/5] Waiting for API health...")
        if not wait_for_health():
            raise RuntimeError("API did not become healthy within " +
                               str(config.HEALTH_TIMEOUT) + "s after promotion.")

        print("[4/5] Verifying loaded model version...")
        ok, detail = check_model_version(candidate)
        if not ok:
            raise RuntimeError("Version mismatch: " + str(detail))
        print("  [version] OK: " + str(detail))

        print("[5/5] Running smoke test...")
        smoke_ok, smoke_results = run_smoke_test(candidate)
        report["smoke_test"] = smoke_results
        _print_smoke(smoke_results)
        if not smoke_ok:
            raise RuntimeError("Smoke test failed. See smoke_test in report.")

        report["status"]      = "promoted"
        report["finished_at"] = _utc_now()
        print("\n[SUCCESS] v" + str(candidate) + " is live and verified.")

    except Exception as exc:
        report["error"] = str(exc)
        print("\n[FAIL] " + str(exc))
        print("Starting rollback to v" + str(old_version) + "...")

        try:
            _set_alias(client, old_version)
            restart_api()

            if not wait_for_health():
                raise RuntimeError("API not healthy after rollback.")

            ok, detail = check_model_version(old_version)
            if not ok:
                raise RuntimeError("Rollback version mismatch: " + str(detail))

            report["status"]          = "rolled_back"
            report["rollback_version"] = old_version
            print("[ROLLBACK] v" + str(old_version) + " restored and verified.")

        except Exception as rb_exc:
            report["status"]         = "rollback_failed"
            report["rollback_error"] = str(rb_exc)
            print("[CRITICAL] Rollback failed: " + str(rb_exc))
            print("Manual recovery required:")
            print("  1. mlflow models set-alias --name " + config.MODEL_NAME +
                  " --alias Production --version " + str(old_version))
            print("  2. docker compose restart " + config.API_SERVICE)

        report["finished_at"] = _utc_now()
        _save_report(report)
        return False

    _save_report(report)
    return True


def _print_smoke(results):
    for r in results:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print("  " + status + " " + r["test"] + ": " + str(r["detail"])[:120])


def main():
    parser = argparse.ArgumentParser(description="RealGuard promotion controller")
    parser.add_argument("--candidate", required=True, help="MLflow model version to promote")
    parser.add_argument("--approve",   action="store_true",
                        help="Explicitly approve deployment (required)")
    parser.add_argument("--tracking-uri", default=None,
                        help="Override MLflow tracking URI")
    args = parser.parse_args()

    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri

    if not args.approve:
        print("Promotion not started. Review the candidate, then re-run with --approve.")
        return 2

    try:
        ok = promote(args.candidate)
        return 0 if ok else 1
    except Exception as exc:
        print("Promotion aborted: " + str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
