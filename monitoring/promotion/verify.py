"""
monitoring/promotion/verify.py - API health, version, and smoke-test checks.

All functions return (success: bool, detail: str | dict).
"""
import json
import subprocess
import time
import urllib.error
import urllib.request

from monitoring.promotion import config


# --------------------------------------------------------------------------- #
#  Health check                                                                #
# --------------------------------------------------------------------------- #

def check_health():
    """Return (True, response_dict) on HTTP 200 + {status:healthy}, else (False, reason)."""
    url = config.API_BASE_URL + config.HEALTH_PATH
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return False, "HTTP " + str(resp.status)
            data = json.loads(resp.read().decode())
            if not isinstance(data, dict):
                return False, "Response is not a JSON object"
            if data.get("status") != "healthy":
                return False, "Unexpected health body: " + str(data)
            return True, data
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


# --------------------------------------------------------------------------- #
#  Model-version check                                                         #
# --------------------------------------------------------------------------- #

def check_model_version(expected_version):
    """Confirm the API is serving exactly expected_version via /model-info."""
    url = config.API_BASE_URL + config.MODEL_INFO_PATH
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            actual = str(data.get("model_version", ""))
            if actual != str(expected_version):
                return False, "Expected v" + str(expected_version) + " but API has v" + actual
            if data.get("status") != "loaded":
                return False, "Model not marked as loaded: " + str(data)
            return True, data
    except Exception as exc:
        return False, str(exc)


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

# Synthetic test transaction (V1-V28 from creditcard.csv legitimate sample)
_SMOKE_PAYLOAD = {
    "Time": 406.0,
    "V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914,
    "V4": 1.37815522427443, "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507, "V9": 0.363786969611213,
    "V10": 0.0907941719789316, "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879, "V15": 1.46817697209427,
    "V16": -0.470400525259478, "V17": 0.207971241929242, "V18": 0.0257905801458869,
    "V19": 0.403992960255733, "V20": 0.251412098239705, "V21": -0.018306777944153,
    "V22": 0.277837575558899, "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824, "V27": 0.133558376740387,
    "V28": -0.0210530534538215, "Amount": 149.62
}

def run_smoke_test(expected_version):
    """
    Run a 3-point smoke test:
      1. Health check
      2. Model-version check
      3. Valid prediction returns HTTP 200 with prediction field
    Returns (all_passed: bool, results: list[dict])
    """
    results = []

    # 1. Health
    ok, detail = check_health()
    results.append({"test": "health", "passed": ok, "detail": str(detail)})

    # 2. Model version
    ok, detail = check_model_version(expected_version)
    results.append({"test": "model_version", "passed": ok, "detail": str(detail)})

    # 3. Prediction
    url  = config.API_BASE_URL + config.PREDICT_PATH
    body = json.dumps(_SMOKE_PAYLOAD).encode()
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            ok   = "prediction" in data
            results.append({
                "test":   "prediction",
                "passed": ok,
                "detail": data if ok else "Missing 'prediction' field: " + str(data)
            })
    except Exception as exc:
        results.append({"test": "prediction", "passed": False, "detail": str(exc)})

    # 4. Invalid request (missing fields) => expect 4xx
    try:
        bad  = json.dumps({"Time": 0}).encode()
        req  = urllib.request.Request(
            url, data=bad,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5):
            results.append({"test": "invalid_request", "passed": False,
                            "detail": "Expected 4xx but got 2xx"})
    except urllib.error.HTTPError as exc:
        ok = 400 <= exc.code < 500
        results.append({"test": "invalid_request", "passed": ok,
                        "detail": "HTTP " + str(exc.code)})
    except Exception as exc:
        results.append({"test": "invalid_request", "passed": False, "detail": str(exc)})

    all_passed = all(r["passed"] for r in results)
    return all_passed, results


# --------------------------------------------------------------------------- #
#  Docker restart + wait                                                       #
# --------------------------------------------------------------------------- #

def restart_api():
    """Restart the FastAPI Compose service."""
    result = subprocess.run(
        ["docker", "compose", "restart", config.API_SERVICE],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError("docker compose restart failed:\n" + result.stderr)


def wait_for_health():
    """Poll /health until healthy or timeout. Returns True on success."""
    deadline = time.monotonic() + config.HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        ok, detail = check_health()
        if ok:
            print("  [health] OK: " + str(detail))
            return True
        print("  [health] waiting... " + str(detail))
        time.sleep(config.HEALTH_INTERVAL)
    return False
