"""
check_threshold.py – Gate the deployment on a minimum accuracy of 0.85.

Reads the MLflow Run ID from model_info.txt, fetches the logged accuracy
metric, and exits with code 1 (failing the pipeline) if the value is
below the threshold.
"""

import sys
import os
import mlflow

THRESHOLD = 0.85

# ── Read Run ID ─────────────────────────────────────────────────────────────
info_path = "model_info.txt"
if not os.path.exists(info_path):
    print(f"ERROR: {info_path} not found.")
    sys.exit(1)

with open(info_path) as f:
    run_id = f.read().strip()

if not run_id:
    print("ERROR: model_info.txt is empty.")
    sys.exit(1)

print(f"Checking Run ID: {run_id}")

# ── Query MLflow ────────────────────────────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(tracking_uri)

client = mlflow.tracking.MlflowClient()
run_data = client.get_run(run_id)
metrics = run_data.data.metrics

accuracy = metrics.get("accuracy")
if accuracy is None:
    print("ERROR: 'accuracy' metric not found for this run.")
    sys.exit(1)

print(f"Recorded accuracy : {accuracy:.4f}")
print(f"Required threshold: {THRESHOLD}")

# ── Gate ────────────────────────────────────────────────────────────────────
if accuracy < THRESHOLD:
    print(f"FAIL: accuracy {accuracy:.4f} is below the {THRESHOLD} threshold.")
    sys.exit(1)

print(f"PASS: accuracy {accuracy:.4f} meets the threshold. Proceeding to deploy.")