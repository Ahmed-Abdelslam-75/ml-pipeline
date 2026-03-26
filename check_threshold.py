import sys, os
import mlflow

THRESHOLD = 0.85

with open("model_info.txt") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(tracking_uri)

client = mlflow.tracking.MlflowClient()
run_data = client.get_run(run_id)
accuracy = run_data.data.metrics.get("accuracy")

print(f"Recorded accuracy : {accuracy:.4f}")
print(f"Required threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAIL: accuracy {accuracy:.4f} is below the {THRESHOLD} threshold.")
    sys.exit(1)

print(f"PASS: accuracy {accuracy:.4f} meets the threshold. Proceeding to deploy.")