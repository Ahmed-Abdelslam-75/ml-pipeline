"""
train_weak.py – Identical to train.py but deliberately hobbled so the
accuracy drops below 0.85, letting you capture a "failed pipeline" screenshot.

Usage:
  Temporarily rename this file to train.py, push, and let the Action run.
  The deploy job will fail at the threshold check.
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("iris-classifier")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── Weak model: depth=1 produces a very shallow tree ───────────────────────
N_ESTIMATORS = 100
MAX_DEPTH = None             # <── forces underfitting → accuracy < 0.85

with mlflow.start_run() as run:
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")

    run_id = run.info.run_id
    print(f"Training complete  │  accuracy={accuracy:.4f}  │  run_id={run_id}")

with open("model_info.txt", "w") as f:
    f.write(run_id)

print(f"Run ID written to model_info.txt")