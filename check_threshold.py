import sys, os

THRESHOLD = 0.85

# Read Run ID
with open("model_info.txt") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# Read accuracy from file (exported by train.py)
if not os.path.exists("accuracy.txt"):
    print("ERROR: accuracy.txt not found.")
    sys.exit(1)

with open("accuracy.txt") as f:
    accuracy = float(f.read().strip())

print(f"Recorded accuracy : {accuracy:.4f}")
print(f"Required threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAIL: accuracy {accuracy:.4f} is below the {THRESHOLD} threshold.")
    sys.exit(1)

print(f"PASS: accuracy {accuracy:.4f} meets the threshold. Proceeding to deploy.")