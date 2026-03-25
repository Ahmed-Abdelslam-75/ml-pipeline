# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile – Lightweight inference container for the trained model
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# The MLflow Run ID is injected at build time so the container knows which
# model artifact to pull from the registry.
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

# Install only what the inference service needs
RUN pip install --no-cache-dir mlflow scikit-learn

# Simulate downloading the model from MLflow.
# Replace the echo with a real `mlflow artifacts download` when you have a
# publicly reachable tracking server.
RUN echo "Downloading model for MLflow Run ID: ${RUN_ID}" && \
    mkdir -p /app/model && \
    echo "${RUN_ID}" > /app/model/run_id.txt

COPY . /app

EXPOSE 8000

# Default command – in production this would start a Flask / FastAPI server.
CMD echo "Serving model from Run ID: ${RUN_ID}" && \
    python -c "print('Inference server ready.')"