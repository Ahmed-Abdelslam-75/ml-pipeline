FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

RUN pip install --no-cache-dir mlflow scikit-learn

RUN echo "Downloading model for MLflow Run ID: ${RUN_ID}" && \
    mkdir -p /app/model && \
    echo "${RUN_ID}" > /app/model/run_id.txt

COPY . /app

EXPOSE 8000

CMD echo "Serving model from Run ID: ${RUN_ID}" && \
    python -c "print('Inference server ready.')"