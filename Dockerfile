FROM python:3.11-slim

WORKDIR /app

COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

# Symlink for any hardcoded Windows paths in MLflow artifact store
RUN mkdir -p "/C:/Users/mohit/Documents/mlops/RealGuard/ml" && \
    ln -s /app/ml/mlruns "/C:/Users/mohit/Documents/mlops/RealGuard/ml/mlruns" 2>/dev/null || true

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
