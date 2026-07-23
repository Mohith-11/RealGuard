FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Create symlink mapping for Windows absolute path in DB
RUN mkdir -p /C:/Users/mohit/Documents/mlops/RealGuard/ml && \
    ln -s /app/ml/mlruns /C:/Users/mohit/Documents/mlops/RealGuard/ml/mlruns

COPY . .


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
