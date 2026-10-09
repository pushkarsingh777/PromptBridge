FROM python:3.11-slim

WORKDIR /app

# Upgrade pip first (cheap, separate cacheable layer)
RUN pip install --no-cache-dir --upgrade pip

# CPU-only PyTorch — biggest download, isolated layer so it caches independently
RUN pip install --no-cache-dir \
    --timeout 300 \
    --retries 3 \
    --index-url https://download.pytorch.org/whl/cpu \
    "numpy==1.26.4" "torch==2.5.1"

# All other dependencies (including prometheus-fastapi-instrumentator inline)
COPY requirements-api.txt .
RUN pip install --no-cache-dir \
    --timeout 120 \
    --retries 3 \
    -r requirements-api.txt \
    prometheus-fastapi-instrumentator

COPY rag_service/ .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
