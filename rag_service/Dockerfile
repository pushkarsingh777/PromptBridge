FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# PromptBridge API dependencies
RUN pip install --no-cache-dir -r requirements-api.txt \
    && pip install --no-cache-dir prometheus-fastapi-instrumentator

COPY rag_service/ .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]