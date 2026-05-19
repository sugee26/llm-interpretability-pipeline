# Hugging Face Spaces Dockerfile for llm-interpretability-pipeline
# Runs the FastAPI demo from examples/serve_demo.py on port 7860 (HF default).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface

WORKDIR /app

# System deps kept minimal: torch wheels include the right CPU runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt fastapi uvicorn pydantic

COPY . .

# HF Spaces listens on 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python3", "examples/serve_demo.py"]
