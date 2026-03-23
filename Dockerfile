# python:3.11-slim — correct base for a TensorFlow project.
# For GPU-accelerated training use nvidia/cuda:12.3-runtime-ubuntu22.04
# and install Python + requirements on top of it instead.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Suppress TF informational C++ logs (keep warnings/errors)
    TF_CPP_MIN_LOG_LEVEL=2 \
    # NVIDIA Container Toolkit: expose all GPUs at runtime when available
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default: run Streamlit dashboard (override CMD for training/preprocessing)
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
