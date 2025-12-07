# Dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install environment
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# set env variables
# PYTHONUNBUFFERED=1 removes python stdout buffering
# CUBLAS_WORKSPACE_CONFIG=:16:8 \ makes operations deterministic
# ---> use :4096:8 for training
# NVIDIA_VISIBLE_DEVICES=all \ shows all gpus from the host
ENV PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:16:8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["echo", "Hello"]
