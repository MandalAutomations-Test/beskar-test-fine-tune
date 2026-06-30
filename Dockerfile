# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — QLoRA LLM fine-tuning on an RTX 5060 (Blackwell / sm_120)
#
# Build:
#   docker build -t fine-tune .
#
# Run (single GPU):
#   docker run --gpus all --rm -it \
#     -v $(pwd)/output:/app/output \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     fine-tune
#
# Override any config value:
#   docker run --gpus all --rm -it \
#     -v $(pwd)/output:/app/output \
#     fine-tune --epochs 1 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
# ──────────────────────────────────────────────────────────────────────────────

# CUDA 12.8 + cuDNN 9 on Ubuntu 22.04
# RTX 5060 (Blackwell) requires CUDA ≥ 12.8 and compute capability sm_120.
FROM nvidia/cuda:12.8.1-cudnn9-devel-ubuntu22.04

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        wget \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Upgrade pip first, then install PyTorch with CUDA 12.8 wheels, then the
# remaining dependencies from requirements.txt.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir \
        --no-deps bitsandbytes==0.45.5 \
    && pip install --no-cache-dir \
        transformers==4.51.3 \
        datasets==3.6.0 \
        tokenizers==0.21.1 \
        accelerate==1.6.0 \
        peft==0.15.2 \
        trl==0.17.0 \
        huggingface-hub==0.30.2 \
        safetensors==0.5.3 \
        pyyaml==6.0.2 \
        numpy==2.2.5 \
        scipy==1.15.3

# ── Application code ──────────────────────────────────────────────────────────
COPY finetune.py .
COPY config.yaml .

# ── Runtime environment ───────────────────────────────────────────────────────
# Allow the HuggingFace cache to be mounted at runtime.
ENV HF_HOME=/root/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
# Prevent NCCL warnings when running with a single GPU.
ENV NCCL_P2P_DISABLE=1

# ── Default command ───────────────────────────────────────────────────────────
ENTRYPOINT ["python", "finetune.py"]
# Pass extra CLI arguments after the image name, e.g.:
#   docker run ... fine-tune --epochs 1
