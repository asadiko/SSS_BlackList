# ─── Blacklist face-recognition microservice ──────────────────────────────────
# CUDA base so the GPU is used when present; runs unchanged on CPU-only hosts
# (onnxruntime falls back to the CPU execution provider automatically).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FACEWEIGHTS=/opt/faceweights

# python3.11 from deadsnakes (stock 22.04 ships an RC build); build-essential
# is needed to compile the insightface extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
        build-essential libglib2.0-0 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python3.11 -m ensurepip --upgrade && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Hard deps first (cache-friendly).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Preferred backend — best-effort: if it fails to build, the service falls
# back to the OpenCV (YuNet+SFace) backend baked below.
RUN pip install "insightface==0.7.3" \
    || echo "WARNING: insightface unavailable — OpenCV fallback will be used"

# Bake model weights so the container needs NO network at runtime.
COPY docker/predownload.py docker/predownload.py
RUN python docker/predownload.py && chmod -R a+rX "$FACEWEIGHTS"

COPY . .

# Non-root runtime user; entrypoint fixes bind-mount ownership then drops root.
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/docker/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

EXPOSE 8100
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "main.py"]
