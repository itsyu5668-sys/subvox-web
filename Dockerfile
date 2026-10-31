# Subvox Web — Docker image for Render (free plan to start) / HF Spaces.
# Layout: /app is SUBVOX_LOCAL_ROOT, so
#   /app/models            <- weights downloaded at build time (1.55 GB)
#   /app/src/subvox_local  <- vendored package (ROOT resolves to /app)
#   /app/lipflow-upstream  <- vendored ESPnet beam search + LipReader
# Models are baked into the image so the same image works when the plan
# is upgraded — no rebuild needed to go from free to Standard/Pro.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SUBVOX_LOCAL_ROOT=/app

# curl: model download at build. libgl1/libglib2.0-0: mediapipe + opencv.
# build-essential/cmake: fallback if llama-cpp-python has no wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libglib2.0-0 build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# llama-cpp-python ships NO wheels on PyPI (sdist-only) — a source build
# is slow and OOM-prone on small builders. Use the project's own prebuilt
# manylinux wheel from its GitHub release instead (same version, 0.3.36).
RUN pip install --no-cache-dir \
    https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/llama_cpp_python-0.3.36-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY lipflow-upstream ./lipflow-upstream
COPY frontend ./frontend
COPY scripts ./scripts

# Required web-only weights (~1.55 GB). Public URLs, no token needed.
RUN bash scripts/download-models-web.sh

# Lipflow reads models from <lipflow-upstream>/models; the loader creates
# this symlink at runtime, pre-create it so /doctor sees it immediately.
RUN ln -s /app/models /app/lipflow-upstream/models

EXPOSE 8000
# Render injects $PORT; default 8000 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
