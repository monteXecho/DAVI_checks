# syntax=docker/dockerfile:1
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps:
# - libgl1 + libglib2.0-0: needed by opencv-python in many environments
# - build-essential: safe for packages that might compile (often not needed but avoids surprises)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better Docker cache)
COPY requirements.txt constraints.txt ./

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install PaddlePaddle CPU from the Paddle index (as in your docs)
# NOTE: pin exactly to 3.0.0 per your instructions
RUN python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install the rest (respect constraints.txt)
RUN python -m pip install -r requirements.txt -c constraints.txt

# Copy app code
COPY . .

EXPOSE 18895

# Production default: Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "18895"]
