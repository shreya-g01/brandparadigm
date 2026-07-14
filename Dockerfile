FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY brandparadigm ./brandparadigm
COPY configs ./configs
RUN pip install -e .

# api/ and dashboard/ are added once they exist (Phase 5/6); the final
# service CMD is set in Phase 7 alongside the deployment guide.
