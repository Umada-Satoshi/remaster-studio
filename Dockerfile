FROM python:3.12-slim AS base

LABEL maintainer="Remaster Studio"
LABEL description="High-quality audio remastering tool with AI auto-optimization + Numba JIT"

# ── System deps ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────
WORKDIR /app

# requirements.txt を先にコピーしてレイヤーキャッシュ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────
COPY app.py .
COPY dsp_optimized.py .
COPY remaster.py .

# ── Data directories ─────────────────────────────────────────────
RUN mkdir -p /app/data/uploads /app/data/outputs

# ── Numba JIT キャッシュディレクトリ ────────────────────────────
ENV NUMBA_CACHE_DIR=/app/.numba_cache
RUN mkdir -p /app/.numba_cache

# ── Environment ──────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV REMASTER_DATA_DIR=/app/data
ENV PYTHONPATH=/app

# ── Port ─────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# ── Startup ──────────────────────────────────────────────────────
# gunicorn: worker timeout 300s (large file processing)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "2", "--timeout", "300", "--access-logfile", "-", "app:app"]
