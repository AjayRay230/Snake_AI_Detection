# /snake-ai-api/Dockerfile

# --- STAGE 1: Build Environment ---
FROM python:3.10 as base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for efficient Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn and Uvicorn
RUN pip install --no-cache-dir gunicorn uvicorn[standard] gdown

# --- STAGE 2: Final Production Image ---
FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from base
COPY --from=base /usr/local /usr/local

# Copy API code
COPY api /app/api

# ✅ Copy the correct download script
COPY download_files.py /app/download_files.py

# --- Download model and dataset files automatically ---
RUN python download_files.py

# Expose the API port
EXPOSE 80

# Run FastAPI using Gunicorn + Uvicorn workers
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80", "api.main:app"]
