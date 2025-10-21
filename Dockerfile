# ---- Base image ----
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set work directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Uvicorn and Gunicorn (for production serving)
RUN pip install --no-cache-dir gunicorn uvicorn[standard] gdown

# Copy application code
COPY api /app/api
COPY download_files.py /app/download_files.py

# Download model/data during build (optional; can be moved to runtime if large)
RUN python download_files.py || echo "Download script failed — continuing build"

# Expose Hugging Face default port
EXPOSE 7860

# Command to start FastAPI with Gunicorn + Uvicorn
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "api.main:app"]
